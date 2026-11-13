/**
 * @file main.cpp
 * @brief RISC-V Architecture Research Platform. 
 *
 * Assembles a .s source file, executes it on a cycle-accurate pipelined
 * CPU with a configurable cache hierarchy, and reports architecture
 * metrics. All configuration comes from a .cfg file.
 *
 * @par Usage
 * @code
 *   rvsim program.s                            # Run with default config
 *   rvsim program.s --config arch.cfg          # Run with config file
 *   rvsim program.s --config sweep.cfg --csv   # Sweep with CSV output
 * @endcode
 */

#include "core/assembler.hpp"
#include "core/cache.hpp"
#include "core/config.hpp"
#include "core/pipeline.hpp"
#include "core/system.hpp"

#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace riscv;

static constexpr const char* kVersion = "0.5.0";

// ── Enum parsing ───────────────────────────────────────────────────────

static ReplacementPolicy to_replacement(const std::string& s)
{
    if (s == "mru")    return ReplacementPolicy::MRU;
    if (s == "plru")   return ReplacementPolicy::PLRU;
    if (s == "fifo")   return ReplacementPolicy::FIFO;
    if (s == "random") return ReplacementPolicy::RANDOM;
    return ReplacementPolicy::LRU;
}

static WritePolicy to_write_policy(const std::string& s)
{
    if (s == "writethrough") return WritePolicy::WRITE_THROUGH;
    return WritePolicy::WRITE_BACK;
}

static WriteAllocate to_alloc_policy(const std::string& s)
{
    if (s == "no_allocate") return WriteAllocate::NO_ALLOCATE;
    return WriteAllocate::ALLOCATE;
}

static ForwardingPolicy to_forwarding(const std::string& s)
{
    if (s == "none")    return ForwardingPolicy::NONE;
    return ForwardingPolicy::PARTIAL;
}

static BranchPredictor to_branch_pred(const std::string& s)
{
    if (s == "not_taken")      return BranchPredictor::NOT_TAKEN;
    if (s == "always_taken")   return BranchPredictor::ALWAYS_TAKEN;
    if (s == "backward_taken") return BranchPredictor::BACKWARD_TAKEN;
    if (s == "bimodal_1bit")   return BranchPredictor::BIMODAL_1BIT;
    return BranchPredictor::BIMODAL_2BIT;
}

// ── Build system ───────────────────────────────────────────────────────

using SimSystem = SingleCoreSystem<PipelinedCPU>;

static SystemConfig to_system_config(const SimConfig& cfg)
{
    SystemConfig s;
    s.platform.ram_size = u64{cfg.mem_size_kb} * 1024;

    if (cfg.l3_size > 0)
        s.l3 = {
            .size_bytes = cfg.l3_size, .line_size = cfg.l1_line,
            .associativity = cfg.l3_assoc, .hit_latency = 30, .miss_penalty = 100,
        };
    if (cfg.l2_size > 0)
        s.l2 = {
            .size_bytes = cfg.l2_size, .line_size = cfg.l1_line,
            .associativity = cfg.l2_assoc, .hit_latency = 12, .miss_penalty = 50,
        };
    if (cfg.l1i_size > 0)
        s.l1i = {
            .size_bytes = cfg.l1i_size, .line_size = cfg.l1_line,
            .associativity = cfg.l1i_assoc, .hit_latency = 4, .miss_penalty = 10,
            .replacement = to_replacement(cfg.l1i_replacement),
        };
    if (cfg.l1_size > 0)
        s.l1d = {
            .size_bytes = cfg.l1_size, .line_size = cfg.l1_line,
            .associativity = cfg.l1_assoc, .hit_latency = 4, .miss_penalty = 10,
            .replacement = to_replacement(cfg.l1_replacement),
            .write_pol = to_write_policy(cfg.l1_write),
            .write_alloc = to_alloc_policy(cfg.l1_write_alloc),
        };

    s.pipeline.forwarding             = to_forwarding(cfg.forwarding);
    s.pipeline.predictor              = to_branch_pred(cfg.branch_pred);
    s.pipeline.branch_mispred_penalty = cfg.mispredict_penalty;
    return s;
}

static void apply_mem_init(Memory& mem, const std::vector<MemInitEntry>& inits)
{
    for (auto& entry : inits)
        switch(entry.kind)
        {
            case MemInitEntry::Kind::WORDS:
                for (u32 i = 0; i < entry.words.size(); ++i)
                    (void)mem.write32(entry.address + i * 4, entry.words[i]);
                break;
            case MemInitEntry::Kind::BYTES:
                for (u32 i = 0; i < entry.bytes.size(); ++i)
                    (void)mem.write8(entry.address + i, entry.bytes[i]);
                break;
            case MemInitEntry::Kind::FILL:
                for (u32 i = 0; i < entry.fill_count; ++i)
                    (void)mem.write8(entry.address + i, entry.fill_val);
                break;
            case MemInitEntry::Kind::STRING:
                for (u32 i = 0; i < entry.str.size(); ++i)
                    (void)mem.write8(entry.address + i, static_cast<u8>(entry.str[i]));
                (void)mem.write8(entry.address + static_cast<u32>(entry.str.size()), 0);
                break;
        }
}

// ── Run ────────────────────────────────────────────────────────────────

struct RunResult
{
    cycle_t cycles = 0; u64 instructions = 0; double ipc = 0;
    u64 stalls = 0; u64 flushes = 0;
    double l1_hit_rate = 0; u64 l1_misses = 0;
    double l1i_hit_rate = 0; u64 l1i_misses = 0;
    double l2_hit_rate = 0; u64 l2_misses = 0;
    double l3_hit_rate = 0; u64 l3_misses = 0;
    u64 branches = 0; u64 mispredicts = 0; double branch_accuracy = 0;
};

static RunResult run_once(const SimConfig& cfg, const std::vector<u8>& code)
{
    SimSystem sys(to_system_config(cfg));

    // Load code
    sys.load_program(0, code);

    // Apply memory initialization from config
    apply_mem_init(sys.ram(), cfg.mem_init);

    // Set up CPU
    sys.core().set_pc(0);

    // Apply register initialization from config
    for (auto& ri : cfg.reg_init)
        sys.core().set_reg(static_cast<reg_idx_t>(ri.reg_idx), ri.value);

    // Run
    sys.run(static_cast<u64>(cfg.max_cycles));

    const auto& ps = sys.core().stats();
    RunResult r;
    r.cycles       = ps.cycles;
    r.instructions = ps.instructions_retired;
    r.ipc          = ps.cycles > 0 ? double(ps.instructions_retired) / double(ps.cycles) : 0;
    r.stalls       = ps.stalls_load_use + ps.stalls_raw + ps.stalls_control;
    r.flushes      = ps.branch_mispredicts;

    if (auto* l1 = sys.l1d())
    {
        r.l1_hit_rate = l1->stats().hit_rate();
        r.l1_misses   = l1->stats().misses;
    }
    if (auto* l1i = sys.l1i())
    {
        r.l1i_hit_rate = l1i->stats().hit_rate();
        r.l1i_misses   = l1i->stats().misses;
    }
    if (auto* l2 = sys.l2())
    {
        r.l2_hit_rate = l2->stats().hit_rate();
        r.l2_misses   = l2->stats().misses;
    }
    if (auto* l3 = sys.l3())
    {
        r.l3_hit_rate = l3->stats().hit_rate();
        r.l3_misses   = l3->stats().misses;
    }

    r.branches        = ps.branches;
    r.mispredicts     = ps.branch_mispredicts;
    r.branch_accuracy = ps.branch_accuracy();
    return r;
}

// ── CSV ────────────────────────────────────────────────────────────────

static void csv_header()
{
    std::cout << "cycles,instructions,ipc,stalls,flushes,"
                 "l1_hit_rate,l1_misses,"
                 "l1i_hit_rate,l1i_misses,"
                 "l2_hit_rate,l2_misses,"
                 "l3_hit_rate,l3_misses,"
                 "branches,mispredicts,branch_accuracy\n";
}

static void csv_row(const RunResult& r)
{
    std::cout << std::format("{},{},{:.4f},{},{},{:.4f},{},{:.4f},{},{:.4f},{},{:.4f},{},{},{},{:.4f}\n",
        r.cycles, r.instructions, r.ipc, r.stalls, r.flushes,
        r.l1_hit_rate, r.l1_misses,
        r.l1i_hit_rate, r.l1i_misses,
        r.l2_hit_rate, r.l2_misses,
        r.l3_hit_rate, r.l3_misses,
        r.branches, r.mispredicts, r.branch_accuracy);
}

// ── Sweep ──────────────────────────────────────────────────────────────

static void sweep_recursive(const std::vector<SweepAxis>& axes, u32 depth,
                            SimConfig& cfg, const std::vector<u8>& code,
                            std::vector<std::string>& vals)
{
    if (depth == static_cast<u32>(axes.size()))
    {
        // Validate this configuration's cache geometry - if invalid, skip
        auto skip = [&](const char* reason)
        {
            std::cerr << "  SKIP: " << reason << " (";
            for (u32 k = 0; k < vals.size(); ++k)
            {
                if (k) std::cerr << ", ";
                std::cerr << axes[k].param << "=" << vals[k];
            }
            std::cerr << ")\n";
        };

        if (cfg.l1_size > 0 && cfg.l2_size > 0 && cfg.l2_size <= cfg.l1_size)
        {
            skip("L2 size < L1 size"); return;
        }
        if (cfg.l1i_size > 0 && cfg.l1_size == 0)
        {
            skip("L1I requires L1"); return;
        }
        if (cfg.l1i_size > 0 && cfg.l2_size > 0 && cfg.l2_size <= cfg.l1i_size)
        {
            skip("L2 size < L1I size"); return;
        }
        if (cfg.l1i_size > 0 && (cfg.l1i_size % (cfg.l1_line * cfg.l1i_assoc)) != 0)
        {
            skip("L1I size not divisible by line*assoc"); return;
        }
        if (cfg.l2_size > 0 && cfg.l3_size > 0 && cfg.l3_size < cfg.l2_size)
        {
            skip("L3 size < L2 size"); return;
        }
        if (cfg.l1_size > 0 && (cfg.l1_size % (cfg.l1_line * cfg.l1_assoc)) != 0)
        {
            skip("L1 size not divisible by line*assoc"); return;
        }
        if (cfg.l2_size > 0 && (cfg.l2_size % (cfg.l1_line * cfg.l2_assoc)) != 0)
        {
            skip("L2 size not divisible by line*assoc"); return;
        }
        if (cfg.l3_size > 0 && (cfg.l3_size % (cfg.l1_line * cfg.l3_assoc)) != 0)
        {
            skip("L3 size not divisible by line*assoc"); return;
        }

        auto r = run_once(cfg, code);
        for (auto& v : vals) std::cout << v << ",";
        csv_row(r);
        return;
    }
    for (const auto& val : axes[depth].values)
    {
        SimConfig saved = cfg;
        apply_config_param(cfg, axes[depth].param, val);
        vals.push_back(val);
        sweep_recursive(axes, depth + 1, cfg, code, vals);
        vals.pop_back();
        cfg = saved;
    }
}

// ── File I/O ───────────────────────────────────────────────────────────

static std::string read_file(const char* path)
{
    std::ifstream f(path, std::ios::ate);
    if (!f) return "";

    auto sz = f.tellg(); f.seekg(0);
    std::string s(static_cast<size_t>(sz), '\0');
    f.read(s.data(), sz);
    return s;
}

// ── Usage ──────────────────────────────────────────────────────────────

static void print_usage(const char* prog)
{
    std::cerr
        << std::format("rvsim v{} — RISC-V Microarchitecture Engine\n\n", kVersion)
        << std::format("Usage: {} <source.s> [--config <file.cfg>] [--csv]\n\n", prog)
        << "  <source.s>            RISC-V assembly source file (required)\n"
        << "  --config <file.cfg>   Architecture configuration file\n"
        << "  --csv                 Output results as CSV\n"
        << "  -v, --version         Print version\n"
        << "  -h, --help            Show this help\n"
        << "\n"
        << "All architectures parameters are set via the config file.\n"
        << "Sweep axes are defined in the [sweep] section of the config file.\n";
}

// ── Main ───────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    SimConfig cfg;
    bool csv_mode = false;
    const char* source_path = nullptr;
    const char* config_path = nullptr;

    // ── Parse CLI arguments ─────────────────────────────

    for (int i = 1; i < argc; ++i)
    {
        auto match = [&](const char* s) { return std::strcmp(argv[i], s) == 0; };
        if (match("-h") || match("--help")) { print_usage(argv[0]); return 0; }
        if (match("--csv"))    { csv_mode = true; continue; }
        if (match("--config")) { if (i + 1 < argc) config_path = argv[++i]; continue; }
        if (match("-v") || match("--version"))
        {
            std::cout << std::format("rvsim v{}\n", kVersion);
            return 0;
        }
        if (argv[i][0] != '-') { source_path = argv[i]; continue; }
        std::cerr << std::format("Unknown option: {}\n", argv[i]);
        return 1;
    }

    if (!source_path) { print_usage(argv[0]); return 1; }

    // ── Load config ─────────────────────────────────────

    if (config_path)
    {
        auto cr = parse_config_file(config_path);
        if (!cr.ok)
        {
            std::cerr << std::format("Config errors in {}:\n", config_path);
            for (auto& e : cr.errors)
                std::cerr << std::format("  line {}: {}\n", e.line, e.message);
            return 1;
        }

        cfg = std::move(cr.config);
        std::cerr << std::format("Loaded config from {}\n", config_path);
    }

    bool has_sweeps = !cfg.sweeps.empty();
    if (has_sweeps) csv_mode = true;

    // ── Assemble ────────────────────────────────────────

    std::string source = read_file(source_path);
    if (source.empty())
    {
        std::cerr << std::format("Error: cannot read '{}'\n", source_path);
        return 1;
    }

    Assembler as;
    auto ar = as.assemble(source);
    if (!ar.ok)
    {
        std::cerr << std::format("Assembly errors in {}:\n", source_path);
        for (auto& e : ar.errors)
            std::cerr << std::format("  line {}: {}\n", e.line, e.message);
        return 1;
    }
    std::cerr << std::format("Assembled {} bytes ({} labels) from {}\n",
                             ar.code.size(), ar.labels.size(), source_path);

    // ── Sweep mode ──────────────────────────────────────

    if (has_sweeps)
    {
        for (auto& a : cfg.sweeps) std::cout << a.param << ",";
        csv_header();

        std::vector<std::string> vals;
        sweep_recursive(cfg.sweeps, 0, cfg, ar.code, vals);

        u64 total = 1;
        for (auto& a : cfg.sweeps) total *= a.values.size();
        std::cerr << std::format("Sweep complete: {} configurations\n", total);
        return 0;
    }

    // ── Single run ──────────────────────────────────────

    auto r = run_once(cfg, ar.code);

    if (csv_mode)
    {
        csv_header();
        csv_row(r);
    }
    else
    {
        std::cerr << std::format(
            "\n"
            " -- Architecture --\n"
            "  L1:  {}B, {}-way, line={}, replacement={}, {} {}\n"
            "  L1I: {}B, {}-way, replacement={} {}\n"
            "  L2:  {}B, {}-way {}\n"
            "  L3:  {}B, {}-way {}\n"
            "  Pipeline: forwarding={}, branch_pred={}, mispredict_penalty={}\n"
            "\n",
            cfg.l1_size, cfg.l1_assoc, cfg.l1_line, cfg.l1_replacement, cfg.l1_write,
            cfg.l1_size == 0 ? "  (disabled)" : "",
            cfg.l1i_size, cfg.l1i_assoc, cfg.l1i_replacement,
            cfg.l1i_size == 0 ? "  (disabled: L1 is unified)" : "",
            cfg.l2_size, cfg.l2_assoc, cfg.l2_size == 0 ? "  (disabled)" : "",
            cfg.l3_size, cfg.l3_assoc, cfg.l3_size == 0 ? "  (disabled)" : "",
            cfg.forwarding, cfg.branch_pred, cfg.mispredict_penalty);

        std::cerr << std::format(
            "\n"
            " -- Results --\n"
            "  Cycles:           {}\n"
            "  Instructions:     {}\n"
            "  IPC:              {:.4f}\n"
            "  Stalls:           {}\n"
            "  Flushes:          {}\n"
            "  L1 hit rate:      {:.2f}%  ({} misses)\n"
            "  L1I hit rate:     {:.2f}%  ({} misses)\n"
            "  L2 hit rate:      {:.2f}%  ({} misses)\n"
            "  L3 hit rate:      {:.2f}%  ({} misses)\n"
            "  Branches:         {} ({} mispredicts, {:.1f}% accuracy)\n"
            "\n",
            r.cycles, r.instructions, r.ipc,
            r.stalls, r.flushes,
            r.l1_hit_rate * 100.0, r.l1_misses,
            r.l1i_hit_rate * 100.0, r.l1i_misses,
            r.l2_hit_rate * 100.0, r.l2_misses,
            r.l3_hit_rate * 100.0, r.l3_misses,
            r.branches, r.mispredicts, r.branch_accuracy * 100.0);
    }

    return 0;
}
