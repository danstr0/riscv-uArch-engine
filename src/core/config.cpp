/**
 * @file config.cpp
 * @brief Configuration file parser implementation
 */

#include "config.hpp"
#include "assembler.hpp"
#include "platform.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <sstream>

namespace riscv {

// ── String utilities ───────────────────────────────────────────────────

static std::string_view sv_trim(std::string_view s)
{
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.remove_suffix(1);
    return s;
}

static std::string to_lower(std::string_view s)
{
    std::string r(s);
    for (auto& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

static u32 parse_size_val(std::string_view s)
{
    s = sv_trim(s);
    u32 val = 0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
    if (ec != std::errc{}) return 0;

    std::string_view suffix(ptr, static_cast<size_t>(s.data() + s.size() - ptr));
    suffix = sv_trim(suffix);
    if (!suffix.empty())
    {
        char c = static_cast<char>(std::toupper(static_cast<unsigned char>(suffix[0])));
        if (c == 'K') val *= 1024;
        else if (c == 'M') val *= 1024 * 1024;
    }
    return val;
}

static bool parse_u32(std::string_view s, u32& out)
{
    s = sv_trim(s);
    if (s.empty()) return false;

    // Hex
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
    {
        auto [ptr, ec] = std::from_chars(s.data() + 2, s.data() + s.size(), out, 16);
        return ec == std::errc{} && ptr == s.data() + s.size();
    }

    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    return ec == std::errc{} && ptr == s.data() + s.size();
}

// ── Register name parsing ───────────────────────────────────────────────────

static bool parse_reg_name(std::string_view name, u32& idx)
{
    auto n = to_lower(name);

    if (n.size() >= 2 && n[0] == 'x')
    {
        u32 v = 0;
        auto [ptr, ec] = std::from_chars(n.data() + 1, n.data() + n.size(), v);
        if (ec == std::errc{} && ptr == n.data() + n.size() && v < 32)
        {
            idx = v;
            return true;
        }
    }

    static const std::map<std::string, u32> abi =
    {
        {"zero", 0}, {"ra", 1}, {"sp", 2}, {"gp", 3}, {"tp", 4},
        {"t0", 5}, {"t1", 6}, {"t2", 7},
        {"s0", 8}, {"fp", 8}, {"s1", 9},
        {"a0", 10}, {"a1", 11}, {"a2", 12}, {"a3", 13},
        {"a4", 14}, {"a5", 15}, {"a6", 16}, {"a7", 17},
        {"s2", 18}, {"s3", 19}, {"s4", 20}, {"s5", 21},
        {"s6", 22}, {"s7", 23}, {"s8", 24}, {"s9", 25},
        {"s10", 26}, {"s11", 27},
        {"t3", 28}, {"t4", 29}, {"t5", 30}, {"t6", 31},
    };
    auto it = abi.find(n);
    if (it != abi.end()) { idx = it->second; return true; }
    return false;
}

// ── Bracketed list parsing ───────────────────────────────────────────── 

static std::vector<std::string> parse_list(std::string_view s)
{
    s = sv_trim(s);
    std::vector<std::string> result;

    // Strip brackets
    if (!s.empty() && s.front() == '[') s.remove_prefix(1);
    if (!s.empty() && s.back()  == ']') s.remove_suffix(1);

    // Split by comma
    while (!s.empty())
    {
        auto comma = s.find(',');
        auto item = sv_trim(s.substr(0, comma));
        if (!item.empty())
            result.push_back(std::string(item));
        if (comma == std::string_view::npos) break;
        s = s.substr(comma + 1);
    }
    return result;
}

// ── Fill directive parsing ─────────────────────────────────────────────

static bool parse_fill(std::string_view s, u8& fill_val, u32& count)
{
    s = sv_trim(s);
    auto lower = to_lower(s);
    if (lower.substr(0, 5) != "fill(") return false;
    auto close = s.find(')');
    if (close == std::string_view::npos) return false;

    auto inner = sv_trim(s.substr(5, close - 5));
    auto comma = inner.find(',');
    if (comma == std::string_view::npos) return false;

    u32 v = 0, c = 0;
    if (!parse_u32(inner.substr(0, comma), v)) return false;
    if (!parse_u32(sv_trim(inner.substr(comma + 1)), c)) return false;
    fill_val = static_cast<u8>(v);
    count = c;
    return true;
}

// ── String literal parsing ─────────────────────────────────────────────

static bool parse_string_literal(std::string_view s, std::string& out)
{
    s = sv_trim(s);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
    {
        out = std::string(s.substr(1, s.size() - 2));
        return true;
    }
    return false;
}

// ── Main parser ────────────────────────────────────────────────────────

ConfigResult parse_config(std::string_view source)
{
    ConfigResult result;
    result.ok = true;

    std::istringstream stream{std::string(source)};
    std::string line;
    std::string current_section;
    u32 line_num = 0;

    while (std::getline(stream, line))
    {
        line_num++;
        auto sv = sv_trim(std::string_view(line));

        // Strip comments
        for (size_t i = 0; i < sv.size(); ++i)
            if (sv[i] == '#' || sv[i] == ';')
            {
                sv = sv.substr(0, i);
                break;
            }
        sv = sv_trim(sv);
        if (sv.empty()) continue;

        // Section header: [section.name]
        if (sv.front() == '[' && sv.back() == ']')
        {
            current_section = to_lower(sv.substr(1, sv.size() - 2));
            continue;
        }

        // Key = value
        auto eq = sv.find('=');
        if (eq == std::string_view::npos)
        {
            result.errors.push_back({line_num, "expected key = value"});
            result.ok = false;
            continue;
        }

        auto key = to_lower(sv_trim(sv.substr(0, eq)));
        auto val_sv = sv_trim(sv.substr(eq + 1));
        auto val = std::string(val_sv);

        auto& cfg = result.config;

        // ── [memory] section ────────────────────────────
        if (current_section == "memory")
        {
            if (key == "size")
            {
                cfg.mem_size_kb = parse_size_val(val) / 1024;
                if (cfg.mem_size_kb == 0) cfg.mem_size_kb = parse_size_val(val); // already in KB
            }
            else if (key == "max_cycles")
            {
                u32 v;
                if (parse_u32(val, v)) cfg.max_cycles = v;
            }
        }
        // ── [memory.init] section ───────────────────────
        else if (current_section == "memory.init")
        {
            // Key is the address, value is the data
            u32 addr = 0;
            if (!parse_u32(key, addr))
            {
                result.errors.push_back({line_num, "invalid address: " + std::string(key)});
                result.ok = false;
                continue;
            }

            MemInitEntry entry;
            entry.address = addr;

            // Check for fill(val, count)
            if (parse_fill(val, entry.fill_val, entry.fill_count))
                entry.kind = MemInitEntry::Kind::FILL;
            // Check for string literal
            else if (parse_string_literal(val, entry.str))
                entry.kind = MemInitEntry::Kind::STRING;
            // Check for bracketed list of u32 values
            else if (!val.empty() && val.front() == '[')
            {
                auto items = parse_list(val);
                entry.kind = MemInitEntry::Kind::WORDS;
                for (auto& item : items)
                {
                    u32 v = 0;
                    if (parse_u32(item, v) || (v = parse_size_val(item)) != 0)
                        entry.words.push_back(v);
                    else
                    {
                        result.errors.push_back({line_num, "invalid value in list: " + item});
                        result.ok = false;
                    }
                }
            }
            else
            {
                // Single u32 value
                u32 v = 0;
                if (parse_u32(val, v))
                {
                    entry.kind = MemInitEntry::Kind::WORDS;
                    entry.words.push_back(v);
                }
                else
                {
                    result.errors.push_back({line_num, "invalid memory init value: " + val});
                    result.ok = false;
                    continue;
                }
            }
            cfg.mem_init.push_back(std::move(entry));
        }
        // ── [cache.l1] section ──────────────────────────
        else if (current_section == "cache.l1")
        {
            if      (key == "size")        cfg.l1_size = parse_size_val(val);
            else if (key == "assoc")       { u32 v; if (parse_u32(val, v)) cfg.l1_assoc = v; }
            else if (key == "line")        cfg.l1_line = parse_size_val(val);
            else if (key == "replacement") cfg.l1_replacement = val;
            else if (key == "write")       cfg.l1_write = val;
            else if (key == "write_alloc") cfg.l1_write_alloc = val;
        }
        // ── [cache.l1i] section ─────────────────────────
        else if (current_section == "cache.l1i")
        {
            if      (key == "size")        cfg.l1i_size = parse_size_val(val);
            else if (key == "assoc")       { u32 v; if (parse_u32(val, v)) cfg.l1i_assoc = v; }
            else if (key == "replacement") cfg.l1i_replacement = val;
        }
        // ── [cache.l2] section ──────────────────────────
        else if (current_section == "cache.l2")
        {
            if      (key == "size")  cfg.l2_size = parse_size_val(val);
            else if (key == "assoc") { u32 v; if (parse_u32(val, v)) cfg.l2_assoc = v; }
        }
        // ── [cache.l3] section ──────────────────────────
        else if (current_section == "cache.l3")
        {
            if      (key == "size")  cfg.l3_size = parse_size_val(val);
            else if (key == "assoc") { u32 v; if (parse_u32(val, v)) cfg.l3_assoc = v; }
        }
        // ── [pipeline] section ──────────────────────────
        else if (current_section == "pipeline")
        {
            if      (key == "forwarding")  cfg.forwarding = val;
            else if (key == "branch_pred") cfg.branch_pred = val;
            else if (key == "mispredict_penalty")
            {
                u32 v;
                if (parse_u32(val, v)) cfg.mispredict_penalty = v;
            }
        }
        // ── [system] section ────────────────────────────
        else if (current_section == "system")
        {
            if      (key == "cpu")        cfg.cpu_type = val;
            else if (key == "cores")      { u32 v; if (parse_u32(val, v)) cfg.num_cores = v; }
            else if (key == "max_cycles") { u32 v; if (parse_u32(val, v)) cfg.max_cycles = v; }
        }
        // ── [registers] section ─────────────────────────
        else if (current_section == "registers")
        {
            u32 reg_idx = 0;
            if (!parse_reg_name(key, reg_idx))
            {
                result.errors.push_back({line_num, "unknown register: " + std::string(key)});
                result.ok = false;
                continue;
            }
            if (reg_idx == 0)
            {
                result.errors.push_back({line_num, "cannot set x0 (hardwired to zero)"});
                result.ok = false;
                continue;
            }
            u32 v = 0;
            if (!parse_u32(val, v))
                v = parse_size_val(val);
            cfg.reg_init.push_back({reg_idx, v});
        }
        // ── [cores.N] section (stepper only) ────────────
        else if (current_section.size() > 6
                 && current_section.substr(0, 6) == "cores.")
        {
            // Parse core index from section name
            u32 core_idx = 0;
            auto idx_str = current_section.substr(6);
            auto [ptr, ec] = std::from_chars(idx_str.data(),
                                             idx_str.data() + idx_str.size(),
                                             core_idx);

            if (ec != std::errc{} || ptr != idx_str.data() + idx_str.size())
            {
                result.errors.push_back({line_num,
                    std::format("bad core index in section [{}]", current_section)});
                result.ok = false;
                continue;
            }
            if (core_idx >= cfg.cores.size())
                cfg.cores.resize(core_idx + 1);
            auto& core = cfg.cores[core_idx];

            if (key == "program")
                core.program = val;
            else if (key == "pc")
            {
                u32 v = 0;
                if (parse_u32(val, v)) core.pc = v;
                else
                {
                    result.errors.push_back({line_num, "bad PC value: " + val});
                    result.ok = false;
                }
            }
            else
            {
                // Register assignment for this core
                u32 reg_idx = 0;
                if (!parse_reg_name(key, reg_idx))
                {
                    result.errors.push_back({line_num,
                        std::format("[cores.{}] unknown key/register '{}'",
                                    core_idx, std::string(key))});
                    result.ok = false;
                    continue;
                }
                if (reg_idx == 0)
                {
                    result.errors.push_back({line_num, "cannot set x0 (hardwired to zero)"});
                    result.ok = false;
                    continue;
                }
                u32 v = 0;
                if (!parse_u32(val, v)) v = parse_size_val(val);
                core.reg_init.push_back({reg_idx, v});
            }
        }
        // ── [sweep] section ─────────────────────────────
        else if (current_section == "sweep")
        {
            if (!val.empty() && val.front() == '[')
            {
                auto items = parse_list(val);
                if (!items.empty())
                    cfg.sweeps.push_back({std::string(key), std::move(items)});
            }
            else
                cfg.sweeps.push_back({std::string(key), {val}});
        }
        // ── Unknown section ─────────────────────────────
        else
        {
            result.errors.push_back({line_num,
                std::format("unknown section: [{}]", current_section)});
            result.ok = false;
        }
    }

    // Run validation if parsing succeeded
    if (result.ok)
        result.ok = validate_config(result.config, result.errors);

    return result;
}

ConfigResult parse_config_file(const char* path)
{
    std::ifstream f(path, std::ios::ate);
    if (!f)
    {
        ConfigResult r;
        r.errors.push_back({0, std::string("cannot open file: ") + path});
        return r;
    }
    auto sz = f.tellg(); f.seekg(0);
    std::string s(static_cast<size_t>(sz), '\0');
    f.read(s.data(), sz);
    return parse_config(s);
}

// ── Parameter name / value validation ──────────────────────────────────

static const std::vector<std::string> kValidParams =
{
    "cache.l1.size", "cache.l1.assoc", "cache.l1.line",
    "cache.l1.replacement", "cache.l1.write", "cache.l1.write_alloc",
    "cache.l1i.size", "cache.l1i.assoc", "cache.l1i.replacement",
    "cache.l2.size", "cache.l2.assoc",
    "cache.l3.size", "cache.l3.assoc",
    "pipeline.forwarding", "pipeline.branch_pred", "pipeline.mispredict_penalty",
    "system.cpu", "system.cores", "system.max_cycles",
};

bool is_valid_param(const std::string& param)
{
    for (auto& p : kValidParams)
        if (p == param) return true;
    return false;
}

static bool is_power_of_2(u32 v) { return v > 0 && (v & (v - 1)) == 0; }

static const std::vector<std::string> kReplacementPolicies =
{
    "lru", "mru", "plru", "fifo", "random"
};
static const std::vector<std::string> kWritePolicies = {"writeback", "writethrough"};
static const std::vector<std::string> kAllocPolicies = {"allocate", "no_allocate"};
static const std::vector<std::string> kForwardingModes = {"none", "partial"};
static const std::vector<std::string> kBranchPredictors =
{
    "not_taken", "always_taken", "backward_taken", "bimodal_1bit", "bimodal_2bit"
};

static bool in_list(const std::string& val, const std::vector<std::string>& list)
{
    for (auto& s : list) if (s == val) return true;
    return false;
}

bool is_valid_param_value(const std::string& param, const std::string& value,
                          std::string& error_msg)
{
    if (param == "cache.l1.size" || param == "cache.l1i.size"
     || param == "cache.l2.size" || param == "cache.l3.size")
    {
        u32 v = parse_size_val(value);
        if (v == 0)            { error_msg = "cache size must be > 0"; return false; }
        if (!is_power_of_2(v)) { error_msg = "cache size must be a power of 2"; return false; }
        return true;
    }
    if (param == "cache.l1.assoc" || param == "cache.l1i.assoc"
     || param == "cache.l2.assoc" || param == "cache.l3.assoc")
    {
        u32 v = 0;
        if (!parse_u32(value, v) || v == 0)
        {
            error_msg = "associativity must be > 0"; return false;
        }
        if (!is_power_of_2(v)) { error_msg = "associativity must be a power of 2"; return false; }
        return true;
    }
    if (param == "cache.l1.line")
    {
        u32 v = parse_size_val(value);
        if (v == 0)            { error_msg = "line size must be > 0"; return false; }
        if (!is_power_of_2(v)) { error_msg = "line size must be a power of 2"; return false; }
        return true;
    }
    if (param == "cache.l1.replacement" || param == "cache.l1i.replacement")
    {
        if (!in_list(value, kReplacementPolicies))
        {
            error_msg = "replacement must be one of: lru, mru, plru, fifo, random";
            return false;
        }
        return true;
    }
    if (param == "cache.l1.write")
    {
        if (!in_list(value, kWritePolicies))
        {
            error_msg = "write policy must be one of: writeback, writethrough";
            return false;
        }
        return true;
    }
    if (param == "cache.l1.write_alloc")
    {
        if (!in_list(value, kAllocPolicies))
        {
            error_msg = "write allocation policy must be one of: allocate, no_allocate";
            return false;
        }
    }
    if (param == "pipeline.forwarding")
    {
        if (!in_list(value, kForwardingModes))
        {
            error_msg = "forwarding must be one of: none, partial";
            return false;
        }
        return true;
    }
    if (param == "pipeline.branch_pred")
    {
        if (!in_list(value, kBranchPredictors))
        {
            error_msg = "branch_pred must be one of: not_taken, always_taken, backward_taken, bimodal_1bit, bimodal_2bit";
            return false;
        }
        return true;
    }
    if (param == "pipeline.mispredict_penalty")
    {
        u32 v = 0;
        if (!parse_u32(value, v) || v < 1)
        {
            error_msg = "mispredict_penalty must be >= 1";
            return false;
        }
        return true;
    }
    if (param == "system.cpu")
    {
        static const std::vector<std::string> types = {"simple", "pipelined", "multicore"};
        if (!in_list(value, types))
        {
            error_msg = "cpu must be one of: simple, pipelined, multicore";
            return false;
        }
        return true;
    }
    if (param == "system.cores")
    {
        u32 v = 0;
        if (!parse_u32(value, v) || v == 0 || v > 16)
        {
            error_msg = "cores must be 1-16";
            return false;
        }
        return true;
    }
    if (param == "system.max_cycles")
    {
        u32 v = 0;
        if (!parse_u32(value, v) || v == 0)
        {
            error_msg = "max_cycles must be > 0";
            return false;
        }
        return true;
    }
    return true; // unknown param - let apply_config_param handle
}

// ── Full config validation ─────────────────────────────────────────────

bool validate_config(const SimConfig& cfg, std::vector<ConfigError>& errors)
{
    bool ok = true;
    auto err = [&](const std::string& msg)
    {
        errors.push_back({0, msg});
        ok = false;
    };

    // ── Memory ──────────────────────────────────────────

    if (cfg.mem_size_kb == 0)
        err("memory size must be > 0");
    if (static_cast<u64>(cfg.mem_size_kb) * 1024 > 0xFFFF'FFFFULL)
        err("memory size exceeds 32-bit address space (max 4GB)");
    if (cfg.max_cycles == 0)
        err("max_cycles must be > 0");

    // ── Memory init bounds ──────────────────────────────

    u64 mem_bytes = static_cast<u64>(cfg.mem_size_kb) * 1024;
    for (auto& entry : cfg.mem_init)
    {
        u64 end = entry.address;
        switch (entry.kind)
        {
            case MemInitEntry::Kind::WORDS:
                end += entry.words.size() * 4;
                if (entry.address % 4 != 0)
                    err(std::format("memory init at 0x{:x}: u32 array must be 4-byte aligned",
                        entry.address));
                break;
            case MemInitEntry::Kind::BYTES:
                end += entry.bytes.size();
                break;
            case MemInitEntry::Kind::FILL:
                end += entry.fill_count;
                break;
            case MemInitEntry::Kind::STRING:
                end += entry.str.size() + 1; // +1 for null terminator
                break;
        }
        if (end > mem_bytes)
            err(std::format("memory init at 0x{:x}: data extends past memory "
                            "(ends at 0x{:x}, mem size 0x{:x})",
                            entry.address, end, mem_bytes));
    }

    // ── Cache hierarchy ─────────────────────────────

    auto validate_cache = [&](const char* name, u32 size, u32 assoc, u32 line)
    {
        if (size == 0) return; // disabled
        if (!is_power_of_2(size))
            err(std::format("{} size ({}) must be a power of 2", name, size));
        if (!is_power_of_2(assoc))
            err(std::format("{} associativity ({}) must be a power of 2", name, assoc));
        if (!is_power_of_2(line))
            err(std::format("{} line size ({}) must be a power of 2", name, line));
        if (line > size)
            err(std::format("{} line size ({}) must be <= cache size ({})", name, line, size));
        if (line > 0 && assoc > 0 && size % (line * assoc) != 0)
            err(std::format("{} size ({}) must be >= line_size * assoc ({} * {} = {})",
                            name, size, line, assoc, line * assoc));
    };

    validate_cache("L1", cfg.l1_size, cfg.l1_assoc, cfg.l1_line);
    validate_cache("L1I", cfg.l1i_size, cfg.l1i_assoc, cfg.l1_line);
    validate_cache("L2", cfg.l2_size, cfg.l2_assoc, cfg.l1_line);
    validate_cache("L3", cfg.l3_size, cfg.l3_assoc, cfg.l1_line);

    // Cache hierarchy ordering - can't have L2 without L1, or L3 without L2
    if (cfg.l2_size > 0 && cfg.l1_size == 0)
        err("L2 cache requires L1 cache");
    if (cfg.l3_size > 0 && cfg.l2_size == 0)
        err("L3 cache requires L2 cache");

    if (cfg.l1i_size > 0 && cfg.l1_size == 0)
        err("L1I cache requires L1 data cache");
    if (cfg.l1i_size > 0 && cfg.l2_size > 0 && cfg.l2_size <= cfg.l1i_size)
        err(std::format("L2 size ({}) must be larger than L1I size ({})",
                        cfg.l2_size, cfg.l1i_size));
    if (!in_list(cfg.l1i_replacement, kReplacementPolicies) && cfg.l1i_size > 0)
        err(std::format("unknown L1I replacement policy '{}' "
                        "(expected: lru, mru, plru, fifo, random)",
                        cfg.l1i_replacement));

    // Cache sizes must increase with distance from CPU
    if (cfg.l1_size > 0 && cfg.l2_size > 0 && cfg.l2_size <= cfg.l1_size)
        err(std::format("L2 size ({}) must be larger than L1 size ({})",
                        cfg.l2_size, cfg.l1_size));
    if (cfg.l2_size > 0 && cfg.l3_size > 0 && cfg.l3_size <= cfg.l2_size)
        err(std::format("L3 size ({}) must be larger than L2 size ({})",
                        cfg.l3_size, cfg.l2_size));

    // Enum string validation
    if (!in_list(cfg.l1_replacement, kReplacementPolicies) && cfg.l1_size > 0)
        err(std::format("unknown replacement policy '{}' "
                        "(expected: lru, mru, plru, fifo, random)",
                        cfg.l1_replacement));
    if (!in_list(cfg.l1_write, kWritePolicies) && cfg.l1_size > 0)
        err(std::format("unknown write policy '{}' (expected: writeback, writethrough)",
                        cfg.l1_write));
    if (!in_list(cfg.l1_write_alloc, kAllocPolicies) && cfg.l1_size > 0)
        err(std::format("unknown write_allow policy '{}' (expected: allocate, no_allocate)",
                        cfg.l1_write_alloc));

    // ── Pipeline ────────────────────────────────────

    if (!in_list(cfg.forwarding, kForwardingModes))
        err(std::format("unknown forwarding mode: '{}' (expected: none, partial)",
                        cfg.forwarding));
    if (!in_list(cfg.branch_pred, kBranchPredictors))
        err(std::format("unknown branch predictor: '{}' "
                        "(expected: not_taken, always_taken, backward_taken, "
                        "bimodal_1bit, bimodal_2bit)",
                        cfg.branch_pred));
    if (cfg.mispredict_penalty < 1)
        err("mispredict_penalty must be >= 1");

    // ── System ──────────────────────────────────────

    static const std::vector<std::string> kCpuTypes = {"simple", "pipelined", "multicore"};
    if (!in_list(cfg.cpu_type, kCpuTypes))
        err(std::format("unknown cpu type '{}' (expected: simple, pipelined, multicore)",
                        cfg.cpu_type));

    if (cfg.cpu_type == "multicore" && cfg.num_cores < 2)
        err("multicore cpu requires at least 2 cores");
    if (cfg.cpu_type != "multicore" && cfg.num_cores > 1)
        err(std::format("cores={} but cpu type is '{}' (use cpu = multicore)",
                        cfg.num_cores, cfg.cpu_type));
    if (cfg.num_cores == 0 || cfg.num_cores > 16)
        err(std::format("cores must be 1-16 (got {})", cfg.num_cores));

    // MultiCoreCPU requires L2 if L1 present
    if (cfg.cpu_type == "multicore" && cfg.l1_size > 0 && cfg.l2_size == 0)
        err("multicore with L1 caches requires a shared L2 cache");

    // ── Physical memory map guard ───────────────────
    // RAM starts at 0; devices sit at fixed addresses, so RAM must end
    // before the first device window. Mem-init entries are already bounded
    // by RAM size above, so they cannot reach a device

    if (mem_bytes > PlatformMap::CLINT_BASE)
        err(std::format("memory size {}KB overlaps the CLINT at 0x{:08x} "
                        "(max {}KB with RAM at address 0)",
                        cfg.mem_size_kb, PlatformMap::CLINT_BASE,
                        PlatformMap::CLINT_BASE / 1024));

    // ── Per-core config ─────────────────────────────

    for (u32 i = 0; i < cfg.cores.size(); ++i)
    {
        auto& core = cfg.cores[i];
        if (i >= cfg.num_cores)
            err(std::format("[cores.{}] core index exceeds num_cores ({})", i, cfg.num_cores));

        // PC must be within memory
        if (core.pc >= mem_bytes)
            err(std::format("[cores.{}] pc=0x{:x} is outside memory (size=0x{:x})",
                            i, core.pc, mem_bytes));

        // PC must also be 4-byte aligned
        if (core.pc & 3)
            err(std::format("[cores.{}] pc=0x{:x} is not 4-byte aligned",
                            i, core.pc));

        // Per-core registers: cannot set x0, cannot have index > 31
        for (auto& ri : core.reg_init)
        {
            if (ri.reg_idx == 0)
                err(std::format("[cores.{}] cannot set x0 (hardwired to zero)", i));
            if (ri.reg_idx >= 32)
                err(std::format("[core.{}] register index {} out of range", i, ri.reg_idx));
        }
    }

    // ── Registers ───────────────────────────────────

    for (auto& ri : cfg.reg_init)
    {
        if (ri.reg_idx == 0)
            err("cannot set x0 (hardwired to zero)");
        if (ri.reg_idx >= 32)
            err(std::format("register index {} out of range (0-31)", ri.reg_idx));
    }

    // ── Sweep axes ──────────────────────────────────

    for (auto& axis : cfg.sweeps)
    {
        // Parameter name must be recognized
        if (!is_valid_param(axis.param))
            err(std::format("[sweep] unknown parameter: '{}'", axis.param));

        // Check for duplicate values - O(n^2) is fine for this
        for (size_t i = 0; i < axis.values.size(); ++i)
            for (size_t j = i + 1; j < axis.values.size(); ++j)
                if (axis.values[i] == axis.values[j])
                    err(std::format("[sweep] {}: duplicate value '{}'",
                                    axis.param, axis.values[i]));

        // Validate each value
        for (auto& val : axis.values)
        {
            std::string msg;
            if (!is_valid_param_value(axis.param, val, msg))
                err(std::format("[sweep] {}: value '{}' - {}", axis.param, val, msg));
        }
    }

    return ok;
}

// ── Apply a qualified parameter ────────────────────────────────────────

void apply_config_param(SimConfig& cfg, const std::string& param,
                        const std::string& value)
{
    if      (param == "cache.l1.size")        cfg.l1_size = parse_size_val(value);
    else if (param == "cache.l1.assoc")       { u32 v; if (parse_u32(value, v)) cfg.l1_assoc = v; }
    else if (param == "cache.l1.line")        cfg.l1_line = parse_size_val(value);
    else if (param == "cache.l1.replacement") cfg.l1_replacement = value;
    else if (param == "cache.l1.write")       cfg.l1_write = value;
    else if (param == "cache.l1.write_alloc") cfg.l1_write_alloc = value;
    else if (param == "cache.l1i.size")        cfg.l1i_size = parse_size_val(value);
    else if (param == "cache.l1i.assoc")       { u32 v; if (parse_u32(value, v)) cfg.l1i_assoc = v; }
    else if (param == "cache.l1i.replacement") cfg.l1i_replacement = value;
    else if (param == "cache.l2.size")        cfg.l2_size = parse_size_val(value);
    else if (param == "cache.l2.assoc")       { u32 v; if (parse_u32(value, v)) cfg.l2_assoc = v; }
    else if (param == "cache.l3.size")        cfg.l3_size = parse_size_val(value);
    else if (param == "cache.l3.assoc")       { u32 v; if (parse_u32(value, v)) cfg.l3_assoc = v; }
    else if (param == "pipeline.forwarding")  cfg.forwarding = value;
    else if (param == "pipeline.branch_pred") cfg.branch_pred = value;
    else if (param == "pipeline.mispredict_penalty")
    {
        u32 v; if (parse_u32(value, v)) cfg.mispredict_penalty = v;
    }
    else if (param == "system.cpu")        cfg.cpu_type = value;
    else if (param == "system.cores")      { u32 v; if (parse_u32(value, v)) cfg.num_cores = v; }
    else if (param == "system.max_cycles") { u32 v; if (parse_u32(value, v)) cfg.max_cycles = v; }
}

} // namespace riscv
