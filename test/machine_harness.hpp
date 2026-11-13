/**
 * @file machine_harness.hpp
 * @brief Run the same assembled program on every core model.
 *
 * Program-level tests describe a program as a set of assembly segments
 * placed at fixed addresses (the assembler has no data directives or
 * relocation, so handlers live at known addresses rather than labels).
 * A @c Machine wraps one concrete system (single-cycle CPU, pipelined CPU
 * with or without caches, multi-core) behind a common interface, and
 * @c for_each_model runs a test body on each so architectural results can
 * be required to match across models.
 *
 * Memory is read back through hart 0's bus, so values still sitting in a
 * write-back cache are observed correctly.
 */

#pragma once

#include "core/assembler.hpp"
#include "core/multicore.hpp"
#include "core/system.hpp"
#include "test_framework.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace harness {

using namespace riscv;

/// An assembly fragment loaded at a fixed address.
struct Segment
{
    addr_t      addr;
    std::string source;
};

enum class Model
{
    Cpu,              ///< Single-cycle CPU, no caches.
    Pipe,             ///< 5-stage pipeline, partial forwarding, no caches.
    PipeNoForward,    ///< 5-stage pipeline, no forwarding.
    PipeCached,       ///< 5-stage pipeline behind unified L1 + L2.
    MultiCore,        ///< MultiCoreCPU, no caches.
    MultiCoreCached,  ///< MultiCoreCPU with private unified L1s, shared L2, MESI.
    CpuSplit,         ///< Single-cycle CPU with split L1I / L1D, no L2.
    PipeSplit,        ///< 5-stage pipeline with split L1I / L1D + L2.
    MultiCoreSplit,   ///< MultiCoreCPU with private split L1s, shared L2, MESI.
};

inline const char* model_name(Model m)
{
    switch (m)
    {
        case Model::Cpu:             return "cpu";
        case Model::Pipe:            return "pipeline";
        case Model::PipeNoForward:   return "pipeline(no-fwd)";
        case Model::PipeCached:      return "pipeline+L1/L2";
        case Model::MultiCore:       return "multicore";
        case Model::MultiCoreCached: return "multicore+L1/L2";
        case Model::CpuSplit:        return "cpu+L1I/L1D";
        case Model::PipeSplit:       return "pipeline+L1I/L1D/L2";
        case Model::MultiCoreSplit:  return "multicore+L1I/L1D/L2";
    }
    return "?";
}

/// Common interface over the concrete system types.
class Machine {
public:
    virtual ~Machine() = default;

    [[nodiscard]] virtual Platform& platform() = 0;
    [[nodiscard]] virtual u32       num_harts() const = 0;
    [[nodiscard]] virtual u32       reg(u32 hart, reg_idx_t r) const = 0;
    [[nodiscard]] virtual CSRFile&  csrs(u32 hart) = 0;
    [[nodiscard]] virtual bool      halted(u32 hart) const = 0;
    [[nodiscard]] virtual u64       interrupts_taken(u32 hart) const = 0;
    [[nodiscard]] virtual cycle_t   cycles() const = 0;
    virtual void set_pc(u32 hart, addr_t pc) = 0;
    /// One step: one instruction (CPU) or one clock (pipelined / multi-core).
    virtual bool step() = 0;
    /// Hart 0's view of memory (through its caches).
    [[nodiscard]] virtual Memory& view() = 0;
    /// The memory a bus-mastering device should use.
    [[nodiscard]] virtual std::shared_ptr<Memory> dma_memory() = 0;

    [[nodiscard]] bool all_halted() const
    {
        for (u32 h = 0; h < num_harts(); ++h)
            if (!halted(h)) return false;
        return true;
    }

    [[nodiscard]] u32 mem32(addr_t a) { return view().read32(a).value; }

    /// Step until @p pred holds or every hart halts; returns @c false on timeout.
    bool run_until(const std::function<bool()>& pred, u64 max_steps = 2'000'000)
    {
        for (u64 i = 0; i < max_steps; ++i)
        {
            if (pred()) return true;
            if (!step() && all_halted()) return pred();
        }
        return false;
    }

    bool run_to_halt(u64 max_steps = 2'000'000)
    {
        return run_until([this] { return all_halted(); }, max_steps);
    }

    /// Assemble and load every segment into RAM. Prints errors
    /// and returns false on failure.
    bool load(const std::vector<Segment>& segments)
    {
        Assembler as;
        for (const auto& seg : segments)
        {
            auto r = as.assemble(seg.source);
            if (!r.ok)
            {
                for (const auto& e : r.errors)
                    std::cerr << std::format("  asm error (segment @0x{:x}) "
                                             "line {}: {}\n",
                                             seg.addr, e.line, e.message);
                return false;
            }
            platform().ram().load(seg.addr, r.code);
        }
        return true;
    }
};

// ── Concrete machines ──────────────────────────────────────────────────

inline CacheConfig small_l1()
{
    return {.size_bytes = 1024, .line_size = 64, .associativity = 4,
            .hit_latency = 1, .miss_penalty = 10};
}
inline CacheConfig small_l2()
{
    return {.size_bytes = 8192, .line_size = 64,
            .hit_latency = 5, .miss_penalty = 40};
}

template <typename Core>
class SingleMachine final : public Machine {
public:
    explicit SingleMachine(SystemConfig cfg) : sys_(cfg) {}

    Platform& platform() override { return sys_.platform(); }
    u32 num_harts() const override { return 1; }
    u32 reg(u32, reg_idx_t r) const override { return sys_.core().reg(r); }
    CSRFile& csrs(u32) override { return sys_.core().csrs(); }
    bool halted(u32) const override
    {
        auto& c = sys_.core();
        if constexpr (std::is_same_v<Core, PipelinedCPU>)
            return c.halted() && c.pipeline_empty();
        else
            return c.halted();
    }
    u64 interrupts_taken(u32) const override
    {
        auto& c = sys_.core();
        if constexpr (std::is_same_v<Core, PipelinedCPU>)
            return c.stats().interrupts_taken;
        else
            return c.interrupts_taken();
    }
    cycle_t cycles() const override { return sys_.cycles(); }
    void set_pc(u32, addr_t pc) override { sys_.core().set_pc(pc); }
    bool step() override { return sys_.step(); }
    Memory& view() override { return sys_.bus(); }
    std::shared_ptr<Memory> dma_memory() override { return sys_.dma_memory(); }

private:
    SingleCoreSystem<Core> sys_;
};

class MultiMachine final : public Machine {
public:
    explicit MultiMachine(MultiCoreConfig cfg) : mc_(cfg) {}

    Platform& platform() override { return mc_.platform(); }
    u32 num_harts() const override { return mc_.num_cores(); }
    u32 reg(u32 h, reg_idx_t r) const override { return mc_.core(h).reg(r); }
    CSRFile& csrs(u32 h) override { return mc_.core(h).csrs(); }
    bool halted(u32 h) const override
    {
        return mc_.core(h).halted() && mc_.core(h).pipeline_empty();
    }
    u64 interrupts_taken(u32 h) const override
    {
        return mc_.core(h).stats().interrupts_taken;
    }
    cycle_t cycles() const override { return mc_.cycles(); }
    void set_pc(u32 h, addr_t pc) override { mc_.set_core_pc(h, pc); }
    bool step() override { return mc_.tick(); }
    Memory& view() override { return mc_.core(0).memory(); }
    std::shared_ptr<Memory> dma_memory() override { return mc_.dma_memory(); }

private:
    MultiCoreCPU mc_;
};

/// Build a machine of the given model with @p harts harts.
inline std::unique_ptr<Machine> make_machine(Model m, u32 harts = 1,
                                             u64 ram_size = 64 * 1024)
{
    SystemConfig sc;
    sc.platform.ram_size = ram_size;

    switch(m)
    {
        case Model::Cpu:
            return std::make_unique<SingleMachine<CPU>>(sc);
        case Model::Pipe:
            return std::make_unique<SingleMachine<PipelinedCPU>>(sc);
        case Model::PipeNoForward:
            sc.pipeline.forwarding = ForwardingPolicy::NONE;
            return std::make_unique<SingleMachine<PipelinedCPU>>(sc);
        case Model::PipeCached:
            sc.l1d = small_l1();
            sc.l2  = small_l2();
            return std::make_unique<SingleMachine<PipelinedCPU>>(sc);
        case Model::CpuSplit:
            sc.l1d = small_l1();
            sc.l1i = small_l1();
            return std::make_unique<SingleMachine<CPU>>(sc);
        case Model::PipeSplit:
            sc.l1d = small_l1();
            sc.l1i = small_l1();
            sc.l2  = small_l2();
            return std::make_unique<SingleMachine<PipelinedCPU>>(sc);
        case Model::MultiCore:
        case Model::MultiCoreCached:
        case Model::MultiCoreSplit:
        {
            MultiCoreConfig mc;
            mc.num_cores        = harts;
            mc.main_memory_size = static_cast<u32>(ram_size);
            mc.l3.size_bytes    = 0;
            if (m == Model::MultiCore)      { mc.l1d.size_bytes = 0; }
            else                            { mc.l1d = small_l1(); mc.l2 = small_l2(); }
            if (m == Model::MultiCoreSplit) { mc.l1i = small_l1(); }
            return std::make_unique<MultiMachine>(mc);
        }
    }
    return nullptr;
}

inline const std::vector<Model> kSingleHartModels =
{
    Model::Cpu, Model::Pipe, Model::PipeNoForward, Model::PipeCached,
    Model::MultiCore, Model::MultiCoreCached,
    Model::CpuSplit, Model::PipeSplit, Model::MultiCoreSplit,
};

inline const std::vector<Model> kMultiHartModels =
{
    Model::MultiCore, Model::MultiCoreCached, Model::MultiCoreSplit,
};

/// Models with a separate L1 I-cache.
inline const std::vector<Model> kSplitModels =
{
    Model::CpuSplit, Model::PipeSplit, Model::MultiCoreSplit,
};

/// Run @p body on each model, printing which model failed.
inline bool for_each_model(const std::vector<Model>& models,
                           const std::function<bool(Machine&, Model)>& body,
                           u32 harts = 1)
{
    for (Model m : models)
    {
        auto machine = make_machine(m, harts);
        if (!body(*machine, m))
        {
            std::cerr << "    ^ on model: " << model_name(m) << "\n";
            return false;
        }
    }
    return true;
}

} // namespace harness

