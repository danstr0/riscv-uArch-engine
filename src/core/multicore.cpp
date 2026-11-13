/**
 * @file multicore.cpp
 * @brief Multi-core CPU system implementation.
 */

#include "multicore.hpp"

#include <cassert>

namespace riscv {

// ── Construction ───────────────────────────────────────────────────────

MultiCoreCPU::MultiCoreCPU(MultiCoreConfig config)
    : config_(config)
{
    assert(config_.num_cores > 0 && config_.num_cores <= 16);

    // ── Platform: RAM, CLINT, PLIC, memory map ──────────

    platform_ = std::make_unique<Platform>(PlatformConfig{
        .num_harts             = config_.num_cores,
        .ram_base              = 0,
        .ram_size              = config_.main_memory_size,
        .mtime_cycles_per_tick = config_.mtime_cycles_per_tick,
    });
    auto main_mem = platform_->ram_ptr();

    // ── Build cache hierarchy ───────────────────────────

    bool use_caches = (config_.l1d.size_bytes > 0);

    std::shared_ptr<Memory> shared_backing = main_mem;

    if (use_caches)
    {
        // Optional L3
        if (config_.l3.size_bytes > 0)
        {
            l3_cache_ = std::make_shared<Cache>(config.l3, main_mem);
            shared_backing = l3_cache_;
        }

        // L2 required when L1 present
        assert(config_.l2.size_bytes > 0);
        l2_cache_ = std::make_shared<Cache>(config_.l2, shared_backing);

        // Per-core L1 caches
        l1d_caches_.reserve(config_.num_cores);
        for (u32 i = 0; i < config_.num_cores; ++i)
            l1d_caches_.push_back(std::make_shared<Cache>(config_.l1d, l2_cache_));

        // Coherence controller
        std::vector<Cache*> l1_ptrs;
        l1_ptrs.reserve(config_.num_cores);
        for (auto& c : l1d_caches_) l1_ptrs.push_back(c.get());

        coherence_ = std::make_unique<CoherenceController>(std::move(l1_ptrs), 
                                                           l2_cache_, 
                                                           config_.l1d.line_size);

        // Wire coherence callbacks
        for (u32 i = 0; i < config_.num_cores; ++i)
        {
            auto* ctrl = coherence_.get();
            u32 core_id = i;

            l1d_caches_[i]->set_on_read_miss([ctrl, core_id](addr_t line_addr) -> u32
            {
                return ctrl->handle_read_miss(core_id, line_addr);
            });

            l1d_caches_[i]->set_on_write([ctrl, core_id](addr_t line_addr) -> u32
            {
                return ctrl->handle_write_miss(core_id, line_addr);
            });

            l1d_caches_[i]->set_on_evict([ctrl, core_id](addr_t line_addr)
            {
                ctrl->notify_eviction(core_id, line_addr);
            });
        }

        // Optional per-core I-caches. Their refills are coherent reads,
        // but their contents are not tracked
        if (config_.l1i.size_bytes > 0)
        {
            if (config_.l1i.line_size != config_.l1d.line_size)
                throw std::invalid_argument("MultiCoreCPU: L1I and L1D line "
                                            "sizes must match");

            config_.l1i.type = CacheType::INSTRUCTION;
            l1i_caches_.reserve(config_.num_cores);
            for (u32 i = 0; i < config_.num_cores; ++i)
            {
                auto l1i = std::make_shared<Cache>(config_.l1i, l2_cache_);
                l1i->set_on_read_miss([ctrl = coherence_.get(), i](addr_t line_addr) -> u32
                {
                    return ctrl->handle_fetch_miss(i, line_addr);
                });
                l1i_caches_.push_back(std::move(l1i));
            }
        }

        // Devices master through the L2, snooping the L1 D-caches first
        dma_ = std::make_shared<DmaPort>(
            l2_cache_, config_.l1d.line_size,
            [ctrl = coherence_.get()](addr_t line) { (void)ctrl->dma_read(line); },
            [ctrl = coherence_.get()](addr_t line) { (void)ctrl->dma_write(line); });
    }
    else
    {
        if (config_.l1i.size_bytes > 0)
            throw std::invalid_argument("MultiCoreCPU: L1I requires L1D "
                                        "and L2 caches");
        dma_ = main_mem;
    }

    // ── Create cores, each with tis own buses and interrupt lines ──

    cores_.reserve(config_.num_cores);
    buses_.reserve(config_.num_cores);
    fetch_buses_.reserve(config_.num_cores);

    for (u32 i = 0; i < config_.num_cores; ++i)
    {
        // RAM accesses go to the core's L1 if present, otherwise straight to RAM;
        // device accesses bypass the caches
        std::shared_ptr<Memory> data_side = use_caches
            ? std::shared_ptr<Memory>(l1d_caches_[i])
            : std::shared_ptr<Memory>(main_mem);
        std::shared_ptr<Memory> fetch_side = l1i_caches_.empty()
            ? data_side
            : std::shared_ptr<Memory>(l1i_caches_[i]);

        auto bus       = platform_->make_hart_bus(std::move(data_side));
        auto fetch_bus = platform_->make_fetch_bus(std::move(fetch_side));
        buses_.push_back(bus);
        fetch_buses_.push_back(fetch_bus);

        cores_.push_back(std::make_unique<PipelinedCPU>(bus, config_.pipeline, fetch_bus));
        auto& core = *cores_.back();

        if (!l1i_caches_.empty())
            core.set_on_fence_i([icache = l1i_caches_[i].get()] { icache->invalidate_all(); });

        // Any store by this hart breaks other harts' LR reservations on it
        core.set_on_store([this, i](addr_t addr, u32 size)
        {
            for (u32 j = 0; j < cores_.size(); ++j)
                if (j != i) cores_[j]->observe_remote_store(addr, size);
        });

        platform_->connect_hart(i, core.csrs());
    }
}

// ── NIC attachment ─────────────────────────────────────────────────────

void MultiCoreCPU::attach_nic(std::shared_ptr<NIC> nic)
{
    platform_->attach_nic(std::move(nic), config_.nic_plic_source);
}

// ── Program loading ────────────────────────────────────────────────────

void MultiCoreCPU::load_program(addr_t addr, std::span<const u8> program)
{
    platform_->ram().load(addr, program);
}

void MultiCoreCPU::load_instruction(addr_t addr, u32 instruction)
{
    const std::array<u8, 4> bytes =
    {
        static_cast<u8>(instruction),
        static_cast<u8>(instruction >> 8),
        static_cast<u8>(instruction >> 16),
        static_cast<u8>(instruction >> 24),
    };
    platform_->ram().load(addr, bytes);
}

// ── Execution ──────────────────────────────────────────────────────────

bool MultiCoreCPU::tick()
{
    cycle_count_++;

    bool any_running = false;
    for (auto& core : cores_)
        if (core->tick()) any_running = true;

    platform_->tick(cycle_count_);

    return any_running;
}

cycle_t MultiCoreCPU::run_cycles(cycle_t n)
{
    for (cycle_t i = 0; i < n; ++i)
        tick();

    return n;
}

cycle_t MultiCoreCPU::run_until_all_halted(cycle_t max_cycles)
{
    cycle_t count = 0;
    while (count < max_cycles)
    {
        if (!tick()) break;
        ++count;
    }
    return count;
}


// ── Core configuration ─────────────────────────────────────────────────

void MultiCoreCPU::set_core_pc(u32 core_id, addr_t pc)
{
    assert(core_id < config_.num_cores);
    cores_[core_id]->set_pc(pc);
}

void MultiCoreCPU::set_core_reg(u32 core_id, reg_idx_t r, u32 value)
{
    assert(core_id < config_.num_cores);
    cores_[core_id]->set_reg(r, value);
}

// ── Cache access ───────────────────────────────────────────────────────

Cache* MultiCoreCPU::l1d(u32 core_id)
{
    if (core_id < l1d_caches_.size())
        return l1d_caches_[core_id].get();
    return nullptr;
}

Cache* MultiCoreCPU::l1i(u32 core_id)
{
    if (core_id < l1i_caches_.size())
        return l1i_caches_[core_id].get();
    return nullptr;
}

// ── Statistics ─────────────────────────────────────────────────────────

MultiCoreStats MultiCoreCPU::get_stats() const
{
    MultiCoreStats s;
    s.core_stats.resize(config_.num_cores);
    s.l1d_stats.resize(config_.num_cores);

    for (u32 i = 0; i < config_.num_cores; ++i)
    {
        s.core_stats[i] = cores_[i]->stats();
        if (i < l1d_caches_.size())
            s.l1d_stats[i] = l1d_caches_[i]->stats();
    }

    for (const auto& l1i : l1i_caches_)
        s.l1i_stats.push_back(l1i->stats());

    s.l2_stats  = l2_cache_ ? l2_cache_->stats() : CacheStats{};
    s.l3_stats  = l3_cache_ ? l3_cache_->stats() : CacheStats{};
    s.coherence = coherence_ ? coherence_->stats() : CoherenceStats{};

    return s;
}

void MultiCoreCPU::reset_stats()
{
    for (auto& l1 : l1d_caches_) l1->stats().reset();
    for (auto& l1 : l1i_caches_) l1->stats().reset();
    if (l2_cache_) l2_cache_->stats().reset();
    if (l3_cache_) l3_cache_->stats().reset();
    if (coherence_) coherence_->reset_stats();
}

} // namespace riscv
