/**
 * @file multicore.cpp
 * @brief Multi-core CPU system implementation.
 */

#include "multicore.hpp"

#include <cassert>

namespace riscv {

// ── Alignment helper ───────────────────────────────────────────────────

static addr_t align_up(addr_t v, addr_t align)
{
    return (v + align - 1) & ~(align - 1);
}

// ── Construction ───────────────────────────────────────────────────────

MultiCoreCPU::MultiCoreCPU(MultiCoreConfig config)
    : config_(config)
{
    assert(config_.num_cores > 0 && config_.num_cores <= 16);

    // ── Main memory ─────────────────────────────────────

    main_mem_ = std::make_shared<SparseMemory>(0, config_.main_memory_size);

    // ── Device MMIO base ────────────────────────────────
    // Immediately above main memory, page-aligned

    device_base_ = align_up(config_.main_memory_size, 0x1000);

    // ── Build cache hierarchy ───────────────────────────

    bool use_caches = (config_.l1d.size_bytes > 0);

    std::shared_ptr<Memory> shared_backing = main_mem_;

    if (use_caches)
    {
        // Optional L3
        if (config_.l3.size_bytes > 0)
        {
            l3_cache_ = std::make_shared<Cache>(config.l3, main_mem_);
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
        }
    }

    // ── Create cores with per-core MMIO buses ───────────

    auto plic_ptr  = std::shared_ptr<Memory>(std::shared_ptr<void>{}, &plic_);
    auto timer_ptr = std::shared_ptr<Memory>(std::shared_ptr<void>{}, &timer_);

    cores_.reserve(config_.num_cores);
    buses_.reserve(config_.num_cores);

    for (u32 i = 0; i < config_.num_cores; ++i)
    {
        auto bus = std::make_shared<MMIOBus>();

        // L1 cache if present, otherwise main memory
        if (use_caches)
            bus->set_default(l1d_caches_[i]);
        else
            bus->set_default(main_mem_);

        // Map devices above main memory
        bus->map(device_base_,          PLIC::REG_SIZE,  plic_ptr,  "plic");
        bus->map(device_base_ + 0x1000, Timer::REG_SIZE, timer_ptr, "timer");

        buses_.push_back(bus);
        cores_.push_back(std::make_unique<PipelinedCPU>(bus, config_.pipeline));
    }

    wire_interrupts();
}

// ── Interrupt wiring ───────────────────────────────────────────────────

void MultiCoreCPU::wire_interrupts()
{
    timer_.set_notify([this](bool pending)
    {
        for (auto& core : cores_)
        {
            if (pending) core->csrs().set_mip_bit(MInterrupt::MTIE);
            else         core->csrs().clear_mip_bit(MInterrupt::MTIE);
        }
    });

    plic_.set_notify([this](bool pending)
    {
        for (auto& core : cores_)
        {
            if (pending) core->csrs().set_mip_bit(MInterrupt::MEIE);
            else         core->csrs().clear_mip_bit(MInterrupt::MEIE);
        }
    });
}

// ── NIC attachment ─────────────────────────────────────────────────────

void MultiCoreCPU::attach_nic(std::shared_ptr<NIC> nic)
{
    nic_ = std::move(nic);
    nic_->plic_source = config_.nic_plic_source;

    u32 source = config_.nic_plic_source;
    nic_->set_interrupt_callback([this, source]()
    {
        plic_.set_pending(source);
    });

    // Map NIC MMIO on each core's bus
    addr_t nic_base = device_base_ + 0x2000;
    for (auto& bus : buses_)
        bus->map(nic_base, NicReg::REG_SIZE, nic_, "nic");
}

// ── Program loading ────────────────────────────────────────────────────

void MultiCoreCPU::load_program(addr_t addr, std::span<const u8> program)
{
    main_mem_->load(addr, program);
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
    main_mem_->load(addr, bytes);
}

// ── Execution ──────────────────────────────────────────────────────────

bool MultiCoreCPU::tick()
{
    cycle_count_++;

    timer_.tick(cycle_count_);

    if (nic_) nic_->tick(cycle_count_);

    bool any_running = false;
    for (auto& core : cores_)
        if (core->tick()) any_running = true;

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

    s.l2_stats  = l2_cache_ ? l2_cache_->stats() : CacheStats{};
    s.l3_stats  = l3_cache_ ? l3_cache_->stats() : CacheStats{};
    s.coherence = coherence_ ? coherence_->stats() : CoherenceStats{};

    return s;
}

void MultiCoreCPU::reset_stats()
{
    for (auto& l1 : l1d_caches_) l1->stats().reset();
    if (l2_cache_) l2_cache_->stats().reset();
    if (l3_cache_) l3_cache_->stats().reset();
    if (coherence_) coherence_->reset_stats();
}

} // namespace riscv
