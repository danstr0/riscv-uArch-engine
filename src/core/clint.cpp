/**
 * @file clint.cpp
 * @brief CLINT implementation.
 */

#include "clint.hpp"

#include <format>
#include <iostream>
#include <stdexcept>

namespace riscv {

CLINT::CLINT(u32 num_harts, u32 cycles_per_tick)
    : cycles_per_tick_(cycles_per_tick)
{
    if (num_harts == 0 || num_harts > MAX_HARTS)
        throw std::invalid_argument(
            std::format("CLINT: num_harts must be 1..{} (got {})",
                        MAX_HARTS, num_harts));
    if (cycles_per_tick == 0)
        throw std::invalid_argument("CLINT: cycles_per_tick must be >= 1");

    harts_.resize(num_harts);
}

// ── Simulation interface ───────────────────────────────────────────────

void CLINT::tick(cycle_t cycle)
{
    last_cycle_ = cycle;
    const u64 now = cycle / cycles_per_tick_ + mtime_offset_;
    if (now == mtime_) return;

    mtime_ = now;
    for (u32 h = 0; h < num_harts(); ++h)
        update_timer_line(h);
}

void CLINT::set_mtime(u64 value)
{
    mtime_offset_ = value - last_cycle_ / cycles_per_tick_;
    mtime_        = value;
    for (u32 h = 0; h < num_harts(); ++h)
        update_timer_line(h);
}

void CLINT::update_timer_line(u32 hart)
{
    auto& hs = harts_[hart];
    const bool level = (mtime_ >= hs.mtimecmp);
    if (level == hs.mtip) return;

    hs.mtip = level;
    if (trace_)
        std::cout << std::format("[CLINT] hart{} MTIP {} | "
                                 "mtime={} mtimecmp={}\n",
                                 hart, level ? "ASSERT" : "CLEAR",
                                 mtime_, hs.mtimecmp);
    if (timer_cb_) timer_cb_(hart, level);
}

void CLINT::set_mtimecmp(u32 hart, u64 cmp)
{
    harts_.at(hart).mtimecmp = cmp;
    update_timer_line(hart);
}

void CLINT::set_msip(u32 hart, bool level)
{
    auto& hs = harts_.at(hart);
    if (hs.msip == level) return;

    hs.msip = level;
    if (trace_)
        std::cout << std::format("[CLINT] hart{} MSIP {}\n", hart,
                                 level ? "ASSERT" : "CLEAR");
    if (software_cb_) software_cb_(hart, level);
}

// ── Address decode ─────────────────────────────────────────────────────

bool CLINT::decode_msip(addr_t addr, u32& hart) const noexcept
{
    if (addr < MSIP_BASE || (addr & 3)) return false;
    const u32 h = (addr - MSIP_BASE) / 4;
    if (h >= num_harts()) return false;
    hart = h;
    return true;
}

bool CLINT::decode_mtimecmp(addr_t addr, u32& hart, bool& high) const noexcept
{
    if (addr < MTIMECMP_BASE || (addr & 3)) return false;
    const u32 h = (addr - MTIMECMP_BASE) / 8;
    if (h >= num_harts()) return false;
    hart = h;
    high = ((addr - MTIMECMP_BASE) & 4) != 0;
    return true;
}

// ── MMIO ───────────────────────────────────────────────────────────────

MemoryResult CLINT::read32(addr_t addr) const
{
    if (addr == MTIME_ADDR)     return {static_cast<u32>(mtime_),       1, true};
    if (addr == MTIME_ADDR + 4) return {static_cast<u32>(mtime_ >> 32), 1, true};

    u32 hart = 0;
    bool high = false;
    if (decode_msip(addr, hart))
        return {harts_[hart].msip ? 1u : 0u, 1, true};
    if (decode_mtimecmp(addr, hart, high))
    {
        const u64 cmp = harts_[hart].mtimecmp;
        return {static_cast<u32>(high ? cmp >> 32 : cmp), 1, true};
    }

    return {0, 1, false};
}

MemoryResult CLINT::write32(addr_t addr, u32 value)
{
    if (addr == MTIME_ADDR)
    {
        set_mtime((mtime_ & 0xFFFF'FFFF'0000'0000ULL) | value);
        return {value, 1, true};
    }
    if (addr == MTIME_ADDR + 4)
    {
        set_mtime((mtime_ & 0x0000'0000'FFFF'FFFFULL) | (u64{value} << 32));
        return {value, 1, true};
    }

    u32 hart = 0;
    bool high = false;
    if (decode_msip(addr, hart))
    {
        set_msip(hart, (value & 1) != 0);  // upper bits are WIRI
                                           // (hardwired zero)
        return {value, 1, true};
    }
    if (decode_mtimecmp(addr, hart, high))
    {
        const u64 cmp = harts_[hart].mtimecmp;
        set_mtimecmp(hart, high ? (cmp & 0x0000'0000'FFFF'FFFFULL) | (u64{value} << 32)
                                : (cmp & 0xFFFF'FFFF'0000'0000ULL) | value);
        return {value, 1, true};
    }

    return {0, 1, false};
}

bool CLINT::valid_address(addr_t addr, size_t size) const
{
    return size > 0 && u64{addr} + size <= REG_SIZE;
}

} // namespace riscv

