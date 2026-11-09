/**
 * @file clint.hpp
 * @brief Core-Local Interruptor (CLINT)
 *
 * Provides one shared 64-bit @c mtime counter plus, for each hart,
 * a 64-bit @c mtimecmp comparator (drives @c mip.MTIP) and and @c msip
 * register (drives @c mip.MSIP, used for inter-processor interrupts).
 *
 * @par MMIO register map (SiFive CLINT layout)
 * @code{.txt}
 *   ┌──────────────────┬──────────────┬───────┬────────────────────────────────┐
 *   │ Offset           │ Name         │ Width │ Description                    │
 *   ├──────────────────┼──────────────┼───────┼────────────────────────────────┤
 *   │ 0x0000 + 4·hart  │ msip[hart]   │ u32   │ Bit 0 drives mip.MSIP          │
 *   │ 0x4000 + 8·hart  │ mtimecmp[h]  │ u64   │ Timer compare (lo @ +0, hi +4) │
 *   │ 0xBFF8           │ mtime        │ u64   │ Shared timer (lo @ +0, hi +4)  │
 *   └──────────────────┴──────────────┴───────┴────────────────────────────────┘
 * @endcode
 *
 * @par Timebase
 * @c mtime advances once every @c cycles_per_tick core cycles (default 1).
 * Real platforms run @c mtime from a fixed-frequency timebase that is
 * independent of the core clock; a divider > 1 models that.
 *
 * @par Writing mtime
 * Software may write @c mtime (as the spec allows). The CLINT then keeps an
 * offset from the simulation clock so @c mtime continues counting from the
 * written value.
 *
 * @see RISC-V Privileged ISA specification, Section 3.2.1 (Machine Timer
 *      Registers); SiFive CLINT documentation for the register layout.
 */

#pragma once

#include "memory.hpp"
#include "types.hpp"

#include <functional>
#include <vector>

namespace riscv {

class CLINT : public Memory {
public:
    static constexpr u32    MAX_HARTS     = 4095;
    static constexpr addr_t MSIP_BASE     = 0x0000;
    static constexpr addr_t MTIMECMP_BASE = 0x4000;
    static constexpr addr_t MTIME_ADDR    = 0xBFF8;
    static constexpr addr_t REG_SIZE      = 0x10000;  ///< MMIO window size.

    /**
     * @param num_harts        Number of harts served (1 .. MAX_HARTS).
     * @param cycles_per_tick  Core cycles per @c mtime increment (>= 1).
     * @throw std::invalid_argument On out-of-range parameters.
     */
    explicit CLINT(u32 num_harts = 1, u32 cycles_per_tick = 1);

    /// @name Memory-mapped interface (32-bit accesses only)
    /// @{
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t) const override { return {0, 1, false}; }
    [[nodiscard]] MemoryResult read8(addr_t)  const override { return {0, 1, false}; }

    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t, u16) override { return {0, 1, false}; }
    MemoryResult write8(addr_t, u8)   override { return {0, 1, false}; }

    void load(addr_t, std::span<const u8>) override {}
    [[nodiscard]] bool valid_address(addr_t addr, size_t = 1) const override;
    /// @}

    /// @name Simulation interface
    /// @{

    /// Advance to core cycle @p cycle; updates @c mtime and re-evaluates
    /// MTIP lines.
    void tick(cycle_t cycle);

    [[nodiscard]] u64 mtime() const noexcept { return mtime_; }
    /// @}

    /// @name Per-hart state (also reachable over MMIO)
    /// @{
    [[nodiscard]] u32 num_harts() const noexcept
    { return static_cast<u32>(harts_.size()); }

    void set_mtimecmp(u32 hart, u64 cmp);
    [[nodiscard]] u64 mtimecmp(u32 hart) const { return harts_.at(hart).mtimecmp; }

    void set_msip(u32 hart, bool level);
    [[nodiscard]] bool msip(u32 hart) const { return harts_.at(hart).msip; }

    /// @return Current level of the hart's MTIP line (mtime >= mtimecmp).
    [[nodiscard]] bool timer_pending(u32 hart) const { return harts_.at(hart).mtip; }
    /// @}

    /// @name Interrupt lines
    /// @{

    /// Called on every level change of a hart's MTIP or MSIP line.
    using LineCallback = std::function<void(u32 hart, bool level)>;
    void on_timer_line(LineCallback cb)    { timer_cb_ = std::move(cb); }
    void on_software_line(LineCallback cb) { software_cb_ = std::move(cb); }
    /// @}

    void set_trace(bool enable) noexcept { trace_ = enable; }

private:
    struct HartState
    {
        u64  mtimecmp = ~u64{0};  ///< Reset value: never fires.
        bool msip     = false;
        bool mtip     = false;
    };

    std::vector<HartState> harts_;
    u32     cycles_per_tick_;
    cycle_t last_cycle_   = 0;
    u64     mtime_        = 0;
    u64     mtime_offset_ = 0;  ///< mtime = cycle / cycles_per_tick + offset (mod 2^64).

    LineCallback timer_cb_;
    LineCallback software_cb_;
    bool trace_ = false;

    void set_mtime(u64 value);
    void update_timer_line(u32 hart);

    /// Decode an MMIO address to (hart, is_high_word); false if not a register.
    [[nodiscard]] bool decode_mtimecmp(addr_t addr, u32& hart, bool& high) const noexcept;
    [[nodiscard]] bool decode_msip(addr_t addr, u32& hart) const noexcept;
};

} // namespace riscv

