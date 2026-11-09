/**
 * @file plic.hpp
 * @brief Platform-Level Interrupt Controller (PLIC) for RISC-V.
 *
 * Models a PLIC with up to 31 interrupt sources (source 0 is reserved as
 * "no interrupt") and one M-mode context per hart, following the register
 * layout of the RISC-V PLIC Specification v1.0.0.
 *
 * @par Interrupt lifecycle
 * A device asserts an interrupt via @c set_pending(). Each context sees
 * the sources it has enabled whose priority exceeds its threshold; when
 * any such source is pending the context's MEIP line is raised. A hart
 * claims an interrupt (receiving the highest-priority pending ID and
 * clearing its pending bit), services it, then writes the ID back to
 * complete.
 *
 * @par MMIO register map
 * @code{.txt}
 *   0x000000 + 4·src          priority[src]        (3 bits)
 *   0x001000                  pending bits 0-31    (read-only)
 *   0x002000 + 0x80·ctx       enable bits 0-31 for context ctx
 *   0x200000 + 0x1000·ctx     priority threshold for context ctx
 *   0x200004 + 0x1000·ctx     claim (read) / complete (write) for context ctx
 * @endcode
 *
 * @par Context API
 * Methods without a context argument operate on context 0, which keeps the
 * single-hart use case terse.
 *
 * @see RISC-V PLIC Specification v1.0.0.
 */

#pragma once

#include "memory.hpp"
#include "types.hpp"

#include <array>
#include <functional>
#include <vector>

namespace riscv {

/// Static PLIC configuration.
struct PLICConfig
{
    /// Maximum interrupt sources. Source 0 is reserved; devices use 1-31.
    static constexpr u32 MAX_SOURCES  = 32;
    /// Maximum contexts supported by this model.
    static constexpr u32 MAX_CONTEXTS = 16;
};

/**
 * @brief MMIO-mapped Platform-Level Interrupt Controller.
 *
 * All registers are 32-bit; sub-word accesses are rejected.
 */
class PLIC : public Memory {
public:
    /// @throw std::invalid_argument If @p num_contexts is @c 0 or above @c MAX_CONTEXTS.
    explicit PLIC(u32 num_contexts = 1);

    /// Register span used by context 0.
    static constexpr addr_t REG_SIZE = 0x200008;
    /// Address window reserved for the PLIC on the bus (spec: 64 MiB).
    static constexpr addr_t WINDOW_SIZE = 0x4000000;

    /// @name Memory-mapped interface
    /// @{
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t) const override { return {0, 1, false}; }
    [[nodiscard]] MemoryResult read8(addr_t)  const override { return {0, 1, false}; }

    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t, u16) override { return {0, 1, false}; }
    MemoryResult write8(addr_t, u8)   override { return {0, 1, false}; }

    void load(addr_t, std::span<const u8>) override {}
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    /// @}

    [[nodiscard]] u32 num_contexts() const noexcept
    { return static_cast<u32>(contexts_.size()); }

    /// @name Device signal interface
    /// @{

    /// Assert an interrupt from source @p source (1-31).
    void set_pending(u32 source);

    /// Manually clear a pending bit.
    void clear_pending(u32 source);

    [[nodiscard]] bool is_pending(u32 source) const;
    /// @}

    /// @name Per-context interface
    /// @{

    /// @return @c true if any source enabled for @p ctx is pending above
    ///         its threshold.
    [[nodiscard]] bool interrupt_pending(u32 ctx) const;


    /**
     * @brief Claim the highest-priority pending interrupt for context @p ctx.
     * @return Source ID (1-31), or @c 0 if none pending.
     */
    [[nodiscard]] u32 claim(u32 ctx);

    /// Complete a previously claimed interrupt, allowing it to trigger again.
    void complete(u32 ctx, u32 source);

    void set_enable(u32 ctx, u32 source, bool enable);
    void set_threshold(u32 ctx, u32 threshold);
    [[nodiscard]] bool enabled(u32 ctx, u32 source) const;
    [[nodiscard]] u32  threshold(u32 ctx) const;
    /// @}

    /// @name Context-0 shorthands
    /// @{
    [[nodiscard]] bool interrupt_pending() const { return interrupt_pending(0); }
    [[nodiscard]] u32  claim()                   { return claim(0); }
    void complete(u32 source)                    { complete(0, source); }
    void set_enable(u32 source, bool enable)     { set_enable(0, source, enable); }
    void set_threshold(u32 threshold)            { set_threshold(0, threshold); }
    [[nodiscard]] bool enabled(u32 source) const { return enabled(0, source); }
    [[nodiscard]] u32  threshold()         const { return threshold(0); }
    /// @}

    /// @name Priorities (global)
    /// @{
    void set_priority(u32 source, u32 priority);
    [[nodiscard]] u32 priority(u32 source) const;
    /// @}

    /// @name Interrupt lines
    /// @{

    /// Callback simulating a context's MEIP wire; invoked on every re-evaluation.
    using ContextCallback = std::function<void(u32 ctx, bool pending)>;
    void on_context_line(ContextCallback cb) { context_cb_ = std::move(cb); }

    /// Context-0 MEIP callback (single-hart shorthand).
    using NotifyCallback = std::function<void(bool pending)>;
    void set_notify(NotifyCallback cb) { notify_cb_ = std::move(cb); }
    /// @}

    /// @name Debug
    /// @{
    void set_trace(bool enable) noexcept { trace_ = enable; }
    /// @}

private:
    struct Context
    {
        u32         enable_bits = 0;  ///< Bit i enables source i for this context.
        u32         threshold   = 0;  ///< Priorities ≤ threshold are masked.
        mutable u32 claimed     = 0;  ///< Source currently claimed (0 = none).
    };

    std::array<u32, PLICConfig::MAX_SOURCES> priorities_{};
    u32                  pending_bits_ = 0;  ///< Bit i = source i pending.
    std::vector<Context> contexts_;

    bool trace_ = false;

    ContextCallback context_cb_;
    NotifyCallback  notify_cb_;

    /// Re-evaluate every context's MEIP line and invoke callbacks.
    void notify() const;

    [[nodiscard]] bool valid_source(u32 s) const noexcept
    { return s > 0 && s < PLICConfig::MAX_SOURCES; }

    /// @name MMIO address map
    /// @{
    static constexpr addr_t PRIORITY_BASE    = 0x000000;
    static constexpr addr_t PENDING_BASE     = 0x001000;
    static constexpr addr_t ENABLE_BASE      = 0x002000;
    static constexpr addr_t ENABLE_STRIDE    = 0x80;
    static constexpr addr_t CONTEXT_BASE     = 0x200000;
    static constexpr addr_t CONTEXT_STRIDE   = 0x1000;
    static constexpr addr_t THRESHOLD_OFFSET = 0x0;
    static constexpr addr_t CLAIM_OFFSET     = 0x4;
    /// @}
};

} // namespace riscv
