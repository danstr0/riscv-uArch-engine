/**
 * @file plic.cpp
 * @brief Multi-context PLIC implementation.
 */

#include "plic.hpp"

#include <format>
#include <iostream>
#include <stdexcept>

namespace riscv {

PLIC::PLIC(u32 num_contexts)
{
    if (num_contexts == 0 || num_contexts > PLICConfig::MAX_CONTEXTS)
        throw std::invalid_argument(
            std::format("PLIC: num_contexts must be 1..{} (got {})",
                        PLICConfig::MAX_CONTEXTS, num_contexts));
    contexts_.resize(num_contexts);
}

// ── MMIO reads ─────────────────────────────────────────────────────────

MemoryResult PLIC::read32(addr_t addr) const
{
    if (addr & 3) return {0, 1, false};

    // Source priorities: 0x000 + source * 4
    if (addr < PRIORITY_BASE + PLICConfig::MAX_SOURCES * 4)
        return {priorities_[(addr - PRIORITY_BASE) / 4], 1, true};

    if (addr == PENDING_BASE)
        return {pending_bits_, 1, true};

    // Enable bits: one 32-bit word per context (sources 0-31)
    if (addr >= ENABLE_BASE
     && addr < ENABLE_BASE + ENABLE_STRIDE * num_contexts())
    {
        const addr_t rel = addr - ENABLE_BASE;
        if (rel % ENABLE_STRIDE != 0) return {0, 1, true};
            return {contexts_[rel / ENABLE_STRIDE].enable_bits, 1, true};
    }

    // Per-context threshold / claim
    if (addr >= CONTEXT_BASE
     && addr < CONTEXT_BASE + CONTEXT_STRIDE * num_contexts())
    {
        const u32    ctx = (addr - CONTEXT_BASE) / CONTEXT_STRIDE;
        const addr_t off = (addr - CONTEXT_BASE) % CONTEXT_STRIDE;
        if (off == THRESHOLD_OFFSET) return {contexts_[ctx].threshold, 1, true};
        if (off == CLAIM_OFFSET) return {const_cast<PLIC*>(this)->claim(ctx), 1, true};
    }

    return {0, 1, false};
}

bool PLIC::valid_address(addr_t addr, size_t size) const
{
    return size > 0
        && u64{addr} + size <= CONTEXT_BASE + CONTEXT_STRIDE * num_contexts();
}

// ── MMIO writes ────────────────────────────────────────────────────────

MemoryResult PLIC::write32(addr_t addr, u32 value)
{
    if (addr & 3) return {0, 1, false};

    if (addr < PRIORITY_BASE + PLICConfig::MAX_SOURCES * 4)
    {
        const u32 source = (addr - PRIORITY_BASE) / 4;
        if (source > 0)
        {
            priorities_[source] = value & 0x7;
            notify();
        }
        return {value, 1, true};
    }

    // Pending bits: read-only
    if (addr == PENDING_BASE)
        return {value, 1 , true};

    if (addr >= ENABLE_BASE
     && addr < ENABLE_BASE + ENABLE_STRIDE * num_contexts())
    {
        const addr_t rel = addr - ENABLE_BASE;
        if (rel % ENABLE_STRIDE == 0)
        {
            contexts_[rel / ENABLE_STRIDE].enable_bits = value & ~1u;
            notify();
        }
        return {value, 1, true};
    }

    if (addr >= CONTEXT_BASE
     && addr < CONTEXT_BASE + CONTEXT_STRIDE * num_contexts())
    {
        const u32    ctx = (addr - CONTEXT_BASE) / CONTEXT_STRIDE;
        const addr_t off = (addr - CONTEXT_BASE) % CONTEXT_STRIDE;
        if (off == THRESHOLD_OFFSET)
        {
            set_threshold(ctx, value);
            return {value, 1, true};
        }
        if (off == CLAIM_OFFSET)
        {
            complete(ctx, value);
            return {value, 1, true};
        }
    }

    return {0, 1, false};
}

// ── Device API ─────────────────────────────────────────────────────────

void PLIC::set_pending(u32 source)
{
    if (!valid_source(source)) return;
    pending_bits_ |= (1u << source);

    if (trace_)
        std::cout << std::format("[PLIC] IRQ{} pending\n", source);
    notify();
}

void PLIC::clear_pending(u32 source)
{
    if (!valid_source(source)) return;
    pending_bits_ &= ~(1u << source);

    if (trace_)
        std::cout << std::format("[PLIC] IRQ{} cleared\n", source);
    notify();
}

bool PLIC::is_pending(u32 source) const
{
    return valid_source(source) && (pending_bits_ & (1u << source));
}

bool PLIC::interrupt_pending(u32 ctx) const
{
    const auto& c = contexts_.at(ctx);
    for (u32 i = 1; i < PLICConfig::MAX_SOURCES; ++i)
        if ((pending_bits_ & c.enable_bits & (1u << i))
         && priorities_[i] > c.threshold)
            return true;
    return false;
}

u32 PLIC::claim(u32 ctx)
{
    auto& c = contexts_.at(ctx);
    u32 best = 0;
    u32 best_prio = 0;
    for (u32 i = 1; i < PLICConfig::MAX_SOURCES; ++i)
        if ((pending_bits_ & c.enable_bits & (1u << i))
         && priorities_[i] > c.threshold
         && priorities_[i] > best_prio)  // ties go to the lowest ID
        {
            best = i;
            best_prio = priorities_[i];
        }

    if (best != 0)
    {
        if (trace_)
            std::cout << std::format("[PLIC] ctx{} claim IRQ{} | prio={}\n",
                                     ctx, best, priorities_[best]);

        pending_bits_ &= ~(1u << best);
        c.claimed = best;
        notify();
    }
    return best;
}

void PLIC::complete(u32 ctx, u32 source)
{
    auto& c = contexts_.at(ctx);
    if (!valid_source(source) || source != c.claimed) return;

    if (trace_)
        std::cout << std::format("[PLIC] ctx{} complete IRQ{}\n",
                                 ctx, source);

    c.claimed = 0;
    notify();
}

void PLIC::set_priority(u32 source, u32 prio)
{
    if (!valid_source(source)) return;
    priorities_[source] = prio & 0x7;

    if (trace_)
        std::cout << std::format("[PLIC] IRQ{} -> priority={}\n",
                                 source, priorities_[source]);
    notify();
}

void PLIC::set_enable(u32 ctx, u32 source, bool enable)
{
    auto& c = contexts_.at(ctx);
    if (!valid_source(source)) return;

    if (trace_)
        std::cout << std::format("[PLIC] ctx{} IRQ{} {}\n",
                                 ctx, source, enable ? "enabled" : "disabled");

    if (enable) c.enable_bits |=  (1u << source);
    else        c.enable_bits &= ~(1u << source);
    notify();
}

void PLIC::set_threshold(u32 ctx, u32 threshold)
{
    contexts_.at(ctx).threshold = threshold & 0x7;

    if (trace_)
        std::cout << std::format("[PLIC] ctx{} threshold={}\n",
                                 ctx, threshold & 0x7);
    notify();
}

u32 PLIC::priority(u32 source) const
{
    return source < PLICConfig::MAX_SOURCES ? priorities_[source] : 0;
}

bool PLIC::enabled(u32 ctx, u32 source) const
{
    return source < PLICConfig::MAX_SOURCES
        && (contexts_.at(ctx).enable_bits & (1u << source));
}

u32 PLIC::threshold(u32 ctx) const
{
    return contexts_.at(ctx).threshold;
}

void PLIC::notify() const
{
    if (notify_cb_)
        notify_cb_(interrupt_pending(0));
    if (context_cb_)
        for (u32 ctx = 0; ctx < num_contexts(); ++ctx)
            context_cb_(ctx, interrupt_pending(ctx));
}

} // namespace riscv

