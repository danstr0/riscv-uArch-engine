/**
 * @file test_clint.cpp
 * @brief Device-level tests for the CLINT.
 *
 * @par Sections
 * @code{.txt}
 *   1 : Basic state
 *   2 : Compare match
 *   3 : Clear via new mtimecmp
 *   4 : MMIO register access
 *   5 : Line callbacks
 *   6 : Multiple harts, msip, timebase
 *   7 : Edge cases
 * @endcode
 */

#include "core/clint.hpp"
#include "test_framework.hpp"

#include <stdexcept>
#include <vector>

using namespace riscv;

namespace {

constexpr addr_t MTIMECMP0_LO = CLINT::MTIMECMP_BASE;
constexpr addr_t MTIMECMP0_HI = CLINT::MTIMECMP_BASE + 4;
constexpr addr_t MTIME_LO     = CLINT::MTIME_ADDR;
constexpr addr_t MTIME_HI     = CLINT::MTIME_ADDR + 4;

} // namespace

// ═══════════════════════════════════════════════════════════════════════
//  1. Basic state
// ═══════════════════════════════════════════════════════════════════════

TEST(clint_no_interrupt_initially)
{
    CLINT clint;
    ASSERT(!clint.timer_pending(0));
    ASSERT(!clint.msip(0));
    return true;
}

TEST(clint_default_compare_is_max)
{
    CLINT clint;
    ASSERT_HEX_EQ(clint.mtimecmp(0), ~u64{0});
    clint.tick(1'000'000);
    ASSERT(!clint.timer_pending(0));
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Compare match
// ═══════════════════════════════════════════════════════════════════════

TEST(clint_fires_at_compare)
{
    CLINT clint;
    clint.set_mtimecmp(0, 100);
    clint.tick(99);
    ASSERT(!clint.timer_pending(0));
    clint.tick(100);
    ASSERT(clint.timer_pending(0));
    return true;
}

TEST(clint_fires_past_compare)
{
    CLINT clint;
    clint.set_mtimecmp(0, 100);
    clint.tick(200);
    ASSERT(clint.timer_pending(0));
    return true;
}

TEST(clint_timer_level_stays_asserted)
{
    CLINT clint;
    clint.set_mtimecmp(0, 10);
    clint.tick(10);
    clint.tick(100);
    ASSERT(clint.timer_pending(0));
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Clear by writing new mtimecmp
// ═══════════════════════════════════════════════════════════════════════

TEST(clint_clear_by_new_compare)
{
    CLINT clint;
    clint.set_mtimecmp(0, 50);
    clint.tick(60);
    ASSERT(clint.timer_pending(0));
    clint.set_mtimecmp(0, 200);
    ASSERT(!clint.timer_pending(0));
    return true;
}

TEST(clint_re_fires_after_clear)
{
    CLINT clint;
    clint.set_mtimecmp(0, 50);
    clint.tick(60);
    clint.set_mtimecmp(0, 100);
    ASSERT(!clint.timer_pending(0));
    clint.tick(99);
    ASSERT(!clint.timer_pending(0));
    clint.tick(100);
    ASSERT(clint.timer_pending(0));
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// 4. MMIO register access
// ═══════════════════════════════════════════════════════════════════════

TEST(clint_mmio_read_mtime)
{
    CLINT clint;
    clint.tick(0x1'2345'6789ULL);
    ASSERT_HEX_EQ(clint.read32(MTIME_LO).value, 0x2345'6789u);
    ASSERT_HEX_EQ(clint.read32(MTIME_HI).value, 0x1u);
    return true;
}

TEST(clint_mmio_write_compare)
{
    CLINT clint;
    ASSERT(clint.write32(MTIMECMP0_HI, 0).ok);
    ASSERT(clint.write32(MTIMECMP0_LO, 100).ok);
    ASSERT_EQ(clint.mtimecmp(0), 100u);
    ASSERT_EQ(clint.read32(MTIMECMP0_LO).value, 100u);
    clint.tick(100);
    ASSERT(clint.timer_pending(0));
    return true;
}

TEST(clint_mmio_write_compare_clears)
{
    CLINT clint;
    clint.set_mtimecmp(0, 50);
    clint.tick(60);
    ASSERT(clint.write32(MTIMECMP0_LO, 200).ok);
    ASSERT(!clint.timer_pending(0));
    return true;
}

TEST(clint_mmio_write_mtime_continues_counting)
{
    CLINT clint;
    clint.tick(42);
    ASSERT(clint.write32(MTIME_LO, 1000).ok);
    ASSERT_EQ(clint.mtime(), 1000u);
    clint.tick(52);                      // 10 cycles later
    ASSERT_EQ(clint.mtime(), 1010u);
    return true;
}

TEST(clint_mmio_invalid_accesses)
{
    CLINT clint;
    ASSERT(!clint.read32(0x0004).ok);    // msip[1] with only 1 hart
    ASSERT(!clint.read32(0x4008).ok);    // mtimecmp[1] with only 1 hart
    ASSERT(!clint.read32(0x0002).ok);    // misaligned
    ASSERT(!clint.read16(MTIME_LO).ok);  // sub-word
    ASSERT(!clint.read32(0x8000).ok);    // hole
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Line callbacks
// ═══════════════════════════════════════════════════════════════════════

TEST(clint_timer_callback_on_edges_only)
{
    CLINT clint;
    std::vector<bool> levels;
    u32 last_hart = 99;
    clint.on_timer_line([&](u32 hart, bool level)
    {
        last_hart = hart;
        levels.push_back(level);
    });

    clint.set_mtimecmp(0, 50);
    clint.tick(50);
    clint.tick(51);                // still high: no callback
    clint.set_mtimecmp(0, 200);
    ASSERT_EQ(levels.size(), 2u);
    ASSERT_EQ(last_hart, 0u);
    ASSERT(levels[0] == true);
    ASSERT(levels[1] == false);
    return true;
}

TEST(clint_software_callback)
{
    CLINT clint(2);
    std::vector<std::pair<u32, bool>> events;
    clint.on_software_line([&](u32 h, bool l)
    {
        events.emplace_back(h, l);
    });

    ASSERT(clint.write32(0x0004, 1).ok);  // msip[1] = 1
    ASSERT(clint.write32(0x0004, 3).ok);  // still 1 (bit 0 only): no edge
    ASSERT(clint.write32(0x0004, 0).ok);
    ASSERT_EQ(events.size(), 2u);
    ASSERT_EQ(events[0].first, 1u);
    ASSERT(events[0].second);
    ASSERT(!events[1].second);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Multiple harts, msip, timebase
// ═══════════════════════════════════════════════════════════════════════

TEST(clint_per_hart_compare_is_independent)
{
    CLINT clint(3);
    clint.set_mtimecmp(1, 100);
    clint.tick(150);
    ASSERT(!clint.timer_pending(0));
    ASSERT(clint.timer_pending(1));
    ASSERT(!clint.timer_pending(2));

    ASSERT(clint.write32(CLINT::MTIMECMP_BASE + 2 * 8, 120).ok);   // hart 2 lo
    ASSERT(clint.write32(CLINT::MTIMECMP_BASE + 2 * 8 + 4, 0).ok); // hart 2 hi
    ASSERT(clint.timer_pending(2));
    return true;
}

TEST(clint_msip_reads_back_bit0_only)
{
    CLINT clint(2);
    ASSERT(clint.write32(0x0000, 0xFFFF'FFFF).ok);
    ASSERT_EQ(clint.read32(0x0000).value, 1u);
    ASSERT(clint.msip(0));
    ASSERT(!clint.msip(1));
    return true;
}

TEST(clint_timebase_divider)
{
    CLINT clint(1, 10);               // mtime ticks every 10 cycles
    clint.tick(95);
    ASSERT_EQ(clint.mtime(), 9u);
    clint.set_mtimecmp(0, 10);
    ASSERT(!clint.timer_pending(0));
    clint.tick(100);
    ASSERT(clint.timer_pending(0));
    return true;
}

TEST(clint_bad_construction_throws)
{
    bool threw = false;
    try { CLINT c(0); } catch (const std::invalid_argument&) { threw = true; }
    ASSERT(threw);
    threw = false;
    try { CLINT c(1, 0); } catch (const std::invalid_argument&) { threw = true; }
    ASSERT(threw);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// 7. Edge cases
// ═══════════════════════════════════════════════════════════════════════

TEST(clint_immediate_match)
{
    CLINT clint;
    clint.tick(100);
    clint.set_mtimecmp(0, 100);
    ASSERT(clint.timer_pending(0));
    return true;
}

TEST(clint_compare_zero_fires_at_reset)
{
    CLINT clint;
    clint.set_mtimecmp(0, 0);        // mtime = 0 >= 0
    ASSERT(clint.timer_pending(0));
    return true;
}

