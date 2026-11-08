/**
 * @file test_sparse_memory.cpp
 * @brief Tests for the demand-allocated SparseMemory model.
 */

#include "test_framework.hpp"

#include "core/cache.hpp"
#include "core/sparse_memory.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace riscv;

// ═══════════════════════════════════════════════════════════════════════
// 1. Allocation behavior
// ═══════════════════════════════════════════════════════════════════════

TEST(sparse_starts_empty)
{
    SparseMemory mem(0, 64 * 1024);
    ASSERT_EQ(mem.pages_allocated(), 0u);
    ASSERT_EQ(mem.bytes_allocated(), 0u);
    return true;
}

TEST(sparse_read_unwritten_is_zero_and_doesnt_allocate)
{
    SparseMemory mem(0, 64 * 1024);
    auto r32 = mem.read32(0x1000);
    auto r16 = mem.read16(0x2002);
    auto r8  = mem.read8(0x3003);
    ASSERT(r32.ok && r16.ok && r8.ok);
    ASSERT_EQ(r32.value, 0u);
    ASSERT_EQ(r16.value, 0u);
    ASSERT_EQ(r8.value, 0u);
    ASSERT_EQ(mem.pages_allocated(), 0u);
    return true;
}

TEST(sparse_write_allocates_one_page)
{
    SparseMemory mem(0, 64 * 1024);
    ASSERT(mem.write32(0x1004, 0xDEAD'BEEF).ok);
    ASSERT_EQ(mem.pages_allocated(), 1u);
    ASSERT(mem.write8(0x1FFF, 0x12).ok);     // same page
    ASSERT_EQ(mem.pages_allocated(), 1u);
    ASSERT(mem.write16(0x2000, 0x3456).ok);  // next page
    ASSERT_EQ(mem.pages_allocated(), 2u);
    ASSERT_EQ(mem.bytes_allocated(), 2u * SparseMemory::kPageSize);
    return true;
}

TEST(sparse_full_address_space_is_cheap)
{
    SparseMemory mem(0, u64{1} << 32);
    ASSERT(mem.write32(0xFFFF'FFFC, 0xCAFE'F00D).ok);
    ASSERT(mem.write32(0x0000'0000, 0x1122'3344).ok);
    ASSERT_HEX_EQ(mem.read32(0xFFFF'FFFC).value, 0xCAFE'F00Du);
    ASSERT_HEX_EQ(mem.read32(0x0000'0000).value, 0x1122'3344u);
    ASSERT_EQ(mem.pages_allocated(), 2u);
    return true;
}

TEST(sparse_clear_releases_pages)
{
    SparseMemory mem(0, 64 * 1024);
    (void)mem.write32(0x100, 42);
    (void)mem.write32(0x5000, 43);
    mem.clear();
    ASSERT_EQ(mem.pages_allocated(), 0u);
    ASSERT_EQ(mem.read32(0x100).value, 0u);
    ASSERT(mem.write32(0x100, 7).ok);        // usable after clear
    ASSERT_EQ(mem.read32(0x100).value, 7u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Data integrity
// ═══════════════════════════════════════════════════════════════════════

TEST(sparse_typed_roundtrip_little_endian)
{
    SparseMemory mem(0, 64 * 1024);
    (void)mem.write32(0x200, 0x1122'3344);
    ASSERT_HEX_EQ(mem.read8(0x200).value, 0x44u);
    ASSERT_HEX_EQ(mem.read8(0x203).value, 0x11u);
    ASSERT_HEX_EQ(mem.read16(0x200).value, 0x3344u);
    ASSERT_HEX_EQ(mem.read16(0x202).value, 0x1122u);

    (void)mem.write16(0x300, 0xBEEF);
    (void)mem.write8(0x302, 0xAD);
    (void)mem.write8(0x303, 0xDE);
    ASSERT_HEX_EQ(mem.read32(0x300).value, 0xDEAD'BEEFu);
    return true;
}

TEST(sparse_lookup_cache_survives_page_alternation)
{
    SparseMemory mem(0, 1024 * 1024);
    for (u32 i = 0; i < 64; ++i)
    {
        addr_t a = i * SparseMemory::kPageSize * 3
                 + (i * 4 % SparseMemory::kPageSize);
        ASSERT_EQ(mem.read32(a + SparseMemory::kPageSize).value, 0u);
        ASSERT(mem.write32(a, i + 1).ok);
    }
    for (u32 i = 0; i < 64; ++i)
    {
        addr_t a = i * SparseMemory::kPageSize * 3
                 + (i * 4 % SparseMemory::kPageSize);
        ASSERT_EQ(mem.read32(a).value, i + 1);
    }
    ASSERT_EQ(mem.pages_allocated(), 64u);
    return true;
}

TEST(sparse_line_ops_cross_page_boundary)
{
    SparseMemory mem(0, 64 * 1024);
    std::array<u8, 32> src{};
    for (u32 i = 0; i < src.size(); ++i) src[i] = static_cast<u8>(0xA0 + i);

    const addr_t a = 0x1FF0;  // 16 bytes in each page
    (void)mem.write_line(a, src.data(), static_cast<u32>(src.size()));
    ASSERT_EQ(mem.pages_allocated(), 2u);

    std::array<u8, 32> dst{};
    (void)mem.read_line(a, dst.data(), static_cast<u32>(dst.size()));
    ASSERT(dst == src);
    return true;
}

TEST(sparse_read_line_mixes_allocated_and_unallocated)
{
    SparseMemory mem(0, 64 * 1024);
    (void)mem.write8(0x1FFF, 0x5A);              // last byte of page 1
    std::array<u8, 4> dst{0xFF, 0xFF, 0xFF, 0xFF};
    (void)mem.read_line(0x1FFE, dst.data(), 4);  // spans into empty page 2
    ASSERT_HEX_EQ(dst[0], 0x00u);
    ASSERT_HEX_EQ(dst[1], 0x5Au);
    ASSERT_HEX_EQ(dst[2], 0x00u);
    ASSERT_HEX_EQ(dst[3], 0x00u);
    ASSERT_EQ(mem.pages_allocated(), 1u);
    return true;
}

TEST(sparse_load_spans_pages)
{
    SparseMemory mem(0, 64 * 1024);
    std::vector<u8> blob(3 * SparseMemory::kPageSize + 100);
    for (size_t i = 0; i < blob.size(); ++i)
        blob[i] = static_cast<u8>(i * 7);

    mem.load(0x0F00, blob);
    ASSERT_EQ(mem.pages_allocated(), 4u);     // 0x0F00..0x3F63 -> pages 0-3
    for (size_t i = 0; i < blob.size(); i += 97)
        ASSERT_EQ(mem.read8(0x0F00 + static_cast<addr_t>(i)).value,
                  blob[i]);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Faults
// ═══════════════════════════════════════════════════════════════════════

TEST(sparse_misaligned_faults)
{
    SparseMemory mem(0, 64 * 1024);
    ASSERT(!mem.read32(0x101).ok);
    ASSERT(!mem.read16(0x101).ok);
    ASSERT(!mem.write32(0x102, 1).ok);
    ASSERT(!mem.write16(0x103, 1).ok);
    ASSERT_EQ(mem.pages_allocated(), 0u);  // faulting writes don't allocate
    return true;
}

TEST(sparse_out_of_range_faults)
{
    SparseMemory mem(0, 64 * 1024);
    ASSERT(!mem.read8(0x10000).ok);
    ASSERT(!mem.read32(0x10000).ok);
    ASSERT(!mem.write8(0x10000, 1).ok);
    ASSERT(!mem.write32(0xFFFF'FFFC, 1).ok);
    ASSERT(mem.read32(0xFFFC).ok);        // last valid word
    ASSERT_EQ(mem.pages_allocated(), 0u);
    return true;
}

TEST(sparse_no_ranges_faults_everything)
{
    SparseMemory mem;
    ASSERT(!mem.read8(0).ok);
    ASSERT(!mem.write8(0, 1).ok);
    ASSERT(!mem.valid_address(0));
    return true;
}

TEST(sparse_load_out_of_range_throws)
{
    SparseMemory mem(0, 4096);
    std::vector<u8> blob(16, 0xAB);
    bool threw = false;
    try { mem.load(4090, blob); }
    catch (const MemoryAccessException& e)
    {
        threw = true;
        ASSERT_HEX_EQ(e.address(), 4090u);
    }
    ASSERT(threw);
    ASSERT_EQ(mem.pages_allocated(), 0u);  // nothing partially written
    return true;
}

TEST(sparse_line_ops_out_of_range_are_noops)
{
    SparseMemory mem(0, 4096);
    std::array<u8, 16> buf{};
    buf.fill(0x77);
    (void)mem.write_line(4088, buf.data(), 16);
    ASSERT_EQ(mem.pages_allocated(), 0u);
    (void)mem.read_line(4088, buf.data(), 16);
    ASSERT_HEX_EQ(buf[0], 0x77u);      // dest untouched, like FlatMemory
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// 4. RAM ranges
// ═══════════════════════════════════════════════════════════════════════

TEST(sparse_multiple_ranges)
{
    SparseMemory mem({{0x0000'0000, 0x1'0000}, {0x8000'0000, 0x10'0000}});
    ASSERT(mem.write32(0x0000'FFFC, 1).ok);
    ASSERT(mem.write32(0x8000'0000, 2).ok);
    ASSERT(mem.write32(0x800F'FFFC, 3).ok);
    ASSERT(!mem.write32(0x0001'0000, 4).ok);     // gap
    ASSERT(!mem.write32(0x8010'0000, 5).ok);     // past second range
    ASSERT(!mem.valid_address(0x0000'FFFC, 8));  // runs into the gap
    ASSERT_EQ(mem.ranges().size(), 2u);
    ASSERT_HEX_EQ(mem.ranges()[0].base, 0u);     // sorted
    return true;
}

TEST(sparse_adjacent_ranges_merge)
{
    SparseMemory mem;
    mem.add_range(0x2000, 0x1000);
    mem.add_range(0x1000, 0x1000);
    ASSERT_EQ(mem.ranges().size(), 1u);
    ASSERT_HEX_EQ(mem.ranges()[0].base, 0x1000u);
    ASSERT_EQ(mem.ranges()[0].size, 0x2000u);
    ASSERT(mem.valid_address(0x1FF0, 0x20));   // spans the seam
    return true;
}

TEST(sparse_invalid_ranges_throw)
{
    auto throws = [](auto&& fn)
    {
        try { fn(); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    ASSERT(throws([] { SparseMemory m(0, 0); }));
    ASSERT(throws([] { SparseMemory m(0xFFFF'F000, 0x2000); }));
    ASSERT(throws([] { SparseMemory m({{0, 0x2000}, {0x1000, 0x2000}}); }));
    ASSERT(!throws([] { SparseMemory m(0, u64{1} << 32); })); // full space OK
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Integration
// ═══════════════════════════════════════════════════════════════════════

TEST(sparse_behind_cache)
{
    auto mem = std::make_shared<SparseMemory>(0, 1024 * 1024);
    CacheConfig c{ .size_bytes = 4096, .line_size = 64, .associativity = 2,
                   .hit_latency = 1, .miss_penalty = 10 };
    Cache l1(c, mem);

    for (u32 i = 0; i < 1024; ++i)
        ASSERT(l1.write32(i * 64, i * 3 + 1).ok);  // 64 KiB stride forces evictions
    for (u32 i = 0; i < 1024; ++i)
        ASSERT_EQ(l1.read32(i * 64).value, i * 3 + 1);
    ASSERT(mem->pages_allocated() <= 16u);         // 64 KiB of footprint
    return true;
}

