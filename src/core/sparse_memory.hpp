/**
 * @file sparse_memory.hpp
 * @brief Demand-allocated physical RAM backed by fixed-size pages.
 *
 * @c SparseMemory models one or more physical RAM ranges inside the 32-bit
 * address space without reserving host memory up front. Storage is carved
 * into 4 KiB blocks that are allocated the first time they are written;
 * reading a never-written location yields zero without allocating.
 *
 * @par Backing pages vs. architectural pages
 * The 4 KiB block size is a host-side storage detail. It is invisible to
 * simulated software and has no effect on timing. Architectural page sizes
 * (Sv32 pages and megapages) belong to a future MMU/TLB model, which will
 * translate virtual addresses before they ever reach physical memory.
 *
 * @par Address validity
 * Only addresses inside a configured RAM range are accessible. Accesses
 * outside every range fail with @c ok == false, exactly like an access
 * past the end of a @c FlatMemory, so programs that run off the end of
 * RAM still fault.
 *
 * @par Thread safety
 * Not thread-safe: reads update an internal last-page lookup cache.
 */

#pragma once

#include "memory.hpp"

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

namespace riscv {

class SparseMemory : public Memory {
public:
    /// log2 of the backing block size.
    static constexpr u32 kPageShift = 12;
    /// Backing block size in bytes (4 KiB).
    static constexpr u32 kPageSize = 1u << kPageShift; 

    /// A contiguous span of physical RAM, [base, base + size).
    struct Range
    {
        addr_t base = 0;
        u64    size = 0;

        [[nodiscard]] u64 end() const noexcept { return u64{base} + size; }
    };

    /// Construct with no RAM ranges; every access faults until one is added.
    SparseMemory() = default;

    /**
     * @brief Construct with a single RAM range.
     * @throw std::invalid_argument If the range is empty or exceeds 4 GiB.
     */
    SparseMemory(addr_t base, u64 size);

    /**
     * @brief Construct with several RAM ranges.
     * @throw std::invalid_argument If any range is empty, exceeds the
     *        address space, or overlaps another.
     */
    explicit SparseMemory(std::vector<Range> ranges);

    /**
     * @brief Add a RAM range. Ranges that touch an existing one are merged.
     * @throw std::invalid_argument If the range is empty, exceeds the
     *        address space, or overlaps an existing range.
     */
    void add_range(addr_t base, u64 size);

    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;

    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write8(addr_t addr, u8 value)   override;

    /// @throw MemoryAccessException If any byte falls outside RAM.
    void load(addr_t addr, std::span<const u8> data) override;
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;

    u32 read_line(addr_t addr, u8* dest, u32 size) const override;
    u32 write_line(addr_t addr, const u8* src, u32 size) override;

    /// Release every allocated page; all RAM reads as zero again.
    void clear() noexcept;

    /// @name Introspection
    /// @{
    [[nodiscard]] const std::vector<Range>& ranges() const noexcept
    { return ranges_; }
    /// Number of backing pages allocated so far (the workload's footprint).
    [[nodiscard]] size_t pages_allocated() const noexcept
    { return pages_.size(); }
    /// Host bytes held by backing pages.
    [[nodiscard]] size_t bytes_allocated() const noexcept
    { return pages_.size() * kPageSize; }
    /// @}

private:
    using Page = std::array<u8, kPageSize>;

    static constexpr u32 kNoPage = ~u32{0};  ///< Sentinel: page numbers < 2^20.

    std::vector<Range> ranges_;  ///< Sorted by base, disjoint.
    std::unordered_map<u32, std::unique_ptr<Page>> pages_;

    // Last-lookup cache. Page objects never move once allocated, so the
    // pointer stays valid across rehashes until clear()
    mutable u32   last_page_no_ = kNoPage;
    mutable Page* last_page_    = nullptr;

    [[nodiscard]] static constexpr u32 page_no(addr_t a) noexcept
    { return a >> kPageShift; }
    [[nodiscard]] static constexpr u32 page_off(addr_t a) noexcept
    { return a & (kPageSize - 1); }

    /// @return The page if allocated, else @c nullptr. Never allocates.
    [[nodiscard]] const Page* find_page(u32 pn) const;
    /// @return The page, allocating a zero-filled one if needed.
    [[nodiscard]] Page& touch_page(u32 pn);

    /// Copy out bytes that may cross page boundaries (unallocated reads as 0).
    void copy_out(addr_t addr, u8* dest, size_t size) const;
    /// Copy in bytes that may cross page boundaries, allocating as needed.
    void copy_in(addr_t addr, const u8* src, size_t size);
};

} // namespace riscv

