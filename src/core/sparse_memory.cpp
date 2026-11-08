/**
 * @file sparse_memory.cpp
 * @brief SparseMemory implementation.
 */

#include "sparse_memory.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>

namespace riscv {

namespace {

constexpr u64 kAddressSpace = u64{1} << 32;

} // namespace

// ── Construction / range management ────────────────────────────────────

SparseMemory::SparseMemory(addr_t base, u64 size)
{
    add_range(base, size);
}

SparseMemory::SparseMemory(std::vector<Range> ranges)
{
    for (const auto& r : ranges)
        add_range(r.base, r.size);
}

void SparseMemory::add_range(addr_t base, u64 size)
{
    if (size == 0)
        throw std::invalid_argument(
            std::format("SparseMemory: empty RAM range at 0x{:08x}", base));

    const u64 end = u64{base} + size;
    if (end > kAddressSpace)
        throw std::invalid_argument(
            std::format("SparseMemory: RAM range 0x{:08x}+0x{:x} exceeds "
                        "the 32-bit address space", base, size));

    for (const auto& r : ranges_)
        if (u64{base} < r.end() && u64{r.base} < end)
            throw std::invalid_argument(
                std::format("SparseMemory: RAM range 0x{:08x}+0x{:x} "
                            "overlaps existing range 0x{:08x}+0x{:x}",
                            base, size, r.base, r.size));

    auto pos = std::upper_bound(ranges_.begin(), ranges_.end(), base,
                                [](addr_t b, const Range& r)
                                {return b < r.base; });
    ranges_.insert(pos, Range{base, size});

    // Merge ranges that now touch, so a span crossing the seam stays valid
    std::vector<Range> merged;
    merged.reserve(ranges_.size());
    for (const auto& r : ranges_)
    {
        if (!merged.empty() && merged.back().end() == u64{r.base})
            merged.back().size += r.size;
        else
            merged.push_back(r);
    }
    ranges_ = std::move(merged);
}

bool SparseMemory::valid_address(addr_t addr, size_t size) const
{
    if (size == 0) return true;
    const u64 end = u64{addr} + size;
    for (const auto& r : ranges_)
        if (addr >= r.base && end <= r.end())
            return true;
    return false;
}

void SparseMemory::clear() noexcept
{
    pages_.clear();
    last_page_no_ = kNoPage;
    last_page_    = nullptr;
}

// ── Page lookup ────────────────────────────────────────────────────────

const SparseMemory::Page* SparseMemory::find_page(u32 pn) const
{
    if (pn == last_page_no_) return last_page_;

    auto it = pages_.find(pn);
    if (it == pages_.end()) return nullptr;

    last_page_no_ = pn;
    last_page_    = it->second.get();
    return last_page_;
}

SparseMemory::Page& SparseMemory::touch_page(u32 pn)
{
    if (pn == last_page_no_) return *last_page_;

    auto& slot = pages_[pn];
    if (!slot) slot = std::make_unique<Page>();

    last_page_no_ = pn;
    last_page_    = slot.get();
    return *slot;
}

void SparseMemory::copy_out(addr_t addr, u8* dest, size_t size) const
{
    while (size > 0)
    {
        const u32 off      = page_off(addr);
        const size_t chunk = std::min<size_t>(size, kPageSize - off);

        if (const Page* p = find_page(page_no(addr)))
            std::memcpy(dest, p->data() + off, chunk);
        else
            std::memset(dest, 0, chunk);

        dest += chunk;
        addr += static_cast<addr_t>(chunk);
        size -= chunk;
    }
}

void SparseMemory::copy_in(addr_t addr, const u8* src, size_t size)
{
    while (size > 0)
    {
        const u32    off   = page_off(addr);
        const size_t chunk = std::min<size_t>(size, kPageSize - off);

        std::memcpy(touch_page(page_no(addr)).data() + off, src, chunk);

        src  += chunk;
        addr += static_cast<addr_t>(chunk);
        size -= chunk;
    }
}

// ── Typed accesses ─────────────────────────────────────────────────────

MemoryResult SparseMemory::read32(addr_t addr) const
{
    if ((addr & 3) || !valid_address(addr, 4)) return {0, 1, false};
    const Page* p = find_page(page_no(addr));
    if (!p) return {0, 1, true};
    u32 value;
    std::memcpy(&value, p->data() + page_off(addr), 4);
    return {value, 1, true};
}

MemoryResult SparseMemory::read16(addr_t addr) const
{
    if ((addr & 1) || !valid_address(addr, 2)) return {0, 1, false};
    const Page* p = find_page(page_no(addr));
    if (!p) return {0, 1, true};
    u16 value;
    std::memcpy(&value, p->data() + page_off(addr), 2);
    return {value, 1, true};
}

MemoryResult SparseMemory::read8(addr_t addr) const
{
    if (!valid_address(addr, 1)) return {0, 1, false};
    const Page* p = find_page(page_no(addr));
    return {p ? (*p)[page_off(addr)] : u32{0}, 1, true};
}

MemoryResult SparseMemory::write32(addr_t addr, u32 value)
{
    if ((addr & 3) || !valid_address(addr, 4)) return {0, 1, false};
    std::memcpy(touch_page(page_no(addr)).data() + page_off(addr), &value, 4);
    return {value, 1, true};
}

MemoryResult SparseMemory::write16(addr_t addr, u16 value)
{
    if ((addr & 1) || !valid_address(addr, 2)) return {0, 1, false};
    std::memcpy(touch_page(page_no(addr)).data() + page_off(addr), &value, 2);
    return {value, 1, true};
}

MemoryResult SparseMemory::write8(addr_t addr, u8 value)
{
    if (!valid_address(addr, 1)) return {0, 1, false};
    touch_page(page_no(addr))[page_off(addr)] = value;
    return {value, 1, true};
}

// ── Bulk accesses ──────────────────────────────────────────────────────

void SparseMemory::load(addr_t addr, std::span<const u8> data)
{
    if (!valid_address(addr, data.size()))
        throw MemoryAccessException(addr, true);
    copy_in(addr, data.data(), data.size());
}

u32 SparseMemory::read_line(addr_t addr, u8* dest, u32 size) const
{
    if (!valid_address(addr, size)) return 1;
    copy_out(addr, dest, size);
    return 1;
}

u32 SparseMemory::write_line(addr_t addr, const u8* src, u32 size)
{
    if (!valid_address(addr, size)) return 1;
    copy_in(addr, src, size);
    return 1;
}

} // namespace riscv

