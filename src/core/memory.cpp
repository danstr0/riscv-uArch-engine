/**
 * @file memory.cpp
 * @brief FlatMemory and MMIOBus implementation.
 */

#include "memory.hpp"

#include <algorithm>
#include <cassert>

namespace riscv {

// ── FlatMemory implementation ──────────────────────────────────────────

FlatMemory::FlatMemory(addr_t base_addr, size_t size)
    : base_addr_(base_addr)
    , ram_(size, 0)
{
    // Ensure addr_t isn't overflowed during range checks later
    assert(base_addr + size >= base_addr && "Memory region wraps around address space");
}

MemoryResult FlatMemory::read32(addr_t addr) const
{
    if (addr & 3) return {0, 1, false};
    if (!in_range(addr, 4)) return {0, 1, false};
    u32 value;
    std::memcpy(&value, &ram_[addr - base_addr_], 4);
    return {value, 1, true};
}

MemoryResult FlatMemory::write32(addr_t addr, u32 value)
{
    if (addr & 3) return {0, 1, false};
    if (!in_range(addr, 4)) return {0, 1, false};
    std::memcpy(&ram_[addr - base_addr_], &value, 4);
    return {value, 1, true};
}

MemoryResult FlatMemory::read16(addr_t addr) const
{
    if (addr & 1) return {0, 1, false};
    if (!in_range(addr, 2)) return {0, 1, false};
    u16 value;
    std::memcpy(&value, &ram_[addr - base_addr_], 2);
    return {value, 1, true};
}

MemoryResult FlatMemory::write16(addr_t addr, u16 value)
{
    if (addr & 1) return {0, 1, false};
    if (!in_range(addr, 2)) return {0, 1, false};
    std::memcpy(&ram_[addr - base_addr_], &value, 2);
    return {value, 1, true};
}

MemoryResult FlatMemory::read8(addr_t addr) const
{
    if (!in_range(addr, 1)) return {0, 1, false};
    return {ram_[addr - base_addr_], 1, true};
}

MemoryResult FlatMemory::write8(addr_t addr, u8 value)
{
    if (!in_range(addr, 1)) return {0, 1, false};
    ram_[addr - base_addr_] = value;
    return {value, 1, true};
}

void FlatMemory::load(addr_t addr, std::span<const u8> data)
{
    if (!in_range(addr, data.size()))
        throw MemoryAccessException(addr, true);
    std::copy(data.begin(), data.end(), ram_.begin() + (addr - base_addr_));
}

bool FlatMemory::valid_address(addr_t addr, size_t size) const
{
    return in_range(addr, size);
}

u32 FlatMemory::read_line(addr_t addr, u8* dest, u32 size) const
{
    if (!in_range(addr, size)) return 1;
    auto offset = static_cast<size_t>(addr - base_addr_);
    std::memcpy(dest, ram_.data() + offset, size);
    return 1;
}

u32 FlatMemory::write_line(addr_t addr, const u8* src, u32 size)
{
    if (!in_range(addr, size)) return 1;
    auto offset = static_cast<size_t>(addr - base_addr_);
    std::memcpy(ram_.data() + offset, src, size);
    return 1;
}

// ── MMIOBus implementation ─────────────────────────────────────────────

void MMIOBus::set_default(std::shared_ptr<Memory> mem)
{
    default_mem_ = std::move(mem);
}

void MMIOBus::map(addr_t base, addr_t size, std::shared_ptr<Memory> device,
                  std::string name)
{
    // Clean up overlapping or identical base mappings
    unmap(base);
    regions_.push_back({base, size, std::move(device), std::move(name)});
}

void MMIOBus::map_passthrough(addr_t base, u64 size, std::shared_ptr<Memory> target,
                              std::string name)
{
    unmap(base);
    const addr_t sz = size > 0xFFFF'FFFFull
                    ? addr_t{0xFFFF'FFFF}
                    : static_cast<addr_t>(size);
    regions_.push_back({base, sz, std::move(target), std::move(name), false});
}

void MMIOBus::unmap(addr_t base)
{
    std::erase_if(regions_, [base](const Region& r) { return r.base == base; });
}

Memory* MMIOBus::find_region(addr_t addr, addr_t* offset) const
{
    for (auto& r : regions_)
        // Check if addr falls within [base, base + size); 64-bit to avoid wrap
        if (addr >= r.base && u64{addr} < u64{r.base} + r.size)
        {
            if (offset) *offset = r.rebase ? addr - r.base : addr;
            return r.device.get();
        }

    // Fallback: route absolute address to default memory if it exists
    if (offset) *offset = addr;
    return default_mem_.get();
}

MemoryResult MMIOBus::read32(addr_t addr) const
{
    addr_t offset;
    Memory* mem = find_region(addr, &offset);
    return mem ? mem->read32(offset) : MemoryResult{0, 1, false};
}

MemoryResult MMIOBus::read16(addr_t addr) const
{
    addr_t offset;
    Memory* mem = find_region(addr, &offset);
    return mem ? mem->read16(offset) : MemoryResult{0, 1, false};
}

MemoryResult MMIOBus::read8(addr_t addr) const
{
    addr_t offset;
    Memory* mem = find_region(addr, &offset);
    return mem ? mem->read8(offset) : MemoryResult{0, 1, false};
}

MemoryResult MMIOBus::write32(addr_t addr, u32 value)
{
    addr_t offset;
    Memory* mem = find_region(addr, &offset);
    return mem ? mem->write32(offset, value) : MemoryResult{0, 1, false};
}

MemoryResult MMIOBus::write16(addr_t addr, u16 value)
{
    addr_t offset;
    Memory* mem = find_region(addr, &offset);
    return mem ? mem->write16(offset, value) : MemoryResult{0, 1, false};
}

MemoryResult MMIOBus::write8(addr_t addr, u8 value)
{
    addr_t offset;
    Memory* mem = find_region(addr, &offset);
    return mem ? mem->write8(offset, value) : MemoryResult{0, 1, false};
}

void MMIOBus::load(addr_t addr, std::span<const u8> data)
{
    addr_t offset;
    Memory* mem = find_region(addr, &offset);
    if (!mem)
        throw MemoryAccessException(addr, true);
    mem->load(offset, data);
}

bool MMIOBus::valid_address(addr_t addr, size_t size) const
{
    if (size == 0) return true;
    addr_t end = addr + static_cast<addr_t>(size);
    if (end < addr) return false;  // overflow

    // Single-region check; does not validate ranges spanning multiple regions.
    for (const auto& r : regions_)
        if (addr >= r.base && end <= r.base + r.size)
            return true;

    return default_mem_ && default_mem_->valid_address(addr, size);
}

} // namespace riscv
