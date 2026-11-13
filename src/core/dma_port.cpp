/**
 * @file dma_port.cpp
 * @brief DmaPort implementation.
 */

#include "dma_port.hpp"

namespace riscv {

DmaPort::DmaPort(std::shared_ptr<Memory> next, u32 line_size,
                 LineHook before_read, LineHook before_write)
    : next_(std::move(next))
    , line_size_(line_size)
    , before_read_(std::move(before_read))
    , before_write_(std::move(before_write))
{
}

void DmaPort::snoop(addr_t addr, u32 size, const LineHook& hook) const
{
    if (!hook || size == 0) return;

    const addr_t mask  = ~static_cast<addr_t>(line_size_ - 1);
    const addr_t first = addr & mask;
    const addr_t last  = (addr + size - 1) & mask;
    for (addr_t line = first; ; line += line_size_)
    {
        hook(line);
        if (line == last) break;
    }
}

MemoryResult DmaPort::read32(addr_t addr) const
{
    if (!valid_address(addr, 4)) return {0, 1, false};
    snoop(addr, 4, before_read_);
    return next_->read32(addr);
}

MemoryResult DmaPort::read16(addr_t addr) const
{
    if (!valid_address(addr, 2)) return {0, 1, false};
    snoop(addr, 2, before_read_);
    return next_->read16(addr);
}

MemoryResult DmaPort::read8(addr_t addr) const
{
    if (!valid_address(addr, 1)) return {0, 1, false};
    snoop(addr, 1, before_read_);
    return next_->read8(addr);
}

MemoryResult DmaPort::write32(addr_t addr, u32 value)
{
    if (!valid_address(addr, 4)) return {0, 1, false};
    snoop(addr, 4, before_write_);
    return next_->write32(addr, value);
}

MemoryResult DmaPort::write16(addr_t addr, u16 value)
{
    if (!valid_address(addr, 2)) return {0, 1, false};
    snoop(addr, 2, before_write_);
    return next_->write16(addr, value);
}

MemoryResult DmaPort::write8(addr_t addr, u8 value)
{
    if (!valid_address(addr, 1)) return {0, 1, false};
    snoop(addr, 1, before_write_);
    return next_->write8(addr, value);
}

void DmaPort::load(addr_t addr, std::span<const u8> data)
{
    if (!valid_address(addr, data.size()))
        throw MemoryAccessException(addr, true);

    snoop(addr, static_cast<u32>(data.size()), before_write_);

    // Byte-wise so a partial line merges with what the next level holds
    // (Cache::load would drop a dirty copy of the surrounding bytes)
    for (size_t i = 0; i < data.size(); ++i)
        (void)next_->write8(addr + static_cast<addr_t>(i), data[i]);
}

bool DmaPort::valid_address(addr_t addr, size_t size) const
{
    return next_->valid_address(addr, size);
}

} // namespace riscv

