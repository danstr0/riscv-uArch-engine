/**
 * @file dma_port.hpp
 * @brief Memory port for bus-mastering devices that keeps DMA cache-coherent.
 *
 * A device given a @c DmaPort instead of raw RAM sees the same data as the
 * CPUs. Before each access, the port tells the system's caches which lines
 * are involved: a read pulls dirty data out of the L1 D-caches, and a write
 * removes every L1 copy. The access itself then goes to the outermost shared
 * level (L2, or RAM when there is no L2).
 *
 * I-caches are not snooped: after DMA writes code, software must run
 * @c fence.i, just as it must after a CPU store to code.
 */

#pragma once

#include "memory.hpp"
#include "types.hpp"

#include <functional>
#include <memory>

namespace riscv {

class DmaPort : public Memory {
public:
    /// Called with a line-aligned address before the line is accessed.
    using LineHook = std::function<void(addr_t line_addr)>;

    DmaPort(std::shared_ptr<Memory> next, u32 line_size,
            LineHook before_read, LineHook before_write);

    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;

    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write8(addr_t addr, u8 value)   override;

    /// Coherent bulk write of @p data.
    /// @throw MemoryAccessException If any byte falls outside memory.
    void load(addr_t addr, std::span<const u8> data) override;

    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;

private:
    std::shared_ptr<Memory> next_;
    u32                     line_size_;
    LineHook                before_read_;
    LineHook                before_write_;

    void snoop(addr_t addr, u32 size, const LineHook& hook) const;
};

} // namespace riscv

