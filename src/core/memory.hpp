/**
 * @file memory.hpp
 * @brief Memory subsystem and bus routing for the RISC-V simulator.
 *
 * Defines a unified memory interface allowing the CPU to interact with 
 * RAM, ROM, and MMIO devices transparently through a common bus.
 *
 * @par Address translation
 * The @c MMIOBus routes requests using absolute addresses, then subtracts
 * the region's base to present each device with a zero-based offset.
 *
 * @par Latency modeling
 * @c MemoryResult includes a @c cycles field for cycle-accurate
 * timing and cache-miss penalty injection.
 */

#pragma once

#include "types.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace riscv {

/**
 * @brief Outcome of a memory transaction.
 * 
 * Callers must verify @c ok before consuming @c value.
 */
struct MemoryResult
{
    u32 value  = 0;     ///< Data retrieved on reads; undefined for writes.
    u32 cycles = 1;     ///< Latency of this transaction.
    bool ok    = true;  ///< @c false if a fault occurred.
};

/**
 * @brief Abstract interface for all memory-mapped entities.
 *
 * All implementations must be byte-addressable. Multi-byte accesses 
 * use host-native (little-endian) byte order.
 */
class Memory {
public:
    virtual ~Memory() = default;

    /// @name Typed reads
    /// @{
    [[nodiscard]] virtual MemoryResult read32(addr_t addr) const = 0;
    [[nodiscard]] virtual MemoryResult read16(addr_t addr) const = 0;
    [[nodiscard]] virtual MemoryResult read8(addr_t addr)  const = 0;
    /// @}

    /// @name Typed writes
    /// @{
    virtual MemoryResult write32(addr_t addr, u32 value) = 0;
    virtual MemoryResult write16(addr_t addr, u16 value) = 0;
    virtual MemoryResult write8(addr_t addr, u8 value)   = 0;
    /// @}

    /**
     * @brief Bulk binary load into this memory region.
     * @param addr  Starting address.
     * @param data  Bytes to copy.
     * @throw std::out_of_range If the data exceeds region boundaries.
     */
    virtual void load(addr_t addr, std::span<const u8> data) = 0;

    /**
     * @brief Bulk read into a buffer.
     * 
     * Default implementation falls back to per-byte reads.
     *
     * @param addr  Starting address.
     * @param dest  Destination buffer (at least @p size bytes).
     * @param size  Number of bytes to read.
     * @return Latency in cycles.
     */
    virtual u32 read_line(addr_t addr, u8* dest, u32 size) const
    {
        u32 cycles = 0;
        for (u32 i = 0; i < size; ++i)
        {
            auto r = read8(addr + i);
            dest[i] = static_cast<u8>(r.value);
            cycles = std::max(cycles, r.cycles);
        }
        return cycles;
    }

    /**
     * @brief Bulk write from a buffer.
     *
     * Default implementation falls back to per-byte writes.
     */
    virtual u32 write_line(addr_t addr, const u8* src, u32 size)
    {
        u32 cycles = 0;
        for (u32 i = 0; i < size; ++i)
        {
            auto r = write8(addr + i, src[i]);
            cycles = std::max(cycles, r.cycles);
        }
        return cycles;
    }

    /// @return @c true if range [addr, addr+size) is accessible.
    [[nodiscard]] virtual bool valid_address(addr_t addr, size_t size = 1) const = 0;
};

/**
 * @brief Contiguous, byte-addressable RAM region.
 *
 * When used behind an MMIOBus, addresses are relative (base is subtracted
 * by the bus before forwarding).
 */
class FlatMemory : public Memory {
public:
    /**
     * @brief Construct a RAM region.
     * @param base_addr  Logical start address.
     * @param size       Size in bytes.
     */
    FlatMemory(addr_t base_addr, size_t size);

    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;

    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write8(addr_t addr, u8 value)   override;

    void load(addr_t addr, std::span<const u8> data) override;
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;

    u32 read_line(addr_t addr, u8* dest, u32 size) const override;
    u32 write_line(addr_t addr, const u8* src, u32 size) override;

    /// @name Instrospection
    /// @{
    [[nodiscard]] addr_t    base() const noexcept { return base_addr_; }
    [[nodiscard]] size_t    size() const noexcept { return ram_.size(); }
    [[nodiscard]] const u8* data() const noexcept { return ram_.data(); }
    [[nodiscard]] u8*       data()       noexcept { return ram_.data(); }
    /// @}

private:
    addr_t          base_addr_;
    std::vector<u8> ram_;

    [[nodiscard]] bool in_range(addr_t addr, size_t len) const noexcept
    {
        if (addr < base_addr_) return false;
        auto offset = static_cast<size_t>(addr - base_addr_);
        return len <= ram_.size() && offset <= ram_.size() - len;
    }
};

/**
 * @brief Address-space router connecting the CPU to memory-mapped devices.
 *
 * For each access the bus identifies the target region by absolute address,
 * subtracts the region base, and forwards the request with the resulting
 * relative offset.
 */
class MMIOBus : public Memory {
public:
    /// A mapping between an address range and a device.
    struct Region
    {
        addr_t                  base;    ///< Absolute start address.
        addr_t                  size;    ///< Size in bytes.
        std::shared_ptr<Memory> device;  ///< The memory-mapped device.
        std::string             name;    ///< Label for debugging (e.g., "RAM", "PLIC").
        bool                    rebase = true;  ///< Subtract @c base before forwarding.
    };

    /// Set a fallback device for accesses that don't hit any mapped region.
    void set_default(std::shared_ptr<Memory> mem);

    /**
     * @brief Map a device into the address space.
     * @param base    Absolute base address.
     * @param size    Size in bytes.
     * @param device  The Memory implementation.
     * @param name    Optional label for debugging.
     */
    void map(addr_t base, addr_t size, std::shared_ptr<Memory> device,
             std::string name = "");

    /**
     * @brief Map a range whose target expects absolute addresses.
     *
     * Used for RAM (or a cache in front of RAM): accesses inside the range
     * are forwarded unchanged instead of being rebased to zero.
     */
    void map_passthrough(addr_t base, u64 size, std::shared_ptr<Memory> target,
                         std::string name = "");

    /// Remove the mapping at @p base.
    void unmap(addr_t base);

    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;
    MemoryResult write32(addr_t addr, u32 value) override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write8(addr_t addr, u8 value)   override;
    void load(addr_t addr, std::span<const u8> data) override;
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;

    /// @return All current region mappings.
    [[nodiscard]] const std::vector<Region>& regions() const noexcept { return regions_; }

private:
    std::vector<Region>     regions_;
    std::shared_ptr<Memory> default_mem_;

    /// Route an absolute address to its target device, writing the relative offset to @p offset.
    [[nodiscard]] Memory* find_region(addr_t addr, addr_t* offset = nullptr) const;
};

} // namespace riscv
