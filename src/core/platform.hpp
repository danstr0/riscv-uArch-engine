/**
 * @file platform.hpp
 * @brief The physical platform shared by all core models: RAM, devices,
 *        the physical memory map, and interrupt wiring.
 *
 * A @c Platform owns everything outside the cores: demand-allocated RAM,
 * a CLINT (timer + software interrupts), a PLIC (external interrupts), and
 * an optional NIC. It builds one MMIO bus per hart and routes each hart's
 * MTIP / MSIP / MEIP lines to that hart's CSR file. Any core model that
 * exposes a @c CSRFile can be attached, so the single-cycle @c CPU, the
 * @c PipelinedCPU, and @c MultiCoreCPU all see the same platform.
 *
 * @par Default physical memory map
 * @code{.txt}
 *   ┌──────────────┬───────────┬─────────┬──────────────────────────────┐
 *   │ Base         │ Size      │ Region  │ Notes                        │
 *   ├──────────────┼───────────┼─────────┼──────────────────────────────┤
 *   │ ram_base (0) │ ram_size  │ RAM     │ Demand-allocated (cacheable) │
 *   │ 0x0200_0000  │ 64 KiB    │ CLINT   │ msip / mtimecmp / mtime      │
 *   │ 0x0C00_0000  │ 64 MiB    │ PLIC    │ One M-mode context per hart  │
 *   │ 0x1000_0000  │ 4 KiB     │ NIC     │ Only when a NIC is attached  │
 *   └──────────────┴───────────┴─────────┴──────────────────────────────┘
 * @endcode
 * The device addresses follow the common SiFive / QEMU @c virt convention
 * and do not depend on RAM size. Accesses that hit no region fault.
 *
 * @par Bus topology
 * Device regions are uncached: a hart's bus routes them straight to the
 * device. Only the RAM range is forwarded to the hart's "downstream"
 * memory (its L1 cache, or RAM itself when caches are disabled).
 *
 * @par PLIC contexts
 * The cores only implement M-mode, so PLIC context @c h is hart @c h's
 * M-mode context.
 */

#pragma once

#include "clint.hpp"
#include "csr.hpp"
#include "memory.hpp"
#include "nic.hpp"
#include "plic.hpp"
#include "sparse_memory.hpp"

#include <memory>
#include <string>
#include <vector>

namespace riscv {

/// Default device base addresses.
namespace PlatformMap
{
    constexpr addr_t CLINT_BASE = 0x0200'0000;
    constexpr addr_t PLIC_BASE  = 0x0C00'0000;
    constexpr addr_t NIC_BASE   = 0x1000'0000;
    constexpr addr_t NIC_SIZE   = 0x1000;
}

/// Platform construction parameters.
struct PlatformConfig
{
    u32    num_harts = 1;          ///< 1 .. PLICConfig::MAX_CONTEXTS.
    addr_t ram_base  = 0;
    u64    ram_size  = 64 * 1024;  ///< Bounds valid RAM; nothing is preallocated.

    /// Core cycles per @c mtime increment (models a slower timebase clock).
    u32 mtime_cycles_per_tick = 1;

    addr_t clint_base = PlatformMap::CLINT_BASE;
    addr_t plic_base  = PlatformMap::PLIC_BASE;
    addr_t nic_base   = PlatformMap::NIC_BASE;
};

class Platform {
public:
    /// One entry of the physical memory map.
    struct Region
    {
        std::string name;
        addr_t      base;
        u64         size;
    };

    /// @throw std::invalid_argument On a bad hart count or overlapping /
    /// out-of-range regions.
    explicit Platform(PlatformConfig config = {});

    // Interrupt callbacks capture `this`
    Platform(const Platform&)            = delete;
    Platform& operator=(const Platform&) = delete;
    Platform(Platform&&)                 = delete;
    Platform& operator=(Platform&&)      = delete;

    /// @name Building the system
    /// @{

    /**
     * @brief Create the MMIO bus a hart issues its loads, stores, and
     *        fetches to.
     * @param downstream  Where RAM accesses go: the hart's L1, or
     *                    @c ram() if uncached.
     */
    [[nodiscard]] std::shared_ptr<MMIOBus>
    make_hart_bus(std::shared_ptr<Memory> downstream);

    /// Connect hart @p hart's interrupt lines to @p csrs (which must
    /// outlive the platform's use).
    void connect_hart(u32 hart, CSRFile& csrs);

    /// Attach a NIC: maps it on every hart bus and routes its interrupt
    /// to PLIC @p plic_source.
    void attach_nic(std::shared_ptr<NIC> nic, u32 plic_source = 1);
    /// @}

    /// Advance devices to core cycle @p cycle.
    void tick(cycle_t cycle);

    /// @name Component access
    /// @{
    [[nodiscard]] const PlatformConfig&   config() const noexcept { return config_; }
    [[nodiscard]] SparseMemory&           ram()          noexcept { return *ram_; }
    [[nodiscard]] std::shared_ptr<SparseMemory> ram_ptr() const noexcept { return ram_; }
    [[nodiscard]] CLINT&                  clint()        noexcept { return *clint_; }
    [[nodiscard]] PLIC&                   plic()         noexcept { return *plic_; }
    [[nodiscard]] NIC*                    nic()          noexcept { return nic_.get(); }

    /// The physical memory map (RAM first, then devices).
    [[nodiscard]] std::vector<Region> memory_map() const;
    /// @}

private:
    PlatformConfig config_;

    std::shared_ptr<SparseMemory> ram_;
    std::shared_ptr<CLINT>        clint_;
    std::shared_ptr<PLIC>         plic_;
    std::shared_ptr<NIC>          nic_;

    std::vector<std::shared_ptr<MMIOBus>> buses_;
    std::vector<CSRFile*>                 harts_;  ///< nullptr until connected.

    void set_line(u32 hart, u32 mip_bit, bool level);
    void validate_map() const;
};

} // namespace riscv

