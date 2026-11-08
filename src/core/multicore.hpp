/**
 * @file multicore.hpp
 * @brief Synchronous multi-Core SoC simulation environment.
 *
 * Models a cycle-synchronous shared-memory multiprocessor with
 * directory-based coherence, a shared cache hierarchy, and
 * memory-mapped I/O devices (PLIC, Timer, NIC).
 *
 * @par Memory layout
 * Device MMIO addresses are placed immediately above main memory:
 * - @c device_base = align_up(mem_size, 0x1000)
 * - PLIC: device_base + 0x0000
 * - Timer: device_base + 0x1000
 * - NIC: device_base + 0x2000
 *
 * @par Interrupt delivery
 * External devices signal the PLIC, which performs priority arbitration
 * and asserts @c MEIP in the target core's @c mip register.
 */

#pragma once

#include "cache.hpp"
#include "coherence.hpp"
#include "csr.hpp"
#include "memory.hpp"
#include "nic.hpp"
#include "pipeline.hpp"
#include "plic.hpp"
#include "sparse_memory.hpp"
#include "timer.hpp"
#include "types.hpp"

#include <memory>
#include <vector>

namespace riscv {

/// System-level configuration for the multi-core SoC.
struct MultiCoreConfig
{
    u32 num_cores = 2;
    PipelineConfig pipeline = {};

    CacheConfig l1d = CacheConfig::L1_typical();  ///< Set @c size_bytes=0 to disable all caches.
    CacheConfig l2  = CacheConfig::L2_typical();  ///< Required when L1 is enabled.
    CacheConfig l3  = CacheConfig::L3_typical();  ///< Set @c size_bytes=0 to disable.

    u32 main_memory_size = 64 * 1024; // 64 KiB default

    u32 nic_plic_source = 1;
};

/// Aggregated statistics across all cores, caches, and coherence.
struct MultiCoreStats
{
    std::vector<PipelineStats> core_stats;
    CoherenceStats             coherence;
    std::vector<CacheStats>    l1d_stats;
    CacheStats                 l2_stats;
    CacheStats                 l3_stats;
};

/**
 * @brief Multi-core SoC simulation container.
 * 
 * Each call to @c tick() advances all pipelines, caches, devices,
 * and the coherence directory by one global clock cycle.
 */
class MultiCoreCPU {
public:
    explicit MultiCoreCPU(MultiCoreConfig config = {});

    /// @name Core access
    /// @{
    [[nodiscard]] u32 num_cores() const noexcept { return config_.num_cores; }
    [[nodiscard]]       PipelinedCPU& core(u32 id)       { return *cores_.at(id); }
    [[nodiscard]] const PipelinedCPU& core(u32 id) const { return *cores_.at(id); }
    /// @}

    /// @name Program loading
    /// @{
    void load_program(addr_t addr, std::span<const u8> program);
    void load_instruction(addr_t addr, u32 instruction);
    /// @}

    /// @name Execution
    /// @{

    /// Advance all components by one cycle. Returns false if all cores halted.
    bool tick();

    cycle_t run_cycles(cycle_t n);
    cycle_t run_until_all_halted(cycle_t max_cycles = 10'000'000);

    [[nodiscard]] cycle_t cycles() const noexcept { return cycle_count_; }
    /// @}

    /// @name Device access
    /// @{
    [[nodiscard]] PLIC&   plic()        noexcept { return plic_; }
    [[nodiscard]] Timer&  timer()       noexcept { return timer_; }
    [[nodiscard]] Memory& main_memory() noexcept { return *main_mem_; }

    /// Attch a nic: maps into the address space and wires its interrupt to the PLIC.
    void attach_nic(std::shared_ptr<NIC> nic);
    [[nodiscard]] NIC* nic() noexcept { return nic_.get(); }

    /// Get the computed device MMIO base address.
    [[nodiscard]] addr_t device_base() const noexcept { return device_base_; }
    /// @}

    /// @name Cache hierarchy
    /// @{
    [[nodiscard]] Cache* l1d(u32 core_id);
    [[nodiscard]] Cache* l2() { return l2_cache_.get(); }
    [[nodiscard]] Cache* l3() { return l3_cache_.get(); }
    [[nodiscard]] CoherenceController* coherence() { return coherence_.get(); }
    [[nodiscard]] bool has_caches() const noexcept { return !l1d_caches_.empty(); }
    /// @}

    /// @name Statistics
    /// @{
    [[nodiscard]] MultiCoreStats get_stats() const;
    void reset_stats();
    /// @}

    /// @name Configuration
    /// @{
    [[nodiscard]] const MultiCoreConfig& config() const noexcept { return config_; }
    void set_core_pc(u32 core_id, addr_t pc);
    void set_core_reg(u32 core_id, reg_idx_t r, u32 value);
    /// @}

private:
    MultiCoreConfig config_;
    addr_t device_base_ = 0;

    std::shared_ptr<SparseMemory>                main_mem_;
    std::shared_ptr<Cache>                     l3_cache_;
    std::shared_ptr<Cache>                     l2_cache_;
    std::vector<std::shared_ptr<Cache>>        l1d_caches_;
    std::vector<std::shared_ptr<MMIOBus>>      buses_;
    std::unique_ptr<CoherenceController>       coherence_;
    std::vector<std::unique_ptr<PipelinedCPU>> cores_;

    PLIC                 plic_;
    Timer                timer_;
    std::shared_ptr<NIC> nic_;

    cycle_t cycle_count_ = 0;

    /// Wire timer and PLIC interrupt lines to core CSR files.
    void wire_interrupts();
};

} // namespace riscv
