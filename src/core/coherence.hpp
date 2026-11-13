/**
 * @file coherence.hpp
 * @brief Directory-based MESI cache coherence controller.
 *
 * Implements a full-map directory protocol serving as the serialization
 * point for all L1 cache misses in a multi-core configuration.
 *
 * @par MESI states
 * - Modified: sole dirty owner.
 * - Exclusive: sole clean owner.
 * - Shared: clean, possibly held by multiple cores.
 * - Invalid: not cached.
 *
 * @par Directory architecture
 * Uses per-line shadow tags rather than bus-based snooping, sending
 * invalidation/snoop messages only to cores known to hold a copy.
 *
 * @see Hennessy & Patterson, "Computer Organization and Design (RISC-V Edition)",
 *      Section 5.12.
 */

#pragma once

#include "cache.hpp"
#include "types.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace riscv {

/// MESI coherence states.
enum class MESIState : u8
{
    Invalid   = 0,  ///< Line is not cached.
    Shared    = 1,  ///< Clean; other cores may also hold it.
    Exclusive = 2,  ///< Clean; this core is the sole owner.
    Modified  = 3,  ///< Dirty; this core is the sole owner.
};

/// Coherence transaction counters.
struct CoherenceStats
{
    u64 read_misses        = 0;
    u64 write_misses       = 0;
    u64 upgrades           = 0;  ///< S/E → M transitions.
    u64 invalidations_sent = 0;
    u64 writebacks_forced  = 0;  ///< M → E/S/I transitions requiring writeback.
    u64 shared_transfers   = 0;  ///< Cache-to-cache intervention transfers.
    u64 l2_fetches         = 0;  ///< Requests satisfied by shared L2/RAM.
    u64 fetch_misses       = 0;  ///< I-cache refills seen by the directory.
    u64 dma_reads          = 0;  ///< Coherent DMA reads (per byte-range access).
    u64 dma_writes         = 0;  ///< Coherent DMA writes (per byte-range access).

    void reset() noexcept { *this = CoherenceStats{}; }
};

/// Per-line directory entry tracking coherence state across all cores.
struct DirectoryEntry
{
    std::vector<MESIState> core_states;

    explicit DirectoryEntry(u32 num_cores)
        : core_states(num_cores, MESIState::Invalid) {}
};

/**
 * @brief Centralized directory-based coherence controller.
 *
 * Manages MESI state transitions for all L1 caches. Each L1 registers
 * read-miss and write callbacks that route through this controller.
 */
class CoherenceController {
public:
    CoherenceController(std::vector<Cache*> l1_caches,
                        std::shared_ptr<Memory> shared_mem,
                        u32 line_size = 64);

    /// @name Core request interface
    /// @{

    /**
     * @brief Handle a read miss from @p core_id for address @p addr.
     *
     * If the line is Modified in another core, that core writes back
     * and both transition to Shared.
     *
     * @return Transaction latency in cycles.
     */
    u32 handle_read_miss(u32 core_id, addr_t addr);

    /**
     * @brief Handle a write miss or upgrade from @p core_id.
     * @return Transaction latency in cycles.
     */
    u32 handle_write_miss(u32 core_id, addr_t addr);

    /// Notify the directory that @p core_id is evicting the line at @p addr.
    void notify_eviction(u32 core_id, addr_t addr);

    /**
     * @brief Handle an I-cache refill from @p core_id.
     *
     * The refill must see the newest data, so a @c Modified owner (including
     * the requesting core's own D-cache) writes back and drops to @c Exclusive.
     * I-cache copies are deliberately not tracked: as RISC-V allows, a line
     * already in an I-cache can go stale until that hart runs @c fence.i.
     *
     * @return Transaction latency in cycles.
     */
    u32 handle_fetch_miss(u32 core_id, addr_t addr);

    /// A device is about to read the line: a Modified owner writes back.
    u32 dma_read(addr_t addr);

    /// A device is about to write the line: all L1 copies are written
    /// back and invalidated.
    u32 dma_write(addr_t addr);
    /// @}

    /// @name Statistics
    /// @{
    [[nodiscard]] const CoherenceStats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_.reset(); }
    /// @}

    /// @name State Inspection
    /// @{
    [[nodiscard]] MESIState line_state(u32 core_id, addr_t addr) const;
    [[nodiscard]] u32 num_cores()         const noexcept { return num_cores_; }
    [[nodiscard]] u32 num_tracked_lines() const noexcept { return static_cast<u32>(directory_.size()); }
    /// @}

private:
    std::vector<Cache*>     l1_caches_;
    std::shared_ptr<Memory> shared_mem_;
    u32                     line_size_;
    u32                     num_cores_;

    std::unordered_map<addr_t, DirectoryEntry> directory_;
    CoherenceStats stats_;

    /// Get or create the directory entry for the line containing @p addr.
    DirectoryEntry& get_entry(addr_t addr);

    /// Align address down to cache line boundary.
    [[nodiscard]] addr_t line_align(addr_t addr) const noexcept
    {
        return addr & ~static_cast<addr_t>(line_size_ - 1);
    }

    /// Invalidate all copies except in @p requesting_core. Returns snoop latency.
    u32 invalidate_others(u32 requesting_core, addr_t line_addr, DirectoryEntry& entry);

    /// Write back a Modified owner of @p line_addr (M → E), if any.
    /// Returns snoop latency.
    u32 clean_modified_owner(addr_t line_addr);

    /// @return Core ID holding the line in Modified state, or @c -1 if none.
    [[nodiscard]] i32 find_modified_owner(const DirectoryEntry& entry) const;

    /// @return Any core holding the line in Shared or Exclusive (excluding @p exclude_core), or -1.
    [[nodiscard]] i32 find_any_sharer(const DirectoryEntry& entry, u32 exclude_core) const;

    /// @name Latency constants
    /// @{
    static constexpr u32 kSnoopLatency        = 2;  ///< Cycles per snoop message.
    static constexpr u32 kTransferLatency     = 5;  ///< Cycles for cache-to-cache transfer.
    static constexpr u32 kInvalidationLatency = 2;  ///< Cycles per invalidation.
    /// @}
};

} // namespace riscv
