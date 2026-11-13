/**
 * @file cache.hpp
 * @brief Parameterised set-associative cache hierarchy for cycle-accurate memory modeling.
 *
 * This provides a flexible, set-associative cache simulation designed for
 * Pareto-front architectural sweeps. It allows for detailed benchmarking
 * across size, associativity, and replacement/write policies.
 *
 * @par Architecture
 * Cache represents a single level (L1, L2, L3) with flat-vector storage.
 * CacheHierarchy chains multiple levels ending in a backing memory,
 * implementing the Memory interface for drop-in CPU integration.
 *
 * @par Address decomposition
 * For an address A: block offset = A mod LineSize,
 * set index = (A / LineSize) mod NumSets,
 * tag = A / (LineSize * NumSets).
 *
 * @par Parameterization
 * - Geometry: size, line size, associativity.
 * - Replacement policy: LRU, PLRU, MRU, FIFO, Random.
 * - Write policy: write-back, write-through.
 * - Allocation policy: write-allocate, no-write-allocate.
 *
 * @see Patterson & Hennessy "Computer Organization and Design (RISC-V Edition)",
 * Sections 5.3-5.4.
 */

#pragma once

#include "memory.hpp"
#include "types.hpp"

#include <bit>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace riscv {

/// Replacement policy for cache eviction.
enum class ReplacementPolicy : u8
{
    LRU,     ///< Least recently used.
    MRU,     ///< Most recently used.
    PLRU,    ///< Pseudo-LRU (tree-based).
    RANDOM,  ///< Uniform random.
    FIFO,    ///< First-in first-out.
};

/// Write policy controlling when data propagates to the next level.
enum class WritePolicy: u8
{
    WRITE_BACK,     ///< Write to cache only; writeback on eviction.
    WRITE_THROUGH,  ///< Write to cache and next level on every write.
};

/// Allocation policy on write misses.
enum class WriteAllocate : u8
{
    ALLOCATE,     ///< Fetch line into cache, then write.
    NO_ALLOCATE,  ///< Write directly to next level.
};

/// Cache role in a split-cache architecture.
enum class CacheType : u8
{
    UNIFIED,      ///< Serves both instructions and data.
    INSTRUCTION,  ///< I-cache (reads only; writes assert).
    DATA,         ///< D-cache (reads and writes).
};

/// Configuration for a single cache level.
struct CacheConfig
{
    u32 size_bytes;          ///< Total cache size.
    u32 line_size     = 64;  ///< Cache line size in bytes.
    u32 associativity = 8;   ///< N-way set associative.
    u32 hit_latency   = 4;   ///< Cycles for a hit.
    u32 miss_penalty  = 10;  ///< Additional cycles on a miss.

    ReplacementPolicy replacement = ReplacementPolicy::LRU;
    WritePolicy       write_pol   = WritePolicy::WRITE_BACK;
    WriteAllocate     write_alloc = WriteAllocate::ALLOCATE;
    CacheType         type        = CacheType::UNIFIED;

    /// @name Derived geometry
    /// @{
    [[nodiscard]] constexpr u32 num_sets() const noexcept
    {
        return size_bytes / (line_size * associativity);
    }

    [[nodiscard]] constexpr u32 num_lines() const noexcept
    {
        return size_bytes / line_size;
    }
    /// @}

    /**
     * @brief Validate self-consistency.
     *
     * All sizes must be powers of two; total size must equal
     * line size * associativity * num_sets.
     */
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return line_size > 0
            && associativity > 0
            && size_bytes > 0
            && std::has_single_bit(line_size)
            && std::has_single_bit(num_sets())
            && (size_bytes % (line_size * associativity)) == 0;
    }

    /// @name Static presets
    /// @{
    static constexpr CacheConfig L1_typical()
    {
        return {
            .size_bytes    = 32 * 1024, 
            .line_size     = 64,
            .associativity = 8,
            .hit_latency   = 4,
            .miss_penalty  = 10,
        };
    }

    static constexpr CacheConfig L2_typical()
    {
        return {
            .size_bytes    = 256 * 1024,
            .line_size     = 64,
            .associativity = 8,
            .hit_latency   = 12,
            .miss_penalty  = 50,
        };
    }

    static constexpr CacheConfig L3_typical()
    {
        return {
            .size_bytes    = 8 * 1024 * 1024,
            .line_size     = 64,
            .associativity = 16,
            .hit_latency   = 40,
            .miss_penalty  = 100,
        };
    }
    /// @}
};

/**
 * @brief Metadata for a single cache line.
 *
 * Payload data lives in @c Cache::storage_, not here.
 */
struct CacheLine
{
    bool        valid       = false;
    bool        dirty       = false;
    addr_t      tag         = 0;
    addr_t      line_addr   = 0;  ///< Full address of the line start.
    mutable u64 last_access = 0;  ///< Timestamp for LRU/MRU.
    mutable u32 fifo_order  = 0;  ///< Insertion order for FIFO.
};

/// Per-level cache performance counters.
struct CacheStats
{
    u64 reads           = 0;
    u64 writes          = 0;
    u64 hits            = 0;
    u64 misses          = 0;
    u64 evictions       = 0;
    u64 dirty_evictions = 0;  ///< Evictions requiring a writeback.
    u64 total_latency   = 0;  ///< Sum of per-access latencies.

    [[nodiscard]] double hit_rate() const noexcept
    {
        u64 total = hits + misses;
        return total > 0
            ? static_cast<double>(hits) / static_cast<double>(total)
            : 0.0;
    }

    [[nodiscard]] double miss_rate() const noexcept
    {
        return 1.0 - hit_rate();
    }

    [[nodiscard]] double avg_latency() const noexcept
    {
        u64 accesses = reads + writes;
        return accesses > 0
            ? static_cast<double>(total_latency) / static_cast<double>(accesses)
            : 0.0;
    }

    void reset() noexcept { *this = CacheStats{}; }
};

/**
 * @brief Single cache level with configurable geometry, replacement, and write policies.
 *
 * Uses flat-vector storage for line payloads. Implements the Memory
 * interface so it can be chained with other levels or used standalone.
 */
class Cache : public Memory {
public:
    Cache(CacheConfig, std::shared_ptr<Memory> next_level);

    /// @name Memory interface
    /// @{
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    MemoryResult write8(addr_t addr, u8 value)   override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write32(addr_t addr, u32 value) override;
    void load(addr_t addr, std::span<const u8> data) override;
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    u32 read_line(addr_t addr, u8* dest, u32 size) const override;
    u32 write_line(addr_t addr, const u8* src, u32 size) override;    
    /// @}

    /// @name Cache control
    /// @{
    [[nodiscard]] bool lookup(addr_t addr) const;
    void invalidate(addr_t addr);
    void invalidate_all();
    void writeback(addr_t addr);  ///< Flush dirty line without invalidating.
    void writeback_all();         ///< Flush all dirty lines without invalidating.
    void flush(addr_t addr);      ///< Writeback + invalidate.
    void flush_all();             ///< Writeback + invalidate all lines.
    /// @}

    /// @name Coherence snooping
    /// @{

    /// Check if this cache holds the line containing @p addr.
    [[nodiscard]] bool snoop_has_line(addr_t addr, bool* dirty = nullptr) const;

    /** 
     * @brief Supply line data to another cache (Modified/Exclusive → Shared).
     *
     * Copies line data to @p dest and marks the local copy clean.
     * @return @c false if the line is not present.
     */
    bool snoop_share_line(addr_t addr, u8* dest, u32 size);

    /// Invalidate a line, writing back to next level first if dirty.
    void snoop_invalidate(addr_t addr);
    /// @}

    /// @name Accessors
    /// @{
    [[nodiscard]] const CacheStats&  stats()  const noexcept { return stats_; }
    [[nodiscard]]       CacheStats&  stats()        noexcept { return stats_; }
    [[nodiscard]] const CacheConfig& config() const noexcept { return config_; }
    /// @}

    /// @name Coherence callbacks
    /// @{
    using MissCallback  = std::function<u32(addr_t)>;  ///< Invoked on read miss before fetch.
    using WriteCallback = std::function<u32(addr_t)>;  ///< Invoked on every write.
    using EvictCallback = std::function<void(addr_t)>; ///< Invoked when this cache gives up a line.

    void set_on_read_miss(MissCallback cb) { on_read_miss_ = std::move(cb); }
    void set_on_write(WriteCallback cb)    { on_write_ = std::move(cb); }

    /**
     * @brief Register a callback fired when a line leaves this cache on
     *        its own.
     *
     * Covers replacement evictions, explicit invalidation, and a write miss
     * that did not allocate. Not fired by @c snoop_invalidate(), since the
     * coherence controller initiated that and already knows.
     */
    void set_on_evict(EvictCallback cb) { on_evict_ = std::move(cb); }
    /// @}

    /// @name Debug
    /// @{
    void dump() const;
    void set_trace(bool enable) noexcept { trace_ = enable; }
    /// @}

private:
    CacheConfig             config_;
    std::shared_ptr<Memory> next_level_;
    MissCallback            on_read_miss_;
    WriteCallback           on_write_;
    EvictCallback           on_evict_;
    bool                    trace_ = false;

    /*
     * Flat storage: all line payloads in one contiguous allocation.
     * Line i's data starts at storage_[i * config_.line_size].
     */
    mutable std::vector<u8>                     storage_;
    mutable std::vector<std::vector<CacheLine>> sets_;
    mutable CacheStats stats_;
    mutable u64        access_counter_ = 0;
    mutable std::vector<u32>             fifo_counters_;  ///< Per-set FIFO counters.
    mutable std::vector<std::vector<u8>> plru_bits_;      ///< Per-set PLRU tree bits.

    /// @name Address decomposition
    /// @{
    [[nodiscard]] u32    offset_of(addr_t addr)    const noexcept;
    [[nodiscard]] u32    index_of(addr_t addr)     const noexcept;
    [[nodiscard]] addr_t tag_of(addr_t addr)       const noexcept;
    [[nodiscard]] addr_t line_addr_of(addr_t addr) const noexcept;
    /// @}

    /// @name Internal helpers
    /// @{
    [[nodiscard]] u8*       line_data(u32 set_idx, u32 way) noexcept;
    [[nodiscard]] const u8* line_data(u32 set_idx, u32 way) const noexcept;

    struct LineRef      { CacheLine* meta; u8* data; };
    struct ConstLineRef { const CacheLine* meta; const u8* data; };

    [[nodiscard]] std::optional<LineRef>      find_line(addr_t addr);
    [[nodiscard]] std::optional<ConstLineRef> find_line(addr_t addr) const;

    LineRef allocate_line(addr_t addr);
    void    evict_line(u32 set_idx, u32 way);
    void    writeback_line(u32 set_idx, u32 way);
    void    fetch_line(addr_t line_addr, u32 set_idx, u32 way);
    [[nodiscard]] u32 find_victim(u32 set_idx) const;

    /// Update replacement metadata after accessing @p way in @p set_idx.
    void touch_way(u32 set_idx, u32 way) const;

    [[nodiscard]] MemoryResult read_bytes(addr_t addr, u32 size) const;
    MemoryResult write_bytes(addr_t addr, const u8* bytes, u32 size);

    /// Flat storage index: set_idx * associativity + way.
    [[nodiscard]] u32 line_index(u32 set_idx, u32 way) const noexcept
    {
        return set_idx * config_.associativity + way;
    }
    /// @}
};

/// Configuration for a multi-level cache hierarchy.
struct CacheHierarchyConfig
{
    std::vector<CacheConfig> levels;

    static CacheHierarchyConfig typical_2level()
    {
        return { .levels = { CacheConfig::L1_typical(), CacheConfig::L2_typical() } };
    }

    static CacheHierarchyConfig typical_3level()
    {
        return { .levels = { CacheConfig::L1_typical(), CacheConfig::L2_typical(),
                             CacheConfig::L3_typical(), } };
    }
};

/**
 * @brief Multi-level cache chain backed by a main memory.
 *
 * All accesses enter through L1. On a miss, L1's next level (L2)
 * is consulted, and so on. Implements the Memory interface for
 * drop-in use. 
 *
 * @par Latency model
 * Returned latency is the sum of @c hit_latency at every level hit,
 * plus @c miss_penalty at every level that missed.
 */
class CacheHierarchy : public Memory {
public:
    CacheHierarchy(CacheHierarchyConfig config, std::shared_ptr<Memory> main_memory);

    /// @name Memory interface
    /// @{
    [[nodiscard]] MemoryResult read8(addr_t addr)  const override;
    [[nodiscard]] MemoryResult read16(addr_t addr) const override;
    [[nodiscard]] MemoryResult read32(addr_t addr) const override;
    MemoryResult write8(addr_t addr, u8 value)   override;
    MemoryResult write16(addr_t addr, u16 value) override;
    MemoryResult write32(addr_t addr, u32 value) override;
    void load(addr_t addr, std::span<const u8> data) override;
    [[nodiscard]] bool valid_address(addr_t addr, size_t size = 1) const override;
    /// @}

    /// @name Cache control
    /// @{
    void flush_all();
    void invalidate_all();
    /// @}

    /// @name Introspection
    /// @{
    [[nodiscard]] size_t       num_levels() const noexcept { return caches_.size(); }
    [[nodiscard]]       Cache& level(size_t n)       { return *caches_.at(n); }
    [[nodiscard]] const Cache& level(size_t n) const { return *caches_.at(n); }

    /// Aggregated statistics across all cache levels.
    struct HierarchyStats
    {
        std::vector<CacheStats> level_stats;
        u64 total_accesses = 0;
        u64 total_latency  = 0;

        [[nodiscard]] double avg_latency() const noexcept
        {
            return total_accesses > 0
                ? static_cast<double>(total_latency) / static_cast<double>(total_accesses)
                : 0.0;
        }
    };

    [[nodiscard]] HierarchyStats get_stats() const;
    void reset_stats();
    /// @}

private:
    std::vector<std::shared_ptr<Cache>> caches_;
    std::shared_ptr<Memory>             main_memory_;
};

} // namespace riscv
