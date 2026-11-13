/**
 * @file cache.cpp
 * @brief Cache and CacheHierarchy implementation.
 */

#include "cache.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <format>
#include <iostream>
#include <random>

namespace riscv {

// ── Construction ───────────────────────────────────────────────────────

Cache::Cache(CacheConfig config, std::shared_ptr<Memory> next_level)
    : config_(config)
    , next_level_(std::move(next_level))
    , storage_(static_cast<size_t>(config.num_lines()) * config.line_size, 0)
    , fifo_counters_(config.num_sets(), 0)
    , plru_bits_(config.num_sets(),
                 std::vector<u8>(config.associativity > 1 ? config.associativity - 1 : 0, 0))
{
    assert(config_.valid() && "Cache geometry must be power-of-2 and self-consistent.");

    u32 ns = config_.num_sets();
    sets_.resize(ns);
    for (auto& set : sets_)
        set.resize(config_.associativity);
}

// ── Address decomposition ──────────────────────────────────────────────

u32 Cache::offset_of(addr_t addr) const noexcept
{
    return addr & (config_.line_size - 1);
}

u32 Cache::index_of(addr_t addr) const noexcept
{
    return (addr >> std::countr_zero(config_.line_size)) & (config_.num_sets() - 1);
}

addr_t Cache::tag_of(addr_t addr) const noexcept
{
    unsigned shift = static_cast<unsigned>(std::countr_zero(config_.line_size)
                                         + std::countr_zero(config_.num_sets()));
    return addr >> shift;
}

addr_t Cache::line_addr_of(addr_t addr) const noexcept
{
    return addr & ~static_cast<addr_t>(config_.line_size - 1);
}

// ── Flat storage access ────────────────────────────────────────────────

u8* Cache::line_data(u32 set_idx, u32 way) noexcept
{
    return &storage_[static_cast<size_t>(line_index(set_idx, way)) * config_.line_size];
}

const u8* Cache::line_data(u32 set_idx, u32 way) const noexcept
{
    return &storage_[static_cast<size_t>(line_index(set_idx, way)) * config_.line_size];
}

// ── Line lookup ────────────────────────────────────────────────────────

std::optional<Cache::LineRef> Cache::find_line(addr_t addr)
{
    u32 idx    = index_of(addr);
    addr_t tag = tag_of(addr);

    for (u32 w = 0; w < config_.associativity; ++w)
    {
        auto& meta = sets_[idx][w];
        if (meta.valid && meta.tag == tag)
            return LineRef{&meta, line_data(idx, w)};
    }
    return std::nullopt;
}

std::optional<Cache::ConstLineRef> Cache::find_line(addr_t addr) const
{
    u32 idx    = index_of(addr);
    addr_t tag = tag_of(addr);

    for (u32 w = 0; w < config_.associativity; ++w)
    {
        const auto& meta = sets_[idx][w];
        if (meta.valid && meta.tag == tag)
            return ConstLineRef{&meta, line_data(idx, w)};
    }
    return std::nullopt;
}

// ── Victim selection ───────────────────────────────────────────────────

u32 Cache::find_victim(u32 set_idx) const
{
    const auto& set = sets_[set_idx];
    u32 N = config_.associativity;

    // Always prefer an invalid (empty) slot.
    for (u32 w = 0; w < N; ++w)
        if (!set[w].valid) return w;

    switch(config_.replacement)
    {
        case ReplacementPolicy::LRU:
        {
            u32 victim = 0;
            u64 oldest = set[0].last_access;
            for (u32 w = 1; w < N; ++w)
                if (set[w].last_access < oldest)
                {
                    oldest = set[w].last_access;
                    victim = w;
                }
            return victim;
        }

        case ReplacementPolicy::MRU:
        {
            u32 victim = 0;
            u64 newest = set[0].last_access;
            for (u32 w = 1; w < N; ++w)
                if (set[w].last_access > newest)
                {
                    newest = set[w].last_access;
                    victim = w;
                }
            return victim;
        }

        case ReplacementPolicy::PLRU:
        {
            const auto& bits = plru_bits_[set_idx];
            u32 node = 0;
            u32 way  = 0;
            u32 span = N;  // number of ways in the current subtree

            while (span > 1)
            {
                u32 half = span / 2;
                if (bits[node] == 0)
                    node = 2 * node + 1;
                // way stays the same (leftmost of current subtree)
                else
                {
                    node = 2 * node + 2;
                    way += half;
                }
                span = half;
            }
            return way;
        }

        case ReplacementPolicy::FIFO:
        {
            u32 victim = 0;
            u32 earliest = set[0].fifo_order;
            for (u32 w = 1; w < N; ++w)
                if (set[w].fifo_order < earliest)
                {
                    earliest = set[w].fifo_order;
                    victim = w;
                }
            return victim;
        }

        case ReplacementPolicy::RANDOM:
        {
            static std::mt19937 rng{42};
            return static_cast<u32>(rng() % N);
        } 
    }

    return 0;  // unreachable
}

// ── Replacement metadata update ────────────────────────────────────────

void Cache::touch_way(u32 set_idx, u32 way) const
{
    if (config_.replacement != ReplacementPolicy::PLRU) return;

    /*
     * Walk the tree from root to leaf, flipping each bit to point
     * away from the accessed way
     */
    auto& bits = plru_bits_[set_idx];
    u32 N    = config_.associativity;
    u32 node = 0;
    u32 base = 0;  // leftmost way index of the current subtree
    u32 span = N;

    while (span > 1)
    {
        u32 half = span / 2;
        if (way < base + half)
        {
            // Accessed way is in the left subtree; point toward right
            bits[node] = 1;
            node = 2 * node + 1;
        }
        else
        {
            // Accessed way is in the right subtree; point toward left
            bits[node] = 0;
            node = 2 * node + 2;
            base += half;
        }
        span = half;
    }
}

// ── Writeback (without invalidation) ───────────────────────────────────

void Cache::writeback_line(u32 set_idx, u32 way)
{
    auto& meta = sets_[set_idx][way];
    if (!meta.valid || !meta.dirty) return;

    if (trace_)
        std::cout << std::format("[WRITEBACK] line=0x{:08x} | set={} | way={}\n",
                                 meta.line_addr, set_idx, way);

    const u8* data = line_data(set_idx, way);
    next_level_->write_line(meta.line_addr, data, config_.line_size);
    meta.dirty = false;
}

// ── Eviction (writeback + invalidate) ──────────────────────────────────

void Cache::evict_line(u32 set_idx, u32 way)
{
    auto& meta = sets_[set_idx][way];
    if (!meta.valid) return;

    stats_.evictions++;

    if (meta.dirty)
    {
        stats_.dirty_evictions++;
        writeback_line(set_idx, way);
    }

    if (trace_)
        std::cout << std::format("[EVICT] line=0x{:08x} | set={} | way={} | dirty={}\n",
                                 meta.line_addr, set_idx, way, meta.dirty);

    meta.valid = false;
    meta.dirty = false;

    if (on_evict_) on_evict_(meta.line_addr);
}

// ── Fetch line from next level ─────────────────────────────────────────

void Cache::fetch_line(addr_t line_addr, u32 set_idx, u32 way)
{
    u8* dest = line_data(set_idx, way);
    next_level_->read_line(line_addr, dest, config_.line_size);

    auto& meta       = sets_[set_idx][way];
    meta.valid       = true;
    meta.dirty       = false;
    meta.tag         = tag_of(line_addr);
    meta.line_addr   = line_addr;
    meta.last_access = access_counter_;
    meta.fifo_order  = fifo_counters_[set_idx]++;

    if (trace_)
        std::cout << std::format("[REFILL] line=0x{:08x} | set={} | way={}\n",
                                 line_addr, set_idx, way);
}

// ── Allocate (evict victim, fetch new line) ────────────────────────────

Cache::LineRef Cache::allocate_line(addr_t addr)
{
    u32 idx    = index_of(addr);
    u32 victim = find_victim(idx);

    evict_line(idx, victim);
    fetch_line(line_addr_of(addr), idx, victim);

    if (trace_)
        std::cout << std::format("[ALLOC] line=0x{:08x} | set={} | way={}\n",
                                 line_addr_of(addr), idx, victim);

    return {&sets_[idx][victim], line_data(idx, victim)};
}

// ── Core read / write (handles cross-line splits) ──────────────────────

MemoryResult Cache::read_bytes(addr_t addr, u32 size) const
{
    // Cross-line check: if the access spans two lines, split it
    u32 off = offset_of(addr);
    if (off + size > config_.line_size)
    {
        // Recursive split - each half is within a single line
        u32 first_part = config_.line_size - off;
        if (trace_)
            std::cout << std::format("[READ_SPLIT] addr=0x{:08x} | "
                                     "size={} -> {}+{} bytes\n",
                                     addr, size,
                                     first_part, size - first_part);

        auto r1 = read_bytes(addr, first_part);
        auto r2 = read_bytes(addr + first_part, size - first_part);

        u32 value = r1.value | (r2.value << (first_part * 8));
        return {value, r1.cycles + r2.cycles, r1.ok && r2.ok};
    }

    access_counter_++;
    stats_.reads++;

    auto found = find_line(addr);
    u32 latency;

    const u8* data_ptr;

    if (found)
    {
        stats_.hits++;
        latency = config_.hit_latency;
        found->meta->last_access = access_counter_;
        u32 hit_way = static_cast<u32>(found->meta - sets_[index_of(addr)].data());
        touch_way(index_of(addr), hit_way);
        data_ptr = found->data;
    }
    else
    {
        stats_.misses++;
        u32 coherence_latency = 0;
        if (on_read_miss_)
            coherence_latency = on_read_miss_(line_addr_of(addr));

        auto& self = const_cast<Cache&>(*this);
        auto ref = self.allocate_line(addr);
        latency = config_.hit_latency + config_.miss_penalty + coherence_latency;
        u32 alloc_way = static_cast<u32>(ref.meta - sets_[index_of(addr)].data());
        touch_way(index_of(addr), alloc_way);
        data_ptr = ref.data;
    }

    stats_.total_latency += latency;

    u32 value = 0;
    for (u32 i = 0; i < size; ++i)
        value |= static_cast<u32>(data_ptr[off + i]) << (i * 8);

    if (trace_)
        std::cout << std::format("[READ] addr=0x{:08x} | set={} | tag=0x{:x} | "
                                 "{} | latency={}\n",
                                 addr, index_of(addr), tag_of(addr),
                                 found ? "HIT" : "MISS",
                                 latency);

    return {value, latency, true};
}

MemoryResult Cache::write_bytes(addr_t addr, const u8* bytes, u32 size)
{
    assert(config_.type != CacheType::INSTRUCTION && "I-cache is read-only");

    // Cross-line check
    u32 off = offset_of(addr);
    if (off + size > config_.line_size)
    {
        u32 first_part = config_.line_size - off;
        if (trace_)
            std::cout << std::format("[WRITE_SPLIT] addr=0x{:08x} | "
                                     "size={} -> {}+{} bytes\n",
                                     addr, size,
                                     first_part, size - first_part);

        auto r1 = write_bytes(addr, bytes, first_part);
        auto r2 = write_bytes(addr + first_part, bytes + first_part, size - first_part);
        return {0, r1.cycles + r2.cycles, r1.ok && r2.ok};
    }

    access_counter_++;
    stats_.writes++;

    u32 coherence_latency = 0;
    if (on_write_)
        coherence_latency = on_write_(line_addr_of(addr));

    auto log_write = [&](std::string_view hit_or_miss, u32 latency)
    {
        if (trace_)
            std::cout << std::format("[WRITE] addr=0x{:08x} | size={} | "
                                     "set={} | tag=0x{:x} | {} | latency={}\n",
                                     addr, size, index_of(addr), tag_of(addr),
                                     hit_or_miss, latency);
    };

    // Write-no-allocate path
    if (config_.write_alloc == WriteAllocate::NO_ALLOCATE)
    {
        auto found = find_line(addr);
        if (!found)
        {
            stats_.misses++;
            for (u32 i = 0; i < size; ++i)
                next_level_->write8(addr + i, bytes[i]);
            u32 latency = config_.hit_latency + config_.miss_penalty + coherence_latency;
            stats_.total_latency += latency;

            if (on_evict_) on_evict_(line_addr_of(addr));

            log_write("MISS", latency);
            return {0, latency, true};
        }

        stats_.hits++;
        found->meta->last_access = access_counter_;
        u32 hit_way = static_cast<u32>(found->meta - sets_[index_of(addr)].data());
        touch_way(index_of(addr), hit_way);
        std::memcpy(found->data + off, bytes, size);

        if (config_.write_pol == WritePolicy::WRITE_THROUGH)
            for (u32 i = 0; i < size; ++i)
                next_level_->write8(addr + i, bytes[i]);
        else
            found->meta->dirty = true;

        u32 latency = config_.hit_latency + coherence_latency;
        stats_.total_latency += latency;

        log_write("HIT", latency);
        return {0, latency, true};
    }

    // Write-allocate path. A write-through line is never dirty; the next
    // level is updated below on every write
    const bool mark_dirty = (config_.write_pol == WritePolicy::WRITE_BACK);
    auto found = find_line(addr);
    u32 latency;
    u8* dest;
    u32 way_idx;

    if (found)
    {
        stats_.hits++;
        latency = config_.hit_latency;
        found->meta->last_access = access_counter_;
        way_idx = static_cast<u32>(found->meta - sets_[index_of(addr)].data());
        dest = const_cast<u8*>(found->data);
        found->meta->dirty |= mark_dirty;
    }
    else
    {
        stats_.misses++;
        auto ref = allocate_line(addr);
        latency = config_.hit_latency + config_.miss_penalty;
        way_idx = static_cast<u32>(ref.meta - sets_[index_of(addr)].data());
        dest = ref.data;
        ref.meta->dirty = mark_dirty;
    }

    touch_way(index_of(addr), way_idx);
    std::memcpy(dest + off, bytes, size);

    if (config_.write_pol == WritePolicy::WRITE_THROUGH)
        for (u32 i = 0; i < size; ++i)
            next_level_->write8(addr + i, bytes[i]);

    latency += coherence_latency;
    stats_.total_latency += latency;

    log_write(found ? "HIT" : "MISS", latency);
    return {0, latency, true};
}

// ── Memory interface ───────────────────────────────────────────────────

MemoryResult Cache::read8(addr_t addr) const  { return read_bytes(addr, 1); }
MemoryResult Cache::read16(addr_t addr) const { return read_bytes(addr, 2); }
MemoryResult Cache::read32(addr_t addr) const { return read_bytes(addr, 4); }

MemoryResult Cache::write8(addr_t addr, u8 value)
{
    return write_bytes(addr, &value, 1);
}

MemoryResult Cache::write16(addr_t addr, u16 value)
{
    std::array<u8, 2> buf =
    {
        static_cast<u8>(value),
        static_cast<u8>(value >> 8),
    };
    return write_bytes(addr, buf.data(), 2);
}

MemoryResult Cache::write32(addr_t addr, u32 value)
{
    std::array<u8, 4> buf =
    {
        static_cast<u8>(value),
        static_cast<u8>(value >> 8),
        static_cast<u8>(value >> 16),
        static_cast<u8>(value >> 24),
    };
    return write_bytes(addr, buf.data(), 4);
}

void Cache::load(addr_t addr, std::span<const u8> data)
{
    // Bypass cache and write directly to backing memory
    next_level_->load(addr, data);

    // Invalidate any cached copies that overlap
    addr_t start = line_addr_of(addr);
    addr_t end   = line_addr_of(addr + static_cast<addr_t>(data.size()) - 1);
    for (addr_t la = start; la <= end; la += config_.line_size)
        invalidate(la);
}

bool Cache::valid_address(addr_t addr, size_t size) const
{
    return next_level_->valid_address(addr, size);
}

// ── Bulk line read/write ───────────────────────────────────────────────

u32 Cache::read_line(addr_t addr, u8* dest, u32 size) const
{
    auto found = find_line(addr);
    if (found)
    {
        found->meta->last_access = access_counter_++;
        u32 idx = index_of(addr);
        u32 way = static_cast<u32>(found->meta - sets_[idx].data());
        touch_way(idx, way);
        std::memcpy(dest, found->data, std::min(size, config_.line_size));

        stats_.hits++;
        stats_.reads++;
        stats_.total_latency += config_.hit_latency;
        return config_.hit_latency;
    }

    stats_.misses++;
    stats_.reads++;
    auto& self = const_cast<Cache&>(*this);
    auto ref = self.allocate_line(addr);
    u32 idx = index_of(addr);
    u32 way = static_cast<u32>(ref.meta - sets_[idx].data());
    touch_way(idx, way);
    std::memcpy(dest, ref.data, std::min(size, config_.line_size));

    u32 latency = config_.hit_latency + config_.miss_penalty;
    stats_.total_latency += latency;
    return latency;
}

u32 Cache::write_line(addr_t addr, const u8* src, u32 size)
{
    auto found = find_line(addr);
    u8* dest;
    u32 latency;

    if (found)
    {
        stats_.hits++;
        latency = config_.hit_latency;
        found->meta->last_access = access_counter_++;

        u32 idx = index_of(addr);
        u32 way = static_cast<u32>(found->meta - sets_[idx].data());
        touch_way(idx, way);
        dest = const_cast<u8*>(found->data);
    }
    else
    {
        stats_.misses++;
        auto ref = allocate_line(addr);
        latency = config_.hit_latency + config_.miss_penalty;
        u32 idx = index_of(addr);
        u32 way = static_cast<u32>(ref.meta - sets_[idx].data());
        touch_way(idx, way);
        dest = ref.data;
    }

    std::memcpy(dest, src, std::min(size, config_.line_size));
    stats_.writes++;

    if (config_.write_pol == WritePolicy::WRITE_THROUGH)
        next_level_->write_line(addr, src, size);
    else
    {
        auto ref = find_line(addr);
        if (ref) ref->meta->dirty = true;
    }

    stats_.total_latency += latency;
    return latency;
}

// ── Coherence snoop support ─────────────────────────────────────────────

bool Cache::snoop_has_line(addr_t addr, bool* dirty) const
{
    auto found = find_line(addr);
    if (!found) return false;

    if (dirty) *dirty = found->meta->dirty;
    return true;
}

bool Cache::snoop_share_line(addr_t addr, u8* dest, u32 size)
{
    auto found = find_line(addr);
    if (!found) return false;

    std::memcpy(dest, found->data, std::min(size, config_.line_size));

    if (found->meta->dirty)
    {
        next_level_->write_line(found->meta->line_addr, found->data, config_.line_size);
        found->meta->dirty = false;
    }

    return true;
}

void Cache::snoop_invalidate(addr_t addr)
{
    auto found = find_line(addr);
    if (!found) return;

    if (found->meta->dirty)
    {
        u32 idx = index_of(addr);
        u32 way = static_cast<u32>(found->meta - sets_[idx].data());
        writeback_line(idx, way);
    }
    found->meta->valid = false;
    found->meta->dirty = false;
}

// ── Cache control ──────────────────────────────────────────────────────

bool Cache::lookup(addr_t addr) const
{
    return find_line(addr).has_value();
}

void Cache::invalidate(addr_t addr)
{
    auto found = find_line(addr);
    if (found)
    {
        const addr_t la = found->meta->line_addr;
        found->meta->valid = false;
        found->meta->dirty = false;
        if (on_evict_) on_evict_(la);
    }
}

void Cache::invalidate_all()
{
    for (auto& set : sets_)
        for (auto& meta : set)
        {
            const bool was_valid = meta.valid;
            meta.valid = false;
            meta.dirty = false;
            if (was_valid && on_evict_) on_evict_(meta.line_addr);
        }
}

void Cache::writeback(addr_t addr)
{
    u32 idx = index_of(addr);
    addr_t tag = tag_of(addr);
    for (u32 w = 0; w < config_.associativity; ++w)
        if (sets_[idx][w].valid && sets_[idx][w].tag == tag)
        {
            writeback_line(idx, w);
            return;
        }
}

void Cache::writeback_all()
{
    for (u32 s = 0; s < sets_.size(); ++s)
        for (u32 w = 0; w < config_.associativity; ++w)
            writeback_line(s, w);
}

void Cache::flush(addr_t addr)
{
    writeback(addr);
    invalidate(addr);
}

void Cache::flush_all()
{
    writeback_all();
    invalidate_all();
}

void Cache::dump() const
{
    u32 valid_lines = 0;
    u32 dirty_lines = 0;
    for (const auto& set : sets_)
        for (const auto& meta : set)
        {
            if (meta.valid) ++valid_lines;
            if (meta.valid && meta.dirty) ++dirty_lines;
        }

    std::cout << std::format(
        "Cache: {} KB, {}-way, {}-byte lines, {} sets\n"
        "  Hit rate:    {:.2f}%\n"
        "  Avg latency: {:.1f} cycles\n"
        "  Valid lines: {}/{}\n"
        "  Dirty lines: {}\n",
        config_.size_bytes / 1024, config_.associativity,
        config_.line_size, config_.num_sets(),
        stats_.hit_rate() * 100.0, stats_.avg_latency(),
        valid_lines, config_.num_lines(), dirty_lines);
}

// ── CacheHierarchy ─────────────────────────────────────────────────────

CacheHierarchy::CacheHierarchy(CacheHierarchyConfig config,
                               std::shared_ptr<Memory> main_memory)
    : main_memory_(std::move(main_memory))
{
    assert(!config.levels.empty());

    /*
     * Build the chain from outermost -> innermost so each level's
     * next_level is already constructed
     */
    std::shared_ptr<Memory> next = main_memory_;

    for (auto it = config.levels.rbegin(); it != config.levels.rend(); ++it)
    {
        auto cache = std::make_shared<Cache>(*it, next);
        caches_.insert(caches_.begin(), cache);
        next = cache;
    }
}

// ── CacheHierarchy memory interface ────────────────────────────────────

MemoryResult CacheHierarchy::read8(addr_t addr)  const { return caches_[0]->read8(addr); }
MemoryResult CacheHierarchy::read16(addr_t addr) const { return caches_[0]->read16(addr); }
MemoryResult CacheHierarchy::read32(addr_t addr) const { return caches_[0]->read32(addr); }

MemoryResult CacheHierarchy::write8(addr_t addr, u8 value)   { return caches_[0]->write8(addr, value); }
MemoryResult CacheHierarchy::write16(addr_t addr, u16 value) { return caches_[0]->write16(addr, value); }
MemoryResult CacheHierarchy::write32(addr_t addr, u32 value) { return caches_[0]->write32(addr, value); }

void CacheHierarchy::load(addr_t addr, std::span<const u8> data)
{
    main_memory_->load(addr, data);
    for (auto& cache : caches_)
    {
        addr_t start = cache->config().line_size > 0
                     ? (addr & ~static_cast<addr_t>(cache->config().line_size - 1))
                     : addr;
        addr_t end = addr + static_cast<addr_t>(data.size());

        for (addr_t la = start; la < end; la += cache->config().line_size)
            cache->invalidate(la);
    }
}

bool CacheHierarchy::valid_address(addr_t addr, size_t size) const
{
    return main_memory_->valid_address(addr, size);
}

// ── Cache control ──────────────────────────────────────────────────────

void CacheHierarchy::flush_all()
{
    for (auto& cache : caches_)
        cache->flush_all();
}

void CacheHierarchy::invalidate_all()
{
    for (auto& cache : caches_)
        cache->invalidate_all();
}

// ── Statistics ─────────────────────────────────────────────────────────

CacheHierarchy::HierarchyStats CacheHierarchy::get_stats() const
{
    HierarchyStats hs;
    hs.level_stats.reserve(caches_.size());

    for (const auto& cache : caches_)
        hs.level_stats.push_back(cache->stats());

    if (!caches_.empty())
    {
        const auto& l1 = caches_[0]->stats();
        hs.total_accesses = l1.reads + l1.writes;
        hs.total_latency = l1.total_latency;
    }

    return hs;
}

void CacheHierarchy::reset_stats()
{
    for (auto& cache : caches_)
        cache->stats().reset();
}

} // namespace riscv
