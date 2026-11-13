/**
 * @file coherence.cpp
 * @brief MESI coherence controller implementation.
 */

#include "coherence.hpp"

#include <algorithm>
#include <cassert>

namespace riscv {

CoherenceController::CoherenceController(std::vector<Cache*> l1_caches,
                                         std::shared_ptr<Memory> shared_mem,
                                         u32 line_size)
    : l1_caches_(std::move(l1_caches))
    , shared_mem_(std::move(shared_mem))
    , line_size_(line_size)
    , num_cores_(static_cast<u32>(l1_caches_.size()))
{
    assert(num_cores_ > 0);
    assert(std::has_single_bit(line_size_));
}

// ── Directory management ───────────────────────────────────────────────

DirectoryEntry& CoherenceController::get_entry(addr_t addr)
{
    addr_t la = line_align(addr);
    auto it = directory_.find(la);
    if (it == directory_.end())
    {
        auto [inserted, _] = directory_.emplace(la, DirectoryEntry(num_cores_));
        return inserted->second;
    }
    return it->second;
}

MESIState CoherenceController::line_state(u32 core_id, addr_t addr) const
{
    addr_t la = line_align(addr);
    auto it = directory_.find(la);
    if (it == directory_.end()) return MESIState::Invalid;
    if (core_id >= it->second.core_states.size()) return MESIState::Invalid;

    return it->second.core_states[core_id];
}

// ── Helpers: find cores with specific states ───────────────────────────

i32 CoherenceController::find_modified_owner(const DirectoryEntry& entry) const
{
    for (u32 i = 0; i < num_cores_; ++i)
        if (entry.core_states[i] == MESIState::Modified)
            return static_cast<i32>(i);
    return -1;
}

i32 CoherenceController::find_any_sharer(const DirectoryEntry& entry, u32 exclude_core) const
{
    for (u32 i = 0; i < num_cores_; ++i)
    {
        if (i == exclude_core) continue;
        if (entry.core_states[i] == MESIState::Shared ||
            entry.core_states[i] == MESIState::Exclusive)
            return static_cast<i32>(i);
    }
    return -1;
}

// ── Invalidate other cores ─────────────────────────────────────────────

u32 CoherenceController::invalidate_others(u32 requesting_core, addr_t line_addr, DirectoryEntry& entry)
{
    u32 latency = 0;

    for (u32 i = 0; i < num_cores_; ++i)
    {
        if (i == requesting_core) continue;

        MESIState s = entry.core_states[i];
        if (s == MESIState::Invalid) continue;

        if (s == MESIState::Modified)
        {
            // Other core has dirty data; force writeback
            l1_caches_[i]->snoop_invalidate(line_addr);
            stats_.writebacks_forced++;
            latency += kTransferLatency;
        }
        else
        {
            l1_caches_[i]->snoop_invalidate(line_addr);
            latency += kInvalidationLatency;
        }

        entry.core_states[i] = MESIState::Invalid;
        stats_.invalidations_sent++;
    }

    return latency;
}

// ── Read miss handling ─────────────────────────────────────────────────

u32 CoherenceController::handle_read_miss(u32 core_id, addr_t addr)
{
    assert(core_id < num_cores_);
    addr_t la = line_align(addr);
    auto& entry = get_entry(la);
    stats_.read_misses++;

    u32 latency = kSnoopLatency;

    // Check if another core has the line Modified
    i32 mod_owner = find_modified_owner(entry);
    u32 u_mod_owner = static_cast<u32>(mod_owner);
    if (mod_owner >= 0 && u_mod_owner != core_id)
    {
        // The owner writes its data back to the shared level, and the
        // requester's refill then picks it up from there
        std::vector<u8> line_data(line_size_);
        l1_caches_[u_mod_owner]->snoop_share_line(la, line_data.data(), line_size_);
        entry.core_states[u_mod_owner] = MESIState::Shared;
        entry.core_states[core_id]     = MESIState::Shared;
        stats_.shared_transfers++;
        stats_.writebacks_forced++;
        latency += kTransferLatency;

        return latency;
    }

    // Check if another core has the line Shared or Exclusive
    i32 sharer = find_any_sharer(entry, core_id);
    if (sharer >= 0)
    {
        u32 u_sharer = static_cast<u32>(sharer);

        if (entry.core_states[u_sharer] == MESIState::Exclusive)
            entry.core_states[u_sharer] = MESIState::Shared;

        entry.core_states[core_id] = MESIState::Shared;
        stats_.l2_fetches++;
        return latency;
    }

    // No other core has it - fetch from L2/memory
    entry.core_states[core_id] = MESIState::Exclusive;
    stats_.l2_fetches++;
    return latency;
}

// ── Write miss handling ────────────────────────────────────────────────

u32 CoherenceController::handle_write_miss(u32 core_id, addr_t addr)
{
    assert(core_id < num_cores_);
    addr_t la = line_align(addr);
    auto& entry = get_entry(la);

    MESIState current = entry.core_states[core_id];

    if (current == MESIState::Modified)
        return 0;

    if (current == MESIState::Exclusive)
    {
        entry.core_states[core_id] = MESIState::Modified;
        stats_.upgrades++;
        return 0;  // no additional latency
    }

    stats_.write_misses++;
    u32 latency = kSnoopLatency;

    if (current == MESIState::Shared)
    {
        latency += invalidate_others(core_id, la, entry);
        entry.core_states[core_id] = MESIState::Modified;
        stats_.upgrades++;
        return latency;
    }

    // Invalid
    latency += invalidate_others(core_id, la, entry);
    entry.core_states[core_id] = MESIState::Modified;
    stats_.l2_fetches++;
    return latency;
}

// ── I-cache refills and DMA ────────────────────────────────────────────

u32 CoherenceController::clean_modified_owner(addr_t line_addr)
{
    auto it = directory_.find(line_addr);
    if (it == directory_.end()) return 0;

    const i32 owner = find_modified_owner(it->second);
    if (owner < 0) return 0;

    const u32 u_owner = static_cast<u32>(owner);
    std::vector<u8> line_data(line_size_);
    l1_caches_[u_owner]->snoop_share_line(line_addr, line_data.data(), line_size_);
    it->second.core_states[u_owner] = MESIState::Exclusive; // clean + still the only L1D copy
    stats_.writebacks_forced++;
    return kTransferLatency;
}

u32 CoherenceController::handle_fetch_miss(u32 core_id, addr_t addr)
{
    assert(core_id < num_cores_);
    (void)core_id;
    stats_.fetch_misses++;
    return kSnoopLatency + clean_modified_owner(line_align(addr));
}

u32 CoherenceController::dma_read(addr_t addr)
{
    stats_.dma_reads++;
    return kSnoopLatency + clean_modified_owner(line_align(addr));
}

u32 CoherenceController::dma_write(addr_t addr)
{
    stats_.dma_writes++;

    addr_t la = line_align(addr);
    auto it = directory_.find(la);
    if (it == directory_.end()) return kSnoopLatency;

    // num_cores_ is not a valid core ID, so every holder is invalidated
    const u32 latency = kSnoopLatency + invalidate_others(num_cores_, la, it->second);
    directory_.erase(it);
    return latency;
}

// ── Eviction notification ──────────────────────────────────────────────

void CoherenceController::notify_eviction(u32 core_id, addr_t addr)
{
    addr_t la = line_align(addr);
    auto it = directory_.find(la);
    if (it == directory_.end()) return;

    auto& entry = it->second;
    if (core_id < entry.core_states.size())
        entry.core_states[core_id] = MESIState::Invalid;

    if (std::none_of(entry.core_states.begin(), entry.core_states.end(),
                     [](MESIState s) { return s != MESIState::Invalid; }))
        directory_.erase(it);
}

} // namespace riscv
