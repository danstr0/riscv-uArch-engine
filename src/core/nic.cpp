/**
 * @file nic.cpp
 * @brief NIC implementation with DMA timing, interrupt coalescing, and RSS.
 */

#include "nic.hpp"

#include <algorithm>
#include <iostream>

namespace riscv {

// ── Constructor / reset ────────────────────────────────────────────────

NIC::NIC(std::shared_ptr<Memory> system_memory, NicTiming timing)
    : sys_mem_(std::move(system_memory))
    , timing_(timing)
{
    reset();
}

void NIC::reset()
{
    regs_.fill(0);
    regs_[NicReg::STATUS / 4] = NicCtrl::LINK_UP;

    interrupt_mask_        = 0;
    interrupt_pending_     = false;
    for (auto& r : rx_rings_) r = DescriptorRing{};
    for (auto& r : tx_rings_) r = DescriptorRing{};
    pending_dma_.clear();
    rx_queue_.clear();
    for (auto& q : tx_complete_) q.clear();
    stats_.reset();
    current_cycle_  = 0;
    rss_.enabled    = false;
    rss_.num_queues = 1;

    if (trace_)
        std::cout << "[NIC] RESET\n";
}

// ── RSS configuration ──────────────────────────────────────────────────

void NIC::set_rss(const RSSConfig& cfg)
{
    rss_ = cfg;

    if (trace_)
        std::cout << std::format("[NIC] RSS {} | queues={}\n",
                                 cfg.enabled ? "enabled" : "disabled",
                                 cfg.num_queues);
}

void NIC::configure_rx_queue(u32 qid, addr_t base, u32 ring_size, u32 tail)
{
    if (qid >= MAX_QUEUES) return;

    auto& r = rx_rings_[qid];
    r.base_addr = base;
    r.ring_size = ring_size;
    r.head      = 0;
    r.tail      = tail;

    if (trace_)
        std::cout << std::format("[NIC] RX ring q={} | base=0x{:08x} | "
                                 "size={} | tail={}\n",
                                 qid, base, ring_size, tail);
}

void NIC::configure_tx_queue(u32 qid, addr_t base, u32 ring_size)
{
    if (qid >= MAX_QUEUES) return;

    auto& r = tx_rings_[qid];
    r.base_addr = base;
    r.ring_size = ring_size;
    r.head      = 0;
    r.tail      = 0;

    if (trace_)
        std::cout << std::format("[NIC] TX ring q={} | base=0x{:08x} | size={}\n",
                                 qid, base, ring_size);
}

// ── RSS hash ────────────────────────────────────────────────────────────

u32 NIC::compute_rss_hash(const Packet& pkt) const
{
    if (pkt.data.size() < 20) return 0;

    u32 src_ip = (static_cast<u32>(pkt.data[12]) << 24)
               | (static_cast<u32>(pkt.data[13]) << 16)
               | (static_cast<u32>(pkt.data[14]) << 8)
               |  static_cast<u32>(pkt.data[15]);

    u32 dst_ip = (static_cast<u32>(pkt.data[16]) << 24)
               | (static_cast<u32>(pkt.data[17]) << 16)
               | (static_cast<u32>(pkt.data[18]) << 8)
               |  static_cast<u32>(pkt.data[19]);

    u32 hash = src_ip ^ dst_ip;
    if (pkt.data.size() >= 24 && (pkt.data[9] == 6 || pkt.data[9] == 17))
    {
        u16 src_port = (static_cast<u16>(pkt.data[20]) << 8) | pkt.data[21];
        u16 dst_port = (static_cast<u16>(pkt.data[22]) << 8) | pkt.data[23];

        hash ^= (static_cast<u32>(src_port) << 16) | dst_port;
    }
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    return hash;
}

u32 NIC::select_rx_queue(const Packet& pkt) const
{
    if (!rss_.enabled || rss_.num_queues <= 1) return 0;
    u32 hash = compute_rss_hash(pkt);
    u32 idx  = hash & (static_cast<u32>(rss_.indirection_table.size()) - 1);

    return rss_.indirection_table[idx] % rss_.num_queues;
}

// ── MMIO register access ───────────────────────────────────────────────

u32 NIC::reg(addr_t offset) const noexcept
{
    if (offset < NicReg::REG_SIZE && (offset & 3) == 0)
        return regs_[offset / 4];
    return 0;

}

MemoryResult NIC::read32(addr_t addr) const
{
    if (addr >= NicReg::REG_SIZE || (addr & 3) != 0)
        return {0, 1, false};

    u32 value = regs_[addr / 4];

    // ICR: read-on-clear per 82599 spec
    if (addr == NicReg::ICR)
        regs_[NicReg::ICR / 4] = 0;

    return {value, timing_.mmio_read_cycles, true};
}

MemoryResult NIC::write32(addr_t addr, u32 value)
{
    if (addr >= NicReg::REG_SIZE || (addr & 3) != 0)
        return {0, 1, false};

    switch (addr)
    {
        case NicReg::CTRL:
            if (value & NicCtrl::RST) reset();
            else
                regs_[NicReg::CTRL / 4] = value & ~NicCtrl::LINK_UP;
            break;
        case NicReg::STATUS: break;
        case NicReg::IMC: interrupt_mask_ &= ~value; break;
        case NicReg::IMS: interrupt_mask_ |= value;  break;
        case NicReg::ICR: regs_[NicReg::ICR / 4] &= ~value; break;
        case NicReg::ITR:
            coalesce_.max_delay_cycles = value & 0xFFFF;
            coalesce_.max_packets      = (value >> 16) & 0xFF;
            coalesce_.enabled          = (coalesce_.max_delay_cycles > 0
                                       || coalesce_.max_packets > 1);
            regs_[addr / 4] = value;
            break;
        case NicReg::RDBAL:
            regs_[addr / 4] = value;
            rx_rings_[0].base_addr = value;
            break;
        case NicReg::RDLEN:
            regs_[addr / 4] = value;
            rx_rings_[0].ring_size = value / 16;
            break;
        case NicReg::RDT:
            regs_[addr / 4] = value;
            rx_rings_[0].tail = value;
            process_rx_queue();
            break;
        case NicReg::TDBAL:
            regs_[addr / 4] = value;
            tx_rings_[0].base_addr = value;
            break;
        case NicReg::TDLEN:
            regs_[addr / 4] = value;
            tx_rings_[0].ring_size = value / 16;
            break;
        case NicReg::TDT:
            regs_[addr / 4] = value;
            tx_rings_[0].tail = value;
            process_tx_ring(0);
            break;
        case NicReg::RXPKT: case NicReg::TXPKT:
        case NicReg::RXBYTES: case NicReg::TXBYTES:
            break;
        default:
            regs_[addr / 4] = value;
            break;
    }

    return {value, timing_.mmio_write_cycles, true};
}

MemoryResult NIC::read16(addr_t addr) const
{
    if ((addr & ~3u) == NicReg::ICR) return {0, 1, false};

    auto r = read32(addr & ~3u);
    if (!r.ok) return {0, 1, false};

    u32 shift = (addr & 2) * 8;
    return {(r.value >> shift) & 0xFFFF, r.cycles, true};
}

MemoryResult NIC::write16(addr_t addr, u16 value)
{
    if ((addr & ~3u) == NicReg::ICR) return {0, 1, false};

    auto r = read32(addr & ~3u);
    if (!r.ok) return {0, 1, false};

    u32 shift = (addr & 2) * 8;
    u32 mask = 0xFFFF << shift;
    return write32(addr & ~3u, (r.value & ~mask) | (static_cast<u32>(value) << shift));
}

MemoryResult NIC::read8(addr_t addr) const
{
    if ((addr & ~3u) == NicReg::ICR) return {0, 1, false};

    auto r = read32(addr & ~3u);
    if (!r.ok) return {0, 1, false};

    u32 shift = (addr & 3) * 8;
    return {(r.value >> shift) & 0xFF, r.cycles, true};
}

MemoryResult NIC::write8(addr_t addr, u8 value)
{
    if ((addr & ~3u) == NicReg::ICR) return {0, 1, false};

    auto r = read32(addr & ~3u);
    if (!r.ok) return {0, 1, false};

    u32 shift = (addr & 3) * 8;
    u32 mask = 0xFF << shift;
    return write32(addr & ~3u, (r.value & ~mask) | (static_cast<u32>(value) << shift));
}

bool NIC::valid_address(addr_t addr, size_t size) const
{
    return size > 0 && addr + size <= NicReg::REG_SIZE;
}

// ── Simulation tick ────────────────────────────────────────────────────

void NIC::tick(cycle_t current_cycle)
{
    current_cycle_ = current_cycle;
    complete_dma();

    if (rx_enabled())
        process_rx_queue();

    if (coalesce_.enabled && coalesce_.max_delay_cycles > 0)
        for (u32 q = 0; q < rss_.num_queues; ++q)
        {
            auto& ring = rx_rings_[q];

            if (ring.coalesce_pending > 0
                && (current_cycle_ - ring.coalesce_first) >= coalesce_.max_delay_cycles)
            {
                if (trace_)
                    std::cout << std::format("[NIC] Coalescing timeout | q={}\n", q);

                stats_.coalesced_packets += ring.coalesce_pending - 1;
                ring.coalesce_pending = 0;
                raise_interrupt(NicInt::TIMER);
            }
        }
}

// ── Packet injection / retrieval ───────────────────────────────────────

void NIC::inject_packet(const Packet& pkt)
{
    Packet p = pkt;
    p.arrival_cycle = current_cycle_;
    rx_queue_.push_back(std::move(p));

    if (trace_)
        std::cout << std::format("[NIC] RX packet injected | {} bytes\n",
                                 pkt.data.size());

    if (rx_enabled())
        process_rx_queue();
}

std::optional<Packet> NIC::poll_tx(u32 qid)
{
    if (qid>= MAX_QUEUES || tx_complete_[qid].empty())
        return std::nullopt;

    Packet pkt = std::move(tx_complete_[qid].front());
    tx_complete_[qid].pop_front();
    return pkt;
}

// ── Interrupt handling ─────────────────────────────────────────────────

void NIC::raise_interrupt(u32 cause)
{
    regs_[NicReg::ICR / 4] |= cause;

    if ((cause & interrupt_mask_) && !interrupt_pending_)
    {
        interrupt_pending_ = true;
        stats_.interrupts_raised++;

        if (trace_)
            std::cout << std::format("[NIC] Interrupt cause=0x{:08x}\n", cause);

        if (interrupt_cb_)
            interrupt_cb_();
    }
}

void NIC::check_coalescing(u32 qid, u32 cause)
{
    auto& ring = rx_rings_[qid];
    ring.coalesce_pending++;

    if (!coalesce_.enabled)
    {
        raise_interrupt(cause);
        ring.coalesce_pending = 0;
        return;
    }

    if (ring.coalesce_pending == 1)
    {
        if (trace_)
            std::cout << std::format("[NIC] Coalescing started | q={}\n", qid);

        ring.coalesce_first = current_cycle_;
    }

    if (ring.coalesce_pending >= coalesce_.max_packets)
    {
        if (trace_)
            std::cout << std::format("[NIC] Coalescing threshold reached | "
                                     "q={} | packets={}\n",
                                     qid, ring.coalesce_pending);

        stats_.coalesced_packets += ring.coalesce_pending - 1;
        ring.coalesce_pending = 0;
        raise_interrupt(cause);
    }
}

// ── RX processing ──────────────────────────────────────────────────────

void NIC::process_rx_queue()
{
    if (!rx_enabled() || rx_queue_.empty()) return;

    while (!rx_queue_.empty())
    {
        Packet pkt = std::move(rx_queue_.front());
        rx_queue_.pop_front();

        u32 qid = select_rx_queue(pkt);
        auto& ring = rx_rings_[qid];

        if (ring.ring_size == 0)
        {
            if (trace_)
                std::cout << std::format("[NIC] RX drop (queue {} disabled)\n", qid);

            stats_.rx_dropped++;
            continue;
        }

        u32 pending_rx = 0;
        for (const auto& d : pending_dma_)
            if (d.type == PendingDma::Type::RX && d.queue_id == qid) ++pending_rx;

        u32 pending_head = (ring.head + pending_rx) % ring.ring_size;
        if (pending_head == ring.tail)
        {
            if (trace_)
                std::cout << std::format("[NIC] RX drop (ring full q={})\n", qid);

            stats_.rx_dropped++;
            continue;
        }

        pkt.dma_start_cycle = current_cycle_;
        u32 dma_time = timing_.dma_cycles(static_cast<u32>(pkt.data.size()));

        if (trace_)
            std::cout << std::format("[NIC] RX enqueue | q={} | "
                                     "desc={} | len={} | dma={} cycles\n",
                                     qid, pending_head, pkt.data.size(), dma_time);

        PendingDma dma;
        dma.type           = PendingDma::Type::RX;
        dma.queue_id       = qid;
        dma.desc_idx       = pending_head;
        dma.packet         = std::move(pkt);
        dma.complete_cycle = current_cycle_ + dma_time;
        pending_dma_.push_back(std::move(dma));
    }
}

// ── TX processing ──────────────────────────────────────────────────────

void NIC::process_tx_ring(u32 qid)
{
    if (!tx_enabled()) return;

    auto& ring = tx_rings_[qid];
    if (ring.ring_size == 0) return;

    while (ring.head != ring.tail)
    {
        TxDescriptor desc = read_tx_desc(qid, ring.head);

        if (desc.length == 0 || !sys_mem_->valid_address(desc.buffer_addr, desc.length))
        {
            if (trace_)
                std::cout << std::format("[NIC] TX drop | q={} | idx={} | "
                                         "invalid descriptor\n",
                                         qid, ring.head);

            stats_.tx_dropped++;
            ring.head = (ring.head + 1) % ring.ring_size;
            continue;
        }

        if (trace_)
            std::cout << std::format("[NIC] TX desc | q={} | idx={} | len={}\n",
                                     qid, ring.head, desc.length);

        Packet pkt;
        pkt.data.resize(desc.length);
        pkt.dma_start_cycle = current_cycle_;

        PendingDma dma;
        dma.type           = PendingDma::Type::TX;
        dma.queue_id       = qid;
        dma.desc_idx       = ring.head;
        dma.packet         = std::move(pkt);
        dma.complete_cycle = current_cycle_ + timing_.dma_cycles(desc.length);
        pending_dma_.push_back(std::move(dma));

        ring.head = (ring.head + 1) % ring.ring_size;
    }
}

// ── DMA completion ─────────────────────────────────────────────────────

void NIC::complete_dma()
{
    while (!pending_dma_.empty()
         && pending_dma_.front().complete_cycle <= current_cycle_)
    {
        PendingDma dma = std::move(pending_dma_.front());
        pending_dma_.pop_front();

        if (dma.type == PendingDma::Type::RX)
        {
            auto& ring = rx_rings_[dma.queue_id];
            RxDescriptor desc = read_rx_desc(dma.queue_id, dma.desc_idx);

            sys_mem_->load(desc.buffer_addr,
                           std::span<const u8>(dma.packet.data.data(),
                                               dma.packet.data.size()));

            // Update descriptor status
            desc.length = static_cast<u16>(dma.packet.data.size());
            desc.status = RxDescriptor::STATUS_DD | RxDescriptor::STATUS_EOP;
            desc.errors = 0;
            write_rx_desc(dma.queue_id, dma.desc_idx, desc);

            // Advance head
            ring.head = (dma.desc_idx + 1) % ring.ring_size;
            regs_[NicReg::RDH / 4] = ring.head;

            // Statistics
            stats_.rx_packets++;
            stats_.rx_bytes += dma.packet.data.size();
            regs_[NicReg::RXPKT / 4]++;
            regs_[NicReg::RXBYTES / 4] += static_cast<u32>(dma.packet.data.size());

            // Latency tracking
            dma.packet.dma_done_cycle = current_cycle_;
            cycle_t latency = current_cycle_ - dma.packet.arrival_cycle;
            stats_.total_rx_latency += latency;
            stats_.min_rx_latency = std::min(stats_.min_rx_latency, latency);
            stats_.max_rx_latency = std::max(stats_.max_rx_latency, latency);

            if (trace_)
                std::cout << std::format("[NIC] RX DMA complete | "
                                         "q={} | desc={} | len={}\n",
                                         dma.queue_id, dma.desc_idx,
                                         dma.packet.data.size());

            // Interrupt
            check_coalescing(dma.queue_id, NicInt::RXQ0);
        }
        else
        {
            // TX: Read packet data from buffer
            auto& ring = tx_rings_[dma.queue_id];
            TxDescriptor desc = read_tx_desc(dma.queue_id, dma.desc_idx);

            for (u32 i = 0; i < desc.length; ++i)
            {
                auto r = sys_mem_->read8(desc.buffer_addr + i);

                if (i < dma.packet.data.size())
                    dma.packet.data[i] = r.ok ? static_cast<u8>(r.value) : 0;
            }

            // Mark descriptor done
            desc.status = TxDescriptor::STATUS_DD;
            write_tx_desc(dma.queue_id, dma.desc_idx, desc);

            // Advance head
            ring.head = (dma.desc_idx + 1) % ring.ring_size;
            if (dma.queue_id == 0) regs_[NicReg::TDH / 4] = ring.head;

            // Statistics
            stats_.tx_packets++;
            stats_.tx_bytes += dma.packet.data.size();
            regs_[NicReg::TXPKT / 4]++;
            regs_[NicReg::TXBYTES / 4] += static_cast<u32>(dma.packet.data.size());

            if (trace_)
                std::cout << std::format("[NIC] TX DMA complete | "
                                         "q={} | desc{} | len={}\n",
                                         dma.queue_id, dma.desc_idx, desc.length);

            tx_complete_[dma.queue_id].push_back(std::move(dma.packet));
            if (desc.cmd & TxDescriptor::CMD_RS)
                check_coalescing(dma.queue_id, NicInt::TXQ0);
        }
    }
}

// ── Descriptor ring access (via system memory) ─────────────────────────

RxDescriptor NIC::read_rx_desc(u32 qid, u32 idx) const
{
    addr_t addr = rx_rings_[qid].base_addr + idx * 16;

    RxDescriptor d{};
    d.buffer_addr    = sys_mem_->read32(addr).value;
    d.buffer_addr_hi = sys_mem_->read32(addr + 4).value;

    u32 w2 = sys_mem_->read32(addr + 8).value;
    u32 w3 = sys_mem_->read32(addr + 12).value;

    d.length   = static_cast<u16>(w2         & 0xFFFF);
    d.checksum = static_cast<u16>((w2 >> 16) & 0xFFFF);
    d.status   = static_cast<u8>(w3          & 0xFF);
    d.errors   = static_cast<u8>((w3 >> 8)   & 0xFF);
    d.vlan     = static_cast<u16>((w3 >> 16) & 0xFFFF);
    return d;
}

void NIC::write_rx_desc(u32 qid, u32 idx, const RxDescriptor& d)
{
    addr_t addr = rx_rings_[qid].base_addr + idx * 16;

    sys_mem_->write32(addr,      d.buffer_addr);
    sys_mem_->write32(addr + 4,  d.buffer_addr_hi);
    sys_mem_->write32(addr + 8,  d.length | (static_cast<u32>(d.checksum) << 16));
    sys_mem_->write32(addr + 12, d.status | (static_cast<u32>(d.errors) << 8)
                                          | (static_cast<u32>(d.vlan) << 16));
}

TxDescriptor NIC::read_tx_desc(u32 qid, u32 idx) const
{
    addr_t addr = tx_rings_[qid].base_addr + idx * 16;

    TxDescriptor d{};
    d.buffer_addr    = sys_mem_->read32(addr).value;
    d.buffer_addr_hi = sys_mem_->read32(addr + 4).value;

    u32 w2 = sys_mem_->read32(addr + 8).value;
    u32 w3 = sys_mem_->read32(addr + 12).value;

    d.length = static_cast<u16>(w2         & 0xFFFF);
    d.cso    = static_cast<u8>((w2 >> 16)  & 0xFF);
    d.cmd    = static_cast<u8>((w2 >> 24)  & 0xFF);
    d.status = static_cast<u8>(w3          & 0xFF);
    d.css    = static_cast<u8>((w3 >> 8)   & 0xFF);
    d.vlan   = static_cast<u16>((w3 >> 16) & 0xFFFF);
    return d;
}

void NIC::write_tx_desc(u32 qid, u32 idx, const TxDescriptor& d)
{
    addr_t addr = tx_rings_[qid].base_addr + idx * 16;

    sys_mem_->write32(addr,      d.buffer_addr);
    sys_mem_->write32(addr + 4,  d.buffer_addr_hi);
    sys_mem_->write32(addr + 8,  d.length | (static_cast<u32>(d.cso) << 16)
                                          | (static_cast<u32>(d.cmd) << 24));
    sys_mem_->write32(addr + 12, d.status | (static_cast<u32>(d.css) << 8)
                                          | (static_cast<u32>(d.vlan) << 16));
}

// ── Loopback ───────────────────────────────────────────────────────────

void NicLoopback::tick(cycle_t current_cycle)
{
    while (auto pkt = nic_.poll_tx())
    {
        DelayedPacket dp;
        dp.pkt           = std::move(*pkt);
        dp.deliver_cycle = current_cycle + delay_;
        pending_.push_back(std::move(dp));
    }

    while (!pending_.empty() && pending_.front().deliver_cycle <= current_cycle)
    {
        nic_.inject_packet(pending_.front().pkt);
        pending_.pop_front();
    }
}

} // namespace riscv
