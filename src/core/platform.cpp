/**
 * @file platform.cpp
 * @brief Platform implementation.
 */

#include "platform.hpp"

#include <format>
#include <stdexcept>

namespace riscv {

namespace {

constexpr u64 kAddressSpace = u64{1} << 32;

} // namespace

Platform::Platform(PlatformConfig config)
    : config_(config)
{
    if (config_.num_harts == 0 || config_.num_harts > PLICConfig::MAX_CONTEXTS)
        throw std::invalid_argument(
            std::format("Platform: num_harts must be 1..{} (got {})",
                        PLICConfig::MAX_CONTEXTS, config_.num_harts));

    validate_map();

    ram_   = std::make_shared<SparseMemory>(config_.ram_base, config_.ram_size);
    clint_ = std::make_shared<CLINT>(config_.num_harts, config_.mtime_cycles_per_tick);
    plic_  = std::make_shared<PLIC>(config_.num_harts);
    harts_.assign(config_.num_harts, nullptr);

    clint_->on_timer_line([this](u32 hart, bool level)
    {
        set_line(hart, MInterrupt::MTIP, level);
    });
    clint_->on_software_line([this](u32 hart, bool level)
    {
        set_line(hart, MInterrupt::MSIP, level);
    });
    plic_->on_context_line([this](u32 ctx, bool level)
    {
        set_line(ctx, MInterrupt::MEIP, level);
    });
}

// ── Memory map ─────────────────────────────────────────────────────────

std::vector<Platform::Region> Platform::memory_map() const
{
    std::vector<Region> map =
    {
        {"ram",   config_.ram_base,   config_.ram_size},
        {"clint", config_.clint_base, CLINT::REG_SIZE},
        {"plic",  config_.plic_base,  PLIC::WINDOW_SIZE},
    };
    if (nic_)
        map.push_back({"nic", config_.nic_base, PlatformMap::NIC_SIZE});
    return map;
}

void Platform::validate_map() const
{
    // Check the NIC window even when no NIC is attached yet, so a later
    // attach_nic() can never create an overlap
    const std::vector<Region> map =
    {
        {"ram",   config_.ram_base,   config_.ram_size},
        {"clint", config_.clint_base, CLINT::REG_SIZE},
        {"plic",  config_.plic_base,  PLIC::WINDOW_SIZE},
        {"nic",   config_.nic_base,   PlatformMap::NIC_SIZE},
    };

    for (const auto& r : map)
    {
        if (r.size == 0)
            throw std::invalid_argument(
                std::format("Platform: {} region is empty", r.name));
        if (u64{r.base} + r.size > kAddressSpace)
            throw std::invalid_argument(
                std::format("Platform: {} region 0x{:08x}+0x{:x} exceeds "
                            "the 32-bit address space",
                            r.name, r.base, r.size));
    }

    for (size_t i = 0; i < map.size(); ++i)
        for (size_t j = i + 1; j < map.size(); ++j)
        {
            const auto& a = map[i];
            const auto& b = map[j];
            if (u64{a.base} < u64{b.base} + b.size
             && u64{b.base} < u64{a.base} + a.size)
                throw std::invalid_argument(
                    std::format("Platform: {} region 0x{:08x}+0x{:x} "
                                "overlaps {} region 0x{:08x}+0x{:x}",
                                a.name, a.base, a.size,
                                b.name, b.base, b.size));
        }
}

// ── Building the system ────────────────────────────────────────────────

std::shared_ptr<MMIOBus> Platform::make_hart_bus(std::shared_ptr<Memory> downstream)
{
    if (!downstream)
        throw std::invalid_argument("Platform::make_hart_bus: downstream memory is null");

    auto bus = std::make_shared<MMIOBus>();

    // RAM first
    bus->map_passthrough(config_.ram_base, config_.ram_size,
                         std::move(downstream), "ram");
    bus->map(config_.clint_base, CLINT::REG_SIZE,   clint_, "clint");
    bus->map(config_.plic_base,  PLIC::WINDOW_SIZE, plic_,  "plic");
    if (nic_)
        bus->map(config_.nic_base, PlatformMap::NIC_SIZE, nic_, "nic");

    buses_.push_back(bus);
    return bus;
}

std::shared_ptr<MMIOBus> Platform::make_fetch_bus(std::shared_ptr<Memory> downstream)
{
    if (!downstream)
        throw std::invalid_argument("Platform::make_fetch_bus: downstream memory is null");

    auto bus = std::make_shared<MMIOBus>();
    bus->map_passthrough(config_.ram_base, config_.ram_size,
                         std::move(downstream), "ram");
    return bus;
}

void Platform::connect_hart(u32 hart, CSRFile& csrs)
{
    if (hart >= config_.num_harts)
        throw std::out_of_range(
            std::format("Platform::connect_hart: hart {} >= num_harts {}",
                        hart, config_.num_harts));

    harts_[hart] = &csrs;
    csrs.set_hartid(hart);

    // Sync lines that may already be asserted
    set_line(hart, MInterrupt::MTIP, clint_->timer_pending(hart));
    set_line(hart, MInterrupt::MSIP, clint_->msip(hart));
    set_line(hart, MInterrupt::MEIP, plic_->interrupt_pending(hart));
}

void Platform::attach_nic(std::shared_ptr<NIC> nic, u32 plic_source)
{
    if (!nic)
        throw std::invalid_argument("Platform::attach_nic: nic is null");
    if (plic_source == 0 || plic_source >= PLICConfig::MAX_SOURCES)
        throw std::invalid_argument(
            std::format("Platform::attach_nic: PLIC source must be 1..{} (got {})",
                        PLICConfig::MAX_SOURCES - 1, plic_source));

    nic_ = std::move(nic);
    nic_->plic_source = plic_source;

    PLIC* plic = plic_.get();
    nic_->set_interrupt_callback([plic, plic_source]()
                                 {plic->set_pending(plic_source); });

    for (auto& bus : buses_)
        bus->map(config_.nic_base, PlatformMap::NIC_SIZE, nic_, "nic");
}

// ── Simulation ─────────────────────────────────────────────────────────

void Platform::tick(cycle_t cycle)
{
    clint_->tick(cycle);
    if (nic_) nic_->tick(cycle);
}

void Platform::set_line(u32 hart, u32 mip_bit, bool level)
{
    if (hart >= harts_.size() || harts_[hart] == nullptr) return;
    if (level) harts_[hart]->set_mip_bit(mip_bit);
    else       harts_[hart]->clear_mip_bit(mip_bit);
}

} // namespace riscv

