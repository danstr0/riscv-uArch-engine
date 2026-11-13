/**
 * @file system.hpp
 * @brief A complete single-hart system.
 *
 * @c SingleCoreSystem<Core> assembles a @c Platform (RAM + CLINT + PLIC),
 * an optional cache hierarchy in front of RAM, and one core of type @c Core
 * (@c CPU or @c PipelinedCPU). The core's loads and stores go through the
 * platform's bus, so programs reach devices at their memory-mapped addresses,
 * and the platform's interrupt lines drive the core's @c mip register.
 *
 * @par Split or unified L1
 * With @c l1i disabled (the default), @c l1d is a unified L1 serving both
 * fetches and data. With @c l1i enabled, fetches go to a separate read-only
 * I-cache. An I-cache refill sees the newest data, including dirty D-cache
 * lines, but a line already in the I-cache can go stale after a store until
 * the hart runs @c fence.i, as RISC-V permits.
 *
 * @par Time
 * Each @c step() advances the core (one instruction for @c CPU, one clock
 * for @c PipelinedCPU) and then advances the devices to the core's cycle
 * count, so @c mtime tracks simulated time for both models.
 *
 * @code{.cpp}
 *   SingleCoreSystem<PipelinedCPU> sys({ .platform = { .ram_size = 1 << 20 } });
 *   sys.load_program(0, code);
 *   sys.run_until([&] { return sys.core().halted(); });
 * @endcode
 */

#pragma once

#include "cache.hpp"
#include "cpu.hpp"
#include "dma_port.hpp"
#include "pipeline.hpp"
#include "platform.hpp"

#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace riscv {

/// Configuration for @c SingleCoreSystem.
struct SystemConfig
{
    PlatformConfig platform{};

    /// Set `size_bytes > 0` to enable a level; L2 requires L1, L3 requires L2.
    CacheConfig l1d{.size_bytes = 0};  ///< L1 data cache, or the unified L1.
    CacheConfig l1i{.size_bytes = 0};  ///< L1 instruction cache; requires @c l1d.
    CacheConfig l2{.size_bytes = 0};
    CacheConfig l3{.size_bytes = 0};

    /// Pipeline knobs (ignored by the single-cycle @c CPU).
    PipelineConfig pipeline{};
};

template <typename Core>
class SingleCoreSystem {
    static_assert(std::is_same_v<Core, CPU> || std::is_same_v<Core, PipelinedCPU>,
                  "SingleCoreSystem supports CPU and PipelinedCPU");

public:
    explicit SingleCoreSystem(SystemConfig config = {})
        : config_(config)
    {
        validate();

        config_.platform.num_harts = 1;
        platform_ = std::make_unique<Platform>(config_.platform);

        std::shared_ptr<Memory> shared = platform_->ram_ptr();
        if (config_.l3.size_bytes) { l3_ = std::make_shared<Cache>(config_.l3, shared); shared = l3_; }
        if (config_.l2.size_bytes) { l2_ = std::make_shared<Cache>(config_.l2, shared); shared = l2_; }

        std::shared_ptr<Memory> data_side = shared;
        if (config_.l1d.size_bytes)
        {
            config_.l1d.type = config_.l1i.size_bytes ? CacheType::DATA
                                                      : CacheType::UNIFIED;
            l1d_ = std::make_shared<Cache>(config_.l1d, shared);
            data_side = l1d_;
        }

        std::shared_ptr<Memory> fetch_side = data_side;
        if (config_.l1i.size_bytes)
        {
            config_.l1i.type = CacheType::INSTRUCTION;
            l1i_ = std::make_shared<Cache>(config_.l1i, shared);
            fetch_side = l1i_;

            // A refill must see code just written through the D-cache
            l1i_->set_on_read_miss([d = l1d_.get()](addr_t line) -> u32
            {
                d->writeback(line);
                return 0;
            });
        }

        dma_ = make_dma_port(shared);

        bus_       = platform_->make_hart_bus(data_side);
        fetch_bus_ = platform_->make_fetch_bus(fetch_side);

        if constexpr (std::is_same_v<Core, PipelinedCPU>)
            core_ = std::make_unique<Core>(bus_, config_.pipeline, fetch_bus_);
        else
            core_ = std::make_unique<Core>(bus_, fetch_bus_);

        if (l1i_)
            core_->set_on_fence_i([i = l1i_.get()] { i->invalidate_all(); });

        platform_->connect_hart(0, core_->csrs());
    }

    /// @name Loading (writes RAM directly, bypassing caches)
    /// @{
    void load_program(addr_t addr, std::span<const u8> program)
    {
        platform_->ram().load(addr, program);
    }
    /// @}

    /// @name Execution
    /// @{

    /// Advance the core by one step, then the devices. Returns @c false
    /// once halted.
    bool step()
    {
        bool running;
        if constexpr (std::is_same_v<Core, PipelinedCPU>)
            running = core_->tick();
        else
            running = core_->step();

        platform_->tick(cycles());
        return running;
    }

    /// Step until @p pred is true, the core halts, or @p max_steps elapse.
    /// Returns steps taken.
    template <typename Pred>
    u64 run_until(Pred&& pred, u64 max_steps = 10'000'000)
    {
        u64 n = 0;
        while (n < max_steps && !pred())
        {
            if (!step()) break;
            ++n;
        }
        return n;
    }

    /// Step until the core halts or @p max_steps elapse.
    u64 run(u64 max_steps = 10'000'000)
    {
        return run_until([] { return false; }, max_steps);
    }

    [[nodiscard]] cycle_t cycles() const noexcept
    {
        if constexpr (std::is_same_v<Core, PipelinedCPU>)
            return core_->stats().cycles;
        else
            return core_->cycles();
    }
    /// @}

    /// @name Component access
    /// @{
    [[nodiscard]] Core&         core()       noexcept { return *core_; }
    [[nodiscard]] const Core&   core() const noexcept { return *core_; }
    [[nodiscard]] Platform&     platform()   noexcept { return *platform_; }
    [[nodiscard]] SparseMemory& ram()        noexcept { return platform_->ram(); }
    [[nodiscard]] MMIOBus&      bus()        noexcept { return *bus_; }
    [[nodiscard]] Cache* l1d()               noexcept { return l1d_.get(); }
    [[nodiscard]] Cache* l1i()               noexcept { return l1i_.get(); }
    [[nodiscard]] Cache* l2()                noexcept { return l2_.get(); }
    [[nodiscard]] Cache* l3()                noexcept { return l3_.get(); }
    [[nodiscard]] const SystemConfig& config() const noexcept { return config_; }

    /// The memory a bus-mastering device (e.g., the NIC) should use;
    /// coherent with the D-cache.
    [[nodiscard]] std::shared_ptr<Memory> dma_memory() const noexcept { return dma_; }
    /// @}

private:
    SystemConfig              config_;
    std::unique_ptr<Platform> platform_;
    std::shared_ptr<Cache>    l3_, l2_, l1d_, l1i_;
    std::shared_ptr<Memory>   dma_;
    std::shared_ptr<MMIOBus>  bus_, fetch_bus_;
    std::shared_ptr<Core>     core_;

    void validate() const
    {
        if (config_.l1d.size_bytes == 0 && config_.l1i.size_bytes)
            throw std::invalid_argument("SingleCoreSystem: L1I requires an L1D cache");
        if (config_.l1d.size_bytes == 0
        && (config_.l2.size_bytes || config_.l3.size_bytes))
            throw std::invalid_argument("SingleCoreSystem: L2/L3 require an L1 cache");
        if (config_.l2.size_bytes == 0 && config_.l3.size_bytes)
            throw std::invalid_argument("SingleCoreSystem: L3 requires an L2 cache");
        if (config_.l1i.size_bytes && config_.l1i.line_size != config_.l1d.line_size)
            throw std::invalid_argument("SingleCoreSystem: L1I and L1D line sizes must match");
    }

    [[nodiscard]] std::shared_ptr<Memory>
    make_dma_port(std::shared_ptr<Memory> shared)
    {
        if (!l1d_) return shared;

        Cache* d = l1d_.get();
        return std::make_shared<DmaPort>(
            std::move(shared), config_.l1d.line_size,
            [d](addr_t line) { d->writeback(line); },
            [d](addr_t line) { d->snoop_invalidate(line); });
    }
};

} // namespace riscv

