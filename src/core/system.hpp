/**
 * @file system.hpp
 * @brief A complete single-hart system.
 *
 * @c SingleCoreSystem<Core> assembles a @c Platform (RAM + CLINT + PLIC),
 * an optional L1/L2/L3 hierarchy in front of RAM, and one core of type
 * @c Core (@c CPU or @c PipelinedCPU). The core's loads, stores, and
 * fetches go through the platform's bus, so programs reach devices at
 * their memory-mapped addresses, and the platform's interrupt lines drive
 * the core's @c mip register.
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
#include "pipeline.hpp"
#include "platform.hpp"

#include <memory>
#include <span>
#include <type_traits>

namespace riscv {

/// Configuration for @c SingleCoreSystem.
struct SystemConfig
{
    PlatformConfig platform{};

    /// Set `size_bytes > 0` to enable a level; L2 requires L1, L3 requires L2.
    CacheConfig l1{.size_bytes = 0};
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
        config_.platform.num_harts = 1;
        platform_ = std::make_unique<Platform>(config_.platform);

        std::shared_ptr<Memory> backing = platform_->ram_ptr();
        if (config_.l1.size_bytes == 0
        && (config_.l2.size_bytes || config_.l3.size_bytes))
            throw std::invalid_argument("SingleCoreSystem: L2/L3 require L1");
        if (config_.l2.size_bytes == 0 && config_.l3.size_bytes)
            throw std::invalid_argument("SingleCoreSystem: L3 requires L2");

        if (config_.l3.size_bytes)
        { l3_ = std::make_shared<Cache>(config_.l3, backing); backing = l3_; }
        if (config_.l2.size_bytes)
        { l2_ = std::make_shared<Cache>(config_.l2, backing); backing = l2_; }
        if (config_.l1.size_bytes)
        { l1_ = std::make_shared<Cache>(config_.l1, backing); backing = l1_;}

        bus_ = platform_->make_hart_bus(backing);

        if constexpr (std::is_same_v<Core, PipelinedCPU>)
            core_ = std::make_unique<Core>(bus_, config_.pipeline);
        else
            core_ = std::make_unique<Core>(bus_);

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
    [[nodiscard]] Cache* l1() noexcept { return l1_.get(); }
    [[nodiscard]] Cache* l2() noexcept { return l2_.get(); }
    [[nodiscard]] Cache* l3() noexcept { return l3_.get(); }
    [[nodiscard]] const SystemConfig& config() const noexcept { return config_; }
    /// @}

private:
    SystemConfig              config_;
    std::unique_ptr<Platform> platform_;
    std::shared_ptr<Cache>    l3_, l2_, l1_;
    std::shared_ptr<MMIOBus>  bus_;
    std::shared_ptr<Core>     core_;
};

} // namespace riscv

