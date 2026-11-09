/**
 * @file pipeline.hpp
 * @brief Cycle-accurate 5-stage in-order RISC-V pipeline simulator.
 *
 * Implements a classic IF/ID/EX/MEM/WB pipeline with a synchronous
 * next-state evaluation model: pipeline registers are double-buffered 
 * and committed atomically at the end of each clock cycle.
 *
 * @par Hazard Resolution
 * Data hazards (RAW) are resolved via configurable forwarding paths
 * or interlock stalls. Load-use hazards always incur a 1-cycle stall.
 * Control hazards use configurable branch predictors with a
 * parameterized misprediction penalty.
 *
 * @par Parameterization
 * - Fowarding:          none, partial.
 * - Branch Prediction:  not taken, always taken, backward-taken,
 *                       1 bit bimodal, 2 bit bimodal.
 * - Mispredict Penalty: configurable cycle count.
 *
 * @see Patterson & Hennessy, "Computer Organization and Design (RISC-V Edition)",
 *      Sections 4.5-4.8. 
 */

#pragma once

#include "csr.hpp"
#include "decoder.hpp"
#include "memory.hpp"
#include "types.hpp"
#include "vector_state.hpp"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace riscv {

/// Pipeline stage identifiers.
enum class Stage : u8
{
    IF  = 0,  ///< Instruction fetch.
    ID  = 1,  ///< Decode and register read.
    EX  = 2,  ///< ALU execution and branch resolution.
    MEM = 3,  ///< Data memory access.
    WB  = 4,  ///< Register file writeback.
};

inline constexpr int kNumStages = 5;

/// State held in the latch between two pipeline stages.
struct PipelineReg
{
    bool        valid  = false;
    DecodedInst inst{};
    addr_t      pc     = 0;

    /// @name Datapath values
    /// @{
    u32 rs1_val    = 0;  ///< Source register 1 value (post-forwarding).
    u32 rs2_val    = 0;  ///< Source register 2 value (post-forwarding).
    u32 alu_result = 0;  ///< EX stage ALU output.
    u32 mem_result = 0;  ///< Data loaded from memory in MEM stage.
    u32 rd_val     = 0;  ///< Final value to write to @c rd in WB.
    /// @}

    /// @name Control signals (decoded in ID)
    /// @{
    bool   branch_taken  = false;
    addr_t branch_target = 0;
    bool   mem_read      = false;  ///< Load enable.
    bool   mem_write     = false;  ///< Store enable.
    bool   reg_write     = false;  ///< Register file write enable.
    /// @}

    void clear() noexcept { *this = PipelineReg{}; }
};

/// Forwarding path availability.
enum class ForwardingPolicy : u8
{
    NONE,     ///< All RAW hazards stall until WB.
    PARTIAL,  ///< Forward from MEM→EX only.
};

/// Branch prediction strategy.
enum class BranchPredictor : u8
{
    NOT_TAKEN,       ///< Static: always predict not taken.
    ALWAYS_TAKEN,    ///< Static: always predict taken.
    BACKWARD_TAKEN,  ///< Static: taken if target < PC.
    BIMODAL_1BIT,    ///< Dynamic: 1-bit saturating counter.
    BIMODAL_2BIT,    ///< Dynamic: 2-bit saturating counter.
};

/// Pipeline configuration knobs.
struct PipelineConfig
{
    ForwardingPolicy forwarding = ForwardingPolicy::PARTIAL;
    BranchPredictor  predictor  = BranchPredictor::NOT_TAKEN;
    u32 branch_mispred_penalty  = 2;    ///< Cycles lost on a misprediction.
    u32 bht_size                = 256;  ///< Branch history table entries.
};

/// Pipeline performance counters.
struct PipelineStats
{
    cycle_t cycles               = 0;
    u64     instructions_retired = 0;

    /// @name Hazard accounting
    /// @{ 
    u64 stalls_load_use = 0;  ///< Load-use dependency stalls.
    u64 stalls_raw      = 0;  ///< RAW hazard stalls (no forwarding).
    u64 stalls_control  = 0;  ///< Branch misprediction penalty cycles.
    u64 bubbles         = 0;  ///< Total non-functional pipeline cycles.
    /// @}

    /// @name Branch, jump, and forwarding metrics
    /// @{
    u64 jumps              = 0;
    u64 branches           = 0;
    u64 branches_taken     = 0;
    u64 branch_mispredicts = 0;
    u64 forwards_mem_ex    = 0;
    /// @}

    u64 interrupts_taken   = 0;  ///< Asynchronous traps entered.

    [[nodiscard]] double ipc() const noexcept
    {
        return cycles > 0
            ? static_cast<double>(instructions_retired) / static_cast<double>(cycles)
            : 0.0;
    }

    [[nodiscard]] double branch_accuracy() const noexcept
    {
        return branches > 0
            ? 1.0 - static_cast<double>(branch_mispredicts) / static_cast<double>(branches)
            : 1.0;
    }

    void reset() noexcept { *this = PipelineStats{}; }
};

/**
 * @brief 5-stage pipelined RISC-V CPU.
 *
 * Each call to tick() advances the pipeline by one clock cycle using
 * double-buffered state (snapshot → compute next → commit).
 */
class PipelinedCPU {
public:
    explicit PipelinedCPU(std::shared_ptr<Memory> memory, 
                          PipelineConfig config = {});

    /// @name Program loading
    /// @{
    void load_program(addr_t addr, std::span<const u8> program);
    void load_instruction(addr_t addr, u32 instruction);
    /// @}

    /// @name PC and register access
    /// @{
    void set_pc(addr_t pc) noexcept { pc_ = pc; };
    [[nodiscard]] addr_t pc() const noexcept { return pc_; }

    void set_reg(reg_idx_t r, u32 value) noexcept { if (r != 0) regs_[r] = value; }
    [[nodiscard]] u32 reg(reg_idx_t r) const noexcept { return regs_[r]; }
    /// @}

    /// @name Execution
    /// @{

    /// Advance the pipeline by one clock cycle. Returns @c false if halted and drained.
    bool tick();

    cycle_t run_cycles(cycle_t n);
    cycle_t run_instructions(u64 n);
    cycle_t run_until_pc(addr_t target);

    template<typename Pred>
    cycle_t run_until(Pred&& pred, cycle_t max_cycles = 10'000'000)
    {
        cycle_t start = stats_.cycles;
        while (stats_.cycles - start < max_cycles && !pred())
            if (!tick()) break;
        return stats_.cycles - start;
    }
    /// @}

    /// @name Inspection
    /// @{
    [[nodiscard]] const PipelineStats&  stats()        const noexcept { return stats_; }
    [[nodiscard]] const PipelineConfig& config()       const noexcept { return config_; }
    [[nodiscard]] const PipelineReg&    stage(Stage s) const noexcept { return stages_[static_cast<int>(s)]; }
    [[nodiscard]] bool  pipeline_empty()               const noexcept;
    [[nodiscard]] bool  halted()                       const noexcept { return halted_; }
    /// @}

    /// @name Debug
    /// @{
    void dump_stats() const;
    void dump_regs() const;
    void set_trace(bool enable) noexcept { trace_ = enable; }
    [[nodiscard]] Memory& memory() noexcept { return *memory_; }
    /// @}

    /// @name Subsystem access
    /// @{
    [[nodiscard]] const VectorState& vstate() const noexcept { return vstate_; }
    [[nodiscard]]       VectorState& vstate()       noexcept { return vstate_; } 
    [[nodiscard]] const CSRFile&     csrs()   const noexcept { return csrs_; }
    [[nodiscard]]       CSRFile&     csrs()         noexcept { return csrs_; }
    /// @}

private:
    std::shared_ptr<Memory> memory_;
    PipelineConfig          config_;

    std::array<u32, 32>     regs_{};
    addr_t                  pc_ = 0;

    std::array<PipelineReg, kNumStages> stages_{};

    bool   halted_         = false;
    bool   faulted_        = false;  ///< Stopped by a memory/alignment fault.
    bool   flush_pipeline_ = false;
    addr_t flush_target_   = 0;

    mutable PipelineStats stats_{};
    bool                  trace_ = false;

    std::optional<addr_t> reservation_;  ///< LR/SC reservation.
    VectorState vstate_;
    CSRFile csrs_;

    /// @name Branch prediction
    /// @{
    std::vector<u8> bht_;  ///< Branch history table (saturating counters).

    [[nodiscard]] bool predict_branch(addr_t pc, i32 offset) const;
    void               update_predictor(addr_t pc, bool taken);
    [[nodiscard]] u32  bht_index(addr_t pc) const noexcept;
    /// @}

    /// @name Forwarding
    /// @{
    struct ForwardResult
    {
        bool available = false;
        u32  value     = 0;
    };

    [[nodiscard]] ForwardResult try_forward(reg_idx_t reg,
                           [[maybe_unused]] const std::array<PipelineReg, kNumStages>& cur,
                                            const std::array<PipelineReg, kNumStages>& next) const;
    [[nodiscard]] u32 resolve_rs1(const PipelineReg& ex,
                                  const std::array<PipelineReg, kNumStages>& cur,
                                  const std::array<PipelineReg, kNumStages>& next) const;
    [[nodiscard]] u32 resolve_rs2(const PipelineReg& ex,
                                  const std::array<PipelineReg, kNumStages>& cur,
                                  const std::array<PipelineReg, kNumStages>& next) const;
    /// @}

    /// @name Hazard detection
    /// @{
    [[nodiscard]] bool detect_data_stall(const std::array<PipelineReg, kNumStages>& cur) const;
    /// @}

    /// @name ALU and branch helpers
    /// @{
    static u32  alu_execute(Op op, u32 rs1, u32 rs2, i32 imm);
    static void execute_vector(Op op, VectorState& vs,
                               u32 rd, u32 vs2, u32 vs1,
                               u32 rs1_val,
                               [[maybe_unused]] u32 rs2_val);
    static bool branch_check(Op op, u32 rs1, u32 rs2);
    /// @}
};

} // namespace riscv
