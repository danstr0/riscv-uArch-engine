/**
 * @file executor.hpp
 * @brief Execution engine for decoded RISC-V instructions.
 *
 * Maintains the architectural state of the hart: 32 integer registers
 * (x0 hardwired to zero), the program counter, CSR file, and vector unit.
 *
 * @par PC management
 * The exeuctor reads the current PC for relative calculations but does
 * @b not commit the new PC. The caller updates PC using @c ExecuteResult::next_pc.
 *
 * @see RISC-V Unprivileged ISA Specification v20260120, §2.1, §12.1, §13.1, §30.1.
 */

#pragma once

#include "csr.hpp"
#include "decoder.hpp"
#include "memory.hpp"
#include "vector_state.hpp"

#include <array>
#include <limits>
#include <optional>

namespace riscv {

/// Performance and telemetry counters.
struct CpuStats
{
    cycle_t cycles         = 0;
    u64     instructions   = 0;
    u64     loads          = 0;
    u64     stores         = 0;
    u64     branches       = 0;
    u64     branches_taken = 0;
    u64     jumps          = 0;

    /// @return Instructions per cycle.
    [[nodiscard]] double ipc() const noexcept
    {
        return cycles > 0
            ? static_cast<double>(instructions) / static_cast<double>(cycles) 
            : 0.0;
    }

    /// @return Fraction of branches that changed the PC.
    [[nodiscard]] double branch_taken_rate() const noexcept
    {
        return branches > 0
            ? static_cast<double>(branches_taken) / static_cast<double>(branches)
            : 0.0;
    }

    void reset() noexcept { *this = CpuStats{}; }
};

/**
 * @brief Result returned by the executor after processing one instruction.
 *
 * Informs the CPU loop how to transition to the next state.
 */
struct [[nodiscard]] ExecuteResult
{
    bool   ok           = true;   ///< @c false if a fault occurred.
    u32    cycles       = 1;      ///< Total latency of this instruction.
    bool   branch_taken = false;  ///< Whether a branch/jump was taken.
    addr_t next_pc      = 0;      ///< Next instruction address.  

    bool   ecall        = false;  ///< Environment call trap.
    bool   ebreak       = false;  ///< Breakpoint trap.
    bool   fence_i      = false;  ///< Instruction-fetch synchronization requested.

    std::optional<u32> rd_value;  ///< Value written to @c rd, if any.
};

/**
 * @brief Instruction executor maintaining hart architectural state.
 *
 * Executes decoded instructions against the register file and memory,
 * returning an @c ExecuteResult describing the state transition.
 */
class Executor {
public:
    explicit Executor(Memory& memory);

    /**
     * @brief Execute a decoded instruction.
     * @param inst  The instruction to execute.
     * @return Execution metadata including the next PC.
     */
    [[nodiscard]] ExecuteResult execute(const DecodedInst& inst);

    /// @name Register access
    /// @{
    [[nodiscard]] u32 reg(reg_idx_t r) const noexcept { return regs_[r]; }

    void set_reg(reg_idx_t r, u32 value) noexcept
    {
        if (r != 0) regs_[r] = value;
    }

    [[nodiscard]] const std::array<u32, 32>& regs() const noexcept { return regs_; }
    /// @}

    /// @name Program counter
    /// @{
    [[nodiscard]] addr_t pc() const noexcept { return pc_; }
    void set_pc(addr_t pc) noexcept { pc_ = pc; }
    /// @}

    /// @name Statistics
    /// @{
    [[nodiscard]] const CpuStats& stats() const noexcept { return stats_; }
    [[nodiscard]]       CpuStats& stats()       noexcept { return stats_; }
    /// @}

    /// @name Subsystem access
    /// @{
    [[nodiscard]] const VectorState& vstate() const noexcept { return vstate_; }
    [[nodiscard]]       VectorState& vstate()       noexcept { return vstate_; }
    [[nodiscard]] const CSRFile&     csrs()   const noexcept { return csrs_; }
    [[nodiscard]]       CSRFile&     csrs()         noexcept { return csrs_; }
    /// @}

    /// @name Debug
    /// @{
    void dump_regs() const;
    void set_trace(bool enable) noexcept { trace_ = enable; }
    /// @}

private:
    Memory&             memory_;
    std::array<u32, 32> regs_{};
    addr_t              pc_ = 0;
    CpuStats            stats_;

    bool trace_ = false;

    /// LR/SC reservation address. Set by LR.W, cleared by SC.W or overlapping stores.
    std::optional<addr_t> reservation_;

    VectorState vstate_;
    CSRFile csrs_;

    /// Vector instruction dispatch.
    ExecuteResult execute_vector(const DecodedInst& inst, u32 rs1, [[maybe_unused]] u32 rs2);

    /// @name RV32I ALU
    /// @{
    static constexpr u32 alu_add(u32 a, u32 b) noexcept { return a + b; }
    static constexpr u32 alu_sub(u32 a, u32 b) noexcept { return a - b; }
    static constexpr u32 alu_and(u32 a, u32 b) noexcept { return a & b; }
    static constexpr u32 alu_or(u32 a, u32 b)  noexcept { return a | b; }
    static constexpr u32 alu_xor(u32 a, u32 b) noexcept { return a ^ b; }
    static constexpr u32 alu_sll(u32 a, u32 b) noexcept { return a << (b & 0x1Fu); }
    static constexpr u32 alu_srl(u32 a, u32 b) noexcept { return a >> (b & 0x1Fu); }

    /// Arithmetic right shift with portable sign-bit replication.
    static constexpr u32 alu_sra(u32 a, u32 b) noexcept
    { 
        u32 shamt = b & 0x1Fu;
        u32 shifted = a >> shamt;
        if ((a & 0x8000'0000u) && shamt > 0)
            shifted |= ~u32{0} << (32 - shamt);
        return shifted; 
    }

    static constexpr u32 alu_slt(u32 a, u32 b) noexcept  { return static_cast<i32>(a)
                                                                < static_cast<i32>(b)
                                                                ? 1u
                                                                : 0u; }
    static constexpr u32 alu_sltu(u32 a, u32 b) noexcept { return a < b ? 1u : 0u; }
    /// @}

    /// @name RV32M multiply/divide
    /// @{
    static constexpr u32 alu_mul(u32 a, u32 b) noexcept { return a * b; }

    static constexpr u32 alu_mulh(u32 a, u32 b) noexcept
    {
        i64 result = static_cast<i64>(static_cast<i32>(a))
                   * static_cast<i64>(static_cast<i32>(b));

        return static_cast<u32>(static_cast<u64>(result) >> 32);
    }

    static constexpr u32 alu_mulhsu(u32 a, u32 b) noexcept
    {
        i64 result = static_cast<i64>(static_cast<i32>(a))
                   * static_cast<i64>(static_cast<u64>(b));

        return static_cast<u32>(static_cast<u64>(result) >> 32);
    }

    static constexpr u32 alu_mulhu(u32 a, u32 b) noexcept
    {
        u64 result = static_cast<u64>(a) * static_cast<u64>(b);

        return static_cast<u32>(result >> 32);
    }

    static constexpr u32 alu_div(u32 a, u32 b) noexcept
    {
        if (b == 0) return ~u32{0};
        auto sa = static_cast<i32>(a);
        auto sb = static_cast<i32>(b);

        if (sa == std::numeric_limits<i32>::min() && sb == -1)
            return static_cast<u32>(sa);

        return static_cast<u32>(sa / sb);
    }

    static constexpr u32 alu_divu(u32 a, u32 b) noexcept
    {
        return b == 0 ? ~u32{0} : a / b;
    }

    static constexpr u32 alu_rem(u32 a, u32 b) noexcept
    {
        if (b == 0) return a;
        auto sa = static_cast<i32>(a);
        auto sb = static_cast<i32>(b);

        if (sa == std::numeric_limits<i32>::min() && sb == -1)
            return 0;

        return static_cast<u32>(sa % sb);
    }

    static constexpr u32 alu_remu(u32 a, u32 b) noexcept
    {
        return b == 0 ? a : a % b;
    }
    /// @}

    /// @name Branch conditions
    /// @{
    static constexpr bool cond_eq(u32 a, u32 b)  noexcept { return a == b; }
    static constexpr bool cond_ne(u32 a, u32 b)  noexcept { return a != b; }
    static constexpr bool cond_lt(u32 a, u32 b)  noexcept { return static_cast<i32>(a)
                                                                 < static_cast<i32>(b); }
    static constexpr bool cond_ge(u32 a, u32 b)  noexcept { return static_cast<i32>(a)
                                                                >= static_cast<i32>(b); }
    static constexpr bool cond_ltu(u32 a, u32 b) noexcept { return a < b; }
    static constexpr bool cond_geu(u32 a, u32 b) noexcept { return a >= b; }
    /// @}

    /// @name Memory sub-executors
    /// @{
    ExecuteResult execute_load(const DecodedInst& inst);
    ExecuteResult execute_store(const DecodedInst& inst);
    /// @}
};

} // namespace riscv
