/**
 * @file cpu.cpp
 * @brief Implementation of the top-level CPU fetch-decode-execute loop.
 */

#include "cpu.hpp"

#include <cassert>
#include <format>
#include <iomanip>
#include <iostream>

namespace riscv {

CPU::CPU(std::shared_ptr<Memory> memory, std::shared_ptr<Memory> imem)
    : memory_(std::move(memory))
    , imem_(imem ? std::move(imem) : memory_)
    , executor_(*memory_)
{
    assert(memory_ != nullptr && "CPU initialized with null memory backend");
}

void CPU::load_program(addr_t addr, std::span<const u8> program)
{
    memory_->load(addr, program);
}

void CPU::load_instruction(addr_t addr, u32 instruction)
{
    const std::array<u8, 4> bytes =
    {
        static_cast<u8>(instruction),
        static_cast<u8>(instruction >> 8),
        static_cast<u8>(instruction >> 16),
        static_cast<u8>(instruction >> 24),
    };
    memory_->load(addr, bytes);
}

bool CPU::step()
{
    if (halted_) [[unlikely]]
        return false;

    // ── 0. Interrupts ───────────────────────────────────
    auto& csrs = executor_.csrs();
    if (const u32 cause = csrs.pending_cause(); cause != 0)
    {
        csrs.enter_trap(executor_.pc(), cause);
        executor_.set_pc(csrs.trap_target(cause));
        ++interrupts_taken_;
    }

    // ── 1. Fetch ────────────────────────────────────────
    addr_t current_pc = executor_.pc();
    auto fetch_result = imem_->read32(current_pc);

    if (!fetch_result.ok)
    {
        std::cerr << std::format("[CPU] Fetch error at PC 0x{:08x}\n", current_pc);
        halted_ = true;
        return false;
    }

    // Accumulate fetch latency
    executor_.stats().cycles += fetch_result.cycles;

    // ── 2. Decode ───────────────────────────────────────
    last_inst_ = Decoder::decode(fetch_result.value, current_pc);

    if (last_inst_.op == Op::INVALID)
    {
        std::cerr << std::format("[ERROR] Illegal instruction 0x{:08x} at PC 0x{:08x}\n",
                                 fetch_result.value, current_pc); 
        halted_ = true;
        return false;
    }

    // ── 3. Execute ──────────────────────────────────────
    last_result_ = executor_.execute(last_inst_);

    if (!last_result_.ok)
    {
        std::cerr << std::format("[CPU] Execution error at PC {:08x}\n", current_pc);
        halted_ = true;
        return false;
    }

    // Update PC
    executor_.set_pc(last_result_.next_pc);

    if (last_result_.fence_i && on_fence_i_)
        on_fence_i_();

    // ── 4. Post-execution state update ──────────────────
    if (last_result_.ebreak)
    {
        halted_ = true;
        return false;
    }

    // Note: ECALL is handled by the caller checking last_result().ecall

    return true;
}

} // namespace riscv
