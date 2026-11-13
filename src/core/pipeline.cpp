/**
 * @file pipeline.cpp
 * @brief 5-stage pipeline implementation.
 *
 * Uses double-buffered pipeline registers: each @c tick() snapshots the
 * current state, computes the next state, then commits atomically.
 */

#include "pipeline.hpp"

#include <cassert>
#include <format>
#include <iostream>
#include <limits>

namespace riscv {

// ── Construction / lifecycle ───────────────────────────────────────────

PipelinedCPU::PipelinedCPU(std::shared_ptr<Memory> memory, PipelineConfig config,
                           std::shared_ptr<Memory> imem)
    : memory_(std::move(memory))
    , imem_(imem ? std::move(imem) : memory_)
    , config_(config)
    , bht_(config.bht_size, 0)
{
    assert(memory_ != nullptr);

    regs_.fill(0);
    pc_ = 0;
    for (auto& s : stages_) s.clear();
    halted_ = false;
    flush_pipeline_ = false;
    reservation_.reset();
    vstate_.reset();
    csrs_.reset();
    stats_.reset();
    std::fill(bht_.begin(), bht_.end(), u8{0});
}

void PipelinedCPU::load_program(addr_t addr, std::span<const u8> program)
{
    memory_->load(addr, program);
}

void PipelinedCPU::load_instruction(addr_t addr, u32 instruction)
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

void PipelinedCPU::observe_remote_store(addr_t addr, u32 size) noexcept
{
    if (!reservation_.has_value()) return;

    const u64 res = *reservation_;
    if (u64{addr} < res + 4 && u64{addr} + size > res)
        reservation_.reset();
}

bool PipelinedCPU::pipeline_empty() const noexcept
{
    for (const auto& s : stages_)
        if (s.valid) return false;

    return true;
}

// ── Main tick ──────────────────────────────────────────────────────────

bool PipelinedCPU::tick()
{
    if (halted_ && pipeline_empty())
        return false;

    stats_.cycles++;
    if (trace_ )
        std::cout << std::format("\n[Cycle {}] PC=0x{:08x}\n",
                                 stats_.cycles, pc_);

    // ── 1. Hardware snapshot ─────────────────────────────────
    auto cur = stages_;
    std::array<PipelineReg, kNumStages> next{};

    // A fault in MEM or EX halts the core; squash the faulting instruction
    // and everything younger, but keep whatever older instruction has
    // already been staged into WB this cycle so it still retires.
    // Committing here lets the pipeline drain; otherwise, the faulting
    // instruction re-executes forever
    auto fault_stop = [&]() -> bool
    {
        const PipelineReg retiring = next[static_cast<int>(Stage::WB)];
        for (auto& s : stages_) s.clear();
        stages_[static_cast<int>(Stage::WB)] = retiring;
        halted_ = true;
        faulted_ = true;
        return true;
    };

    // 2. WB: Retire and commit to architectural state ─────────
    {
        const auto& wb = cur[static_cast<int>(Stage::WB)];
        if (wb.valid)
        {
            if (wb.reg_write && wb.inst.rd != 0)
            {
                regs_[wb.inst.rd] = wb.rd_val;

                if (trace_)
                    std::cout << std::format("[WB] x{} <= 0x{:08x} ({})\n",
                                             wb.inst.rd, wb.rd_val,
                                             wb.inst.disassemble());
            }
            stats_.instructions_retired++;
        }
    }

    // ── 2b. Precise interrupt check ──────────────────────────
    const u32 irq_cause = faulted_ ? 0 : csrs_.pending_cause();
    const bool take_interrupt = (irq_cause != 0);
    addr_t irq_epc = pc_;
    if (take_interrupt)
        for (Stage s : {Stage::EX, Stage::ID, Stage::IF})
            if (cur[static_cast<int>(s)].valid)
            {
                irq_epc = cur[static_cast<int>(s)].pc;
                break;
            }

    // ── 3. MEM: Memory access and load result generation ─────
    {
        const auto& mem_in = cur[static_cast<int>(Stage::MEM)];
        auto& wb_out = next[static_cast<int>(Stage::WB)];

        if (mem_in.valid)
        {
            wb_out = mem_in;
            addr_t addr = mem_in.alu_result;

            auto log_mem_op = [&](std::string_view op_name,
                                std::string_view direction,
                                u32 val)
            {
                if (trace_)
                    std::cout << std::format("[{}] {} | addr={:08x} {} 0x{:08x}\n",
                                             op_name, mem_in.inst.disassemble(),
                                             addr, direction, val);
            };

            if (mem_in.mem_read)
            {
                MemoryResult r{};
                switch (mem_in.inst.op)
                {
                case Op::LB:
                    r = memory_->read8(addr);
                    wb_out.rd_val = static_cast<u32>(sign_extend<8>(r.value));
                    log_mem_op("LOAD", "->", wb_out.rd_val);
                    break;
                case Op::LH:
                    r = memory_->read16(addr);
                    wb_out.rd_val = static_cast<u32>(sign_extend<16>(r.value));
                    log_mem_op("LOAD", "->", wb_out.rd_val);
                    break;
                case Op::LW:
                    r = memory_->read32(addr);
                    wb_out.rd_val = r.value;
                    log_mem_op("LOAD", "->", wb_out.rd_val);
                    break;
                case Op::LBU:
                    r = memory_->read8(addr);
                    wb_out.rd_val = r.value & 0xFFu;
                    log_mem_op("LOAD", "->", wb_out.rd_val);
                    break;
                case Op::LHU:
                    r = memory_->read16(addr);
                    wb_out.rd_val = r.value & 0xFFFFu;
                    log_mem_op("LOAD", "->", wb_out.rd_val);
                    break;

                // ── RVV vector load ─────────────────
                case Op::VLE32:
                    if (addr & 3)
                    {
                        std::cerr << std::format("[ERROR] Misaligned vector load "
                                                 "at 0x{:08x} (PC 0x{:08x})\n",
                                                 addr, mem_in.pc);
                        wb_out.valid = false;
                        return fault_stop();
                    }
                    for (u32 i = 0; i < vstate_.vl; ++i)
                    {
                        r = memory_->read32(addr + i * 4);
                        vstate_.regs.set_elem32(mem_in.inst.rd, i, r.value);
                    }
                    if (trace_)
                        std::cout << std::format("[VLOAD] vd=v{} | base=0x{:08x} | vl={}\n",
                                                 mem_in.inst.rd, addr, vstate_.vl);
                    break;

                // ── RV32A memory operation ──────────
                case Op::LR_W:
                {
                    r = memory_->read32(addr);
                    wb_out.rd_val = r.value;
                    reservation_ = addr;

                    if (trace_)
                        std::cout << std::format("[LR] addr=0x{:08x} | value=0x{:08x} | "
                                                 "reservation=set\n",
                                                 addr, r.value);
                    break;
                }
                case Op::SC_W:
                {
                    if (reservation_.has_value() && reservation_.value() == addr)
                    {
                        memory_->write32(addr, mem_in.rs2_val);
                        if (on_store_) on_store_(addr, 4);
                        wb_out.rd_val = 0; // success
                    }
                    else
                        wb_out.rd_val = 1; // failure

                    if (trace_)
                        std::cout << std::format("[SC] addr=0x{:08x} | {} | "
                                                 "value=0x{:08x}\n",
                                                 addr,
                                                 wb_out.rd_val ? "FAIL" : "SUCCESS",
                                                 mem_in.rs2_val);
                    reservation_.reset();
                    break;
                }
                case Op::AMOSWAP_W: case Op::AMOADD_W: case Op::AMOXOR_W:
                case Op::AMOAND_W:  case Op::AMOOR_W:
                case Op::AMOMIN_W:  case Op::AMOMAX_W:
                case Op::AMOMINU_W: case Op::AMOMAXU_W:
                {
                    r = memory_->read32(addr);
                    u32 old_val = r.value;
                    u32 rs2v    = mem_in.rs2_val;
                    u32 new_val;
                    switch (mem_in.inst.op)
                    {
                    case Op::AMOSWAP_W: new_val = rs2v; break;
                    case Op::AMOADD_W:  new_val = old_val + rs2v; break;
                    case Op::AMOXOR_W:  new_val = old_val ^ rs2v; break;
                    case Op::AMOAND_W:  new_val = old_val & rs2v; break;
                    case Op::AMOOR_W:   new_val = old_val | rs2v; break;
                    case Op::AMOMIN_W:  new_val = (static_cast<i32>(old_val)
                                                 < static_cast<i32>(rs2v))
                                                 ? old_val 
                                                 : rs2v;
                                        break;
                    case Op::AMOMAX_W:  new_val = (static_cast<i32>(old_val)
                                                 > static_cast<i32>(rs2v))
                                                 ? old_val 
                                                 : rs2v;
                                        break;
                    case Op::AMOMINU_W: new_val = (old_val < rs2v) ? old_val : rs2v; break;
                    case Op::AMOMAXU_W: new_val = (old_val > rs2v) ? old_val : rs2v; break;
                    default: new_val = old_val; break;
                    }
                    memory_->write32(addr, new_val);
                    if (on_store_) on_store_(addr, 4);
                    wb_out.rd_val = old_val;

                    if (trace_)
                        std::cout << std::format("[AMO] {} | addr=0x{:08x} | "
                                                 "old=0x{:08x} | new=0x{:08x}\n",
                                                 mem_in.inst.disassemble(), addr,
                                                 old_val, new_val);
                    break;
                }

                default: break;
                }
                if (!r.ok)
                {
                    std::cerr << std::format("[ERROR] Memory {} fault at address 0x{:08x} "
                                             "(PC 0x{:08x})\n",
                                             mem_in.mem_read ? "read" : "write", addr, mem_in.pc);
                    wb_out.valid = false;
                    return fault_stop();
                }
            }
            else if (mem_in.mem_write)
            {
                MemoryResult r{};
                u32 val = mem_in.rs2_val;
                switch (mem_in.inst.op)
                {
                case Op::SB:
                    r = memory_->write8(addr, static_cast<u8>(val));
                    log_mem_op("STORE", "<-", val);
                    break;
                case Op::SH:
                    r = memory_->write16(addr, static_cast<u16>(val));
                    log_mem_op("STORE", "<-", val);
                    break;
                case Op::SW:
                    r = memory_->write32(addr, val);
                    log_mem_op("STORE", "<-", val);
                    break;

                case Op::VSE32:
                    if (addr & 3)
                    {
                        std::cerr << std::format("[ERROR] Misaligned vector store "
                                                 "at 0x{:08x} (PC 0x{:08x})\n",
                                                 addr, mem_in.pc);
                        wb_out.valid = false;
                        return fault_stop();
                    }
                    for (u32 i = 0; i < vstate_.vl; ++i)
                        memory_->write32(addr + i * 4,
                                         vstate_.regs.get_elem32(mem_in.inst.rd, i));

                    if (trace_)
                        std::cout << std::format("[VSTORE] vd=v{} | base=0x{:08x} | vl={}\n",
                                                 mem_in.inst.rd, addr, vstate_.vl);
                    break;

                default: break;
                }
                if (!r.ok)
                {
                    std::cerr << std::format("[ERROR] Memory {} fault at address 0x{:08x} "
                                             "(PC 0x{:08x})\n",
                                             mem_in.mem_read ? "read" : "write", addr, mem_in.pc);
                    wb_out.valid = false;
                    return fault_stop();
                }
                u32 store_size;
                switch(mem_in.inst.op)
                {
                    case Op::SB:    store_size = 1; break;
                    case Op::SH:    store_size = 2; break;
                    case Op::SW:    store_size = 4; break;
                    case Op::VSE32: store_size = vstate_.vl * 4; break;
                    default:        store_size = 0; break;
                }
                if (on_store_) on_store_(addr, store_size);

                if (reservation_.has_value())
                {
                    addr_t res = reservation_.value();
                    if (addr < res + 4 && addr + store_size > res)
                    {
                        reservation_.reset();

                        if (trace_)
                            std::cout << std::format("[LR/SC] reservation cleared by store "
                                                     "addr=0x{:08x}\n", addr);
                    }
                }
            }
            else
                wb_out.rd_val = mem_in.alu_result;
        }
    }

    // ── 4. Hazard detection and interlock logic ──────────────
    const auto& id_stage = cur[static_cast<int>(Stage::ID)];
    bool stall = detect_data_stall(cur);

    // ── 5. EX: ALU execution and forwarding resolution ───────
    {
        const auto& ex_in = cur[static_cast<int>(Stage::EX)];
        auto& mem_out = next[static_cast<int>(Stage::MEM)];

        if (ex_in.valid && !take_interrupt)
        {
            mem_out = ex_in;

            u32 rs1 = resolve_rs1(ex_in, cur, next);
            u32 rs2 = resolve_rs2(ex_in, cur, next);

            mem_out.rs2_val = rs2;

            const auto& inst = ex_in.inst;
            u32 uimm = static_cast<u32>(inst.imm);

            if (trace_)
                std::cout << std::format("[PIPELINE] Executing: {}\n",
                                         inst.disassemble());

            auto exec_csr = [&](auto compute_new_val, bool do_write)
            {
                const u32 csr_addr = uimm & 0xFFF;
                const u32 old_val  = csrs_.read(csr_addr);

                if (do_write)
                {
                    const u32 new_val = compute_new_val(old_val);

                    csrs_.write(csr_addr, new_val);

                    if (trace_)
                        std::cout << std::format("[CSR] csr=0x{:03x} | "
                                                 "old=0x{:08x} | new=0x{:08x}\n",
                                                 csr_addr, old_val, new_val);
                }

                mem_out.alu_result = old_val;
                mem_out.rd_val     = old_val;
                mem_out.reg_write  = (inst.rd != 0);
            };

            auto exec_jump = [&](addr_t target)
            {
                mem_out.alu_result    = inst.pc + 4;
                mem_out.rd_val        = inst.pc + 4;
                mem_out.branch_taken  = true;
                mem_out.branch_target = target;

                if (trace_)
                    std::cout << std::format("[JUMP] 0x{:08x} -> 0x{:08x}\n",
                                             inst.pc, target);

                flush_pipeline_ = true;
                flush_target_   = target;

                stats_.jumps++;
            };

            switch (inst.op)
            {
                // ── Branches ────────────────────────────
                case Op::BEQ: case Op::BNE: case Op::BLT:
                case Op::BGE: case Op::BLTU: case Op::BGEU:
                {
                    bool taken = branch_check(inst.op, rs1, rs2);
                    addr_t target = inst.pc + static_cast<addr_t>(inst.imm);

                    if (taken && (target & 3))
                    {
                        std::cerr << std::format("[ERROR] Misaligned branch target 0x{:08x} "
                                                 "(PC 0x{:08x})\n",
                                                 target, inst.pc);
                        return fault_stop();
                    }

                    bool predicted_taken = predict_branch(inst.pc, inst.imm);

                    if (trace_)
                        std::cout << std::format("[BRANCH] pc=0x{:08x} "
                                                 "predicted={} actual={} "
                                                 "target=0x{:08x}\n",
                                                 inst.pc, predicted_taken, taken,
                                                 target);

                    stats_.branches++;
                    if (taken) stats_.branches_taken++;
                    update_predictor(inst.pc, taken);

                    if (taken != predicted_taken)
                    {
                        if (trace_)
                            std::cout << std::format("[FLUSH] branch mispredict "
                                                     "-> pc=0x{:08x}\n",
                                                     taken ? target : inst.pc + 4);

                        stats_.branch_mispredicts++;
                        stats_.stalls_control += config_.branch_mispred_penalty;
                        flush_pipeline_ = true;
                        flush_target_ = taken ? target : (inst.pc + 4);
                    }

                    mem_out.branch_taken = taken;
                    mem_out.branch_target = target;
                    break;
                }

                // ── Jumps ────────────────────────────────────
                case Op::JAL:
                {
                    addr_t target = inst.pc + static_cast<addr_t>(inst.imm);
                    if (target & 3)
                    {
                        std::cerr << std::format("[Pipeline] Misaligned JAL target 0x{:08x} "
                                                 "(PC 0x{:08x})\n",
                                                 target, inst.pc);
                        return fault_stop();
                    }
                    exec_jump(target);
                    break;
                }   
                case Op::JALR:
                {
                    addr_t target = (rs1 + uimm) & ~1u;
                    if (target & 3)
                    {
                        std::cerr << std::format("[Pipeline] Misaligned JALR target 0x{:08x} "
                                                 "(PC 0x{:08x})\n",
                                                 target, inst.pc);
                        return fault_stop();
                    }
                    exec_jump(target);
                    break;
                }

                // ── Upper immediate ──────────────────────────
                case Op::LUI:   mem_out.alu_result = uimm; break;
                case Op::AUIPC: mem_out.alu_result = inst.pc + uimm; break;

                // ── Load / store address ─────────────────────
                case Op::LB: case Op::LH: case Op::LW:
                case Op::LBU: case Op::LHU:
                case Op::SB: case Op::SH: case Op::SW:
                    mem_out.alu_result = rs1 + uimm; break;

                // ── ALU ──────────────────────────────────────
                case Op::ADDI: case Op::SLTI: case Op::SLTIU:
                case Op::XORI: case Op::ORI:  case Op::ANDI:
                case Op::SLLI: case Op::SRLI: case Op::SRAI:
                case Op::ADD:  case Op::SUB:  case Op::SLL:
                case Op::SLT:  case Op::SLTU: case Op::XOR:
                case Op::SRL:  case Op::SRA:  case Op::OR:
                case Op::AND:
                case Op::MUL:  case Op::MULH: case Op::MULHSU:
                case Op::MULHU:case Op::DIV:  case Op::DIVU:
                case Op::REM:  case Op::REMU:
                    mem_out.alu_result = alu_execute(inst.op, rs1, rs2, inst.imm);
                    break;

                // ── RV32A instructions ───────────────────────
                case Op::LR_W:
                case Op::SC_W:
                case Op::AMOSWAP_W: case Op::AMOADD_W: case Op::AMOXOR_W:
                case Op::AMOAND_W:  case Op::AMOOR_W:
                case Op::AMOMIN_W:  case Op::AMOMAX_W:
                case Op::AMOMINU_W: case Op::AMOMAXU_W:
                    mem_out.alu_result = rs1;
                    mem_out.rs2_val    = rs2;
                    mem_out.mem_read   = true;
                    mem_out.reg_write  = true;
                    break;

                // ── RVV instructions ─────────────────────────
                case Op::VSETVLI:
                {
                    u32 avl = (inst.rs1 == 0 && inst.rd == 0)
                            ? vstate_.vl
                            : (inst.rs1 == 0) ? ~u32{0} : rs1;
                    u32 new_vl = vstate_.vsetvli(avl, uimm);
                    mem_out.alu_result = new_vl;
                    mem_out.rd_val     = new_vl;
                    mem_out.reg_write  = (inst.rd != 0);
                    break;
                }

                case Op::VLE32:
                    mem_out.alu_result = rs1;
                    mem_out.mem_read   = true;
                    break;
                case Op::VSE32:
                    mem_out.alu_result = rs1;
                    mem_out.mem_write  = true;
                    break;

                case Op::VADD_VV:  case Op::VSUB_VV:
                case Op::VAND_VV:  case Op::VOR_VV:  case Op::VXOR_VV:
                case Op::VADD_VX:  case Op::VSUB_VX:
                case Op::VAND_VX:  case Op::VOR_VX:  case Op::VXOR_VX:
                case Op::VSLL_VX:  case Op::VSRL_VX:
                case Op::VMSEQ_VV: case Op::VMSEQ_VX:
                case Op::VMSLT_VV: case Op::VMSLTU_VV:
                case Op::VMAND_MM: case Op::VMNAND_MM: case Op::VMANDN_MM:
                case Op::VMXOR_MM: case Op::VMOR_MM:   case Op::VMNOR_MM:
                case Op::VMORN_MM: case Op::VMXNOR_MM:
                case Op::VREDSUM_VS:
                case Op::VMV_V_X:
                    execute_vector(inst.op, vstate_,
                                   inst.rd, inst.rs2, inst.rs1,
                                   rs1, rs2);
                    break;

                case Op::VMV_X_S:
                    mem_out.alu_result = vstate_.regs.get_elem32(inst.rs2, 0);
                    mem_out.rd_val     = mem_out.alu_result;
                    mem_out.reg_write  = true;
                    break;

                // ── System ───────────────────────────────────
                case Op::FENCE: case Op::ECALL: case Op::EBREAK:
                    break;

                // ── FENCE.I ──────────────────────────────────
                case Op::FENCE_I:
                    if (on_fence_i_) on_fence_i_();
                    flush_pipeline_ = true;
                    flush_target_   = inst.pc + 4;
                    break;

                // ── CSR instructions ─────────────────────────
                case Op::CSRRW:
                    exec_csr([&](u32) { return rs1; }, true);
                    break;
                case Op::CSRRS:
                    exec_csr([&](u32 old) { return old | rs1; }, inst.rs1 != 0);
                    break;
                case Op::CSRRC:
                    exec_csr([&](u32 old) { return old & ~rs1; }, inst.rs1 != 0);
                    break;
                case Op::CSRRWI:
                    exec_csr([&](u32) { return inst.rs1; }, true);
                    break;
                case Op::CSRRSI:
                    exec_csr([&](u32 old) { return old | inst.rs1; }, inst.rs1 != 0);
                    break;
                case Op::CSRRCI:
                    exec_csr([&](u32 old) { return old & ~inst.rs1; }, inst.rs1 != 0);
                    break;

                // ── MRET: return from trap ───────────────────
                case Op::MRET:
                {
                    addr_t return_pc = csrs_.mret();
                    flush_pipeline_  = true;
                    flush_target_    = return_pc;
                    break;
                }

                default: [[unlikely]] break;
            }

            // Default: rd_val = ALU result (loads overwrite in MEM stage)
            if (!mem_out.mem_read)
                mem_out.rd_val = mem_out.alu_result;
        }
    }

    // ── 5b. Interrupt entry ──────────────────────────────────
    // EX did not execute this cycle; squash EX/ID/IF and redirect fetch.
    // Skipping ID/IF also keeps squashed instructions from setting halted_
    // (EBREAK / illegal decode) or advancing the predicting PC.
    if (take_interrupt)
    {
        const addr_t handler = csrs_.trap_target(irq_cause);

        if (trace_)
            std::cout << std::format("[TRAP] cause=0x{:08x} epc=0x{:08x} "
                                     "-> handler=0x{:08x}\n",
                                     irq_cause, irq_epc, handler);

        csrs_.enter_trap(irq_epc, irq_cause);
        stats_.interrupts_taken++;

        next[static_cast<int>(Stage::IF)].clear();
        next[static_cast<int>(Stage::ID)].clear();
        next[static_cast<int>(Stage::EX)].clear();
        pc_             = handler;
        halted_         = false;
        flush_pipeline_ = false;

        stages_ = next;
        return true;
    }

    // ── 6. ID: Decode and pipeline register management ───────
    if (!stall)
    {
        if (id_stage.valid)
            next[static_cast<int>(Stage::EX)] = id_stage;
    }
    else
    {
        // Bubble into EX; keep ID and IF frozen
        next[static_cast<int>(Stage::EX)].clear();
        stats_.bubbles++;
    }

    // ── 7. Speculative fetch and PC update ───────────────────
    if (!stall)
    {
        const auto& if_in = cur[static_cast<int>(Stage::IF)];
        auto& id_out = next[static_cast<int>(Stage::ID)];

        if (if_in.valid)
        {
            id_out.valid = true;
            id_out.pc    = if_in.pc;
            id_out.inst  = Decoder::decode(if_in.inst.raw, if_in.pc);

            if (id_out.inst.op == Op::INVALID) halted_ = true;
            if (id_out.inst.op == Op::EBREAK)  halted_ = true;

            id_out.rs1_val   = regs_[id_out.inst.rs1];
            id_out.rs2_val   = regs_[id_out.inst.rs2];
            id_out.mem_read  = id_out.inst.is_load();
            id_out.mem_write = id_out.inst.is_store();
            id_out.reg_write = id_out.inst.writes_rd();
        }

        // Fetch into IF
        auto& if_out = next[static_cast<int>(Stage::IF)];
        if (!halted_)
        {
            auto r = imem_->read32(pc_);
            if (r.ok)
            {
                if_out.valid    = true;
                if_out.pc       = pc_;
                if_out.inst.raw = r.value;

                // Real hardware does this in a pre-decode stage
                auto opcode = static_cast<u8>(r.value & 0x7F);
                if (opcode == 0b1100011)  // BRANCH opcode
                {
                    auto quick = Decoder::decode(r.value, pc_);
                    if (predict_branch(pc_, quick.imm))
                        pc_ = pc_ + static_cast<addr_t>(quick.imm);
                    else
                        pc_ += 4;
                }
                else
                    pc_ += 4;
            }
            else
                halted_ = true;
        }
    }
    else
    {
        // Stall: keep IF and ID unchanged
        next[static_cast<int>(Stage::IF)] = cur[static_cast<int>(Stage::IF)];
        next[static_cast<int>(Stage::ID)] = cur[static_cast<int>(Stage::ID)];
    }

    // ── 8. Synchronous flush (misprediction recovery) ────────
    if (flush_pipeline_)
    {
        if (trace_)
            std::cout << std::format("[PIPELINE] flush -> pc=0x{:08x}\n",
                                     flush_target_);

        next[static_cast<int>(Stage::IF)].clear();
        next[static_cast<int>(Stage::ID)].clear();
        next[static_cast<int>(Stage::EX)].clear();
        pc_ = flush_target_;
        flush_pipeline_ = false;
        halted_ = false;
    }

    // ── 9. Commit next state ─────────────────────────────────
    stages_ = next;

    return true;
}

// ── Run helpers ────────────────────────────────────────────────────────

cycle_t PipelinedCPU::run_cycles(cycle_t n)
{
    cycle_t count = 0;
    while (count < n)
    {
        if (!tick()) break;
        ++count;
    }
    return count;
}

cycle_t PipelinedCPU::run_instructions(u64 n)
{
    u64 start = stats_.instructions_retired;
    while (stats_.instructions_retired - start < n)
        if (!tick()) break;

    return stats_.cycles;
}

cycle_t PipelinedCPU::run_until_pc(addr_t target)
{
    return run_until([this, target]()
    {
        const auto& wb = stages_[static_cast<int>(Stage::WB)];
        return wb.valid && wb.pc == target;
    });
}

// ── Hazard detection ───────────────────────────────────────────────────

bool PipelinedCPU::detect_data_stall(const std::array<PipelineReg, kNumStages>& cur) const
{
    const auto& id = cur[static_cast<int>(Stage::ID)];
    const auto& ex = cur[static_cast<int>(Stage::EX)];

    if (!id.valid) return false;

    /*
     * Load-use hazard: EX has a load whose rd is needed by ID.
     *
     * This always requires a 1-cycle stall - the load result
     * isn't available until the end of MEM.
     */
    if (ex.valid && ex.mem_read && ex.inst.rd != 0)
    {
        bool needs_rs1 = id.inst.reads_rs1() && id.inst.rs1 == ex.inst.rd;
        bool needs_rs2 = id.inst.reads_rs2() && id.inst.rs2 == ex.inst.rd;

        if (needs_rs1 || needs_rs2)
        {
            if (trace_)
                std::cout << std::format("[STALL] load-use hazard: ID needs x{}, "
                                         "EX loading it\n",
                                         ex.inst.rd);

            stats_.stalls_load_use++;
            return true;
        }
    }

    /*
     * RAW hazard without forwarding: if forwarding is NONE, stall
     * whenever ID reads a register that EX or MEM will write.
     */
    if (config_.forwarding == ForwardingPolicy::NONE)
    {
        auto check = [&](reg_idx_t r) -> bool
        {
            if (r == 0) return false;
            if (ex.valid && ex.reg_write && ex.inst.rd == r) return true;
            const auto& mem = cur[static_cast<int>(Stage::MEM)];
            if (mem.valid && mem.reg_write && mem.inst.rd == r) return true;
            return false;
        };
        bool hazard = false;
        if (id.inst.reads_rs1() && check(id.inst.rs1)) hazard = true;
        if (id.inst.reads_rs2() && check(id.inst.rs2)) hazard = true;
        if (hazard)
        {
            if (trace_)
                std::cout << "[STALL] RAW hazard\n";

            stats_.stalls_raw++;
            return true;
        }
    }

    // Partial forwarding: stall on EX->EX hazards (only MEM->EX is available)
    if (config_.forwarding == ForwardingPolicy::PARTIAL)
        if (ex.valid && ex.reg_write && !ex.mem_read && ex.inst.rd != 0)
        {
            bool needs_rs1 = id.inst.reads_rs1() && id.inst.rs1 == ex.inst.rd;
            bool needs_rs2 = id.inst.reads_rs2() && id.inst.rs2 == ex.inst.rd;

            if (needs_rs1 || needs_rs2)
            {
                stats_.stalls_raw++;
                return true;
            }
        }

    return false;
}

// ── Forwarding ─────────────────────────────────────────────────────────

PipelinedCPU::ForwardResult PipelinedCPU::try_forward(
        reg_idx_t reg,
        [[maybe_unused]] const std::array<PipelineReg, kNumStages>& cur,
        const std::array<PipelineReg, kNumStages>& next) const
{
    if (reg == 0) return {};

    // MEM->EX: forward from the value MEM just produced this cycle
    if (config_.forwarding != ForwardingPolicy::NONE)
    {
        const auto& wb_out = next[static_cast<int>(Stage::WB)];
        if (wb_out.valid
         && wb_out.reg_write
         && wb_out.inst.rd == reg)
        {
            if (trace_)
                std::cout << std::format("[FWD] MEM->EX x{} = 0x{:08x}\n",
                                         reg, wb_out.rd_val);

            stats_.forwards_mem_ex++;
            return {true, wb_out.rd_val};
        }
    }

    return {};
}

u32 PipelinedCPU::resolve_rs1(const PipelineReg& ex,
                              const std::array<PipelineReg, kNumStages>& cur,
                              const std::array<PipelineReg, kNumStages>& next) const
{
    u32 val = regs_[ex.inst.rs1];
    if (!ex.inst.reads_rs1()) return val;
    auto fwd = try_forward(ex.inst.rs1, cur, next);
    return fwd.available ? fwd.value : val;
}

u32 PipelinedCPU::resolve_rs2(const PipelineReg& ex,
                              const std::array<PipelineReg, kNumStages>& cur,
                              const std::array<PipelineReg, kNumStages>& next) const
{
    u32 val = regs_[ex.inst.rs2];
    if (!ex.inst.reads_rs2()) return val;
    auto fwd = try_forward(ex.inst.rs2, cur, next);
    return fwd.available ? fwd.value : val;
}

// ── Branch prediction ──────────────────────────────────────────────────

u32 PipelinedCPU::bht_index(addr_t pc) const noexcept
{
    return (pc >> 2) & (static_cast<u32>(bht_.size()) - 1);
}

bool PipelinedCPU::predict_branch(addr_t pc, i32 offset) const
{
    switch(config_.predictor)
    {
        case BranchPredictor::NOT_TAKEN:
            return false;

        case BranchPredictor::ALWAYS_TAKEN:
            return true;

        case BranchPredictor::BACKWARD_TAKEN:
            return offset < 0;

        case BranchPredictor::BIMODAL_1BIT:
            return bht_[bht_index(pc)] != 0;

        case BranchPredictor::BIMODAL_2BIT:
            // Predict taken if counter >= 2 (weakly or strongly taken)
            return bht_[bht_index(pc)] >= 2;
    }
    return false;
}

void PipelinedCPU::update_predictor(addr_t pc, bool taken)
{
    u32 idx = bht_index(pc);

    switch (config_.predictor)
    {
        case BranchPredictor::NOT_TAKEN:
        case BranchPredictor::ALWAYS_TAKEN:
        case BranchPredictor::BACKWARD_TAKEN:
            // Static predictors have no state to update
            break;

        case BranchPredictor::BIMODAL_1BIT:
            bht_[idx] = taken ? 1 : 0;
            break;

        case BranchPredictor::BIMODAL_2BIT:
            if (taken)
            {
                if (bht_[idx] < 3) bht_[idx]++;
            }
            else
                if (bht_[idx] > 0) bht_[idx]--;
            break;
    }
}

// ── ALU / branch helpers ───────────────────────────────────────────────

u32 PipelinedCPU::alu_execute(Op op, u32 rs1, u32 rs2, i32 imm)
{
    u32 uimm = static_cast<u32>(imm);
    switch(op)
    {
        case Op::ADD:   return rs1 + rs2;
        case Op::SUB:   return rs1 - rs2;
        case Op::SLL:   return rs1 << (rs2 & 0x1Fu);
        case Op::SLT:   return static_cast<i32>(rs1) < static_cast<i32>(rs2) ? 1u : 0u;
        case Op::SLTU:  return rs1 < rs2 ? 1u : 0u;
        case Op::XOR:   return rs1 ^ rs2;
        case Op::SRL:   return rs1 >> (rs2 & 0x1Fu);
        case Op::SRA:
        {
            u32 shamt = rs2 & 0x1Fu;
            u32 shifted = rs1 >> shamt;
            if ((rs1 & 0x8000'0000u) && shamt > 0)
                shifted |= ~u32{0} << (32 - shamt);
            return shifted;
        }
        case Op::OR:    return rs1 | rs2;
        case Op::AND:   return rs1 & rs2;

        case Op::ADDI:  return rs1 + uimm;
        case Op::SLTI:  return static_cast<i32>(rs1) < imm ? 1u : 0u;
        case Op::SLTIU: return rs1 < uimm ? 1u : 0u;
        case Op::XORI:  return rs1 ^ uimm;
        case Op::ORI:   return rs1 | uimm;
        case Op::ANDI:  return rs1 & uimm;
        case Op::SLLI:  return rs1 << (imm & 0x1F);
        case Op::SRLI:  return rs1 >> (imm & 0x1F);
        case Op::SRAI:
        {
            u32 shamt = uimm & 0x1Fu;
            u32 shifted = rs1 >> shamt;
            if ((rs1 & 0x8000'0000u) && shamt > 0)
                shifted |= ~u32{0} << (32 - shamt);
            return shifted;
        }

        case Op::MUL:   return rs1 * rs2;
        case Op::MULH:
        {
            i64 r = static_cast<i64>(static_cast<i32>(rs1))
                  * static_cast<i64>(static_cast<i32>(rs2));
            return static_cast<u32>(static_cast<u64>(r) >> 32);
        }
        case Op::MULHSU:
        {
            i64 r = static_cast<i64>(static_cast<i32>(rs1))
                  * static_cast<i64>(static_cast<u64>(rs2));
            return static_cast<u32>(static_cast<u64>(r) >> 32);
        }
        case Op::MULHU:
        {
            u64 r = static_cast<u64>(rs1) * static_cast<u64>(rs2);
            return static_cast<u32>(r >> 32);
        }
        case Op::DIV:
        {
            if (rs2 == 0) return ~u32{0};
            auto sa = static_cast<i32>(rs1);
            auto sb = static_cast<i32>(rs2);
            if (sa == std::numeric_limits<i32>::min() && sb == -1)
                return static_cast<u32>(sa);
            return static_cast<u32>(sa / sb);
        }
        case Op::DIVU:
            return rs2 == 0 ? ~u32{0} : rs1 / rs2;
        case Op::REM:
        {
            if (rs2 == 0) return rs1;
            auto sa = static_cast<i32>(rs1);
            auto sb = static_cast<i32>(rs2);
            if (sa == std::numeric_limits<i32>::min() && sb == -1)
                return 0;
            return static_cast<u32>(sa % sb);
        }
        case Op::REMU:
            return rs2 == 0 ? rs1 : rs1 % rs2;

        case Op::LUI:   return uimm;
        case Op::AUIPC: return 0; // caller adds PC

        default: [[unlikely]] return 0;
    }
}

void PipelinedCPU::execute_vector(Op op, VectorState& vs,
                                  u32 vd, u32 vs2, u32 vs1,
                                  u32 rs1_val, [[maybe_unused]] u32 rs2_val)
{
    u32 vl      = vs.vl;
    auto& vregs = vs.regs;

    auto exec_vv = [&](auto vv_op)
    {
        for (u32 i = 0; i < vl; ++i)
        {
            vregs.set_elem32(vd, i,
                             vv_op(vregs.get_elem32(vs2, i),
                                   vregs.get_elem32(vs1, i)));
        }
    };

    auto exec_vx = [&](auto vx_op)
    {
        for (u32 i = 0; i < vl; ++i)
        {
            const u32 lhs = vregs.get_elem32(vs2, i);
            vregs.set_elem32(vd, i, vx_op(lhs, rs1_val));
        }
    };

    auto exec_mask = [&](auto pred)
    {
        for (u32 i = 0; i < vl; ++i)
            vregs.set_mask_bit(vd, i, pred(i));
    };

    switch (op)
    {
        // ── VV arithmetic ────────────────────────────────
        case Op::VADD_VV: exec_vv([](u32 a, u32 b) { return a + b; }); break;
        case Op::VSUB_VV: exec_vv([](u32 a, u32 b) { return a - b; }); break;
        case Op::VAND_VV: exec_vv([](u32 a, u32 b) { return a & b; }); break;
        case Op::VOR_VV:  exec_vv([](u32 a, u32 b) { return a | b; }); break;
        case Op::VXOR_VV: exec_vv([](u32 a, u32 b) { return a ^ b; }); break;

        // ── VX arithmetic ────────────────────────────────
        case Op::VADD_VX: exec_vx([](u32 a, u32 b) { return a + b; }); break;
        case Op::VSUB_VX: exec_vx([](u32 a, u32 b) { return a - b; }); break;
        case Op::VAND_VX: exec_vx([](u32 a, u32 b) { return a & b; }); break;
        case Op::VOR_VX:  exec_vx([](u32 a, u32 b) { return a | b; }); break;
        case Op::VXOR_VX: exec_vx([](u32 a, u32 b) { return a ^ b; }); break;
        case Op::VSLL_VX: exec_vx([](u32 a, u32 b) { return a << (b & 0x1F); }); break;
        case Op::VSRL_VX: exec_vx([](u32 a, u32 b) { return a >> (b & 0x1F); }); break;

        // ── Comparisons ─────────────────────────────────
        case Op::VMSEQ_VV:
            exec_mask([&](u32 i)
            {
                return vregs.get_elem32(vs2, i) ==
                       vregs.get_elem32(vs1, i);
            });
            break;
        case Op::VMSEQ_VX:
            exec_mask([&](u32 i)
            {
                return vregs.get_elem32(vs2, i) == rs1_val;
            });
            break;
        case Op::VMSLT_VV:
            exec_mask([&](u32 i)
            {
                return static_cast<i32>(vregs.get_elem32(vs2, i)) <
                       static_cast<i32>(vregs.get_elem32(vs1, i));
            });
            break;
        case Op::VMSLTU_VV:
            exec_mask([&](u32 i)
            {
                return vregs.get_elem32(vs2, i) <
                       vregs.get_elem32(vs1, i);
            });
            break;

        // ── Mask operations ─────────────────────────────
        case Op::VMAND_MM: case Op::VMNAND_MM: case Op::VMANDN_MM:
        case Op::VMXOR_MM: case Op::VMOR_MM:   case Op::VMNOR_MM:
        case Op::VMORN_MM: case Op::VMXNOR_MM:
        {
            for (u32 i = 0; i < vl; ++i)
            {
                u32 b2 = vregs.get_elem32(vs2, i / 32);
                u32 b1 = vregs.get_elem32(vs1, i / 32);
                u32 bit_pos = i % 32;
                bool bit2 = (b2 >> bit_pos) & 1;
                bool bit1 = (b1 >> bit_pos) & 1;
                bool bit_result;

                switch (op)
                {
                    case Op::VMAND_MM:  bit_result = bit2 & bit1;    break;
                    case Op::VMNAND_MM: bit_result = !(bit2 & bit1); break;
                    case Op::VMANDN_MM: bit_result = bit2 & (!bit1); break;
                    case Op::VMXOR_MM:  bit_result = bit2 ^ bit1;    break;
                    case Op::VMOR_MM:   bit_result = bit2 | bit1;    break;
                    case Op::VMNOR_MM:  bit_result = !(bit2 | bit1); break;
                    case Op::VMORN_MM:  bit_result = bit2 | (!bit1); break;
                    case Op::VMXNOR_MM: bit_result = !(bit2 ^ bit1); break;
                    default: [[unlikely]] bit_result = false;        break;
                }

                u32 dst_elem = vregs.get_elem32(vd, i / 32);
                if (bit_result) dst_elem |=  (1u << bit_pos);
                else            dst_elem &= ~(1u << bit_pos);
                vregs.set_elem32(vd, i / 32, dst_elem);
            }
            break;
        }

        // ── Reduction ───────────────────────────────────
        case Op::VREDSUM_VS:
        {
            u32 acc = vregs.get_elem32(vs1, 0);
            for (u32 i = 0; i < vl; ++i)
                acc += vregs.get_elem32(vs2, i);
            vregs.set_elem32(vd, 0, acc);
            break;
        }

        // ── Splat ───────────────────────────────────────
        case Op::VMV_V_X:
            for (u32 i = 0; i < vl; ++i)
                vregs.set_elem32(vd, i, rs1_val);
            break;

        default: break;
    }
}

bool PipelinedCPU::branch_check(Op op, u32 rs1, u32 rs2)
{
    switch (op)
    {
        case Op::BEQ:  return rs1 == rs2;
        case Op::BNE:  return rs1 != rs2;
        case Op::BLT:  return static_cast<i32>(rs1) < static_cast<i32>(rs2);
        case Op::BGE:  return static_cast<i32>(rs1) >= static_cast<i32>(rs2);
        case Op::BLTU: return rs1 < rs2;
        case Op::BGEU: return rs1 >= rs2;
        default: [[unlikely]] return false;
    }
}

// ── Debug output ───────────────────────────────────────────────────────

void PipelinedCPU::dump_stats() const
{
    const auto& s = stats_;

    std::cout
        << std::format("Retired={} IPC={:.3f}\n"
                       "Stalls(LU={}, RAW={}, CTRL={}) "
                       "Bubbles={}\n"
                       "Branches={} Taken={} Mispred={} "
                       "Acc={:.2f}%\n"
                       "Forwards(MEM->EX={})\n",
                       s.instructions_retired, s.ipc(),
                       s.stalls_load_use, s.stalls_raw, s.stalls_control, s.bubbles,
                       s.branches, s.branches_taken, s.branch_mispredicts,
                       s.branch_accuracy() * 100.0, s.forwards_mem_ex);
}

void PipelinedCPU::dump_regs() const
{
    std::cout << std::format("PC: 0x{:08x}\n", pc_);
    for (size_t i = 0; i < 32; i += 4)
    {
        for (size_t j = 0; j < 4; ++j)
            std::cout << std::format("{:>4s}: 0x{:08x}  ",
                                     reg_name(static_cast<reg_idx_t>(i + j)),
                                     regs_[i + j]);
        std::cout << "\n";
    }
}

} // namespace riscv
