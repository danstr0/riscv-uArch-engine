/**
 * @file executor.cpp
 * @brief Implementation of the RISC-V instruction executor.
 */

#include "executor.hpp"

#include <format>
#include <iostream>

namespace riscv {

Executor::Executor(Memory& memory)
    : memory_(memory)
{
    regs_.fill(0);
    pc_ = 0;
    reservation_.reset();
    vstate_.reset();
    csrs_.reset();
    stats_.reset();
}

void Executor::dump_regs() const
{
    std::cout << std::format("--- Architectural State (PC: 0x{:08x}) ---\n", pc_);
    for (size_t i = 0; i < 32; i += 4)
    {
        for (size_t j = 0; j < 4; ++j)
            std::cout << std::format("{:>4s}: 0x{:08x}  ",
                                     reg_name(static_cast<reg_idx_t>(i + j)),
                                     regs_[i + j]);
        std::cout << "\n";
    }
}

// ── Memory operations ──────────────────────────────────────────────────

ExecuteResult Executor::execute_load(const DecodedInst& inst)
{
    ExecuteResult result;

    /*
     * Address calculation: rs1 + sign-extended immediate.
     *
     * Spec §2.1.6: "The effective address is obtained by adding register r1
     * to the sign-extended 12-bit offset."
     */
    addr_t addr = regs_[inst.rs1] + static_cast<u32>(inst.imm);
    MemoryResult mem;

    switch(inst.op)
    {
        case Op::LB:
            mem = memory_.read8(addr);
            if (mem.ok) set_reg(inst.rd, static_cast<u32>(sign_extend<8>(mem.value)));
            break;
        case Op::LH:
            mem = memory_.read16(addr);
            if (mem.ok) set_reg(inst.rd, static_cast<u32>(sign_extend<16>(mem.value)));
            break;
        case Op::LW:
            mem = memory_.read32(addr);
            if (mem.ok) set_reg(inst.rd, mem.value);
            break;
        case Op::LBU:
            mem = memory_.read8(addr);
            if (mem.ok) set_reg(inst.rd, mem.value & 0xFFu);
            break;
        case Op::LHU:
            mem = memory_.read16(addr);
            if (mem.ok) set_reg(inst.rd, mem.value & 0xFFFFu);
            break;
        default:  [[unlikely]]
            result.ok = false;
            return result;
    }

    result.ok     = mem.ok;
    result.cycles = mem.cycles;
    if (mem.ok)
    {
        result.rd_value = regs_[inst.rd];

        if (trace_)
            std::cout << std::format("[LOAD] addr=0x{:08x} -> 0x{:08x}\n",
                                     addr, regs_[inst.rd]);
    }

    return result;
}

ExecuteResult Executor::execute_store(const DecodedInst& inst)
{
    ExecuteResult result;
    addr_t addr = regs_[inst.rs1] + static_cast<u32>(inst.imm);
    u32 value   = regs_[inst.rs2];
    MemoryResult mem;

    switch (inst.op)
    {
        case Op::SB: mem = memory_.write8(addr, static_cast<u8>(value)); break;
        case Op::SH: mem = memory_.write16(addr, static_cast<u16>(value)); break;
        case Op::SW: mem = memory_.write32(addr, value); break;
        default: [[unlikely]]
            result.ok = false;
            return result;
    }

    result.ok     = mem.ok;
    result.cycles = mem.cycles;

    if (trace_)
        std::cout << std::format("[STORE] addr={:08x} <- 0x{:08x}\n",
                                 addr, memory_.read32(addr).value);

    // Any store to the reserved address invalidates the reservation
    if (reservation_.has_value())
    {
        addr_t res = reservation_.value();
        addr_t store_size;
        switch(inst.op)
        {
            case Op::SB: store_size = 1; break;
            case Op::SH: store_size = 2; break;
            case Op::SW: store_size = 4; break;
            default:     store_size = 0; break;
        }
        // Check if [addr, addr+store_size) overlaps [res, res+4)
        if (addr < res + 4 && addr + store_size > res)
        {
            reservation_.reset();

            if (trace_)
                std::cout << std::format("[LR/SC] reservation cleared by store | "
                                         "addr=0x{:08x}\n", addr);
        }
    }

    return result;
}

// ── Main execute dispatch ──────────────────────────────────────────────

ExecuteResult Executor::execute(const DecodedInst& inst)
{
    if (trace_)
        std::cout << std::format("[CPU] Executing: {}\n",
                                 inst.disassemble());

    ExecuteResult result;
    result.next_pc = pc_ + 4;

    u32 rs1 = regs_[inst.rs1];
    u32 rs2 = regs_[inst.rs2];
    u32 uimm = static_cast<u32>(inst.imm);

    auto exec_jump = [&](addr_t target)
    {
        result.rd_value = pc_ + 4;
        set_reg(inst.rd, *result.rd_value);
        result.next_pc  = target;

        if (trace_)
            std::cout << std::format("[JUMP] 0x{:08x} -> 0x{:08x}\n",
                                     pc_, target);

        stats_.jumps++;
    };

    auto exec_alu = [&](auto fn, u32 lhs, u32 rhs)
    {
        set_reg(inst.rd, fn(lhs, rhs));
    };

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

        set_reg(inst.rd, old_val);
        result.rd_value = old_val;
    };

    switch (inst.op)
    {
        // ── Loads & stores ──────────────────────────────
        case Op::LB: case Op::LH: case Op::LW:
        case Op::LBU: case Op::LHU:
            result = execute_load(inst);
            result.next_pc = pc_ + 4;
            if (result.ok) stats_.loads++;
            break;

        case Op::SB: case Op::SH: case Op::SW:
            result = execute_store(inst);
            result.next_pc = pc_ + 4;
            if (result.ok) stats_.stores++;
            break;

        // ── Branches ────────────────────────────────────
        case Op::BEQ:  result.branch_taken = cond_eq(rs1, rs2);  goto branch_common;
        case Op::BNE:  result.branch_taken = cond_ne(rs1, rs2);  goto branch_common;
        case Op::BLT:  result.branch_taken = cond_lt(rs1, rs2);  goto branch_common;
        case Op::BGE:  result.branch_taken = cond_ge(rs1, rs2);  goto branch_common;
        case Op::BLTU: result.branch_taken = cond_ltu(rs1, rs2); goto branch_common;
        case Op::BGEU: result.branch_taken = cond_geu(rs1, rs2); goto branch_common;
        branch_common:
        {
            addr_t target = pc_ + static_cast<addr_t>(inst.imm);

            result.next_pc = result.branch_taken 
                           ? target
                           : (pc_ + 4);
            if (result.next_pc & 3)
            {
                std::cerr << std::format("[ERROR] Misaligned branch target 0x{:08x} "
                                         "(pc=0x{:08x})\n",
                                         target, pc_);

                result.ok = false;
                return result;
            }

            if (trace_)
                std::cout << std::format("[BRANCH] pc=0x{:08x} -> 0x{:08x}\n",
                                         pc_, result.next_pc);

            stats_.branches++;
            if (result.branch_taken) stats_.branches_taken++;
            break;
        }

        // ── Jumps ───────────────────────────────────────
        case Op::JAL:
        {
            addr_t target = pc_ + static_cast<addr_t>(inst.imm);
            if (target & 3)
            {
                result.ok = false;
                return result;
            }
            exec_jump(target);
            break;
        }

        case Op::JALR:
        {
            /*
             * Spec §2.1.5.1: The target address is (rs1 + imm), and the least-significant
             * bit is forced to zero.
             */
            addr_t target = (rs1 + uimm) & ~1u;
            if (target & 3)
            {
                result.ok = false;
                return result;
            }
            exec_jump(target);
            break;
        }

        // ── Upper immediates ────────────────────────────
        case Op::LUI:
            result.rd_value = uimm;
            set_reg(inst.rd, *result.rd_value);
            break;

        case Op::AUIPC:
            result.rd_value = pc_ + uimm;
            set_reg(inst.rd, *result.rd_value);
            break;

        // ── Arithmetic (immediate) ──────────────────────
        case Op::ADDI:   exec_alu(alu_add,    rs1, uimm); break;
        case Op::SLTI:   exec_alu(alu_slt,    rs1, uimm); break;
        case Op::SLTIU:  exec_alu(alu_sltu,   rs1, uimm); break;
        case Op::XORI:   exec_alu(alu_xor,    rs1, uimm); break;
        case Op::ORI:    exec_alu(alu_or,     rs1, uimm); break;
        case Op::ANDI:   exec_alu(alu_and,    rs1, uimm); break;
        case Op::SLLI:   exec_alu(alu_sll,    rs1, uimm); break;
        case Op::SRLI:   exec_alu(alu_srl,    rs1, uimm); break;
        case Op::SRAI:   exec_alu(alu_sra,    rs1, uimm); break;

        // ── Arithmetic (register) ───────────────────────
        case Op::ADD:    exec_alu(alu_add,    rs1, rs2); break;
        case Op::SUB:    exec_alu(alu_sub,    rs1, rs2); break;
        case Op::SLL:    exec_alu(alu_sll,    rs1, rs2); break;
        case Op::SLT:    exec_alu(alu_slt,    rs1, rs2); break;
        case Op::SLTU:   exec_alu(alu_sltu,   rs1, rs2); break;
        case Op::XOR:    exec_alu(alu_xor,    rs1, rs2); break;
        case Op::SRL:    exec_alu(alu_srl,    rs1, rs2); break;
        case Op::SRA:    exec_alu(alu_sra,    rs1, rs2); break;
        case Op::OR:     exec_alu(alu_or,     rs1, rs2); break;
        case Op::AND:    exec_alu(alu_and,    rs1, rs2); break;

        // ── RV32M instructions ──────────────────────────
        case Op::MUL:    exec_alu(alu_mul,    rs1, rs2); break;
        case Op::MULH:   exec_alu(alu_mulh,   rs1, rs2); break;
        case Op::MULHSU: exec_alu(alu_mulhsu, rs1, rs2); break;
        case Op::MULHU:  exec_alu(alu_mulhu,  rs1, rs2); break;
        case Op::DIV:    exec_alu(alu_div,    rs1, rs2); break;
        case Op::DIVU:   exec_alu(alu_divu,   rs1, rs2); break;
        case Op::REM:    exec_alu(alu_rem,    rs1, rs2); break;
        case Op::REMU:   exec_alu(alu_remu,   rs1, rs2); break;

        // ── RV32A instructions ──────────────────────────
        case Op::LR_W:
        {
            auto mem = memory_.read32(rs1);
            if (mem.ok)
            {
                set_reg(inst.rd, mem.value);
                reservation_ = rs1;
                result.rd_value = mem.value;
            }

            if (trace_)
                std::cout << std::format("[LR] addr=0x{:08x} | value=0x{:08x} | "
                                         "reservation SET\n",
                                         rs1, mem.value);

            result.ok = mem.ok;
            result.cycles = mem.cycles;
            break;
        }
        case Op::SC_W:
        {
            if (reservation_.has_value() && reservation_.value() == rs1)
            {
                auto mem = memory_.write32(rs1, rs2);
                set_reg(inst.rd, 0);  // success
                result.rd_value = 0;
                result.ok = mem.ok;
                result.cycles = mem.cycles;
            }
            else
            {
                set_reg(inst.rd, 1);  // failure
                result.rd_value = 1;
            }

            if (trace_)
                std::cout << std::format("[SC] addr=0x{:08x} | value=0x{:08x} | {}\n",
                                         rs1, rs2,
                                         result.rd_value ? "FAIL" : "SUCCESS");
            reservation_.reset();
            break;
        }
        case Op::AMOSWAP_W: case Op::AMOADD_W: case Op::AMOXOR_W:
        case Op::AMOAND_W:  case Op::AMOOR_W:
        case Op::AMOMIN_W:  case Op::AMOMAX_W:
        case Op::AMOMINU_W: case Op::AMOMAXU_W:
        {
            auto mem = memory_.read32(rs1);
            if (!mem.ok) { result.ok = false; break; }
            u32 old_val = mem.value;
            u32 new_val;
            switch(inst.op)
            {
                case Op::AMOSWAP_W: new_val = rs2; break;
                case Op::AMOADD_W:  new_val = old_val + rs2; break;
                case Op::AMOXOR_W:  new_val = old_val ^ rs2; break;
                case Op::AMOAND_W:  new_val = old_val & rs2; break;
                case Op::AMOOR_W:   new_val = old_val | rs2; break;
                case Op::AMOMIN_W:  new_val = (static_cast<i32>(old_val)
                                             < static_cast<i32>(rs2))
                                             ? old_val
                                             : rs2;
                                    break;
                case Op::AMOMAX_W:  new_val = (static_cast<i32>(old_val)
                                             > static_cast<i32>(rs2))
                                             ? old_val
                                             : rs2;
                                    break;
                case Op::AMOMINU_W: new_val = (old_val < rs2) ? old_val : rs2; break;
                case Op::AMOMAXU_W: new_val = (old_val > rs2) ? old_val : rs2; break;
                default: [[unlikely]] new_val = old_val; break;
            }
            memory_.write32(rs1, new_val);
            set_reg(inst.rd, old_val);
            result.rd_value = old_val;
            result.cycles = mem.cycles;

            if (trace_)
                std::cout << std::format("[AMO] addr=0x{:08x} | "
                                         "old=0x{:08x} | new=0x{:08x}\n",
                                         rs1, old_val, new_val);
            break;
        }

        // ── System ──────────────────────────────────────
        case Op::ECALL:  result.ecall = true; break;
        case Op::EBREAK: result.ebreak = true; break;
        case Op::FENCE:  break;  // NOP in single-threaded context
        case Op::FENCE_I: result.fence_i = true; break;

        // CSR instructions ───────────────────────────────
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
            exec_csr([&](u32) { return inst.rs1; } , true);
            break;
        case Op::CSRRSI:
            exec_csr([&](u32 old) { return old | inst.rs1; }, inst.rs1 != 0);
            break;
        case Op::CSRRCI:
            exec_csr([&](u32 old) { return old & ~inst.rs1; }, inst.rs1 != 0);
            break;

        // ── MRET: return from trap ──────────────────────
        case Op::MRET:
        {
            result.next_pc = csrs_.mret();
            break;
        }

        // ── RVV operations ──────────────────────────────
        case Op::VSETVLI:
        case Op::VLE32:    case Op::VSE32:
        case Op::VADD_VV:  case Op::VADD_VX:
        case Op::VSUB_VV:  case Op::VSUB_VX:
        case Op::VAND_VV:  case Op::VAND_VX:
        case Op::VOR_VV:   case Op::VOR_VX:
        case Op::VXOR_VV:  case Op::VXOR_VX:
        case Op::VSLL_VX:  case Op::VSRL_VX:
        case Op::VMSEQ_VV: case Op::VMSEQ_VX:
        case Op::VMSLT_VV: case Op::VMSLTU_VV:
        case Op::VMAND_MM: case Op::VMNAND_MM: case Op::VMANDN_MM:
        case Op::VMXOR_MM: case Op::VMOR_MM:   case Op::VMNOR_MM:
        case Op::VMORN_MM: case Op::VMXNOR_MM:
        case Op::VREDSUM_VS:
        case Op::VMV_V_X:  case Op::VMV_X_S:
            return execute_vector(inst, rs1, rs2);

        case Op::INVALID:
        default: [[unlikely]]
            result.ok = false;
            break;
    }

    // ── Post-execution metadata update ──────────────────
    if (result.ok)
    {
        if (inst.writes_rd() && !result.rd_value.has_value())
            result.rd_value = regs_[inst.rd];
        stats_.instructions++;
        stats_.cycles += result.cycles;
    }

    return result;
}

// ── Vector execution ──────────────────────────────────── 

ExecuteResult Executor::execute_vector(const DecodedInst& inst,
                                       u32 rs1, [[maybe_unused]] u32 rs2)
{
    ExecuteResult result;
    result.next_pc = pc_ + 4;

    auto& vs    = vstate_;
    u32   vl    = vs.vl;
    auto& vregs = vs.regs;

    const u32 vd   = inst.rd;
    const u32 vs1  = inst.rs1;
    const u32 vs2  = inst.rs2;
    const u32 uimm = static_cast<u32>(inst.imm);

    auto exec_vv = [&](auto op)
    {
        for (u32 i = 0; i < vl; ++i)
        {
            const u32 lhs = vregs.get_elem32(vs2, i);
            const u32 rhs = vregs.get_elem32(vs1, i);

            vregs.set_elem32(vd, i, op(lhs, rhs));
        }
    };

    auto exec_vx = [&](auto op)
    {
        for (u32 i = 0; i < vl; ++i)
        {
            const u32 lhs = vregs.get_elem32(vs2, i);

            vregs.set_elem32(vd, i, op(lhs, rs1));
        }
    };

    auto exec_mask = [&](auto pred)
    {
        for (u32 i = 0; i < vl; ++i)
            vregs.set_mask_bit(vd, i, pred(i));
    };

    switch (inst.op)
    {
        // ── Configuration ───────────────────────────────
        case Op::VSETVLI:
        {
            u32 avl = (inst.rs1 == 0 && inst.rd == 0)
                    ? vs.vl            // keep current vl
                    : (inst.rs1 == 0)
                    ? ~u32{0}          // set vl=VLMAX 
                    : rs1;

            u32 new_vl = vs.vsetvli(avl, uimm);
            set_reg(inst.rd, new_vl);
            result.rd_value = new_vl;
            break;
        }

        // ── Vector load (unit-stride, SEW=32) ───────────
        case Op::VLE32:
        {
            addr_t base = rs1;
            if (base & 3)
            {
                std::cerr << std::format("[ERROR] Misaligned vector load "
                                         "at 0x{:08x} (pc=0x{:08x})\n",
                                         base, pc_);

                result.ok = false;
                return result;
            }
            for (u32 i = 0; i < vl; ++i)
            {
                auto r = memory_.read32(base + i * 4);
                if (!r.ok) { result.ok = false; return result; }
                vregs.set_elem32(vd, i, r.value);
            }

            if (trace_)
                std::cout << std::format("[VLOAD] vd=v{} | base=0x{:08x} | vl={}\n",
                                         inst.rd, base, vstate_.vl);

            result.cycles = vl;  // 1 cycle per element
            break;
        }

        // ── Vector store (unit-stride, SEW=32) ──────────
        case Op::VSE32:
        {
            addr_t base = rs1;
            if (base & 3)
            {
                std::cerr << std::format("[ERROR] Misaligned vector store "
                                         "at 0x{:08x} (pc=0x{:08x})\n",
                                         base, pc_);

                result.ok = false;
                return result;
            }
            for (u32 i = 0; i < vl; ++i)
            {
                u32 val = vregs.get_elem32(inst.rd, i);
                auto r = memory_.write32(base + i * 4, val);
                if (!r.ok) { result.ok = false; return result; }
            }

            if (trace_)
                std::cout << std::format("[VSTORE] vd=v{} | base=0x{:08x} | vl={}\n",
                                         inst.rd, base, vstate_.vl);

            result.cycles = vl;
            break;
        }

        // ── Arithmetic VV ───────────────────────────────
        case Op::VADD_VV: exec_vv([](u32 a, u32 b) { return a + b; }); break;
        case Op::VSUB_VV: exec_vv([](u32 a, u32 b) { return a - b; }); break;
        case Op::VAND_VV: exec_vv([](u32 a, u32 b) { return a & b; }); break;
        case Op::VOR_VV:  exec_vv([](u32 a, u32 b) { return a | b; }); break;
        case Op::VXOR_VV: exec_vv([](u32 a, u32 b) { return a ^ b; }); break;

        // ── Arithmetic VX (scalar broadcast) ────────────
        case Op::VADD_VX: exec_vx([](u32 a, u32 b) { return a + b; }); break;
        case Op::VSUB_VX: exec_vx([](u32 a, u32 b) { return a - b; }); break;
        case Op::VAND_VX: exec_vx([](u32 a, u32 b) { return a & b; }); break;
        case Op::VOR_VX:  exec_vx([](u32 a, u32 b) { return a | b; }); break;
        case Op::VXOR_VX: exec_vx([](u32 a, u32 b) { return a ^ b; }); break;
        case Op::VSLL_VX: exec_vx([](u32 a, u32 b) { return a << (b & 0x1F); }); break;
        case Op::VSRL_VX: exec_vx([](u32 a, u32 b) { return a >> (b & 0x1F); }); break;

        // ── Comparisons (write mask to vd) ──────────────
        case Op::VMSEQ_VV:
            exec_mask([&](u32 i)
            {
                return vregs.get_elem32(vs2, i)
                    == vregs.get_elem32(vs1, i);
            });
            break;
        case Op::VMSEQ_VX:
            exec_mask([&](u32 i)
            {
                return vregs.get_elem32(vs2, i) == rs1;
            });
            break;
        case Op::VMSLT_VV:
            exec_mask([&](u32 i)
            {
                return static_cast<i32>(vregs.get_elem32(vs2, i))
                     < static_cast<i32>(vregs.get_elem32(vs1, i));
            });
            break;
        case Op::VMSLTU_VV:
            exec_mask([&](u32 i)
            {
                return vregs.get_elem32(vs2, i)
                     < vregs.get_elem32(vs1, i);
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

                switch(inst.op)
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

                if (bit_result) dst_elem |= (1u << bit_pos);
                else            dst_elem &= ~(1u << bit_pos);
                vregs.set_elem32(vd, i / 32, dst_elem);
            }
            break;
        }

        // ── Reduction ───────────────────────────────────
        case Op::VREDSUM_VS:
        {
            // vd[0] = vs1[0] + sum(vs2[0..vl-1])
            u32 acc = vregs.get_elem32(vs1, 0);
            for (u32 i = 0; i < vl; ++i)
                acc += vregs.get_elem32(vs2, i);
            vregs.set_elem32(vd, 0, acc);
            break;
        }

        // ── Move ────────────────────────────────────────
        case Op::VMV_V_X:
            // Splat scalar rs1 to all elements of vd
            for (u32 i = 0; i < vl; ++i)
                vregs.set_elem32(vd, i, rs1);
            break;
        case Op::VMV_X_S:
            // Extract element 0 of vs2 into scalar rd
            set_reg(inst.rd, vs.regs.get_elem32(vs2, 0));
            result.rd_value = regs_[inst.rd];
            break;

        default: [[unlikely]]
            result.ok = false;
            break;
    }

    if (result.ok)
    {
        stats_.instructions++;
        stats_.cycles += result.cycles;
    }

    return result;
}

} // namespace riscv
