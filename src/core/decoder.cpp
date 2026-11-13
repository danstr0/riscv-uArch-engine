/**
 * @file decoder.cpp
 * @brief Instruction decoder implementation.
 */

#include "decoder.hpp"

#include <format>

namespace riscv {

// ── Mnemonic lookup ────────────────────────────────────────────────────

const char* op_name(Op op)
{
    switch (op)
    {
        case Op::LB:         return "lb";
        case Op::LH:         return "lh";
        case Op::LW:         return "lw";
        case Op::LBU:        return "lbu";
        case Op::LHU:        return "lhu";
        case Op::SB:         return "sb";
        case Op::SH:         return "sh";
        case Op::SW:         return "sw";

        case Op::BEQ:        return "beq";
        case Op::BNE:        return "bne";
        case Op::BLT:        return "blt";
        case Op::BGE:        return "bge";
        case Op::BLTU:       return "bltu";
        case Op::BGEU:       return "bgeu";

        case Op::JAL:        return "jal";
        case Op::JALR:       return "jalr";

        case Op::LUI:        return "lui";
        case Op::AUIPC:      return "auipc";

        case Op::ADDI:       return "addi";
        case Op::SLTI:       return "slti";
        case Op::SLTIU:      return "sltiu";
        case Op::XORI:       return "xori";
        case Op::ORI:        return "ori";
        case Op::ANDI:       return "andi";
        case Op::SLLI:       return "slli";
        case Op::SRLI:       return "srli";
        case Op::SRAI:       return "srai";

        case Op::ADD:        return "add";
        case Op::SUB:        return "sub";
        case Op::SLL:        return "sll";
        case Op::SLT:        return "slt";
        case Op::SLTU:       return "sltu";
        case Op::XOR:        return "xor";
        case Op::SRL:        return "srl";
        case Op::SRA:        return "sra";
        case Op::OR:         return "or";
        case Op::AND:        return "and";

        case Op::MUL:        return "mul";
        case Op::MULH:       return "mulh";
        case Op::MULHSU:     return "mulhsu";
        case Op::MULHU:      return "mulhu";
        case Op::DIV:        return "div";
        case Op::DIVU:       return "divu";
        case Op::REM:        return "rem";
        case Op::REMU:       return "remu";

        case Op::LR_W:       return "lr.w";
        case Op::SC_W:       return "sc.w";
        case Op::AMOSWAP_W:  return "amoswap.w";
        case Op::AMOADD_W:   return "amoadd.w";
        case Op::AMOXOR_W:   return "amoxor.w";
        case Op::AMOAND_W:   return "amoand.w";
        case Op::AMOOR_W:    return "amoor.w";
        case Op::AMOMIN_W:   return "amomin.w";
        case Op::AMOMAX_W:   return "amomax.w";
        case Op::AMOMINU_W:  return "amominu.w";
        case Op::AMOMAXU_W:  return "amomaxu.w";

        case Op::VSETVLI:    return "vsetvli";
        case Op::VLE32:      return "vle32.v";
        case Op::VSE32:      return "vse32.v";
        case Op::VADD_VV:    return "vadd.vv";
        case Op::VADD_VX:    return "vadd.vx";
        case Op::VSUB_VV:    return "vsub.vv";
        case Op::VSUB_VX:    return "vsub.vx";
        case Op::VAND_VV:    return "vand.vv";
        case Op::VAND_VX:    return "vand.vx";
        case Op::VOR_VV:     return "vor.vv";
        case Op::VOR_VX:     return "vor.vx";
        case Op::VXOR_VV:    return "vxor.vv";
        case Op::VXOR_VX:    return "vxor.vx";
        case Op::VSLL_VX:    return "vsll.vx";
        case Op::VSRL_VX:    return "vsrl.vx";
        case Op::VMSEQ_VV:   return "vmseq.vv";
        case Op::VMSEQ_VX:   return "vmseq.vx";
        case Op::VMSLT_VV:   return "vmslt.vv";
        case Op::VMSLTU_VV:  return "vmsltu.vv";
        case Op::VMAND_MM:   return "vmand.mm";
        case Op::VMNAND_MM:  return "vmnand.mm";
        case Op::VMANDN_MM:  return "vmandn.mm";
        case Op::VMXOR_MM:   return "vmxor.mm"; 
        case Op::VMOR_MM:    return "vmor.mm";
        case Op::VMNOR_MM:   return "vmnor.mm";
        case Op::VMORN_MM:   return "vmorn.mm";
        case Op::VMXNOR_MM:  return "vmxnor.mm";
        case Op::VREDSUM_VS: return "vredsum.vs";
        case Op::VMV_V_X:    return "vmv.v.x";
        case Op::VMV_X_S:    return "vmv.x.s";

        case Op::FENCE:      return "fence";
        case Op::FENCE_I:    return "fence.i";
        case Op::ECALL:      return "ecall";
        case Op::EBREAK:     return "ebreak";
        case Op::CSRRW:      return "csrrw";
        case Op::CSRRS:      return "csrrs";
        case Op::CSRRC:      return "csrrc";
        case Op::CSRRWI:     return "csrrwi";
        case Op::CSRRSI:     return "csrrsi";
        case Op::CSRRCI:     return "csrrci";
        case Op::MRET:       return "mret";

        case Op::INVALID:    return "invalid";
    }
    return "???";
}

// ── Disassembly ────────────────────────────────────────────────────────

std::string DecodedInst::disassemble() const
{
    switch(format)
    {
        case Format::R:
            if (is_atomic())
            {
                if (op == Op::LR_W)
                    return std::format("{} {}, ({})",
                            op_name(op), reg_name(rd), reg_name(rs1));

                return std::format("{} {}, {}, ({})",
                        op_name(op), reg_name(rd), reg_name(rs2), reg_name(rs1));
            }
            if (is_vector())
                switch(op)
                {
                    case Op::VLE32: // vd, (rs1)
                    case Op::VSE32: // vs3, (rs1)
                        return std::format("{} {}, ({})",
                                op_name(op), vreg_name(rd), reg_name(rs1));

                    case Op::VSETVLI: // rd, rs1, vtypei
                        return std::format("{} {}, {}, {}",
                                op_name(op), reg_name(rd), reg_name(rs1), imm);

                    // vd, vs2, vs1
                    case Op::VADD_VV: case Op::VSUB_VV:
                    case Op::VAND_VV: case Op::VOR_VV: case Op::VXOR_VV:
                    case Op::VMSEQ_VV: case Op::VMSLT_VV: case Op::VMSLTU_VV:
                    case Op::VMAND_MM: case Op::VMNAND_MM: case Op::VMANDN_MM:
                    case Op::VMOR_MM: case Op::VMNOR_MM: case Op::VMORN_MM:
                    case Op::VMXOR_MM: case Op::VMXNOR_MM:
                    case Op::VREDSUM_VS:
                        return std::format("{} {}, {}, {}",
                                op_name(op), vreg_name(rd), vreg_name(rs2), vreg_name(rs1));

                    // vd, vs2, rs1
                    case Op::VADD_VX: case Op::VSUB_VX:
                    case Op::VAND_VX: case Op::VOR_VX: case Op::VXOR_VX:
                    case Op::VSLL_VX: case Op::VSRL_VX:
                    case Op::VMSEQ_VX:
                        return std::format("{} {}, {}, {}",
                                op_name(op), vreg_name(rd), vreg_name(rs2), reg_name(rs1));

                    case Op::VMV_V_X: // vd, rs1
                        return std::format("{} {}, {}",
                                op_name(op), vreg_name(rd), reg_name(rs1));

                    case Op::VMV_X_S: // rd, vs2
                        return std::format("{} {}, {}",
                                op_name(op), reg_name(rd), vreg_name(rs2));

                    // Unreachable
                    default:
                        return "???";
                }
            return std::format("{} {}, {}, {}",
                op_name(op), reg_name(rd), reg_name(rs1), reg_name(rs2));

        case Format::I:
            if (is_load() || op == Op::JALR)
                return std::format("{} {}, {}({})",
                    op_name(op), reg_name(rd), imm, reg_name(rs1));

            return std::format("{} {}, {}, {}",
                op_name(op), reg_name(rd), reg_name(rs1), imm);

        case Format::S:
            return std::format("{} {}, {}({})",
                op_name(op), reg_name(rs2), imm, reg_name(rs1));

        case Format::B:
            return std::format("{} {}, {}, 0x{:x}",
                op_name(op), reg_name(rs1), reg_name(rs2),
                pc + static_cast<u32>(imm));

        case Format::U:
            // U-type immediates are displayed as shifted 20-bit hex values.
            return std::format("{} {}, 0x{:x}",
                op_name(op), reg_name(rd), static_cast<u32>(imm) >> 12);

        case Format::J:
            return std::format("{} {}, 0x{:x}",
                op_name(op), reg_name(rd), pc + static_cast<u32>(imm));
    }
    return "???";
}

// ── Immediate extraction ───────────────────────────────────────────────

i32 Decoder::extract_i_imm(u32 inst)
{
    // I-type: imm[11:0] = inst[31:20]
    return sign_extend<12>(inst >> 20);
}

i32 Decoder::extract_s_imm(u32 inst)
{
    // S-type: imm[11:5]=inst[31:25], imm[4:0]=inst[11:7]
    u32 imm = (bits(inst, 31, 25) << 5) | bits(inst, 11, 7);
    return sign_extend<12>(imm);
}

i32 Decoder::extract_b_imm(u32 inst)
{
    // B-type: imm[12]=inst[31], imm[11]=inst[7], imm[10:5]=inst[30:25], imm[4:1]=inst[11:8]
    // Bit 0 is implicitly 0
    u32 imm = (bit(inst, 31)      << 12)
            | (bit(inst, 7)       << 11)
            | (bits(inst, 30, 25) << 5)
            | (bits(inst, 11, 8)  << 1);
    return sign_extend<13>(imm);
}

i32 Decoder::extract_u_imm(u32 inst)
{
    // U-type: imm[31:12] = inst[31:12], lower 12 bits zeroed
    return static_cast<i32>(inst & 0xFFFF'F000u);
}

i32 Decoder::extract_j_imm(u32 inst)
{
    // J-type: imm[20]=inst[31], imm[19:12]=inst[19:12], imm[11]=inst[20], imm[10:1]=inst[30:21]
    // Bit 0 is implicitly 0
    u32 imm = (bit(inst, 31)      << 20)
            | (bits(inst, 19, 12) << 12)
            | (bit(inst, 20)      << 11)
            | (bits(inst, 30, 21) << 1);
    return sign_extend<21>(imm);
}

// ── Per-opcode decode helpers ──────────────────────────────────────────

DecodedInst Decoder::decode_load(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::I;
    d.raw    = inst;
    d.pc     = pc;
    d.rd     = static_cast<reg_idx_t>(bits(inst, 11, 7));
    d.rs1    = static_cast<reg_idx_t>(bits(inst, 19, 15));
    d.imm    = extract_i_imm(inst);

    u32 funct3 = bits(inst, 14, 12);

    switch (funct3)
    {
        case 0b000: d.op = Op::LB;  break;
        case 0b001: d.op = Op::LH;  break;
        case 0b010: d.op = Op::LW;  break;
        case 0b100: d.op = Op::LBU; break;
        case 0b101: d.op = Op::LHU; break;
        default:    d.op = Op::INVALID; break;
    }
    return d;
}

DecodedInst Decoder::decode_store(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::S;
    d.raw    = inst;
    d.pc     = pc;
    d.rs1    = static_cast<reg_idx_t>(bits(inst, 19, 15));
    d.rs2    = static_cast<reg_idx_t>(bits(inst, 24, 20));
    d.imm    = extract_s_imm(inst);

    u32 funct3 = bits(inst, 14, 12);

    switch(funct3)
    {
        case 0b000: d.op = Op::SB; break;
        case 0b001: d.op = Op::SH; break;
        case 0b010: d.op = Op::SW; break;
        default:    d.op = Op::INVALID; break;
    }
    return d;
}

DecodedInst Decoder::decode_branch(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::B;
    d.raw    = inst;
    d.pc     = pc;
    d.rs1    = static_cast<reg_idx_t>(bits(inst, 19, 15));
    d.rs2    = static_cast<reg_idx_t>(bits(inst, 24, 20));
    d.imm    = extract_b_imm(inst);

    u32 funct3 = bits(inst, 14, 12);

    switch (funct3)
    {
        case 0b000: d.op = Op::BEQ;  break;
        case 0b001: d.op = Op::BNE;  break;
        case 0b100: d.op = Op::BLT;  break;
        case 0b101: d.op = Op::BGE;  break;
        case 0b110: d.op = Op::BLTU; break;
        case 0b111: d.op = Op::BGEU; break;
        default:    d.op = Op::INVALID; break;
    }
    return d;
}

DecodedInst Decoder::decode_op_imm(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::I;
    d.raw    = inst;
    d.pc     = pc;
    d.rd     = static_cast<reg_idx_t>(bits(inst, 11, 7));
    d.rs1    = static_cast<reg_idx_t>(bits(inst, 19, 15));
    d.imm    = extract_i_imm(inst);

    u32 funct3 = bits(inst, 14, 12);
    u32 funct7 = bits(inst, 31, 25);

    switch (funct3)
    {
        case 0b000: d.op = Op::ADDI;  break;
        case 0b010: d.op = Op::SLTI;  break;
        case 0b011: d.op = Op::SLTIU; break;
        case 0b100: d.op = Op::XORI;  break;
        case 0b110: d.op = Op::ORI;   break;
        case 0b111: d.op = Op::ANDI;  break;

        case 0b001: // SLLI
            if (funct7 == 0b0000000)
            {
                d.op  = Op::SLLI;
                d.imm = static_cast<i32>(bits(inst, 24, 20)); // shamt
            }
            else
                d.op = Op::INVALID;
            break;

        case 0b101: // SRLI / SRAI
            d.imm = static_cast<i32>(bits(inst, 24, 20)); // shamt
            if (funct7 == 0b0000000)      d.op = Op::SRLI;
            else if (funct7 == 0b0100000) d.op = Op::SRAI;
            else                          d.op = Op::INVALID;
            break;

        default: d.op = Op::INVALID; break;
    }
    return d;
}

DecodedInst Decoder::decode_op(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::R;
    d.raw    = inst;
    d.pc     = pc;
    d.rd     = static_cast<reg_idx_t>(bits(inst, 11, 7));
    d.rs1    = static_cast<reg_idx_t>(bits(inst, 19, 15));
    d.rs2    = static_cast<reg_idx_t>(bits(inst, 24, 20));

    u32 funct3 = bits(inst, 14, 12);
    u32 funct7 = bits(inst, 31, 25);

    if (funct7 == 0b0000001)
    {
        switch (funct3)
        {
            case 0b000: d.op = Op::MUL;    break;
            case 0b001: d.op = Op::MULH;   break;
            case 0b010: d.op = Op::MULHSU; break;
            case 0b011: d.op = Op::MULHU;  break;
            case 0b100: d.op = Op::DIV;    break;
            case 0b101: d.op = Op::DIVU;   break;
            case 0b110: d.op = Op::REM;    break;
            case 0b111: d.op = Op::REMU;   break;
        }
        return d;
    }

    switch (funct3)
    {
        case 0b000:
            d.op = (funct7 == 0b0000000) ? Op::ADD
                 : (funct7 == 0b0100000) ? Op::SUB
                 : Op::INVALID;
            break;
        case 0b001: d.op = (funct7 == 0b0000000) ? Op::SLL  : Op::INVALID; break;
        case 0b010: d.op = (funct7 == 0b0000000) ? Op::SLT  : Op::INVALID; break;
        case 0b011: d.op = (funct7 == 0b0000000) ? Op::SLTU : Op::INVALID; break;
        case 0b100: d.op = (funct7 == 0b0000000) ? Op::XOR  : Op::INVALID; break;
        case 0b101:
            d.op = (funct7 == 0b0000000) ? Op::SRL
                 : (funct7 == 0b0100000) ? Op::SRA
                 : Op::INVALID;
            break;
        case 0b110: d.op = (funct7 == 0b0000000) ? Op::OR   : Op::INVALID; break;
        case 0b111: d.op = (funct7 == 0b0000000) ? Op::AND  : Op::INVALID; break;
        default:    d.op = Op::INVALID; break;
    }
    return d;
}

DecodedInst Decoder::decode_amo(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::R;
    d.raw    = inst;
    d.pc     = pc;
    d.rd     = static_cast<reg_idx_t>(bits(inst, 11, 7));
    d.rs1    = static_cast<reg_idx_t>(bits(inst, 19, 15));
    d.rs2    = static_cast<reg_idx_t>(bits(inst, 24, 20));

    u32 funct3 = bits(inst, 14, 12);
    u32 funct5 = bits(inst, 31, 27);

    // Only .W is supported in RV32A
    if (funct3 != 0b010)
    {
        d.op = Op::INVALID;
        return d;
    }

    switch(funct5)
    {
        case 0b00010: d.op = Op::LR_W;      break;
        case 0b00011: d.op = Op::SC_W;      break;
        case 0b00001: d.op = Op::AMOSWAP_W; break;
        case 0b00000: d.op = Op::AMOADD_W;  break;
        case 0b00100: d.op = Op::AMOXOR_W;  break;
        case 0b01100: d.op = Op::AMOAND_W;  break;
        case 0b01000: d.op = Op::AMOOR_W;   break;
        case 0b10000: d.op = Op::AMOMIN_W;  break;
        case 0b10100: d.op = Op::AMOMAX_W;  break;
        case 0b11000: d.op = Op::AMOMINU_W; break;
        case 0b11100: d.op = Op::AMOMAXU_W; break;
        default:      d.op = Op::INVALID;   break;
    }

    if (d.op == Op::LR_W && d.rs2 != 0)
        d.op = Op::INVALID;

    return d;
}

DecodedInst Decoder::decode_vector(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::R;
    d.raw    = inst;
    d.pc     = pc;
    d.rd     = static_cast<reg_idx_t>(bits(inst, 11, 7));   // vd (or rd for VMV.X.S / VSETVLI)
    d.rs1    = static_cast<reg_idx_t>(bits(inst, 19, 15));  // vs1 or rs1
    d.rs2    = static_cast<reg_idx_t>(bits(inst, 24, 20));  // vs2 or rs2

    u32 opcode = bits(inst, 6, 0);
    u32 funct3 = bits(inst, 14, 12);
    u32 funct6 = bits(inst, 31, 26);

    // Vector load
    if (opcode == 0b0000111)
    {
        /* 
         * Unit-stride load: width in funct3, nf|mew|mop|vm|lumop in upper bits
         * Only VLE32 (funct3 = 110, width = 32) is supported
         */
        if (funct3 == 0b110 && d.rs2 == 0b00000)
            d.op = Op::VLE32;
        else
            d.op = Op::INVALID;
        return d;
    }

    // Vector store
    if (opcode == 0b0100111)
    {
        if (funct3 == 0b110 && d.rs2 == 0b00000)
            d.op = Op::VSE32;
        else
            d.op = Op::INVALID;
        return d;
    }

    // OP-V (opcode = 1010111)

    if (funct3 == 0b111 && bit(inst, 31) == 0)
    {
        d.op  = Op::VSETVLI;
        d.imm = static_cast<i32>(bits(inst, 30, 20));  // zimm[10:0] encodes type
        return d;
    }
    switch(funct3)
    {
        case 0b000:  // OPIVV: vector-vector integer
            switch (funct6)
            {
                case 0b000000: d.op = Op::VADD_VV;   break;
                case 0b000010: d.op = Op::VSUB_VV;   break;
                case 0b001001: d.op = Op::VAND_VV;   break;
                case 0b001010: d.op = Op::VOR_VV;    break;
                case 0b001011: d.op = Op::VXOR_VV;   break;
                case 0b011000: d.op = Op::VMSEQ_VV;  break;
                case 0b011011: d.op = Op::VMSLT_VV;  break;
                case 0b011010: d.op = Op::VMSLTU_VV; break;
                default:       d.op = Op::INVALID;   break;
            }
            break;

        case 0b100:  // OPIVX: vector-scalar integer
            switch (funct6)
            {
                case 0b000000: d.op = Op::VADD_VX;  break;
                case 0b000010: d.op = Op::VSUB_VX;  break;
                case 0b001001: d.op = Op::VAND_VX;  break;
                case 0b001010: d.op = Op::VOR_VX;   break;
                case 0b001011: d.op = Op::VXOR_VX;  break;
                case 0b100101: d.op = Op::VSLL_VX;  break;
                case 0b101000: d.op = Op::VSRL_VX;  break;
                case 0b011000: d.op = Op::VMSEQ_VX; break;
                case 0b010111: d.op = (d.rs2 == 0) ? Op::VMV_V_X : Op::INVALID; break;
                default:       d.op = Op::INVALID;  break;
            }
            break;

        case 0b010:  // OPMVV: vector-vector (mask/reduction/move)
            switch (funct6)
            {
                case 0b000000: d.op = Op::VREDSUM_VS; break;
                case 0b011001: d.op = Op::VMAND_MM;   break;
                case 0b011101: d.op = Op::VMNAND_MM;  break;
                case 0b011000: d.op = Op::VMANDN_MM;  break;
                case 0b011010: d.op = Op::VMOR_MM;    break;
                case 0b011110: d.op = Op::VMNOR_MM;   break;
                case 0b011100: d.op = Op::VMORN_MM;   break;
                case 0b011011: d.op = Op::VMXOR_MM;   break;
                case 0b011111: d.op = Op::VMXNOR_MM;  break;
                case 0b010000: d.op = (d.rs1 == 0) ? Op::VMV_X_S : Op::INVALID; break;
                default:       d.op = Op::INVALID;    break;
            }
            break;

        default:
            d.op = Op::INVALID;
            break;
    }

    return d;
}

DecodedInst Decoder::decode_system(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.format = Format::I;
    d.raw    = inst;
    d.pc     = pc;
    d.rd     = static_cast<reg_idx_t>(bits(inst, 11, 7));
    d.rs1    = static_cast<reg_idx_t>(bits(inst, 19, 15));
    d.imm    = extract_i_imm(inst);

    u32 funct3 = bits(inst, 14, 12);

    if (funct3 == 0b000)
    {
        u32 funct12 = bits(inst, 31, 20);
        constexpr u32 mret_funct12 = 0b001100000010;

        if (inst == 0x00000073)           d.op = Op::ECALL;
        else if (inst == 0x00100073)      d.op = Op::EBREAK;
        else if (funct12 == mret_funct12 
                && d.rd == 0 
                && d.rs1 == 0)            d.op = Op::MRET;
        else                              d.op = Op::INVALID;
    } 
    else
    {
        // CSR instructions
        switch (funct3)
        {
            case 0b001: d.op = Op::CSRRW;   break;
            case 0b010: d.op = Op::CSRRS;   break;
            case 0b011: d.op = Op::CSRRC;   break;
            case 0b101: d.op = Op::CSRRWI;  break;
            case 0b110: d.op = Op::CSRRSI;  break;
            case 0b111: d.op = Op::CSRRCI;  break;
            default:    d.op = Op::INVALID; break;
        }
        d.imm = static_cast<i32>(bits(inst, 31, 20));
    }
    return d;
}

// ── Main decode entry point ────────────────────────────────────────────

DecodedInst Decoder::decode(u32 inst, addr_t pc)
{
    DecodedInst d;
    d.raw = inst;
    d.pc  = pc;

    auto opcode = static_cast<Opcode>(bits(inst, 6, 0));

    switch (opcode)
    {
        case Opcode::LOAD:   return decode_load(inst, pc);
        case Opcode::STORE:  return decode_store(inst, pc);
        case Opcode::BRANCH: return decode_branch(inst, pc);
        case Opcode::OP_IMM: return decode_op_imm(inst, pc);
        case Opcode::OP:     return decode_op(inst, pc);
        case Opcode::AMO:    return decode_amo(inst, pc);
        case Opcode::SYSTEM: return decode_system(inst, pc);

        case Opcode::LUI:
        case Opcode::AUIPC:
            d.format = Format::U;
            d.op     = (opcode == Opcode::LUI) ? Op::LUI : Op::AUIPC;
            d.rd     = static_cast<reg_idx_t>(bits(inst, 11, 7));
            d.imm    = extract_u_imm(inst);
            return d;

        case Opcode::JAL:
            d.format = Format::J;
            d.op     = Op::JAL;
            d.rd     = static_cast<reg_idx_t>(bits(inst, 11, 7));
            d.imm    = extract_j_imm(inst);
            return d;

        case Opcode::JALR:
            if (bits(inst, 14, 12) != 0b000) { d.op = Op::INVALID; return d; }
            d.format = Format::I;
            d.op     = Op::JALR;
            d.rd     = static_cast<reg_idx_t>(bits(inst, 11, 7));
            d.rs1    = static_cast<reg_idx_t>(bits(inst, 19, 15));
            d.imm    = extract_i_imm(inst);
            return d;

        case Opcode::MISC_MEM:
            d.format = Format::I;
            d.op     = (bits(inst, 14, 12) == 0b001) ? Op::FENCE_I : Op::FENCE;
            d.rd     = static_cast<reg_idx_t>(bits(inst, 11, 7));
            d.rs1    = static_cast<reg_idx_t>(bits(inst, 19, 15));
            d.imm    = extract_i_imm(inst);
            return d;

        case Opcode::VL:
        case Opcode::VS:
        case Opcode::OP_V:
            return decode_vector(inst, pc);

        default:
            d.op = Op::INVALID;
            return d;
    }
}

} // namespace riscv
