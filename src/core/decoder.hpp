/**
 * @file decoder.hpp
 * @brief Instruction decoder for the RISC-V simulator.
 *
 * Decodes 32-bit instruction words into a structured representation
 * covering RV32IMA and a subset of RVV.
 *
 * @par Decode Process
 * Bits [6:0] select the major opcode, which dispatches to a format-aware
 * helper that extracts register fields, reconstructs the sign-extended
 * immediate, and maps @c funct3 / @c funct7 to a concrete @c Op tag.
 *
 * @see RISC-V Unprivileged ISA Specification, Sections 2.1, 6.1.1, 12.1,
 *      13.1, 30.1.
 */

#pragma once

#include "types.hpp"

#include <string>

namespace riscv {

/// Primary Opcodes (bits [6:0] of the instruction word).
enum class Opcode : u8
{
    LOAD      = 0b0000011,  ///< LB, LH, LW, LBU, LHU
    MISC_MEM  = 0b0001111,  ///< FENCE, FENCE.I
    OP_IMM    = 0b0010011,  ///< ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
    AUIPC     = 0b0010111,  ///< AUIPC
    STORE     = 0b0100011,  ///< SB, SH, SW
    AMO       = 0b0101111,  ///< LR.W, SC.W, AMO*
    OP        = 0b0110011,  ///< ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
    LUI       = 0b0110111,  ///< LUI
    BRANCH    = 0b1100011,  ///< BEQ, BNE, BLT, BGE, BLTU, BGEU
    JALR      = 0b1100111,  ///< JALR
    JAL       = 0b1101111,  ///< JAL
    SYSTEM    = 0b1110011,  ///< ECALL, EBREAK, CSR*, MRET
    VL        = 0b0000111,  ///< Vector loads (VLE32)
    VS        = 0b0100111,  ///< Vector stores (VSE32)
    OP_V      = 0b1010111,  ///< Vector ALU, VSETVLI
};

/// Decoded operation tag identifying the specific architectural operation.
enum class Op : u8
{
    // ── Loads (I-type) ──────────────────────────────────
    LB, LH, LW, LBU, LHU,

    // ── Stores (S-type) ─────────────────────────────────
    SB, SH, SW,

    // ── Branches (B-type) ───────────────────────────────
    BEQ, BNE, BLT, BGE, BLTU, BGEU,

    // ── Jumps ───────────────────────────────────────────
    JAL,   // J-type
    JALR,  // I-type

    // ── Upper immediate (U-type) ────────────────────────
    LUI, AUIPC,

    // ── Register-immediate ALU (I-type) ─────────────────
    ADDI, SLTI, SLTIU, XORI, ORI, ANDI,
    SLLI, SRLI, SRAI,

    // ── Register-register ALU (R-type) ──────────────────
    ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND,

    // ── M extension (R-type, funct7 = 0000001) ──────────
    MUL, MULH, MULHSU, MULHU,
    DIV, DIVU, REM, REMU,

    // ── A extension (R-type) ────────────────────────────
    LR_W, SC_W,
    AMOSWAP_W, AMOADD_W, AMOXOR_W, AMOAND_W, AMOOR_W,
    AMOMIN_W, AMOMAX_W, AMOMINU_W, AMOMAXU_W,

    // ── V extension subset ──────────────────────────────
    VSETVLI,
    VLE32, VSE32,
    VADD_VV, VADD_VX, VSUB_VV, VSUB_VX,
    VAND_VV, VAND_VX, VOR_VV, VOR_VX, VXOR_VV, VXOR_VX,
    VSLL_VX, VSRL_VX,
    VMSEQ_VV, VMSEQ_VX, VMSLT_VV, VMSLTU_VV,
    VMAND_MM, VMNAND_MM, VMANDN_MM, VMXOR_MM,
    VMOR_MM,VMNOR_MM, VMORN_MM, VMXNOR_MM,
    VREDSUM_VS,
    VMV_V_X, VMV_X_S,

    // ── System / synchronization ────────────────────────
    FENCE, FENCE_I, ECALL, EBREAK,
    CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI,
    MRET,

    INVALID,  ///< Sentinel for unrecognized or malformed instructions.
};

/// Instruction format, determining how the 32-bit word is partitioned.
enum class Format : u8
{
    R,  ///< Register-register.
    I,  ///< Immediate.
    S,  ///< Store.
    B,  ///< Branch.
    U,  ///< Upper immediate.
    J,  ///< Jump.
};

/**
 * @brief Fully decoded representation of a single instruction.
 *
 * @par Invariants
 * - @c imm is always sign-extended to 32 bits.
 * - For U-type, @c imm contains the full value (already shifted left by 12).
 * - For shift-immediates, @c imm contains only the 5-bit shamt.
 */
struct DecodedInst
{
    Op     op     = Op::INVALID;
    Format format = Format::R;

    reg_idx_t rd  = 0;  ///< Destination register [11:7].
    reg_idx_t rs1 = 0;  ///< Source register 1 [19:15].
    reg_idx_t rs2 = 0;  ///< Source register 2 [24:20].
    i32       imm = 0;  ///< Sign-extended immediate.

    u32    raw = 0;  ///< Original instruction word.
    addr_t pc  = 0;  ///< Address of this instruction.

    /// @name Classification helpers
    /// @{
    [[nodiscard]] constexpr bool writes_rd() const noexcept
    {
        switch (op)
        {
            case Op::SB: case Op::SH: case Op::SW:
            case Op::BEQ: case Op::BNE: case Op::BLT:
            case Op::BGE: case Op::BLTU: case Op::BGEU:
            case Op::FENCE: case Op::FENCE_I: case Op::ECALL: case Op::EBREAK:
            case Op::INVALID:
                return false;
            default:
                return rd != 0;
        }
    }

    [[nodiscard]] constexpr bool reads_rs1() const noexcept
    {
        switch (op)
        {
            case Op::LUI: case Op::AUIPC: case Op::JAL:
            case Op::FENCE: case Op::FENCE_I: case Op::ECALL: case Op::EBREAK:
            case Op::INVALID:
                return false;
            default:
                return true;
        }
    }

    [[nodiscard]] constexpr bool reads_rs2() const noexcept
    {
        return format == Format::R || format == Format::S || format == Format::B;
    }

    [[nodiscard]] constexpr bool is_load()   const noexcept { return op >= Op::LB && op <= Op::LHU; }
    [[nodiscard]] constexpr bool is_store()  const noexcept { return op >= Op::SB && op <= Op::SW; }
    [[nodiscard]] constexpr bool is_branch() const noexcept { return op >= Op::BEQ && op <= Op::BGEU; }
    [[nodiscard]] constexpr bool is_jump()   const noexcept { return op == Op::JAL || op == Op::JALR; }
    [[nodiscard]] constexpr bool is_atomic() const noexcept { return op >= Op::LR_W && op <= Op::AMOMAXU_W; }
    [[nodiscard]] constexpr bool is_vector() const noexcept { return op >= Op::VSETVLI && op <= Op::VMV_X_S; }
    /// @}

    /// Generate a standard RISC-V assembly string (e.g., "addi a0, sp, 16").
    [[nodiscard]] std::string disassemble() const;
};

/**
 * @brief Stateless instruction decoder.
 *
 * All methods are static. Transforms a raw 32-bit word into a DecodedInst.
 */
class Decoder {
public:
    Decoder() = delete;

    /**
     * @brief Decode a 32-bit RISC-V instruction word.
     * @param instruction Raw word fetched from memory.
     * @param pc Program counter of this instruction.
     * @return Decoded result. Check @c .op == Op::INVALID for decode failures.
     */
    [[nodiscard]] static DecodedInst decode(u32 instruction, addr_t pc = 0);

private:
    static DecodedInst decode_load(u32 inst, addr_t pc);
    static DecodedInst decode_store(u32 inst, addr_t pc);
    static DecodedInst decode_branch(u32 inst, addr_t pc);
    static DecodedInst decode_op_imm(u32 inst, addr_t pc);
    static DecodedInst decode_op(u32 inst, addr_t pc);
    static DecodedInst decode_amo(u32 inst, addr_t pc);
    static DecodedInst decode_vector(u32 inst, addr_t pc);
    static DecodedInst decode_system(u32 inst, addr_t pc);

    /// @name Immediate extraction (see Spec Section 2.1.3)
    /// @{
    [[nodiscard]] static i32 extract_i_imm(u32 inst);
    [[nodiscard]] static i32 extract_s_imm(u32 inst);
    [[nodiscard]] static i32 extract_b_imm(u32 inst);
    [[nodiscard]] static i32 extract_u_imm(u32 inst);
    [[nodiscard]] static i32 extract_j_imm(u32 inst);
    /// @}
};

/// Returns the lowercase mnemonic string for an @c Op tag (e.g., Op::LUI → "lui").
[[nodiscard]] const char* op_name(Op op);

} // namespace riscv
