/**
 * @file assembler.hpp
 * @brief Stateless two-pass RISC-V assembler.
 *
 * Assembles a text source string into a byte vector of machine
 * code suitable for loading directly into the simulator memory.
 *
 * @section asm_syntax Assembly Syntax
 *
 * @subsection asm_labels Labels
 * A label is an identifier followed by a colon at the start of a line.
 * Labels may optionally be followed by an instruction on the same line.
 * @code
 *   loop:
 *   start: addi t0, t0, 1
 * @endcode
 *
 * @subsection asm_comments Comments
 * Everything after:
 * - #,
 * - ;,
 * - or //
 * is ignored.
 *
 * @subsection asm_registers Registers
 * - Integer: @c x0-x31 or ABI names (@c zero, @c ra, @c sp, @c gp, @c tp,
 *   @c t0-t6, @c s0-s11, @c a0-a7, @c fp).
 * - Vector: @c v0-v31.
 * - CSR: named (@c mstatus, @c mie, @c mtvec, @c mscratch, @c mepc,
 *   @c mcause, @c mip, @c mhartid) or hexadecimal (@c 0x300).
 *   Range 0x000-0xFFF.
 *
 * @subsection asm_literals Numeric Literals
 * Decimal (@c 42), hexadecimal (@c 0xFF), or negative (@c -10, @c -0x10).
 *
 * @subsection asm_instructions Supported Instructions
 * @code
 * ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
 * ┃ Category    ┃ Mnemonics                                                  ┃
 * ┣━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
 * ┃ RV32I ALU   │ add, sub, sll, slt, sltu, xor, srl, sra, or, and           ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ RV32I Imm   │ addi, slti, sltiu, xori, ori, andi, slli, srli, srai       ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ Load/Store  │ lb, lh, lw, lbu, lhu, sb, sh, sw                           ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ Branch      │ beq, bne, blt, bge, bltu, bgeu                             ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ Jump        │ jal, jalr                                                  ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ Upper Imm   │ lui, auipc                                                 ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ System      │ ecall, ebreak, mret, fence                                 ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ CSR         │ csrrw, csrrs, csrrc, csrrwi, csrrsi, csrrci                ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ M Extension │ mul, mulh, mulhsu, mulhu, div, divu, rem, remu             ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ A Extension │ lr.w, sc.w, amoswap.w, amoadd.w, amoand.w, amoor.w,        ┃
 * ┃             │ amoxor.w, amomin.w, amomax.w, amominu.w, amomaxu.w         ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ V Extension │ vsetvli, vle32.v, vse32.v, vadd.vv, vsub.vv, vand.vv,      ┃
 * ┃             │ vor.vv, vxor.vv, vmseq.vv, vmslt.vv, vmsltu.vv,            ┃
 * ┃             │ vadd.vx, vsub.vx, vand.vx, vor.vx, vxor.vx, vsll.vx,       ┃
 * ┃             │ vsrl.vx, vmseq.vx, vmv.v.x, vmv.x.s, vredsum.vs,           ┃
 * ┃             │ vmand.mm, vmnand.mm, vmandn.mm, vmor.mm, vmnor.mm,         ┃
 * ┃             │ vmorn.mm, vmxor.mm, vmxnor.mm                              ┃
 * ┠─────────────┼────────────────────────────────────────────────────────────┨
 * ┃ Pseudo-ops  │ nop, li, mv, j, jr, ret, not, neg, beqz, bnez, seqz, snez, ┃
 * ┃             │ csrr, csrw, csrs, csrc, csrwi, csrsi, csrci,               ┃
 * ┃             │ vmmv.m, vmclr.m, vmset.m, vmnot.m                          ┃
 * ┗━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
 * @endcode
 *
 * @subsection asm_encoding Encoding
 * The assembler is two-pass: pass 1 collects labels and computes instruction
 * sizes, and pass 2 encodes. All instructions emit 4 bytes except @c li with
 * an immediate outside [-2048, 2047], which emits 8 (lui + addi).
 */

#pragma once

#include "types.hpp"

#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace riscv {

/// Single assembly error with source location.
struct AsmError
{
    u32         line; ///< 1-based line number.
    std::string message;
};

/// Result of an assembly operation.
struct AsmResult
{
    bool                  ok = false;
    std::vector<u8>       code;   ///< Assembled machine code (little-endian).
    std::vector<AsmError> errors;
    std::unordered_map<std::string, addr_t> labels; ///< Label → address map.

    /// @brief Total assembled code size in bytes.
    [[nodiscard]] u32 size() const noexcept { return static_cast<u32>(code.size()); }
};

/**
 * @brief Stateless two-pass RISC-V assembler.
 *
 * All parsing and encoding helpers are static. The public entry points
 * (@c assemble(), @c assemble_lines()) are const and produce an @c AsmResult
 * containing either the assembled byte stream or a list of errors.
 *
 * @par Parse helper convention
 * All @c parse_* methods return @c true on success (writing the result
 * to an output parameter) and @c false on failure (leaving the output
 * parameter unmodified).
 */
class Assembler {
public:
    /// Assemble a source string into machine code.
    [[nodiscard]] AsmResult assemble(std::string_view source) const;

    /// Assemble from a pre-split vector of source lines.
    [[nodiscard]] AsmResult assemble_lines(const std::vector<std::string>& lines) const;

private:
    /// Intermediate representation of a single source line.
    struct ParsedLine
    {
        std::string              label;    ///< Empty if no label on this line.
        std::string              mnemonic; ///< Lowercase. Empty if label-only or blank.
        std::vector<std::string> operands; ///< Trimmed operand strings.
        u32                      line_num = 0; ///< 1-based source line number.
    };

    /// Parse one source line into label, mnemonic, and operands.
    [[nodiscard]] static ParsedLine parse_line(std::string_view line, u32 line_num);

    /// Resolve an integer register name (x0-x31 or ABI) to its index.
    [[nodiscard]] static bool parse_register(std::string_view name, u32& out);

    /// Resolve a vector register name (v0-v31) to its index.
    [[nodiscard]] static bool parse_vreg(std::string_view name, u32& out);

    /// Parse a numeric literal or label reference into a signed immediate.
    [[nodiscard]] static bool parse_immediate(std::string_view token,
                                              const std::unordered_map<std::string, addr_t>& labels,
                                              i32& out);

    /// Parse @c offset(reg) or @c (reg). Rejects trailing characters.
    [[nodiscard]] static bool parse_mem_operand(std::string_view token,
                                                const std::unordered_map<std::string, addr_t>& labels,
                                                i32& offset, u32& reg);

    /// Resolve a CSR name or hex address (0x000-0xFFF) to its 12-bit address.
    [[nodiscard]] static bool parse_csr(std::string_view name, u32& out);

    /**
     * @brief Encode one parsed instruction into machine code words.
     * @return One or two 32-bit words on success, or an error message string.
     */
    [[nodiscard]] static std::variant<std::vector<u32>, std::string>
    encode_instruction(const ParsedLine& line,
                       const std::unordered_map<std::string, addr_t>& labels,
                       addr_t pc);

    /// Append a 32-bit word to a byte vector in little-endian order.
    static void emit_word(std::vector<u8>& code, u32 word);
};

} // namespace riscv
