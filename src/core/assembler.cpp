/**
 * @file assembler.cpp
 * @brief Two-pass RISC-V assembler implementation.
 */

#include "assembler.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <format>
#include <sstream>

namespace riscv {

// ── String utilities ───────────────────────────────────────────────────

static std::string_view trim(std::string_view s)
{
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.remove_suffix(1);
    return s;
}

static std::string to_lower(std::string_view s)
{
    std::string r(s);
    for (auto& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

// ── Register name table ────────────────────────────────────────────────

bool Assembler::parse_register(std::string_view name, u32& out)
{
    auto n = to_lower(name);

    // x0-x31
    if (n.size() >= 2 && n[0] == 'x')
    {
        u32 v = 0;
        auto [ptr, ec] = std::from_chars(n.data() + 1, n.data() + n.size(), v);

        if (ec == std::errc{} && ptr == n.data() + n.size() && v < 32)
        {
            out = v;
            return true;
        }
    }

    // ABI names
    static const std::unordered_map<std::string, u32> abi =
    {
        {"zero", 0}, {"ra", 1}, {"sp", 2}, {"gp", 3}, {"tp", 4},
        {"t0", 5}, {"t1", 6}, {"t2", 7},
        {"s0", 8}, {"fp", 8}, {"s1", 9},
        {"a0", 10}, {"a1", 11}, {"a2", 12}, {"a3", 13},
        {"a4", 14}, {"a5", 15}, {"a6", 16}, {"a7", 17},
        {"s2", 18}, {"s3", 19}, {"s4", 20}, {"s5", 21},
        {"s6", 22}, {"s7", 23}, {"s8", 24}, {"s9", 25},
        {"s10", 26}, {"s11", 27},
        {"t3", 28}, {"t4", 29}, {"t5", 30}, {"t6", 31},
    };
    auto it = abi.find(n);
    if (it != abi.end())
    { 
        out = it->second;
        return true;
    }

    return false;
}

bool Assembler::parse_vreg(std::string_view name, u32& out)
{
    auto n = to_lower(name);
    if (n.size() >= 2 && n[0] == 'v')
    {
        u32 v = 0;
        auto [ptr, ec] = std::from_chars(n.data() + 1, n.data() + n.size(), v);

        if (ec == std::errc{} && ptr == n.data() + n.size() && v < 32)
        {
            out = v;
            return true;
        }
    }
    return false;
}

bool Assembler::parse_csr(std::string_view name, u32& out)
{
    auto n = to_lower(name);
    static const std::unordered_map<std::string, u32> csrs =
    {
        {"mstatus",  0x300}, {"mie",  0x304}, {"mtvec",  0x305},
        {"mscratch", 0x340}, {"mepc", 0x341}, {"mcause", 0x342}, {"mip", 0x344},
        {"mhartid",  0xF14},
    };
    auto it = csrs.find(n);
    if (it != csrs.end())
    {
        out = it->second;
        return true;
    }

    // Numeric
    u32 v = 0;
    if (n.size() > 2 && n[0] == '0' && n[1] == 'x')
    {
        auto [ptr, ec] = std::from_chars(n.data() + 2, n.data() + n.size(), v, 16);

        if (ec == std::errc{} && ptr == n.data() + n.size() && v <= 0xFFF)
        {
            out = static_cast<u32>(v);
            return true;
        }
    }

    return false;
}

/**
 * Hex literals are parsed as u32 and reinterpreted as i32, so 0x8000'0000
 * yields INT32_MIN. Decimal literals are parsed directly as i32.
 */
bool Assembler::parse_immediate(std::string_view token,
                                const std::unordered_map<std::string, addr_t>& labels,
                                i32& out)
{
    auto s = trim(token);
    if (s.empty()) return false;

    // Label reference
    auto key = std::string(s);
    auto it = labels.find(key);
    if (it != labels.end())
    {
        out = static_cast<i32>(it->second);
        return true;
    }

    // Hex
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
    {
        u32 v = 0;
        auto [ptr, ec] = std::from_chars(s.data() + 2, s.data() + s.size(), v, 16);

        if (ec == std::errc{} && ptr == s.data() + s.size())
        {
            out = static_cast<i32>(v);
            return true;
        }
    }

    // Negative hex
    if (s.size() > 3 && s[0] == '-' && s[1] == '0' && (s[2] == 'x' || s[2] == 'X'))
    {
        u32 v = 0;
        auto [ptr, ec] = std::from_chars(s.data() + 3, s.data() + s.size(), v, 16);

        if (ec == std::errc{} && ptr == s.data() + s.size())
        {
            // Unsigned negation avoids UB on -INT32_MIN
            out = static_cast<i32>(0u - v);
            return true;
        }
    }

    // Decimal
    i32 v = 0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);

    if (ec == std::errc{} && ptr == s.data() + s.size())
    {
        out = v;
        return true;
    }

    return false;
}

bool Assembler::parse_mem_operand(std::string_view token,
                                  const std::unordered_map<std::string, addr_t>& labels,
                                  i32& offset, u32& reg)
{
    auto s = trim(token);
    auto paren = s.find('(');
    if (paren == std::string_view::npos) return false;
    auto close = s.find(')', paren);
    if (close == std::string_view::npos) return false;
    if (!trim(s.substr(close + 1)).empty()) return false;

    auto off_str = trim(s.substr(0, paren));
    auto reg_str = trim(s.substr(paren + 1, close - paren - 1));

    if (off_str.empty())
        offset = 0;
    else if (!parse_immediate(off_str, labels, offset))
        return false;

    return parse_register(reg_str, reg);
}

// ── Line parsing ───────────────────────────────────────────────────────

Assembler::ParsedLine Assembler::parse_line(std::string_view line, u32 line_num)
{
    ParsedLine result;
    result.line_num = line_num;

    // Strip comments
    for (size_t i = 0; i < line.size(); ++i)
    {
        if (line[i] == '#' || line[i] == ';')
        {
            line = line.substr(0, i);
            break;
        }
        if (i + 1 < line.size() && line[i] == '/' && line[i + 1] == '/')
        {
            line = line.substr(0, i);
            break;
        }
    }

    line = trim(line);
    if (line.empty()) return result;

    // Check for label
    auto colon = line.find(':');
    if (colon != std::string_view::npos)
    {
        result.label = std::string(trim(line.substr(0, colon)));

        line = trim(line.substr(colon + 1));
        if (line.empty()) return result;
    }

    // Split into mnemonic + operands
    size_t mend = 0;
    while (mend < line.size() && !std::isspace(static_cast<unsigned char>(line[mend])))
        ++mend;

    result.mnemonic = to_lower(line.substr(0, mend));
    line = trim(line.substr(mend));

    // Parse comma-separated operands
    while (!line.empty())
    {
        size_t end = 0;
        int paren_depth = 0;
        while (end < line.size())
        {
            if (line[end] == '(') paren_depth++;
            else if (line[end] == ')') paren_depth--;
            else if (line[end] == ',' && paren_depth == 0) break;
            end++;
        }

        auto op = trim(line.substr(0, end));
        if (!op.empty())
            result.operands.push_back(std::string(op));

        if (end < line.size()) line = trim(line.substr(end + 1));
        else break;
    }

    return result;
}

// ── Encoding helpers ───────────────────────────────────────────────────

static u32 enc_r(u32 funct7, u32 rs2, u32 rs1, u32 funct3, u32 rd, u32 opcode)
{
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode;
}

static u32 enc_i(u32 imm12, u32 rs1, u32 funct3, u32 rd, u32 opcode)
{
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode;
}

static u32 enc_s(u32 imm12, u32 rs2, u32 rs1, u32 funct3, u32 opcode)
{
    u32 imm_11_5 = (imm12 >> 5) & 0x7F;
    u32 imm_4_0  = imm12 & 0x1F;

    return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | opcode;
}

static u32 enc_b(i32 offset, u32 rs2, u32 rs1, u32 funct3, u32 opcode)
{
    u32 imm = static_cast<u32>(offset);
    u32 bit_12   = (imm >> 12) & 1;
    u32 bit_11   = (imm >> 11) & 1;
    u32 bit_10_5 = (imm >> 5) & 0x3F;
    u32 bit_4_1  = (imm >> 1) & 0xF;

    return (bit_12 << 31) | (bit_10_5 << 25) | (rs2 << 20) | (rs1 << 15)
         | (funct3 << 12) | (bit_4_1 << 8) | (bit_11 << 7) | opcode;
}

static u32 enc_u(u32 imm20, u32 rd, u32 opcode)
{
    return (imm20 & 0xFFFF'F000u) | (rd << 7) | opcode;
}

static u32 enc_j(i32 offset, u32 rd, u32 opcode)
{
    u32 imm = static_cast<u32>(offset);
    u32 bit_20    = (imm >> 20) & 1;
    u32 bit_19_12 = (imm >> 12) & 0xFF;
    u32 bit_11    = (imm >> 11) & 1;
    u32 bit_10_1  = (imm >> 1) & 0x3FF;

    return (bit_20 << 31) | (bit_10_1 << 21) | (bit_11 << 20)
         | (bit_19_12 << 12) | (rd << 7) | opcode;
}

// Opcodes
static constexpr u32 OP_LUI    = 0b0110111;
static constexpr u32 OP_AUIPC  = 0b0010111;
static constexpr u32 OP_JAL    = 0b1101111;
static constexpr u32 OP_JALR   = 0b1100111;
static constexpr u32 OP_BRANCH = 0b1100011;
static constexpr u32 OP_LOAD   = 0b0000011;
static constexpr u32 OP_STORE  = 0b0100011;
static constexpr u32 OP_OPIMM  = 0b0010011;
static constexpr u32 OP_OP     = 0b0110011;
static constexpr u32 OP_SYSTEM = 0b1110011;
static constexpr u32 OP_FENCE  = 0b0001111;
static constexpr u32 OP_AMO    = 0b0101111;
static constexpr u32 OP_V      = 0b1010111;

// ── Instruction encoding (pass 2) ──────────────────────────────────────

/**
 * Encoding is organized by instruction category, in this order:
 * RV32I pseudo-ops, R-type ALU, I-type ALU, LUI/AUIPC, loads,
 * stores, branches, JAL/JALR, system, CSR, A extension, V extension.
 */
std::variant<std::vector<u32>, std::string>
Assembler::encode_instruction(const ParsedLine& line,
                              const std::unordered_map<std::string, addr_t>& labels,
                              addr_t pc)
{
    const auto& m   = line.mnemonic;
    const auto& ops = line.operands;
    auto nops = static_cast<u32>(ops.size());

    // Helper lambdas for common operand patterns
    auto reg = [&](u32 idx, u32& r) -> bool
    {
        return idx < nops && parse_register(ops[idx], r);
    };
    auto vreg = [&](u32 idx, u32& r) -> bool
    {
        return idx < nops && parse_vreg(ops[idx], r);
    };
    auto imm = [&](u32 idx, i32& v) -> bool
    {
        return idx < nops && parse_immediate(ops[idx], labels, v);
    };
    auto mem = [&](u32 idx, i32& off, u32& r) -> bool
    {
        return idx < nops && parse_mem_operand(ops[idx], labels, off, r);
    };
    auto amo_funct7 = [&](u32 funct5, u32 aq = 0, u32 rl = 0)
    {
        return (funct5 << 2) | (aq << 1) | rl;
    };
    auto err = [&](const std::string& msg) -> std::variant<std::vector<u32>, std::string>
    {
        return msg;
    };
    auto ok1 = [](u32 word) -> std::variant<std::vector<u32>, std::string>
    {
        return std::vector<u32>{word};
    };
    auto ok2 = [](u32 w1, u32 w2) -> std::variant<std::vector<u32>, std::string>
    {
        return std::vector<u32>{w1, w2};
    };

    // ── RV32I Pseudo-instructions ───────────────────────

    if (m == "nop" && nops == 0)
    {
        if (nops != 0) return err("nop takes no operands");
        return ok1(enc_i(0, 0, 0, 0, OP_OPIMM));  // addi x0, x0, 0
    }
    if (m == "ret" && nops == 0)
    {
        if (nops != 0) return err("ret takes no operands");
        return ok1(enc_i(0, 1, 0, 0, OP_JALR));   // jalr x0, x1, 0
    }
    if (m == "mv")
    {
        u32 rd, rs;
        if (nops != 2 || !reg(0, rd) || !reg(1, rs)) return err("mv rd, rs");

        return ok1(enc_i(0, rs, 0, rd, OP_OPIMM));  // addi rd, rs, 0
    }
    if (m == "not")
    {
        u32 rd, rs;
        if (nops != 2 || !reg(0, rd) || !reg(1, rs)) return err("not rd, rs");

        return ok1(enc_i(0xFFF, rs, 4, rd, OP_OPIMM));  // xori rd, rs, -1
    }
    if (m == "neg")
    {
        u32 rd, rs;
        if (nops != 2 || !reg(0, rd) || !reg(1, rs)) return err("neg rd, rs");

        return ok1(enc_r(0b0100000, rs, 0, 0, rd, OP_OP));  // sub rd, x0, rs
    }
    if (m == "j")
    {
        i32 target;
        if (nops != 1 || !imm(0, target)) return err("j target");

        i32 offset = target - static_cast<i32>(pc);
        if (offset < -1048576 || offset > 1048575)
            return err("j offset out of range");
        if (offset & 1)
            return err("j target must be 2-aligned");

        return ok1(enc_j(offset, 0, OP_JAL));  // jal x0, offset
    }
    if (m == "jr")
    {
        u32 rs;
        if (nops != 1 || !reg(0, rs)) return err("jr rs");

        return ok1(enc_i(0, rs, 0, 0, OP_JALR));
    }
    if (m == "li")
    {
        u32 rd; i32 v;
        if (nops != 2 || !reg(0, rd) || !imm(1, v)) return err("li rd, imm");

        u32 uv = static_cast<u32>(v);

        if (v >= -2048 && v <= 2047)
            return ok1(enc_i(uv, 0, 0, rd, OP_OPIMM));  // addi rd, x0, imm

        // lui + addi for large values
        u32 hi = uv + 0x800;  // compensate for addi sign-extending bit 11
        u32 lo = uv & 0xFFF;

        return ok2(enc_u(hi, rd, OP_LUI),
                   enc_i(lo, rd, 0, rd, OP_OPIMM));
    }
    if (m == "beqz")
    {
        u32 rs; i32 target;
        if (nops != 2 || !reg(0, rs) || !imm(1, target)) return err("beqz rs, target");

        i32 offset = target - static_cast<i32>(pc);
        if (offset < -4096 || offset > 4095)
            return err("beqz offset out of range");
        if (offset & 1)
            return err("beqz target must be 2-byte aligned");

        return ok1(enc_b(offset, 0, rs, 0, OP_BRANCH));
    }
    if (m == "bnez")
    {
        u32 rs; i32 target;
        if (nops != 2 || !reg(0, rs) || !imm(1, target)) return err("bnez rs, target");

        i32 offset = target - static_cast<i32>(pc);
        if (offset < -4096 || offset > 4095)
            return err("bnez offset out of range");
        if (offset & 1)
            return err("bnez target must be 2-byte aligned");

        return ok1(enc_b(offset, 0, rs, 1, OP_BRANCH));
    }
    if (m == "seqz")
    {
        u32 rd, rs;
        if (nops != 2 || !reg(0, rd) || !reg(1, rs)) return err("seqz rd, rs");

        return ok1(enc_i(1, rs, 3, rd, OP_OPIMM));  // sltiu rd, rs, 1
    }
    if (m == "snez")
    {
        u32 rd, rs;
        if (nops != 2 || !reg(0, rd) || !reg(1, rs)) return err("snez rd, rs");

        return ok1(enc_r(0, rs, 0, 3, rd, OP_OP));  // sltu rd, x0, rs
    }

    // ── RV32IM ALU (R-type) ──────────────────────────────

    auto try_r_type = [&]() -> std::variant<std::vector<u32>, std::string>
    {
        struct REntry { const char* name; u32 funct7; u32 funct3; };
        static const REntry table[] =
        {
            {"add", 0b0000000, 0b000}, {"sub", 0b0100000, 0b000},
            {"sll", 0b0000000, 0b001}, {"slt", 0b0000000, 0b010}, {"sltu", 0b0000000, 0b011},
            {"xor", 0b0000000, 0b100}, {"srl", 0b0000000, 0b101}, {"sra",  0b0100000, 0b101},
            {"or",  0b0000000, 0b110}, {"and", 0b0000000, 0b111},
            // M extension
            {"mul",     0b0000001, 0b000}, {"mulh",  0b0000001, 0b001},
            {"mulhsu",  0b0000001, 0b010}, {"mulhu", 0b0000001, 0b011},
            {"div",     0b0000001, 0b100}, {"divu",  0b0000001, 0b101},
            {"rem",     0b0000001, 0b110}, {"remu",  0b0000001, 0b111},
        };
        for (auto& e : table)
            if (m == e.name)
            {
                u32 rd, rs1, rs2;
                if (nops != 3 || !reg(0, rd) || !reg(1, rs1) || !reg(2, rs2))
                    return err(std::format("{} rd, rs1, rs2", m));

                return ok1(enc_r(e.funct7, rs2, rs1, e.funct3, rd, OP_OP));
            }
        return err("");
    };

    auto r_result = try_r_type();
    if (auto* v = std::get_if<std::vector<u32>>(&r_result)) return *v;
    if (auto* s = std::get_if<std::string>(&r_result); s && !s->empty()) return *s;

    // ── RV32I ALU (I-type) ──────────────────────────────

    {
        struct IEntry { const char* name; u32 funct3; bool is_shift; u32 funct7; };
        static const IEntry table[] =
        {
            {"addi", 0b000, false, 0}, {"slti", 0b010, false, 0}, {"sltiu", 0b011, false, 0},
            {"xori", 0b100, false, 0}, {"ori",  0b110, false, 0}, {"andi",  0b111, false, 0},
            {"slli", 0b001, true, 0}, {"srli", 0b101, true, 0},
            {"srai", 0b101, true, 0b0100000},
        };
        for (auto& e : table)
            if (m == e.name)
            {
                u32 rd, rs1; i32 v;
                if (nops != 3 || !reg(0, rd) || !reg(1, rs1) || !imm(2, v))
                    return err(std::format("{} rd, rs1, imm", m));

                if (e.is_shift)
                {
                    if (v < 0 || v > 31) return err(std::format("{} shamt out of range", m));
                    u32 shamt = static_cast<u32>(v);
                    return ok1(enc_i(e.funct7 << 5 | shamt, rs1, e.funct3, rd, OP_OPIMM));
                }

                if (v < -2048 || v > 2047)
                    return err(std::format("{} immediate out of range", m));

                u32 uv = static_cast<u32>(v);

                return ok1(enc_i(uv, rs1, e.funct3, rd, OP_OPIMM));
            }
    }

    // ── LUI / AUIPC ─────────────────────────────────────

    if (m == "lui" || m == "auipc")
    {
        u32 rd; i32 v;
        if (nops != 2 || !reg(0, rd) || !imm(1, v))
            return err(std::format("{} rd, imm", m));
        if (v < -0x80000 || v > 0xFFFFF)
            return err(std::format("{} immediate out of range", m));

        u32 uv = static_cast<u32>(v);

        return ok1(enc_u(uv << 12, rd, (m == "lui") ? OP_LUI : OP_AUIPC));
    }

    // ── Loads ───────────────────────────────────────────

    {
        struct LEntry { const char* name; u32 funct3; };
        static const LEntry table[] =
        {
            {"lb", 0b000}, {"lh", 0b001}, {"lw", 0b010}, {"lbu", 0b100}, {"lhu", 0b101},
        };
        for (auto& e : table)
            if (m == e.name)
            {
                u32 rd, rs1; i32 off;
                if (nops != 2 || !reg(0, rd) || !mem(1, off, rs1))
                    return err(std::format("{} rd, offset(rs1)", m));
                if (off < -2048 || off > 2047)
                    return err(std::format("{} offset out of range", m));

                u32 uoff = static_cast<u32>(off);

                return ok1(enc_i(uoff, rs1, e.funct3, rd, OP_LOAD));
            }
    }

    // ── Stores ──────────────────────────────────────────

    {
        struct SEntry { const char* name; u32 funct3; };
        static const SEntry table[] = {{"sb", 0b000}, {"sh", 0b001}, {"sw", 0b010}};
        for (auto& e : table)
            if (m == e.name)
            {
                u32 rs2, rs1; i32 off;
                if (nops != 2 || !reg(0, rs2) || !mem(1, off, rs1))
                    return err(std::format("{} rs2, offset(rs1)", m));
                if (off < -2048 || off > 2047)
                    return err(std::format("{} offset out of range", m));

                return ok1(enc_s(static_cast<u32>(off), rs2, rs1, e.funct3, OP_STORE));
            }
    }

    // ── Branches ────────────────────────────────────────

    {
        struct BEntry { const char* name; u32 funct3; };
        static const BEntry table[] =
        {
            {"beq", 0b000}, {"bne",  0b001}, {"blt",  0b100},
            {"bge", 0b101}, {"bltu", 0b110}, {"bgeu", 0b111},
        };
        for (auto& e : table)
            if (m == e.name)
            {
                u32 rs1, rs2; i32 target;
                if (nops != 3 || !reg(0, rs1) || !reg(1, rs2) || !imm(2, target))
                    return err(std::format("{} rs1, rs2, target", m));

                i32 offset = target - static_cast<i32>(pc);
                if (offset < -4096 || offset > 4095)
                    return err(std::format("{} offset out of range", m));
                if (offset & 1)
                    return err(std::format("{} offset must be 2-byte aligned", m));

                return ok1(enc_b(offset, rs2, rs1, e.funct3, OP_BRANCH));
            }
    }

    // ── JAL / JALR ──────────────────────────────────────

    if (m == "jal")
    {
        u32 rd; i32 target;
        if (nops == 1 && imm(0, target))
            rd = 1;
        else if (nops == 2 && reg(0, rd) && imm(1, target))
            ;
        else
            return err("jal target OR jal rd, target");

        i32 offset = target - static_cast<i32>(pc);
        if (offset < -1048576 || offset > 1048575)
            return err("jal offset out of range");
        if (offset & 1)
            return err("jal target must be 2-byte aligned");

        return ok1(enc_j(offset, rd, OP_JAL));
    }
    if (m == "jalr")
    {
        u32 rd, rs1; i32 off;
        if (nops != 2 || !reg(0, rd) || !mem(1, off, rs1))
            return err("jalr rd, offset(rs1)");
        if (off < -2048 || off > 2047)
            return err("jalr offset out of range");

        u32 uoff = static_cast<u32>(off);

        return ok1(enc_i(uoff, rs1, 0, rd, OP_JALR));
    }

    // ── System instructions ─────────────────────────────

    if (m == "ecall")   { if (nops != 0) return err("ecall takes no operands");  return ok1(0x0000'0073); }
    if (m == "ebreak")  { if (nops != 0) return err("ebreak takes no operands"); return ok1(0x0010'0073); }
    if (m == "mret")    { if (nops != 0) return err("mret takes no operands");   return ok1(0x3020'0073); }
    if (m == "fence")   { if (nops != 0) return err("fence takes no operands");  return ok1(0x0FF0'000F); }
    if (m == "fence.i") { if (nops != 0) return err("fence.i takes no operands"); return ok1(0x0000'100F); }

    // ── CSR instructions ────────────────────────────────

    {
        struct CEntry { const char* name; u32 funct3; bool is_imm; };
        static const CEntry table[] =
        {
            {"csrrw",  0b001, false}, {"csrrs",  0b010, false}, {"csrrc",  0b011, false},
            {"csrrwi", 0b101, true},  {"csrrsi", 0b110, true},  {"csrrci", 0b111, true},
        };
        for (auto& e : table)
            if (m == e.name)
            {
                if (nops != 3) return err(std::format("{} requires 3 operands", m));

                u32 rd, csr_addr;
                if (!reg(0, rd)) return err(std::format("{} rd, csr, rs1/imm", m));
                if (!parse_csr(ops[1], csr_addr)) return err(std::format("unknown CSR: {}", ops[1]));

                if (e.is_imm)
                {
                    i32 zimm;
                    if (!imm(2, zimm)) return err(std::format("{} rd, csr, zimm", m));
                    if (zimm < 0 || zimm > 31)
                        return err(std::format("{} zimm out of range", m));

                    return ok1(enc_i(csr_addr,
                                     static_cast<u32>(zimm),
                                     e.funct3, rd, OP_SYSTEM));
                }
                u32 rs1;
                if (!reg(2, rs1)) return err(std::format("{} rd, csr, rs1", m));

                return ok1(enc_i(csr_addr, rs1, e.funct3, rd, OP_SYSTEM));
            }
    }

    // ── CSR pseudo-instructions ─────────────────────────
    // csrr  rd, csr    -> csrrs  rd, csr, x0
    // csrw  csr, rs    -> csrrw  x0, csr, rs
    // csrs  csr, rs    -> csrrs  x0, csr, rs
    // csrc  csr, rs    -> csrrc  x0, csr, rs
    // csrwi csr, zimm  -> csrrwi x0, csr, zimm   (likewise csrsi / csrci)

    if (m == "csrr")
    {
        u32 rd, csr_addr;
        if (nops != 2 || !reg(0, rd)) return err("csrr rd, csr");
        if (!parse_csr(ops[1], csr_addr)) return err(std::format("unknown CSR: {}", ops[1]));
        return ok1(enc_i(csr_addr, 0, 0b010, rd, OP_SYSTEM));
    }
    {
        struct PEntry { const char* name; u32 funct3; bool is_imm; };
        static const PEntry table[] =
        {
            {"csrw",  0b001, false}, {"csrs",  0b010, false}, {"csrc",  0b011, false},
            {"csrwi", 0b101, true},  {"csrsi", 0b110, true},  {"csrci", 0b111, true},
        };
        for (auto& e : table)
            if (m == e.name)
            {
                if (nops != 2) return err(std::format("{} requires 2 operands", m));

                u32 csr_addr;
                if (!parse_csr(ops[0], csr_addr))
                    return err(std::format("unknown CSR: {}", ops[0]));

                if (e.is_imm)
                {
                    i32 zimm;
                    if (!imm(1, zimm)) return err(std::format("{} csr, zimm", m));
                    if (zimm < 0 || zimm > 31)
                        return err(std::format("{} zimm out of range", m));
                    return ok1(enc_i(csr_addr, static_cast<u32>(zimm), e.funct3, 0, OP_SYSTEM));
                }
                u32 rs1;
                if (!reg(1, rs1)) return err(std::format("{} csr, rs1", m));
                return ok1(enc_i(csr_addr, rs1, e.funct3, 0, OP_SYSTEM));
            }
    }

    // ── A extension ─────────────────────────────────────

    if (m == "lr.w")
    {
        u32 rd; i32 off; u32 rs1;
        if (nops != 2 || !reg(0, rd) || !parse_mem_operand(ops[1], labels, off, rs1))
            return err("lr.w rd, (rs1)");
        if (off != 0)
            return err("lr.w does not accept an offset");

        return ok1(enc_r(amo_funct7(0b00010), 0, rs1, 2, rd, OP_AMO));
    }
    if (m == "sc.w")
    {
        u32 rd, rs1, rs2; i32 off;
        if (nops != 3 || !reg(0, rd) || !reg(1, rs2)
            || !parse_mem_operand(ops[2], labels, off, rs1))
            return err("sc.w rd, rs2, (rs1)");
        if (off != 0)
            return err("sc.w does not accept an offset");

        return ok1(enc_r(amo_funct7(0b00011), rs2, rs1, 2, rd, OP_AMO));
    }
    {
        struct AEntry { const char* name; u32 funct5; };
        static const AEntry amo_table[] =
        {
            {"amoswap.w", 0b00001}, {"amoadd.w",  0b00000},
            {"amoand.w",  0b01100}, {"amoor.w",   0b01000},
            {"amoxor.w",  0b00100}, {"amomin.w",  0b10000},
            {"amomax.w",  0b10100}, {"amominu.w", 0b11000},
            {"amomaxu.w", 0b11100},
        };
        for (auto& e : amo_table)
            if (m == e.name)
            {
                u32 rd, rs2, rs1; i32 off;
                if (nops != 3 || !reg(0, rd) || !reg(1, rs2)
                    || !parse_mem_operand(ops[2], labels, off, rs1))
                    return err(std::format("{} rd, rs2, (rs1)", m));
                if (off != 0)
                    return err(std::format("{} does not accept an offset", m));

                return ok1(enc_r(amo_funct7(e.funct5), rs2, rs1, 2, rd, OP_AMO));
            }
    }

    // ── V extension ─────────────────────────────────────

    if (m == "vsetvli")
    {
        u32 rd, rs1; i32 vtypei;
        if (nops != 3 || !reg(0, rd) || !reg(1, rs1) || !imm(2, vtypei))
            return err("vsetvli rd, rs1, vtypei");
        if (vtypei < 0 || vtypei > 0x7FF)
            return err("vsetvli vtypei out of range (0-0x7FF)");

        u32 enc = (0u << 31) | (static_cast<u32>(vtypei) << 20)
                | (rs1 << 15) | (0b111u << 12) | (rd << 7) | OP_V;
        return ok1(enc);
    }
    // ── VLE32.V / VSE32.V ───────────────────────────────
    if (m == "vle32.v")
    {
        u32 vd, rs1; i32 off;
        if (nops != 2 || !vreg(0, vd) || !mem(1, off, rs1))
            return err("vle32.v vd, (rs1)");
        if (off != 0)
            return err("vle32.v does not accept an offset");

        u32 enc = (0b000u << 29) | (0u << 28) | (0b00u << 26) | (1u << 25)
                | (0b00000u << 20) | (rs1 << 15) | (0b110u << 12) | (vd << 7) | 0b000'0111u;
        return ok1(enc);
    }
    if (m == "vse32.v")
    {
        u32 vs3, rs1; i32 off;
        if (nops != 2 || !vreg(0, vs3) || !mem(1, off, rs1))
            return err("vse32.v vs3, (rs1)");
        if (off != 0)
            return err("vse32.v does not accept an offset");

        u32 enc = (0b000u << 29) | (0u << 28) | (0b00u << 26) | (1u << 25)
                | (0b00000u << 20) | (rs1 << 15) | (0b110u << 12) | (vs3 << 7) | 0b010'0111u;
        return ok1(enc);
    }
    // ── .VV / .MM instructions ──────────────────────────
    {
        struct VVEntry { const char* name; u32 funct6; u32 funct3; };
        static const VVEntry vv_table[] =
        {
            {"vadd.vv",   0b000000, 0b000}, {"vsub.vv",   0b000010, 0b000},
            {"vand.vv",   0b001001, 0b000}, {"vor.vv",    0b001010, 0b000},
            {"vxor.vv",   0b001011, 0b000},
            {"vmseq.vv",  0b011000, 0b000}, {"vmslt.vv",  0b011011, 0b000},
            {"vmsltu.vv", 0b011010, 0b000},
            {"vmand.mm",  0b011001, 0b010}, {"vmnand.mm", 0b011101, 0b010},
            {"vmandn.mm", 0b011000, 0b010},
            {"vmor.mm",   0b011010, 0b010}, {"vmnor.mm",  0b011110, 0b010},
            {"vmorn.mm",  0b011100, 0b010},
            {"vmxor.mm",  0b011011, 0b010}, {"vmxnor.mm", 0b011111, 0b010},
        };
        for (auto& e : vv_table)
            if (m == e.name)
            {
                u32 vd, vs2, vs1;
                if (nops != 3 || !vreg(0, vd) || !vreg(1, vs2) || !vreg(2, vs1))
                    return err(std::format("{} vd, vs2, vs1", m));

                u32 enc = (e.funct6 << 26) | (1u << 25) | (vs2 << 20)
                        | (vs1 << 15) | (e.funct3 << 12) | (vd << 7) | OP_V;
                return ok1(enc);
            }
    }
    // ── .VX instructions ────────────────────────────────
    {
        struct VXEntry { const char* name; u32 funct6; };
        static const VXEntry vx_table[] =
        {
            {"vadd.vx",  0b000000}, {"vsub.vx", 0b000010},
            {"vand.vx",  0b001001}, {"vor.vx",  0b001010}, {"vxor.vx", 0b001011},
            {"vsll.vx",  0b100101}, {"vsrl.vx", 0b101000},
            {"vmseq.vx", 0b011000},
        };
        for (auto& e : vx_table)
            if (m == e.name)
            {
                u32 vd, vs2, rs1;
                if (nops != 3 || !vreg(0, vd) || !vreg(1, vs2) || !reg(2, rs1))
                    return err(std::format("{} vd, vs2, rs1", m));

                u32 enc = (e.funct6 << 26) | (1u << 25) | (vs2 << 20)
                        | (rs1 << 15) | (0b100u << 12) | (vd << 7) | OP_V;
                return ok1(enc);
            }
    }
    // ── Mask pseudo-instructions ────────────────────────
    if (m == "vmmv.m")
    {
        u32 vd, vs;
        if (nops != 2 || !vreg(0, vd) || !vreg(1, vs)) return err("vmmv.m vd, vs");

        // vmand.mm vd, vs, vs
        u32 enc = (0b011001 << 26) | (1u << 25) | (vs << 20) | (vs << 15)
                | (0b010u << 12) | (vd << 7) | OP_V;
        return ok1(enc);
    }
    if (m == "vmclr.m")
    {
        u32 vd;
        if (nops != 1 || !vreg(0, vd)) return err("vmclr.m vd");

        // vmxor.mm vd, vd, vd
        u32 enc = (0b011011 << 26) | (1u << 25) | (vd << 20) | (vd << 15)
                | (0b010u << 12) | (vd << 7) | OP_V;
        return ok1(enc);
    }
    if (m == "vmset.m")
    {
        u32 vd;
        if (nops != 1 || !vreg(0, vd))
            return err("vmset.m vd");

        // vmxnor.mm vd, vd, vd
        u32 enc = (0b011111 << 26) | (1u << 25) | (vd << 20) | (vd << 15)
                | (0b010u << 12) | (vd << 7) | OP_V;
        return ok1(enc);
    }
    if (m == "vmnot.m")
    {
        u32 vd, vs;
        if (nops != 2 || !vreg(0, vd) || !vreg(1, vs))
            return err("vmnot.m vd, vs");

        // vmnand.mm vd, vs, vs    
        u32 enc = (0b011101 << 26) | (1u << 25) | (vs << 20) | (vs << 15)
                | (0b010u << 12) | (vd << 7) | OP_V;
        return ok1(enc);
    }
    // ── Reduction and move instructions ─────────────────
    if (m == "vredsum.vs")
    {
        u32 vd, vs2, vs1;
        if (nops != 3 || !vreg(0, vd) || !vreg(1, vs2) || !vreg(2, vs1))
            return err("vredsum.vs vd, vs2, vs1");

        u32 enc = (0b000000 << 26) | (1u << 25) | (vs1 << 20) | (vs2 << 15)
                | (0b010 << 12) | (vd << 7) | OP_V;
        return ok1(enc);
    }
    if (m == "vmv.v.x")
    {
        u32 vd, rs1;
        if (nops != 2 || !vreg(0, vd) || !reg(1, rs1))
            return err("vmv.v.x vd, rs1");

        u32 enc = (0b010111u << 26) | (1u << 25) | (0u << 20) | (rs1 << 15)
                | (0b100u << 12) | (vd << 7) | OP_V;
        return ok1(enc);
    }
    if (m == "vmv.x.s")
    {
        u32 rd, vs2;
        if (nops != 2 || !reg(0, rd) || !vreg(1, vs2))
            return err("vmv.x.s rd, vs2");

        u32 enc = (0b010000u << 26) | (1u << 25) | (vs2 << 20) | (0u << 15)
                | (0b010u << 12) | (rd << 7) | OP_V;
        return ok1(enc);
    }

    return err(std::format("unknown instruction: {}", m));
}

// ── Emit helper ────────────────────────────────────────────────────────

void Assembler::emit_word(std::vector<u8>& code, u32 word)
{
    code.push_back(static_cast<u8>(word));
    code.push_back(static_cast<u8>(word >> 8));
    code.push_back(static_cast<u8>(word >> 16));
    code.push_back(static_cast<u8>(word >> 24));
}

// ── Two-pass assembly ──────────────────────────────────────────────────

AsmResult Assembler::assemble(std::string_view source) const
{
    std::vector<std::string> lines;
    std::istringstream stream{std::string(source)};
    std::string line;

    while (std::getline(stream, line))
        lines.push_back(std::move(line));

    return assemble_lines(lines);
}

/**
 * Pass 1: collect labels; compute PC for each instruction
 * Pass 2: encode instructions using the completed label map.
 *
 * li with an unresolved immediate is conservatively sized at 8 bytes
 * (lui + addi) to guarantee correct label offsets.
 */
AsmResult Assembler::assemble_lines(const std::vector<std::string>& lines) const
{
    AsmResult result;

    // ── Pass 1 ──────────────────────────────────────────

    std::vector<ParsedLine> parsed;
    parsed.reserve(lines.size());
    addr_t pc = 0;

    for (u32 i = 0; i < lines.size(); ++i)
    {
        auto pl = parse_line(lines[i], i + 1);
        parsed.push_back(pl);

        if (!pl.label.empty())
        {
            if (result.labels.count(pl.label))
                result.errors.push_back({pl.line_num, 
                                         std::format("duplicate label: {}", pl.label)});
            else
                result.labels[pl.label] = pc;
        }

        if (pl.mnemonic.empty()) continue;

        if (pl.mnemonic == "li")
        {
            i32 v = 0;
            bool resolved = pl.operands.size() >= 2
                && parse_immediate(pl.operands[1], result.labels, v);

            pc += (!resolved || v < -2048 || v > 2047) ? 8 : 4;
        }
        else
            pc += 4;
    }

    if (!result.errors.empty())
        return result;

    // ── Pass 2 ──────────────────────────────────────────

    pc = 0;
    for (auto& pl : parsed)
    {
        if (pl.mnemonic.empty()) continue;

        auto enc = encode_instruction(pl, result.labels, pc);
        if (auto* words = std::get_if<std::vector<u32>>(&enc))
            for (u32 w : *words)
            {
                emit_word(result.code, w);
                pc += 4;
            }
        else
        {
            auto& msg = std::get<std::string>(enc);
            result.errors.push_back({pl.line_num, msg});
        }
    }

    result.ok = result.errors.empty();
    return result;
}

} // namespace riscv
