/**
 * @file test_csr.cpp
 * @brief Tests for CSR instructions.
 *
 * @par Sections
 * @code
 *   1 : Decoder
 *   2 : Assembler encodings for CSR pseudo-instructions
 *   3 : Read/modify/write semantics
 *   4 : Field masks, WARL, and read-only CSRs
 *   5 : Hazards and ordering
 *   6 : mret
 * @endcode
 */

#include "machine_harness.hpp"
#include "test_framework.hpp"
#include "test_utils.hpp"

using namespace riscv;
using namespace harness;

namespace {

/// Assemble a single line and return its first instruction word
/// (0 on error).
u32 encode(const std::string& line)
{
    auto r = Assembler().assemble(line);
    if (!r.ok || r.code.size() < 4) return 0;
    return u32{r.code[0]} | (u32{r.code[1]} << 8)
        | (u32{r.code[2]} << 16) | (u32{r.code[3]} << 24);
}

/// Run @p program on every single-hart model.
bool run_everywhere(const std::string& program,
                    const std::function<bool(Machine&)>& check)
{
    return for_each_model(kSingleHartModels, [&](Machine& m, Model)
    {
        ASSERT(m.load({{0x0, program}}));
        ASSERT(m.run_to_halt());
        return check(m);
    });
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════
//  1. Decoder
// ═══════════════════════════════════════════════════════════════════════

TEST(csr_decode_csrrw)
{
    auto inst = Decoder::decode(0x3001'10f3u); // csrrw x1, mstatus, x2
    ASSERT_EQ(inst.op, Op::CSRRW);
    ASSERT_EQ(inst.rd, 1);
    ASSERT_EQ(inst.rs1, 2);
    constexpr u32 mstatus = 0x300u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mstatus);
    return true;
}

TEST(csr_decode_csrrs)
{
    auto inst = Decoder::decode(0x3043'22f3u); // csrrs x5, mie, x6
    ASSERT_EQ(inst.op, Op::CSRRS);
    ASSERT_EQ(inst.rd, 5);
    ASSERT_EQ(inst.rs1, 6);
    constexpr u32 mie = 0x304u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mie);
    return true;
}

TEST(csr_decode_csrrc)
{
    auto inst = Decoder::decode(0x3055'34f3u); // csrrc x9, mtvec, x10
    ASSERT_EQ(inst.op, Op::CSRRC);
    ASSERT_EQ(inst.rd, 9);
    ASSERT_EQ(inst.rs1, 10);
    constexpr u32 mtvec = 0x305u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mtvec);
    return true;
}

TEST(csr_decode_csrrwi)
{
    auto inst = Decoder::decode(0x3412'd6f3u); // csrrwi x13, mepc, 5
    ASSERT_EQ(inst.op, Op::CSRRWI);
    ASSERT_EQ(inst.rd, 13);
    ASSERT_EQ(inst.rs1, 5);
    constexpr u32 mepc = 0x341u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mepc);
    return true;
}

TEST(csr_decode_csrrsi)
{
    auto inst = Decoder::decode(0x3421'e7f3u); // csrrsi x15, mcause, 3
    ASSERT_EQ(inst.op, Op::CSRRSI);
    ASSERT_EQ(inst.rd, 15);
    ASSERT_EQ(inst.rs1,3);
    constexpr u32 mcause = 0x342u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mcause);
    return true;
}

TEST(csr_decode_csrrci)
{
    auto inst = Decoder::decode(0x3443'f8f3u); // csrrci x17, mip, 7
    ASSERT_EQ(inst.op, Op::CSRRCI);
    ASSERT_EQ(inst.rd, 17);
    ASSERT_EQ(inst.rs1, 7);
    constexpr u32 mip = 0x344u;
    ASSERT_EQ(static_cast<u32>(inst.imm), mip);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Assembler encodings
// ═══════════════════════════════════════════════════════════════════════

TEST(csrx_pseudo_encodings)
{
    ASSERT_HEX_EQ(encode("csrr a0, mstatus"),   0x3000'2573u);  // csrrs a0, mstatus, x0
    ASSERT_HEX_EQ(encode("csrw mtvec, t0"),     0x3052'9073u);  // csrrw x0, mtvec, t0
    ASSERT_HEX_EQ(encode("csrs mie, t0"),       0x3042'A073u);  // csrrs x0, mie, t0
    ASSERT_HEX_EQ(encode("csrc mie, t0"),       0x3042'B073u);  // csrrc x0, mie, t0
    ASSERT_HEX_EQ(encode("csrwi mscratch, 5"),  0x3402'D073u);  // csrrwi x0, mscratch, 5
    ASSERT_HEX_EQ(encode("csrsi mstatus, 8"),   0x3004'6073u);  // csrrsi x0, mstatus, 8
    ASSERT_HEX_EQ(encode("csrci mstatus, 8"),   0x3004'7073u);  // csrrci x0, mstatus, 8
    ASSERT_HEX_EQ(encode("csrr t1, mhartid"),   0xF140'2373u);
    return true;
}

TEST(csrx_pseudo_errors)
{
    Assembler as;
    ASSERT(!as.assemble("csrr a0").ok);
    ASSERT(!as.assemble("csrw bogus, t0").ok);
    ASSERT(!as.assemble("csrwi mscratch, 32").ok);  // zimm is 5 bits
    ASSERT(!as.assemble("csrsi mstatus, -1").ok);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  3. Read/modify/write semantics
// ═══════════════════════════════════════════════════════════════════════

TEST(csrx_csrrw_returns_old_and_writes_new)
{
    return run_everywhere(R"(
        li    t0, 0x123
        csrrw t1, mscratch, t0     # t1 = 0, mscratch = 0x123
        li    t0, 0x456
        csrrw t2, mscratch, t0     # t2 = 0x123
        csrr  t3, mscratch         # t3 = 0x456
        ebreak
    )", [](Machine& m)
    {
        ASSERT_EQ(m.reg(0, 6), 0u);
        ASSERT_HEX_EQ(m.reg(0, 7), 0x123u);
        ASSERT_HEX_EQ(m.reg(0, 28), 0x456u);
        return true;
    });
}

TEST(csrx_csrrs_csrrc_set_and_clear_bits)
{
    return run_everywhere(R"(
        li    t0, 0xF0
        csrw  mscratch, t0
        li    t0, 0x0F
        csrrs t1, mscratch, t0     # t1 = 0xF0, mscratch = 0xFF
        li    t0, 0x3C
        csrrc t2, mscratch, t0     # t2 = 0xFF, mscratch = 0xC3
        csrr  t3, mscratch
        ebreak
    )", [](Machine& m)
    {
        ASSERT_HEX_EQ(m.reg(0, 6), 0xF0u);
        ASSERT_HEX_EQ(m.reg(0, 7), 0xFFu);
        ASSERT_HEX_EQ(m.reg(0, 28), 0xC3u);
        return true;
    });
}

TEST(csrx_immediate_forms)
{
    return run_everywhere(R"(
        csrrwi t1, mscratch, 31    # t1 = 0, mscratch = 31
        csrrsi t2, mscratch, 0     # zimm = 0: read, no write
        csrrci t3, mscratch, 5     # t3 = 31, mscratch = 26
        csrrsi t4, mscratch, 1     # t4 = 26, mscratch = 27
        csrr   t5, mscratch
        ebreak
    )", [](Machine& m)
    {
        ASSERT_EQ(m.reg(0, 6), 0u);
        ASSERT_EQ(m.reg(0, 7), 31u);
        ASSERT_EQ(m.reg(0, 28), 31u);
        ASSERT_EQ(m.reg(0, 29), 26u);
        ASSERT_EQ(m.reg(0, 30), 27u);
        return true;
    });
}

TEST(csrx_rd_x0_still_writes_and_rs1_x0_does_not)
{
    return run_everywhere(R"(
        li    t0, 77
        csrrw zero, mscratch, t0   # write happens; x0 stays 0
        csrrs t1, mscratch, zero   # read only
        csrrc t2, mscratch, zero   # read only
        csrr  t3, mscratch
        ebreak
    )", [](Machine& m)
    {
        ASSERT_EQ(m.reg(0, 0), 0u);
        ASSERT_EQ(m.reg(0, 6), 77u);
        ASSERT_EQ(m.reg(0, 7), 77u);
        ASSERT_EQ(m.reg(0, 28), 77u);
        return true;
    });
}

TEST(csrx_unimplemented_csr_reads_zero)
{
    return run_everywhere(R"(
        li    t0, -1
        csrw  0x7C0, t0
        csrr  t1, 0x7C0
        ebreak
    )", [](Machine& m)
    {
        ASSERT_EQ(m.reg(0, 6), 0u);
        return true;
    });
}

// ═══════════════════════════════════════════════════════════════════════
//  4. Field masks, WARL, read-only
// ═══════════════════════════════════════════════════════════════════════

TEST(csrx_mstatus_and_mie_write_masks)
{
    return run_everywhere(R"(
        li    t0, -1
        csrw  mie, t0       # interrupts stay disabled globally
        csrr  t2, mie
        csrw  mie, zero
        csrw  mstatus, t0
        csrr  t1, mstatus
        csrw  mstatus, zero
        ebreak
    )", [](Machine& m)
    {
        ASSERT_HEX_EQ(m.reg(0, 6), MStatus::MIE | MStatus::MPIE);  // 0x88
        ASSERT_HEX_EQ(m.reg(0, 7), 0x888u);                        // MSIE | MTIE | MEIE
        return true;
    });
}

TEST(csrx_mip_is_read_only_and_reflects_lines)
{
    // Writing mip cannot fake an interrupt; arming the CLINT comparator
    // makes MTIP visible through csrr
    return run_everywhere(R"(
        li    t0, -1
        csrw  mip, t0
        csrr  t1, mip           # 0: nothing pending, writes ignored
        li    t0, 0x02004000
        sw    zero, 4(t0)
        sw    zero, 0(t0)       # mtimecmp = 0 -> MTIP
        nop
        nop
        csrr  t2, mip
        ebreak
    )", [](Machine& m)
    {
        ASSERT_EQ(m.reg(0, 6), 0u);
        ASSERT_HEX_EQ(m.reg(0, 7), MInterrupt::MTIP);
        return true;
    });
}

TEST(csrx_mtvec_mode_is_warl)
{
    return run_everywhere(R"(
        li    t0, 0x403     # MODE = 3 (reserved) -> DIRECT
        csrw  mtvec, t0
        csrr  t1, mtvec
        li    t0, 0x401     # MODE = 1 (vectored) is kept
        csrw  mtvec, t0
        csrr  t2, mtvec
        ebreak
    )", [](Machine& m)
    {
        ASSERT_HEX_EQ(m.reg(0, 6), 0x400u);
        ASSERT_HEX_EQ(m.reg(0, 7), 0x401u);
        return true;
    });
}

TEST(csrx_mepc_bit0_is_zero)
{
    return run_everywhere(R"(
        li    t0, 0x1235
        csrw  mepc, t0
        csrr  t1, mepc
        ebreak
    )", [](Machine& m)
    {
        ASSERT_HEX_EQ(m.reg(0, 6), 0x1234u);
        return true;
    });
}

TEST(csrx_mhartid_is_read_only)
{
    return run_everywhere(R"(
        li    t0, 5
        csrw  mhartid, t0
        csrr  t1, mhartid
        ebreak
    )", [](Machine& m)
    {
        ASSERT_EQ(m.reg(0, 6), 0u);
        return true;
    });
}

// ═══════════════════════════════════════════════════════════════════════
//  5. Hazards and ordering
// ═══════════════════════════════════════════════════════════════════════

TEST(csrx_csr_result_feeds_next_instruction)
{
    // Back-to-back RAW dependencies through CSR results, and a CSR write
    // whose source was produced by the immediately preceding instruction
    return run_everywhere(R"(
        li    t0, 40
        csrw  mscratch, t0
        csrr  t1, mscratch
        addi  t2, t1, 2            # depends on csrr (EX->EX)
        add   t3, t2, t1           # depends on both
        addi  t4, t3, 1
        csrw  mscratch, t4         # source produced by previous instruction
        csrr  t5, mscratch
        lw    t6, 0(zero)          # load ...
        csrw  mscratch, t6         # ... then load-use into a CSR write
        csrr  a0, mscratch
        ebreak
    )", [](Machine& m)
    {
        ASSERT_EQ(m.reg(0, 6), 40u);
        ASSERT_EQ(m.reg(0, 7), 42u);
        ASSERT_EQ(m.reg(0, 28), 82u);
        ASSERT_EQ(m.reg(0, 30), 83u);
        // 0(zero) holds the first instruction word
        // (li t0, 40 = addi t0, x0, 40)
        ASSERT_HEX_EQ(m.reg(0, 10), 0x02800293u);
        return true;
    });
}

TEST(csrx_swap_sequence_in_order)
{
    // A chain of csrrw swaps must observe each previous write
    // in program order
    return run_everywhere(R"(
        li    a0, 1
        li    a1, 2
        li    a2, 3
        csrrw a3, mscratch, a0     # a3 = 0
        csrrw a4, mscratch, a1     # a4 = 1
        csrrw a5, mscratch, a2     # a5 = 2
        csrr  a6, mscratch         # a6 = 3
        ebreak
    )", [](Machine& m)
    {
        ASSERT_EQ(m.reg(0, 13), 0u);
        ASSERT_EQ(m.reg(0, 14), 1u);
        ASSERT_EQ(m.reg(0, 15), 2u);
        ASSERT_EQ(m.reg(0, 16), 3u);
        return true;
    });
}

// ═══════════════════════════════════════════════════════════════════════
//  6. mret
// ═══════════════════════════════════════════════════════════════════════

TEST(csrx_mret_jumps_to_mepc_and_restores_mie)
{
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({
            {0x0, R"(
                li    t0, 0x100
                csrw  mepc, t0
                li    t0, 0x80     # MPIE = 1, MIE = 0
                csrw  mstatus, t0
                mret
                li    a0, 0xBAD    # skipped
                ebreak
            )"},
            {0x100, R"(
                csrr  a1, mstatus
                li    a2, 1
                ebreak
            )"},
        }));
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(0, 10), 0u);
        ASSERT_EQ(m.reg(0, 12), 1u);
        ASSERT_HEX_EQ(m.reg(0, 11), MStatus::MIE | MStatus::MPIE);
        return true;
    });
}

TEST(csrx_mret_with_mpie_clear_leaves_interrupts_off)
{
    return run_everywhere(R"(
        li    t0, 0x18             # mepc = 0x18 (the csrr below)
        csrw  mepc, t0
        csrw  mstatus, zero        # MPIE = 0
        nop
        mret                       # 0x10
        nop                        # 0x14 (skipped)
        csrr  a1, mstatus          # 0x18
        ebreak
    )", [](Machine& m)
    {
        // MIE = old MPIE (0), MPIE set to 1
        ASSERT_HEX_EQ(m.reg(0, 11), MStatus::MPIE);
        return true;
    });
}

