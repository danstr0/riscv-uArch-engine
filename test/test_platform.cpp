/**
 * @file test_platform.cpp
 * @brief Program-level device integration tests.
 *
 * Every test here runs real RISC-V programs that reach devices only the
 * way software does: loads and stores to memory-mapped registers, CSR
 * instructions, and trap handlers. Device-model APIs are used only to
 * play the role of the "outside world" (e.g., a packet arriving on the
 * wire) and to inspect results. Each program runs on every applicable
 * core model, and the architectural outcome must be identical across them.
 *
 * @par Layout used by the programs
 * @code{.txt}
 *   0x0000  hart 0 main program          0x8000  shared data / flags
 *   0x0200  hart 1 main program          0x9000  scratch array
 *   0x0400  trap handler (mtvec)         0x1000  NIC RX descriptor ring
 *                                        0x2000  NIC RX buffers
 * @endcode
 *
 * @par Sections
 * @code{.txt}
 *   1 : Memory map (each device reachable at its documented address)
 *   2 : Timer interrupts through mtvec / mret
 *   3 : Precise interrupts (no lost or duplicated work)
 *   4 : Software interrupts (IPI) between harts
 *   5 : PLIC external interrupts, per-hart routing
 *   6 : NIC -> PLIC -> CPU end to end
 *   7 : Platform construction / validation
 * @endcode
 */

#include "machine_harness.hpp"

#include <stdexcept>

using namespace riscv;
using namespace harness;

namespace {

constexpr addr_t DATA = 0x8000;

// Trap handler that counts timer interrupts at DATA and re-arms mtimecmp[0]
// to (mtime + interval). Clobbers only t3-t5. Records mcause at DATA+4.
std::string timer_handler(u32 interval)
{
    return std::format(R"(
        li   t5, 0x8000
        lw   t3, 0(t5)
        addi t3, t3, 1
        sw   t3, 0(t5)
        csrr t4, mcause
        sw   t4, 4(t5)
        li   t4, 0x0200BFF8    # mtime (lo)
        lw   t3, 0(t4)
        addi t3, t3, {}
        li   t4, 0x02004000    # mtimecmp[0]
        sw   zero, 4(t4)
        sw   t3, 0(t4)
        mret
    )", interval);
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════
//  1. Memory map
// ═══════════════════════════════════════════════════════════════════════

TEST(plat_devices_reachable_at_documented_addresses)
{
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        m.platform().attach_nic(std::make_shared<NIC>(m.platform().ram_ptr()), 1);
        ASSERT(m.load({{0x0, R"(
            li   s0, 0x8000
            # CLINT: mtimecmp[0] round-trips; mtime is counting
            li   t0, 0x02004000
            li   t1, 0x12345678
            sw   t1, 0(t0)
            lw   t2, 0(t0)
            sw   t2, 0(s0)
            li   t0, 0x0200BFF8
            lw   t2, 0(t0)
            sw   t2, 4(s0)
            # PLIC: priority[5] round-trips (masked to 3 bits)
            li   t0, 0x0C000014
            li   t1, 0xF
            sw   t1, 0(t0)
            lw   t2, 0(t0)
            sw   t2, 8(s0)
            # PLIC context-0 threshold
            li   t0, 0x0C200000
            li   t1, 3
            sw   t1, 0(t0)
            lw   t2, 0(t0)
            sw   t2, 12(s0)
            # NIC STATUS has LINK_UP
            li   t0, 0x10000004
            lw   t2, 0(t0)
            sw   t2, 16(s0)
            ebreak
        )"}}));
        ASSERT(m.run_to_halt());

        ASSERT_HEX_EQ(m.mem32(DATA + 0), 0x1234'5678u);
        ASSERT(m.mem32(DATA + 4) > 0);              // mtime advanced
        ASSERT_EQ(m.mem32(DATA + 8), 7u);
        ASSERT_EQ(m.mem32(DATA + 12), 3u);
        ASSERT(m.mem32(DATA + 16) & NicCtrl::LINK_UP);

        // Device state really changed (the program didn't hit RAM or the wrong device)
        ASSERT_HEX_EQ(m.platform().clint().mtimecmp(0), 0xFFFF'FFFF'1234'5678ull);
        ASSERT_EQ(m.platform().plic().priority(5), 7u);
        ASSERT_EQ(m.platform().plic().threshold(0), 3u);
        return true;
    });
}

TEST(plat_unmapped_load_faults)
{
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({{0x0, R"(
            li   t0, 0x30000000      # hole in the memory map
            lw   t1, 0(t0)
            li   a0, 1               # must never execute
            ebreak
        )"}}));
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(0, 10), 0u);
        return true;
    });
}

TEST(plat_store_past_ram_faults_even_with_caches)
{
    // RAM is 64 KiB; 0x10000 is the first byte past it. With caches, this
    // used to be absorbed silently by the cache.
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({{0x0, R"(
            li   t0, 0x10000
            li   t1, 42
            sw   t1, 0(t0)
            li   a0, 1
            ebreak
        )"}}));
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(0, 10), 0u);
        return true;
    });
}

TEST(plat_nic_window_faults_when_no_nic_attached)
{
    return for_each_model({Model::Cpu, Model::Pipe}, [](Machine& m, Model)
    {
        ASSERT(m.load({{0x0, R"(
            li   t0, 0x10000004
            lw   t1, 0(t0)
            li   a0, 1
            ebreak
        )"}}));
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(0, 10), 0u);
        return true;
    });
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Timer interrupts
// ═══════════════════════════════════════════════════════════════════════

TEST(plat_timer_interrupt_handler_runs_and_returns)
{
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({
            {0x0, R"(
                li   s0, 0x8000
                li   t0, 0x400
                csrw mtvec, t0
                li   s1, 0x02004000
                li   t0, 300
                sw   zero, 4(s1)
                sw   t0, 0(s1)       # first interrupt at mtime = 300
                li   t0, 0x80
                csrs mie, t0         # MTIE
                csrsi mstatus, 8     # MIE
            wait:
                lw   t1, 0(s0)
                li   t2, 3
                blt  t1, t2, wait
                csrci mstatus, 8
                li   a0, 0x600D
                ebreak
            )"},
            {0x400, timer_handler(300)},
        }));
        ASSERT(m.run_to_halt());

        ASSERT_HEX_EQ(m.reg(0, 10), 0x600Du);  // main resumed after each mret
        ASSERT(m.mem32(DATA) >= 3u);
        ASSERT_HEX_EQ(m.mem32(DATA + 4), MCause::M_TIMER);
        ASSERT_EQ(m.interrupts_taken(0), u64{m.mem32(DATA)});
        ASSERT(m.csrs(0).mepc() < 0x400);      // interrupted main, not the handler
        return true;
    });
}

TEST(plat_timer_interrupt_masked_by_mie)
{
    // MTIP is asserted (mtimecmp = 0) but mie.MTIE is clear: no trap
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({
            {0x0, R"(
                li   t0, 0x400
                csrw mtvec, t0
                li   t0, 0x02004000
                sw   zero, 4(t0)
                sw   zero, 0(t0)
                csrsi mstatus, 8
                li   t1, 200
            spin:
                addi t1, t1, -1
                bnez t1, spin
                csrr a1, mip
                ebreak
            )"},
            {0x400, "li a0, 0xBAD\nebreak"},
        }));
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(0, 10), 0u);
        ASSERT(m.reg(0, 11) & MInterrupt::MTIP);  // pending is visible in mip
        ASSERT_EQ(m.interrupts_taken(0), 0u);
        return true;
    });
}

TEST(plat_vectored_mtvec_dispatches_by_cause)
{
    // MODE=1: a timer interrupt (cause 7) goes to BASE + 4*7 = 0x41C
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({
            {0x0, R"(
                li   s0, 0x8000
                li   t0, 0x401       # BASE=0x400, MODE=vectored
                csrw mtvec, t0
                li   t0, 0x02004000
                sw   zero, 4(t0)
                li   t1, 100
                sw   t1, 0(t0)
                li   t0, 0x80
                csrs mie, t0
                csrsi mstatus, 8
            wait:
                lw   t1, 0(s0)
                beqz t1, wait
                ebreak
            )"},
            {0x400, "li a0, 0xBAD\nebreak"},  // base: must not be used for interrupts
            {0x41C, R"(
                li   t5, 0x8000
                li   t3, 1
                sw   t3, 0(t5)
                li   t4, 0x02004000
                li   t3, -1
                sw   t3, 4(t4)       # mtimecmp = far future: deassert MTIP
                mret
            )"},
        }));
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.mem32(DATA), 1u);
        ASSERT_EQ(m.reg(0, 10), 0u);
        ASSERT_HEX_EQ(m.csrs(0).mcause(), MCause::M_TIMER);
        return true;
    });
}

TEST(plat_timebase_divider_slows_mtime)
{
    SystemConfig cfg;
    cfg.platform.mtime_cycles_per_tick = 10;
    SingleCoreSystem<PipelinedCPU> sys(cfg);
    for (int i = 0; i < 1000; ++i) sys.step();
    ASSERT_EQ(sys.platform().clint().mtime(), sys.cycles() / 10);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  3. Precise interrupts
// ═══════════════════════════════════════════════════════════════════════

TEST(plat_precise_interrupts_no_lost_or_repeated_work)
{
    // A loop of non-idempotent work (register accumulate + memory RMW +
    // a load-use hazard + a mispredicted backward branch) is interrupted
    // dozens of times at varying phases. If any instruction were executed
    // twice or skipped across a trap, the totals would be wrong.
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({
            {0x0, R"(
                li   t0, 0x400
                csrw mtvec, t0
                li   t0, 0x02004000
                sw   zero, 4(t0)
                li   t1, 50
                sw   t1, 0(t0)
                li   t0, 0x80
                csrs mie, t0
                csrsi mstatus, 8
                li   s2, 0x9000
                li   a0, 0           # register accumulator
                li   a1, 0           # i
                li   a2, 300         # N
            loop:
                addi a0, a0, 3
                lw   a3, 0(s2)
                addi a3, a3, 1       # load-use dependency
                sw   a3, 0(s2)
                addi a1, a1, 1
                blt  a1, a2, loop
                csrci mstatus, 8
                ebreak
            )"},
            // Re-arm with an interval that varies by count, so interrupts
            // land on different pipeline phases.
            {0x400, R"(
                li   t5, 0x8000
                lw   t3, 0(t5)
                addi t3, t3, 1
                sw   t3, 0(t5)
                andi t3, t3, 15
                addi t3, t3, 45
                li   t4, 0x0200BFF8
                lw   t6, 0(t4)
                add  t3, t3, t6
                li   t4, 0x02004000
                sw   t3, 0(t4)
                mret
            )"},
        }));
        ASSERT(m.run_to_halt());

        ASSERT_EQ(m.reg(0, 10), 900u);     // 300 * 3
        ASSERT_EQ(m.reg(0, 11), 300u);
        ASSERT_EQ(m.mem32(0x9000), 300u);
        ASSERT(m.mem32(DATA) >= 10u);      // really was interrupted a lot
        ASSERT_EQ(m.interrupts_taken(0), u64{m.mem32(DATA)});
        return true;
    });
}

TEST(plat_enabling_mie_traps_before_next_instruction)
{
    // With MTIP already pending, the instruction that sets mstatus.MIE
    // completes and the trap is taken before the following instruction:
    // mepc must be exactly the address after the csrsi.
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({
            {0x0, R"(
                li   t0, 0x400       # 0x00
                csrw mtvec, t0       # 0x04
                li   t0, 0x02004000  # 0x08 (lui+addi)
                sw   zero, 4(t0)     # 0x10
                sw   zero, 0(t0)     # 0x14  mtimecmp = 0: MTIP asserted
                li   t0, 0x80        # 0x18
                csrs mie, t0         # 0x1C
                csrsi mstatus, 8     # 0x20  <- enables; trap before 0x24
                li   a0, 1           # 0x24
                ebreak               # 0x28
            )"},
            {0x400, R"(
                csrr a1, mepc
                csrr a2, mstatus
                ebreak
            )"},
        }));
        ASSERT(m.run_to_halt());
        ASSERT_HEX_EQ(m.reg(0, 11), 0x24u);
        ASSERT_EQ(m.reg(0, 10), 0u);     // 0x24 never executed
        ASSERT_HEX_EQ(m.reg(0, 12) & (MStatus::MIE | MStatus::MPIE), MStatus::MPIE);
        return true;
    });
}

// ═══════════════════════════════════════════════════════════════════════
//  4. Software interrupts between harts
// ═══════════════════════════════════════════════════════════════════════

namespace {

// Shared handler: clear own msip, bump per-hart trap counter at 0x8010+4*hart,
// record cause at 0x8040+4*hart, write ack = hartid+1 to 0x8000.
const char* kIpiHandler = R"(
    csrr t3, mhartid
    slli t4, t3, 2
    li   t5, 0x02000000
    add  t5, t5, t4
    sw   zero, 0(t5)
    li   t5, 0x8010
    add  t5, t5, t4
    lw   t6, 0(t5)
    addi t6, t6, 1
    sw   t6, 0(t5)
    csrr t6, mcause
    sw   t6, 0x30(t5)
    addi t3, t3, 1
    li   t5, 0x8000
    sw   t3, 0(t5)
    mret
)";

const char* kEnableMsi = R"(
    li   t0, 0x400
    csrw mtvec, t0
    li   t0, 8
    csrs mie, t0             # MSIE
    csrsi mstatus, 8
    li   s0, 0x8000
)";

} // namespace

TEST(plat_ipi_reaches_only_target_hart)
{
    return for_each_model(kMultiHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({
            {0x0, std::string(kEnableMsi) + R"(
            ready:
                lw   t1, 0x20(s0)    # wait for hart 1
                beqz t1, ready
                li   t0, 0x02000004  # msip[1]
                li   t1, 1
                sw   t1, 0(t0)
            ack:
                lw   t1, 0(s0)
                beqz t1, ack
                ebreak
            )"},
            {0x200, std::string(kEnableMsi) + R"(
                li   t1, 1
                sw   t1, 0x20(s0)    # ready
            wait:
                lw   t1, 0(s0)
                beqz t1, wait
                ebreak
            )"},
            {0x400, kIpiHandler},
        }));
        m.set_pc(1, 0x200);
        ASSERT(m.run_to_halt());

        ASSERT_EQ(m.mem32(DATA), 2u);           // ack from hart 1 (id + 1)
        ASSERT_EQ(m.mem32(0x8010), 0u);         // hart 0 never trapped
        ASSERT_EQ(m.mem32(0x8014), 1u);
        ASSERT_HEX_EQ(m.mem32(0x8044), MCause::M_SOFTWARE);
        ASSERT_EQ(m.interrupts_taken(0), 0u);
        ASSERT_EQ(m.interrupts_taken(1), 1u);
        ASSERT(!m.platform().clint().msip(1));  // handler cleared it via MMIO
        return true;
    }, 2);
}

TEST(plat_mhartid_distinguishes_harts)
{
    return for_each_model(kMultiHartModels, [](Machine& m, Model)
    {
        const char* prog = "csrr a0, mhartid\nebreak";
        ASSERT(m.load({{0x0, prog}}));
        m.set_pc(1, 0x0);
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(0, 10), 0u);
        ASSERT_EQ(m.reg(1, 10), 1u);
        return true;
    }, 2);
}

TEST(plat_per_hart_timers_are_independent)
{
    // Only hart 1 arms its comparator; only hart 1 takes a timer interrupt
    return for_each_model(kMultiHartModels, [](Machine& m, Model)
    {
        const std::string enable_mti = R"(
            li   t0, 0x400
            csrw mtvec, t0
            li   t0, 0x80
            csrs mie, t0
            csrsi mstatus, 8
            li   s0, 0x8000
        )";
        ASSERT(m.load({
            {0x0, enable_mti + R"(
                li   t1, 3000
            spin:
                addi t1, t1, -1
                bnez t1, spin
                ebreak
            )"},
            {0x200, enable_mti + R"(
                li   t0, 0x02004008  # mtimecmp[1]
                sw   zero, 4(t0)
                li   t1, 200
                sw   t1, 0(t0)
            wait:
                lw   t1, 0(s0)
                beqz t1, wait
                ebreak
            )"},
            {0x400, R"(
                csrr t3, mhartid
                addi t3, t3, 1
                li   t5, 0x8000
                sw   t3, 0(t5)
                slli t3, t3, 3
                li   t4, 0x02003FF8  # mtimecmp[hart] = 0x02004000 + 8*hart
                add  t4, t4, t3
                li   t3, -1
                sw   t3, 4(t4)
                mret
            )"},
        }));
        m.set_pc(1, 0x200);
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.mem32(DATA), 2u);
        ASSERT_EQ(m.interrupts_taken(0), 0u);
        ASSERT_EQ(m.interrupts_taken(1), 1u);
        return true;
    }, 2);
}

// ═══════════════════════════════════════════════════════════════════════
//  5. PLIC external interupts
// ═══════════════════════════════════════════════════════════════════════

namespace {

// Claim via this hart's context, record (claimed id) at 0x8030+4*hart and a
// per-hart trap count at 0x8010+4*hart, complete, then ack at 0x8000.
const char* kPlicHandler = R"(
    csrr t3, mhartid
    slli t4, t3, 12
    li   t5, 0x0C200004      # claim/complete, context 0
    add  t5, t5, t4
    lw   t6, 0(t5)           # claim
    slli t4, t3, 2
    li   a7, 0x8030
    add  a7, a7, t4
    sw   t6, 0(a7)
    lw   a6, -0x20(a7)       # trap count at 0x8010+4*hart
    addi a6, a6, 1
    sw   a6, -0x20(a7)
    sw   t6, 0(t5)           # complete
    li   t5, 0x8000
    li   t6, 1
    sw   t6, 0(t5)
    mret
)";

} // namespace

TEST(plat_plic_routes_to_enabled_context_only)
{
    return for_each_model(kMultiHartModels, [](Machine& m, Model)
    {
        const std::string enable_mei = R"(
            li   t0, 0x400
            csrw mtvec, t0
            li   t0, 0x800
            csrs mie, t0             # MEIE on both harts
            csrsi mstatus, 8
            li   s0, 0x8000
        )";
        ASSERT(m.load({
            {0x0, enable_mei + R"(
                li   t0, 0x0C00000C  # priority[3] = 2
                li   t1, 2
                sw   t1, 0(t0)
                li   t0, 0x0C002080  # context 1 enables: source 3 only
                li   t1, 8
                sw   t1, 0(t0)
                li   t1, 1
                sw   t1, 0x20(s0)    # configured
            wait0:
                lw   t1, 0(s0)
                beqz t1, wait0
                ebreak
            )"},
            {0x200, enable_mei + R"(
            wait1:
                lw   t1, 0(s0)
                beqz t1, wait1
                ebreak
            )"},
            {0x400, kPlicHandler},
        }));
        m.set_pc(1, 0x200);

        ASSERT(m.run_until([&] { return m.mem32(0x8020) == 1; }));
        m.platform().plic().set_pending(3);             // device raises IRQ 3
        ASSERT(m.run_to_halt());

        ASSERT_EQ(m.mem32(0x8034), 3u);                 // hart 1 claimed source 3
        ASSERT_EQ(m.mem32(0x8014), 1u);
        ASSERT_EQ(m.mem32(0x8010), 0u);                 // hart 0 never trapped
        ASSERT_EQ(m.interrupts_taken(0), 0u);
        ASSERT_HEX_EQ(m.csrs(1).mcause(), MCause::M_EXTERNAL);
        ASSERT(!m.platform().plic().is_pending(3));
        ASSERT(!(m.csrs(1).mip() & MInterrupt::MEIP));  // line dropped after claim
        return true;
    }, 2);
}

TEST(plat_plic_threshold_masks_in_software)
{
    // Priority 1 with threshold 1 is masked; raising priority to 2 unmasks
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({
            {0x0, R"(
                li   t0, 0x400
                csrw mtvec, t0
                li   t0, 0x800
                csrs mie, t0
                csrsi mstatus, 8
                li   s0, 0x8000
                li   t0, 0x0C000008  # priority[2] = 1
                li   t1, 1
                sw   t1, 0(t0)
                li   t2, 0x0C200000  # threshold = 1
                sw   t1, 0(t2)
                li   t3, 0x0C002000  # enable source 2
                li   t1, 4
                sw   t1, 0(t3)
                li   t1, 1
                sw   t1, 0x20(s0)
            spin:
                lw   t1, 0x24(s0)    # harness sets this once IRQ is pending
                beqz t1, spin
                li   t1, 300
            delay:
                addi t1, t1, -1
                bnez t1, delay
                lw   a0, 0(s0)       # must still be 0: masked
                li   t1, 2
                sw   t1, 0(t0)       # priority[2] = 2 > threshold
            wait:
                lw   t1, 0(s0)
                beqz t1, wait
                ebreak
            )"},
            {0x400, kPlicHandler},
        }));
        ASSERT(m.run_until([&] { return m.mem32(0x8020) == 1; }));
        m.platform().plic().set_pending(2);
        ASSERT(m.view().write32(0x8024, 1).ok);
        ASSERT(m.run_to_halt());

        ASSERT_EQ(m.reg(0, 10), 0u);
        ASSERT_EQ(m.mem32(0x8030), 2u);
        ASSERT_EQ(m.interrupts_taken(0), 1u);
        return true;
    });
}

// ═══════════════════════════════════════════════════════════════════════
//  6. NIC -> PLIC -> CPU
// ═══════════════════════════════════════════════════════════════════════

TEST(plat_nic_rx_interrupt_end_to_end)
{
    // Software sets up the RX ring and the NIC/PLIC registers over MMIO.
    // A packet arrives on the wire; the NIC DMAs it into RAM and raises
    // PLIC source 1; the CPU traps, claims, reads ICR and the payload,
    // completes, and returns.
    //
    // Uncached models only: NIC DMA writes RAM directly and is not coherent
    // with CPU caches (a CPU-written descriptor can sit in a write-back L1
    // where the NIC never sees it).
    return for_each_model({Model::Cpu, Model::Pipe,
                           Model::PipeNoForward, Model::MultiCore},
                          [](Machine& m, Model)
    {
        NicTiming timing;
        timing.dma_latency_cycles       = 10;
        timing.dma_cycles_per_cacheline = 1;
        m.platform().attach_nic(std::make_shared<NIC>(m.platform().ram_ptr(), timing), 1);

        ASSERT(m.load({
            {0x0, R"(
                li   t0, 0x400
                csrw mtvec, t0
                li   s0, 0x8000
                # 4 RX descriptors at 0x1000 -> buffers at 0x2000 + 256*i
                li   t0, 0x1000
                li   t1, 0x2000
                li   t2, 4
            desc:
                sw   t1, 0(t0)
                sw   zero, 4(t0)
                sw   zero, 8(t0)
                sw   zero, 12(t0)
                addi t0, t0, 16
                addi t1, t1, 256
                addi t2, t2, -1
                bnez t2, desc
                # NIC registers
                li   s1, 0x10000000
                li   t0, 0x1000
                sw   t0, 0x20(s1)    # RDBAL
                li   t0, 64
                sw   t0, 0x28(s1)    # RDLEN = 4 descriptors
                sw   zero, 0x2C(s1)  # RDH
                li   t0, 3
                sw   t0, 0x30(s1)    # RDT
                li   t0, 1
                sw   t0, 0x0C(s1)    # IMS: RXQ0
                li   t0, 6
                sw   t0, 0(s1)       # CTRL: RXEN | TXEN
                # PLIC: source 1 priority 1, enabled for context 0
                li   t0, 0x0C000004
                li   t1, 1
                sw   t1, 0(t0)
                li   t0, 0x0C002000
                li   t1, 2
                sw   t1, 0(t0)
                li   t0, 0x800
                csrs mie, t0
                csrsi mstatus, 8
                li   t1, 1
                sw   t1, 0x20(s0)    # ready
            wait:
                lw   t1, 0(s0)
                beqz t1, wait
                ebreak
            )"},
            {0x400, R"(
                li   t5, 0x0C200004
                lw   t3, 0(t5)       # claim
                li   t4, 0x8000
                sw   t3, 4(t4)
                li   t6, 0x10000010
                lw   t6, 0(t6)       # ICR (read-clear)
                sw   t6, 8(t4)
                li   t6, 0x2000
                lbu  t6, 0(t6)       # first payload byte
                sw   t6, 12(t4)
                li   t6, 0x10000060
                lw   t6, 0(t6)       # RXPKT
                sw   t6, 16(t4)
                sw   t3, 0(t5)       # complete
                li   t6, 1
                sw   t6, 0(t4)
                mret
            )"},
        }));

        ASSERT(m.run_until([&] { return m.mem32(0x8020) == 1; }));
        Packet p;
        p.data = {0xAA, 0xBB, 0xCC, 0xDD};
        m.platform().nic()->inject_packet(p);
        ASSERT(m.run_to_halt());

        ASSERT_EQ(m.mem32(DATA + 4), 1u);      // claimed NIC's source
        ASSERT(m.mem32(DATA + 8) & NicInt::RXQ0);
        ASSERT_HEX_EQ(m.mem32(DATA + 12), 0xAAu);
        ASSERT_EQ(m.mem32(DATA + 16), 1u);
        ASSERT_HEX_EQ(m.csrs(0).mcause(), MCause::M_EXTERNAL);
        ASSERT_EQ(m.interrupts_taken(0), 1u);
        return true;
    });
}

// ═══════════════════════════════════════════════════════════════════════
//  7. Platform construction / validation
// ═══════════════════════════════════════════════════════════════════════

TEST(plat_default_memory_map)
{
    Platform p;
    auto map = p.memory_map();
    ASSERT_EQ(map.size(), 3u);
    ASSERT_HEX_EQ(map[1].base, PlatformMap::CLINT_BASE);
    ASSERT_HEX_EQ(map[2].base, PlatformMap::PLIC_BASE);
    p.attach_nic(std::make_shared<NIC>(p.ram_ptr()), 1);
    ASSERT_EQ(p.memory_map().size(), 4u);
    return true;
}

TEST(plat_rejects_ram_overlapping_devices)
{
    bool threw = false;
    // 32 MiB + 4 KiB hits the CLINT
    try { Platform p({.ram_size = 0x0200'0000 + 0x1000}); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT(threw);

    Platform ok({.ram_size = 0x0200'0000});   // exactly up to the CLINT
    ASSERT_EQ(ok.ram().ranges().size(), 1u);
    return true;
}

TEST(plat_ram_can_move_to_0x80000000)
{
    Platform p({.ram_base = 0x8000'0000, .ram_size = 256u << 20});
    auto bus = p.make_hart_bus(p.ram_ptr());
    ASSERT(bus->write32(0x8000'0000, 7).ok);
    ASSERT(bus->write32(0x8FFF'FFFC, 8).ok);
    ASSERT(!bus->write32(0x0000'0000, 9).ok);  // nothing at 0 any more
    ASSERT_EQ(p.ram().pages_allocated(), 2u);
    return true;
}

TEST(plat_rejects_bad_hart_counts)
{
    bool threw = false;
    try { Platform p({.num_harts = 0}); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT(threw);
    threw = false;
    try { Platform p({.num_harts = 17}); }
    catch (const std::invalid_argument&) { threw = true; }
    ASSERT(threw);
    return true;
}

