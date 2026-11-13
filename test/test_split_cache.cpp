/**
 * @file test_split_cache.cpp
 * @brief Split L1I/L1D and cache coherence tests.
 *
 * @par Sections
 * @code{.txt}
 *   1 : Split versus unified routing of fetches and data
 *   2 : Self-modifying code and fence.i
 *   3 : MESI transitions through real cache accesses
 *   4 : Directory accuracy (evictions, no-allocate writes)
 *   5 : Atomicity across harts (LR/SC, AMOs, locks)
 *   6 : Coherent DMA
 *   7 : Randomized stress against a shadow memory
 * @endcode
 */

#include "core/config.hpp"
#include "machine_harness.hpp"

#include <map>
#include <random>
#include <vector>

using namespace riscv;
using namespace harness;

namespace {

CacheConfig tiny(WritePolicy wp = WritePolicy::WRITE_BACK,
                 WriteAllocate wa = WriteAllocate::ALLOCATE)
{
    return {.size_bytes = 512, .line_size = 64, .associativity = 2,
            .hit_latency = 1, .miss_penalty = 10,
            .write_pol = wp, .write_alloc = wa};
}

MultiCoreConfig mc_config(u32 cores, CacheConfig l1d = tiny())
{
    MultiCoreConfig c;
    c.num_cores        = cores;
    c.main_memory_size = 64 * 1024;
    c.l1d              = l1d;
    c.l2               = small_l2();
    c.l3.size_bytes    = 0;
    return c;
}

std::vector<u8> assemble(const std::string& src)
{
    auto r = Assembler{}.assemble(src);
    if (!r.ok)
        for (const auto& e : r.errors)
            std::cerr << "  asm error line " << e.line << ": " << e.message << "\n";
    return r.code;
}

constexpr addr_t DATA = 0x8000;

} // namespace

// ═══════════════════════════════════════════════════════════════════════
//  1. Split versus unified routing
// ═══════════════════════════════════════════════════════════════════════

TEST(split_fetches_use_icache_and_data_uses_dcache)
{
    SystemConfig cfg;
    cfg.l1d = small_l1();
    cfg.l1i = small_l1();
    SingleCoreSystem<CPU> sys(cfg);

    auto code = assemble(R"(
        li   t0, 0x8000
        li   t1, 7
        sw   t1, 0(t0)
        lw   t2, 0(t0)
        ebreak
    )");
    sys.load_program(0, code);
    sys.run();

    ASSERT_EQ(sys.core().reg(7), 7u);
    ASSERT_EQ(sys.l1d()->stats().reads, 1u);   // only the lw
    ASSERT_EQ(sys.l1d()->stats().writes, 1u);  // only the sw
    ASSERT_EQ(sys.l1i()->stats().writes, 0u);
    ASSERT_EQ(sys.l1i()->stats().reads, sys.core().instructions());
    return true;
}

TEST(split_unified_fetches_go_through_l1d)
{
    SystemConfig cfg;
    cfg.l1d = small_l1();
    SingleCoreSystem<CPU> sys(cfg);
    sys.load_program(0, assemble("addi a0, zero, 1\nebreak"));
    sys.run();

    ASSERT(sys.l1i() == nullptr);
    ASSERT_EQ(sys.l1d()->stats().reads, 2u);  // two fetches
    return true;
}

TEST(split_multicore_has_one_icache_per_core)
{
    auto cfg = mc_config(2);
    cfg.l1i  = small_l1();
    MultiCoreCPU mc(cfg);
    mc.load_program(0, assemble("addi a0, zero, 1\nebreak"));
    mc.set_core_pc(1, 0);
    mc.run_until_all_halted(100);

    ASSERT(mc.l1i(0) != mc.l1i(1));
    ASSERT(mc.l1i(0)->stats().reads > 0);
    ASSERT(mc.l1i(1)->stats().reads > 0);
    ASSERT_EQ(mc.l1d(0)->stats().reads, 0u);
    ASSERT_EQ(mc.get_stats().l1i_stats.size(), 2u);
    ASSERT(mc.coherence()->stats().fetch_misses >= 2u);
    return true;
}

TEST(split_rejects_bad_configurations)
{
    auto throws = [](auto&& fn)
    {
        try { fn(); } catch (const std::invalid_argument&) { return true; }
        return false;
    };

    ASSERT(throws([]
    {
        SystemConfig c; c.l1i = small_l1();                  // L1I without L1D
        SingleCoreSystem<CPU> s(c);
    }));
    ASSERT(throws([]
    {
        SystemConfig c; c.l1d = small_l1(); c.l1i = small_l1();
        c.l1i.size_bytes = 512; c.l1i.line_size = 32;        // line size mismatch
        SingleCoreSystem<CPU> s(c);
    }));
    ASSERT(throws([]
    {
        MultiCoreConfig c; c.num_cores = 2; c.l1d.size_bytes = 0;
        c.l1i = small_l1();
        MultiCoreCPU m(c);
    }));
    return true;
}

TEST(split_fetch_from_device_region_faults)
{
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({{0x0, R"(
            li   t0, 0x02000000
            jalr ra, 0(t0)
            li   a0, 1
            ebreak
        )"}}));
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(0, 10), 0u);
        return true;
    });
}

TEST(split_config_section_parses)
{
    auto r = parse_config(R"(
        [cache.l1]
        size = 8K
        [cache.l1i]
        size = 16K
        assoc = 4
        replacement = plru
    )");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.l1i_size, 16u * 1024);
    ASSERT_EQ(r.config.l1i_assoc, 4u);
    ASSERT(r.config.l1i_replacement == "plru");
    return true;
}

TEST(split_config_defaults_to_unified)
{
    auto r = parse_config("[cache.l1]\nsize = 8K\n");
    ASSERT(r.ok);
    ASSERT_EQ(r.config.l1i_size, 0u);
    return true;
}

TEST(split_config_validation_errors)
{
    // needs L1D
    ASSERT(!parse_config("[cache.l1]\nsize = 0\n[cache.l1i]\nsize = 4K\n").ok);
    // not a power of 2
    ASSERT(!parse_config("[cache.l1]\nsize = 4K\n[cache.l1i]\nsize = 3K\n").ok);
    ASSERT(!parse_config("[cache.l1]\nsize = 4K\n[cache.l1i]\nsize = 8K\nreplacement = nope\n").ok);
    // L2 must exceed L1I
    ASSERT(!parse_config("[cache.l1]\nsize = 4K\n[cache.l1i]\nsize = 16K\n"
                         "[cache.l2]\nsize = 16K\n").ok);
    return true;
}

TEST(split_config_sweep_axis)
{
    auto r = parse_config("[cache.l1]\nsize = 4K\n[cache.l1i]\nsize = 4K\n"
                          "[sweep]\ncache.l1i.size = [4K, 8K]\n");
    ASSERT(r.ok);
    ASSERT(!parse_config("[sweep]\ncache.l1i.bogus = [1, 2]\n").ok);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  2. Self-modifying code and fence.i
// ═══════════════════════════════════════════════════════════════════════

namespace {

// Calls the function at 0x100 (returns 1 in a0), overwrites its first
// instruction with `addi a0, zero, 2`, then calls it again. Results land
// in a1 (before) and a2 (after).
std::string smc_program(const std::string& sync)
{
    return R"(
        li   s0, 0x100
        jalr ra, 0(s0)
        mv   a1, a0
        li   t0, 0x00200513
        sw   t0, 0(s0)
        )" + sync + R"(
        jalr ra, 0(s0)
        mv   a2, a0
        ebreak
    )";
}

const char* kSmcTarget = "addi a0, zero, 1\nret";

} // namespace

TEST(split_fence_i_encoding)
{
    auto code = assemble("fence.i");
    ASSERT_EQ(code.size(), 4u);
    ASSERT_HEX_EQ(u32{code[0]} | (u32{code[1]} << 8), 0x100Fu);
    ASSERT(Decoder::decode(0x0000100F, 0).op == Op::FENCE_I);
    ASSERT(Decoder::decode(0x0FF0000F, 0).op == Op::FENCE);
    ASSERT(!Assembler{}.assemble("fence.i t0").ok);
    return true;
}

TEST(split_fence_i_makes_new_code_visible)
{
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({{0x0, smc_program("fence.i")}, {0x100, kSmcTarget}}));
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(0, 11), 1u);
        ASSERT_EQ(m.reg(0, 12), 2u);
        return true;
    });
}

TEST(split_fence_i_refetches_the_next_instruction)
{
    // The instruction right after fence.i is already in the pipeline when
    // the fence executes, so it has to be squashed and refetched.
    return for_each_model(kSingleHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({{0x0, R"(
            li   t0, 0x00200513      # 0x00 (lui+addi): addi a0, zero, 2
            li   s0, 0x14            # 0x08
            sw   t0, 0(s0)           # 0x0C
            fence.i                  # 0x10
            addi a0, zero, 1         # 0x14: patched above
            ebreak
        )"}}));
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(0, 10), 2u);
        return true;
    });
}

TEST(split_icache_stays_stale_without_fence_i)
{
    // Pins the architectural contract: a store does not reach a line
    // already in a split I-cache until fence.i
    return for_each_model(kSplitModels, [](Machine& m, Model)
    {
        ASSERT(m.load({{0x0, smc_program("nop")}, {0x100, kSmcTarget}}));
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(0, 11), 1u);
        ASSERT_EQ(m.reg(0, 12), 1u);
        return true;
    });
}

TEST(split_fence_i_sees_code_written_by_another_hart)
{
    // Hart 0 patches code that hart 1 has already run and leaves it dirty
    // in hart 0's D-cache. Hart 1's refill after fence.i must still see it
    return for_each_model(kMultiHartModels, [](Machine& m, Model)
    {
        ASSERT(m.load({
            {0x0, R"(
                li   s0, 0x8000
            wait:
                lw   t1, 0(s0)       # hart 1 has run the old code
                beqz t1, wait
                li   t2, 0x100
                li   t0, 0x00200513
                sw   t0, 0(t2)
                fence
                li   t1, 1
                sw   t1, 4(s0)       # patched
                ebreak
            )"},
            {0x200, R"(
                li   s0, 0x8000
                li   s1, 0x100
                jalr ra, 0(s1)
                mv   a1, a0
                li   t1, 1
                sw   t1, 0(s0)
            wait2:
                lw   t1, 4(s0)
                beqz t1, wait2
                fence.i
                jalr ra, 0(s1)
                mv   a2, a0
                ebreak
            )"},
            {0x100, kSmcTarget},
        }));
        m.set_pc(1, 0x200);
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.reg(1, 11), 1u);
        ASSERT_EQ(m.reg(1, 12), 2u);
        return true;
    }, 2);
}

// ═══════════════════════════════════════════════════════════════════════
//  3. MESI transitions through real cache accesses
// ═══════════════════════════════════════════════════════════════════════

TEST(coh_state_transitions_through_real_accesses)
{
    MultiCoreCPU mc(mc_config(2));
    Memory& c0 = mc.core(0).memory();
    Memory& c1 = mc.core(1).memory();
    auto* dir = mc.coherence();
    constexpr addr_t A = 0x4000;

    (void)c0.read32(A);
    ASSERT(dir->line_state(0, A) == MESIState::Exclusive);

    (void)c1.read32(A);
    ASSERT(dir->line_state(0, A) == MESIState::Shared);
    ASSERT(dir->line_state(1, A) == MESIState::Shared);

    ASSERT(c1.write32(A, 11).ok);                      // upgrade
    ASSERT(dir->line_state(0, A) == MESIState::Invalid);
    ASSERT(dir->line_state(1, A) == MESIState::Modified);
    ASSERT(!mc.l1d(0)->lookup(A));

    ASSERT_EQ(c0.read32(A).value, 11u);                // intervention
    ASSERT(dir->line_state(0, A) == MESIState::Shared);
    ASSERT(dir->line_state(1, A) == MESIState::Shared);

    ASSERT(c0.write32(A, 22).ok);
    ASSERT(c1.write32(A, 33).ok);                      // steals Modified
    ASSERT(dir->line_state(0, A) == MESIState::Invalid);
    ASSERT(dir->line_state(1, A) == MESIState::Modified);
    ASSERT_EQ(c0.read32(A).value, 33u);
    return true;
}

TEST(coh_read_from_modified_keeps_l2_copy)
{
    MultiCoreCPU mc(mc_config(2));
    constexpr addr_t A = 0x4040;
    ASSERT(mc.core(0).memory().write32(A, 5).ok);
    (void)mc.core(1).memory().read32(A);
    ASSERT(mc.l2()->lookup(A));                // the transfer goes through L2
    return true;
}

TEST(coh_false_sharing_preserves_both_writers)
{
    return for_each_model(kMultiHartModels, [](Machine& m, Model)
    {
        const auto writer = [](int offset)
        {
            return std::format(R"(
                li   t0, 0x8000
                li   t1, 0
                li   t2, 200
            loop:
                addi t1, t1, 1
                sw   t1, {}(t0)
                blt  t1, t2, loop
                ebreak
            )", offset);
        };
        ASSERT(m.load({{0x0, writer(0)}, {0x200, writer(4)}}));
        m.set_pc(1, 0x200);
        ASSERT(m.run_to_halt());
        ASSERT_EQ(m.mem32(DATA), 200u);
        ASSERT_EQ(m.mem32(DATA + 4), 200u);
        return true;
    }, 2);
}

// ═══════════════════════════════════════════════════════════════════════
//  4. Directory accuracy
// ═══════════════════════════════════════════════════════════════════════

TEST(coh_eviction_clears_directory_state)
{
    MultiCoreCPU mc(mc_config(2));
    Memory& c0 = mc.core(0).memory();
    constexpr addr_t A = 0x4000;

    ASSERT(c0.write32(A, 1).ok);
    ASSERT(mc.coherence()->line_state(0, A) == MESIState::Modified);

    // tiny() has 4 sets of 2 ways: lines 256 bytes apart conflict
    for (u32 k = 1; k <= 4; ++k)
        (void)c0.read32(A + k * 256);

    ASSERT(!mc.l1d(0)->lookup(A));
    ASSERT(mc.coherence()->line_state(0, A) == MESIState::Invalid);
    return true;
}

TEST(coh_evicted_modified_line_is_not_lost)
{
    // Regression: a stale Modified entry made the next reader "transfer"
    // zeros from the evicted owner and write them over memory
    MultiCoreCPU mc(mc_config(2));
    Memory& c0 = mc.core(0).memory();
    Memory& c1 = mc.core(1).memory();
    constexpr addr_t A = 0x4000;

    ASSERT(c0.write32(A, 0xCAFE).ok);
    for (u32 k = 1; k <= 4; ++k)
        (void)c0.read32(A + k * 256);

    ASSERT_HEX_EQ(c1.read32(A).value, 0xCAFEu);
    ASSERT_HEX_EQ(c0.read32(A).value, 0xCAFEu);
    return true;
}

TEST(coh_no_allocate_write_miss_is_not_owned)
{
    MultiCoreCPU mc(mc_config(2, tiny(WritePolicy::WRITE_THROUGH,
                                      WriteAllocate::NO_ALLOCATE)));
    constexpr addr_t A = 0x4000;

    ASSERT(mc.core(0).memory().write32(A, 9).ok);
    ASSERT(!mc.l1d(0)->lookup(A));
    ASSERT(mc.coherence()->line_state(0, A) == MESIState::Invalid);
    ASSERT_EQ(mc.core(1).memory().read32(A).value, 9u);
    return true;
}

TEST(coh_write_through_lines_are_never_dirty)
{
    auto mem = std::make_shared<SparseMemory>(0, 64 * 1024);
    Cache c(tiny(WritePolicy::WRITE_THROUGH), mem);

    ASSERT(c.write32(0x1000, 3).ok);
    bool dirty = true;
    ASSERT(c.snoop_has_line(0x1000, &dirty));
    ASSERT(!dirty);
    ASSERT_EQ(mem->read32(0x1000).value, 3u);

    for (u32 k = 1; k <= 4; ++k)
        (void)c.read32(0x1000 + k * 256);
    ASSERT_EQ(c.stats().dirty_evictions, 0u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  5. Atomicity across harts
// ═══════════════════════════════════════════════════════════════════════

namespace {

constexpr u32 kHarts = 4;
constexpr u32 kIters = 150;

bool run_on_all_harts(Machine& m, const std::string& program)
{
    ASSERT(m.load({{0x0, program}}));
    for (u32 h = 0; h < m.num_harts(); ++h)
        m.set_pc(h, 0);
    ASSERT(m.run_to_halt(4'000'000));
    return true;
}

} // namespace

TEST(coh_sc_fails_after_another_harts_store)
{
    // Both harts reserve, hart 1 stores, then hart 0's SC must fail
    MultiCoreCPU mc(mc_config(2));
    mc.load_program(0, assemble(R"(
        li    t0, 0x8000
        lr.w  t1, (t0)
        li    t2, 0x8100
    wait:
        lw    t3, 0(t2)
        beqz  t3, wait
        li    t4, 77
        sc.w  a0, t4, (t0)
        ebreak
    )"));
    mc.load_program(0x200, assemble(R"(
        li    t0, 0x8000
        li    t1, 40
    delay:
        addi  t1, t1, -1
        bnez  t1, delay
        li    t2, 55
        sw    t2, 0(t0)
        li    t2, 0x8100
        li    t3, 1
        sw    t3, 0(t2)
        ebreak
    )"));
    mc.set_core_pc(1, 0x200);
    mc.run_until_all_halted(100'000);

    ASSERT_EQ(mc.core(0).reg(10), 1u);         // SC reported failure
    ASSERT_EQ(mc.core(0).memory().read32(0x8000).value, 55u);
    return true;
}

TEST(coh_lr_sc_counter_loses_no_updates)
{
    return for_each_model(kMultiHartModels, [](Machine& m, Model)
    {
        ASSERT(run_on_all_harts(m, std::format(R"(
            li    t0, 0x8000
            li    t1, 0
            li    t2, {}
        loop:
            lr.w  t3, (t0)
            addi  t3, t3, 1
            sc.w  t4, t3, (t0)
            bnez  t4, loop
            addi  t1, t1, 1
            blt   t1, t2, loop
            ebreak
        )", kIters)));
        ASSERT_EQ(m.mem32(DATA), kHarts * kIters);
        return true;
    }, kHarts);
}

TEST(coh_amoadd_counter_loses_no_updates)
{
    return for_each_model(kMultiHartModels, [](Machine& m, Model)
    {
        ASSERT(run_on_all_harts(m, std::format(R"(
            li    t0, 0x8000
            li    t1, 0
            li    t2, {}
            li    t5, 1
        loop:
            amoadd.w t3, t5, (t0)
            addi  t1, t1, 1
            blt   t1, t2, loop
            ebreak
        )", kIters)));
        ASSERT_EQ(m.mem32(DATA), kHarts * kIters);
        return true;
    }, kHarts);
}

TEST(coh_spinlock_protects_plain_counter)
{
    // The counter itself is a non-atomic lw/addi/sw on a different line
    // from the lock, so it relies on the lock and on coherence
    return for_each_model(kMultiHartModels, [](Machine& m, Model)
    {
        ASSERT(run_on_all_harts(m, std::format(R"(
            li    s0, 0x8000         # lock
            li    s1, 0x8100         # counter
            li    t1, 0
            li    t2, {}
            li    t5, 1
        loop:
        acquire:
            amoswap.w t3, t5, (s0)
            bnez  t3, acquire
            lw    t4, 0(s1)
            addi  t4, t4, 1
            sw    t4, 0(s1)
            sw    zero, 0(s0)        # release
            addi  t1, t1, 1
            blt   t1, t2, loop
            ebreak
        )", kIters)));
        ASSERT_EQ(m.mem32(0x8100), kHarts * kIters);
        ASSERT_EQ(m.mem32(DATA), 0u);
        return true;
    }, kHarts);
}

// ═══════════════════════════════════════════════════════════════════════
//  6. Coherent DMA
// ═══════════════════════════════════════════════════════════════════════

TEST(coh_dma_read_sees_dirty_l1_data)
{
    MultiCoreCPU mc(mc_config(2));
    constexpr addr_t A = 0x4000;
    ASSERT(mc.core(1).memory().write32(A, 0xABCD).ok);

    ASSERT_HEX_EQ(mc.dma_memory()->read32(A).value, 0xABCDu);
    ASSERT(mc.coherence()->line_state(1, A) == MESIState::Exclusive);
    return true;
}

TEST(coh_dma_write_invalidates_cached_copies)
{
    MultiCoreCPU mc(mc_config(2));
    constexpr addr_t A = 0x4000;
    ASSERT(mc.core(0).memory().write32(A, 1).ok);
    ASSERT(mc.core(0).memory().write32(A + 4, 2).ok);

    ASSERT(mc.dma_memory()->write32(A, 99).ok);

    ASSERT(!mc.l1d(0)->lookup(A));
    ASSERT_EQ(mc.core(0).memory().read32(A).value, 99u);
    ASSERT_EQ(mc.core(1).memory().read32(A + 4).value, 2u);  // rest of the dirty line survived
    return true;
}

TEST(coh_dma_load_merges_partial_lines)
{
    MultiCoreCPU mc(mc_config(1));
    constexpr addr_t A = 0x4000;
    ASSERT(mc.core(0).memory().write32(A, 0x1122'3344).ok);

    const std::vector<u8> bytes = {0xAA, 0xBB};
    mc.dma_memory()->load(A + 8, bytes);

    ASSERT_HEX_EQ(mc.core(0).memory().read32(A).value, 0x1122'3344u);
    ASSERT_HEX_EQ(mc.core(0).memory().read16(A + 8).value, 0xBBAAu);
    return true;
}

TEST(coh_dma_rejects_addresses_outside_ram)
{
    MultiCoreCPU mc(mc_config(1));
    ASSERT(!mc.dma_memory()->read32(0x10000).ok);
    ASSERT(!mc.dma_memory()->write8(0x10000, 1).ok);
    return true;
}

TEST(coh_single_core_dma_is_coherent)
{
    SystemConfig cfg;
    cfg.l1d = tiny();
    cfg.l2  = small_l2();
    SingleCoreSystem<PipelinedCPU> sys(cfg);
    constexpr addr_t A = 0x4000;

    ASSERT(sys.bus().write32(A, 7).ok);
    ASSERT_EQ(sys.dma_memory()->read32(A).value, 7u);
    ASSERT(sys.dma_memory()->write32(A, 8).ok);
    ASSERT_EQ(sys.bus().read32(A).value, 8u);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
//  7. Randomized stress against a shadow memory
// ═══════════════════════════════════════════════════════════════════════

namespace {

/// Every line's directory state must agree with the caches, and at most
/// one core may hold a line writable.
bool check_directory_invariants(MultiCoreCPU& mc, addr_t base, u32 span)
{
    auto* dir = mc.coherence();
    for (addr_t a = base; a < base + span; a += 64)
    {
        u32 writers = 0, holders = 0;
        for (u32 c = 0; c < mc.num_cores(); ++c)
        {
            const MESIState s = dir->line_state(c, a);
            ASSERT_EQ(mc.l1d(c)->lookup(a) ? 1 : 0, s != MESIState::Invalid ? 1 : 0);
            if (s == MESIState::Modified || s == MESIState::Exclusive) ++writers;
            if (s != MESIState::Invalid) ++holders;
        }
        ASSERT(writers <= 1);
        ASSERT(writers == 0 || holders == 1);
    }
    return true;
}

bool stress(CacheConfig l1d, u32 seed, bool with_l3)
{
    auto cfg = mc_config(4, l1d);
    if (with_l3)
    {
        cfg.l3 = {.size_bytes = 32768, .line_size = 64, .associativity = 4,
                  .hit_latency = 10, .miss_penalty = 50};
    }
    MultiCoreCPU mc(cfg);

    constexpr addr_t BASE = 0x4000;
    constexpr u32    SPAN = 8 * 1024;  // far larger than the L1s, so lines churn

    std::map<addr_t, u8> shadow;       // absent = 0
    auto shadow_read = [&](addr_t a, u32 n)
    {
        u32 v = 0;
        for (u32 i = 0; i < n; ++i)
        {
            auto it = shadow.find(a + i);
            v |= u32{it == shadow.end() ? u8{0} : it->second} << (i * 8);
        }
        return v;
    };
    auto shadow_write = [&](addr_t a, u32 n, u32 v)
    {
        for (u32 i = 0; i < n; ++i)
            shadow[a + i] = static_cast<u8>(v >> (i * 8));
    };

    std::mt19937 engine(seed);
    auto rng = [&engine] { return static_cast<u32>(engine()); };

    for (u32 op = 0; op < 60000; ++op)
    {
        const u32 agent = rng() % 5;          // 4 = the DMA port
        Memory&   port  = (agent == 4) ? *mc.dma_memory()
                                       : mc.core(agent).memory();
        const u32 size  = 1u << (rng() % 3);
        const addr_t a  = BASE + ((rng() % SPAN) & ~(size - 1));

        if (rng() % 2)
        {
            const u32 v = rng();
            MemoryResult r;
            if      (size == 4) r = port.write32(a, v);
            else if (size == 2) r = port.write16(a, static_cast<u16>(v));
            else                r = port.write8(a, static_cast<u8>(v));
            ASSERT(r.ok);
            shadow_write(a, size, v);
        }
        else
        {
            MemoryResult r;
            if      (size == 4) r = port.read32(a);
            else if (size == 2) r = port.read16(a);
            else                r = port.read8(a);
            ASSERT(r.ok);
            ASSERT_HEX_EQ(r.value, shadow_read(a, size));
        }

        if (op % 5000 == 0)
            ASSERT(check_directory_invariants(mc, BASE, SPAN));
    }
    ASSERT(check_directory_invariants(mc, BASE, SPAN));

    for (addr_t a = BASE; a < BASE + SPAN; a += 4)
        ASSERT_HEX_EQ(mc.core(0).memory().read32(a).value, shadow_read(a, 4));
    return true;
}

} // namespace

TEST(coh_stress_write_back_allocate)
{
    return stress(tiny(), 1, false) && stress(tiny(), 2, true);
}

TEST(coh_stress_write_through_allocate)
{
    return stress(tiny(WritePolicy::WRITE_THROUGH), 3, false);
}

TEST(coh_stress_write_back_no_allocate)
{
    return stress(tiny(WritePolicy::WRITE_BACK,
                       WriteAllocate::NO_ALLOCATE), 4, false);
}

TEST(coh_stress_write_through_no_allocate)
{
    return stress(tiny(WritePolicy::WRITE_THROUGH,
                       WriteAllocate::NO_ALLOCATE), 5, false);
}

