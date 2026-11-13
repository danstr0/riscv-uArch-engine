============
Architecture
============

This document describes the simulated RISC-V architecture: which
instructions are supported, how the pipeline works, how the cache
hierarchy is organized, and how the multi-core system ties everything
together.


Instruction Set
---------------

The simulator implements a subset of the RV32 ISA sufficient for
general-purpose computation, SIMD processing, multi-core
synchronization, and system-level interrupt handling.

RV32I — Base Integer (37 instructions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 20 50 30
   :header-rows: 1

   * - Category
     - Instructions
     - Encoding
   * - Loads
     - ``LB``, ``LH``, ``LW``, ``LBU``, ``LHU``
     - I-type
   * - Stores
     - ``SB``, ``SH``, ``SW``
     - S-type
   * - Branches
     - ``BEQ``, ``BNE``, ``BLT``, ``BGE``, ``BLTU``, ``BGEU``
     - B-type
   * - Jumps
     - ``JAL``, ``JALR``
     - J-type / I-type
   * - Upper imm
     - ``LUI``, ``AUIPC``
     - U-type
   * - Reg-imm
     - ``ADDI``, ``SLTI``, ``SLTIU``, ``XORI``, ``ORI``, ``ANDI``
     - I-type
   * - Shifts
     - ``SLLI``, ``SRLI``, ``SRAI``
     - I-type (funct7)
   * - Reg-reg
     - ``ADD``, ``SUB``, ``SLL``, ``SLT``, ``SLTU``, ``XOR``, ``SRL``, ``SRA``, ``OR``, ``AND``
     - R-type
   * - System
     - ``ECALL``, ``EBREAK``, ``FENCE``
     - I-type

RV32M — Integer Multiply/Divide (8 instructions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All R-type with funct7 = 0000001:

- ``MUL``, ``MULH``, ``MULHSU``, ``MULHU``
- ``DIV``, ``DIVU``, ``REM``, ``REMU``

RV32A — Atomics (11 instructions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All R-type with opcode = 0101111 (AMO):

- ``LR.W`` — load-reserved (places a reservation on the word)
- ``SC.W`` — store-conditional (succeeds only if reservation is intact)
- ``AMOSWAP.W``, ``AMOADD.W``, ``AMOXOR.W``, ``AMOAND.W``, ``AMOOR.W``
- ``AMOMIN.W``, ``AMOMAX.W``, ``AMOMINU.W``, ``AMOMAXU.W``

``LR``/``SC`` is the foundation for lock-free synchronization — ``SC`` returns
``0`` on success (reservation held) and ``1`` on failure (a store overlapped
the reserved word between ``LR`` and ``SC``, from this core or another).

RVV — Vector Extension Subset (25 instructions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VLEN = 128 bits (configurable), SEW = 32 only, LMUL = 1.  With
SEW=32 and VLEN=128, each vector register holds 4 elements.

.. list-table:: 
   :widths: 20 80
   :header-rows: 1

   * - Category
     - Instructions
   * - Configuration
     - ``VSETVLI``
   * - Load/store
     - ``VLE32.V``, ``VSE32.V`` (unit-stride, 32-bit elements)
   * - Arithmetic
     - ``VADD``, ``VSUB``, ``VAND``, ``VOR``, ``VXOR``
       (``.VV`` and ``.VX`` variants)
   * - Shifts
     - ``VSLL.VX``, ``VSRL.VX``
   * - Comparison
     - ``VMSEQ(.VV|.VX)``, ``VMSLT.VV``, ``VMSLTU.VV``
   * - Mask-mask
     - ``VMAND``, ``VMNAND``, ``VMANDN``, ``VMXOR``, ``VMOR``,
       ``VMNOR``, ``VMORN``, ``VMXNOR`` (``.MM``)
   * - Reduction
     - ``VREDSUM.VS``
   * - Move
     - ``VMV.V.X`` (splat scalar → vector),
       ``VMV.X.S`` (extract element 0 → scalar)

CSR Access (6 instructions + MRET)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CSRRW``, ``CSRRS``, ``CSRRC``, ``CSRRWI``, ``CSRRSI``, ``CSRRCI`` — atomic read-modify-write
of control/status registers. ``MRET`` returns from a machine-mode trap.

.. list-table:: **Supported CSRs** 
   :widths: 15 15 70
   :header-rows: 1

   * - Name
     - Address
     - Purpose
   * - ``mstatus``
     - 0x300
     - Global interrupt enable (MIE, MPIE bits)
   * - ``mie``
     - 0x304
     - Per-source interrupt enable (MEIE, MTIE, MSIE)
   * - ``mtvec``
     - 0x305
     - Trap vector base address
   * - ``mscratch``
     - 0x340
     - Scratch register for trap handlers
   * - ``mepc``
     - 0x341
     - Exception PC (saved on trap entry)
   * - ``mcause``
     - 0x342
     - Trap cause code
   * - ``mip``
     - 0x344
     - Pending interrupt bits (read-only)
   * - ``mhartid``
     - 0xF14
     - Hart ID (read-only)

``mtvec`` supports both MODE values: direct (all traps to BASE) and
vectored (interrupts to BASE + 4 × cause). Reserved MODE values read back
as direct

Interrupt priority follows the privileged spec: external (MEI) > software
(MSI) > timer (MTI).

Interrupts are precise on both core models. The single-cycle CPU takes
them between instructions. The pipeline takes them at the EX boundary:
instructions in MEM and WB complete, the instructions in EX/ID/IF are
squashed, and ``mepc`` holds the oldest squashed instruction, so ``mret``
resumes exactly where the program was interrupted.

The CSR pseudo-instructions ``csrr``, ``csrw``, ``csrs``, ``csrc``,
``csrwi``, ``csrsi``, and ``csrci`` are supported by the assembler.


Pipeline
--------

The simulator offers two CPU modes:

- **Simple CPU** — single-cycle execution, one instruction per call to
  ``step()``.
- **Pipelined CPU** — classic 5-stage in-order pipeline:
  ::

      IF → ID → EX → MEM → WB

  Each stage advances once per ``tick()`` call. The pipeline models
  structural, data, and control hazards with configurable mitigation.

Pipeline stages
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10 10 80
   :header-rows: 1

   * - Stage
     - Name
     - Function
   * - ``IF``
     - Fetch
     - Read instruction from memory at PC. Branch predictor selects
       next PC (fall-through or predicted target).
   * - ``ID``
     - Decode
     - Decode instruction, read register file, detect hazards.
       Insert stall bubble if needed.
   * - ``EX``
     - Execute
     - ALU operation, branch resolution, address calculation.
       Forwarding paths inject bypassed values here.
   * - ``MEM``
     - Memory
     - Data cache access for loads and stores.
   * - ``WB``
     - Writeback
     - Write result to register file.

Forwarding (data hazard mitigation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Mode
     - Behavior
   * - ``none``
     - Stall on every RAW dependency until the producing instruction reaches WB.
       Worst case: 2-cycle stall per dependency.
   * - ``partial``
     - Forward from MEM→EX only. Reduces stalls but still stalls
       on EX→EX dependencies.

Branch prediction
^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Predictor
     - Strategy
   * - ``not_taken``
     - Always predict fall-through. Mispredicts on every taken branch.
   * - ``always_taken``
     - Always predict taken. Good for tight backward loops, bad for
       forward branches (if-then).
   * - ``backward_taken``
     - Predict taken if target < PC (backward branch), not-taken otherwise.
   * - ``bimodal_1bit``
     - 1-bit saturating counter per PC. Adapts to branch history but "ping-pongs"
       on alternating outcomes.
   * - ``bimodal_2bit``
     - 2-bit saturating counter per PC. Requires two consecutive mispredicts to
       change predictor.

Misprediction penalty is configurable (default: 3 cycles). On a
mispredict, the pipeline flushes all stages after EX and restarts
fetch at the correct target.


Cache Hierarchy
---------------

The cache hierarchy is fully configurable at each level. All levels
are optional: the CPU can run directly against main memory (useful
for isolating pipeline effects in benchmarks).
::

    CPU ─┬─ data ──── [L1D] ─┬─ [L2] → [L3] → Main Memory
         └─ fetch ─── [L1I] ─┘

With no L1I configured, the L1 is unified and serves both paths. The L1I
is read-only. Its refills see the newest data, including dirty D-cache
lines, but a line already in the I-cache can go stale after a store until
the hart executes ``fence.i``, which invalidates that hart's I-cache and
refetches every younger instruction. RISC-V allows this. Device regions
are not executable: a fetch from one faults.

L1 cache (per-core in multi-core mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Size**: configurable (e.g., 4K, 8K, 16K, 32K, 64K)
- **Associativity**: configurable (must be power of 2)
- **Line size**: configurable (e.g., 32, 64, 128 bytes)
- **Replacement policy**: LRU, MRU, Pseudo-LRU, FIFO, Random
- **Write policy**: write-back or write-through
- **Write-allocate**: allocate on write miss, or no-allocate

L2 cache (shared in multi-core mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Required when L1 is present in multi-core configurations
- Configurable size and associativity

L3 cache (shared, optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Configurable size and associativity

Geometry constraints (enforced by the config validator)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- All sizes, associativities, and line sizes must be powers of 2
- ``size % (line_size × associativity)`` must equal 0
- Cache sizes must not decrease with distance from CPU (L2 ≥ L1, L3 ≥ L2)
- L2 requires L1; L3 requires L2

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Policy
     - Name
     - Description
   * - ``lru``
     - Least Recently Used
     - Evicts the line accessed longest ago.
   * - ``mru``
     - Most Recently Used
     - Evicts the most recently accessed line.
   * - ``plru``
     - Pseudo-LRU
     - Tree-based approximation of LRU using one bit per internal node.
   * - ``fifo``
     - First-In First-Out
     -  Evicts the oldest line regardless of access pattern.
   * - ``random``
     - Random eviction
     -


Multi-Core System
-----------------

The multi-core system connects N pipelined cores through a shared
memory hierarchy with hardware cache coherence.
::

    Core 0 → MMIOBus ─┬─ RAM range → L1d[0] ──┐
                      └─ devices (CLINT, PLIC, NIC)
                                              ├── Coherence Controller → L2 → L3 → RAM
    Core 1 → MMIOBus ─┬─ RAM range → L1d[1] ──┘
                      └─ devices

Cache modes
^^^^^^^^^^^

1. **No caches** — all cores access main memory directly through
   their MMIO bus. Useful for testing device interactions without
   cache noise.

2. **L1 + L2** — per-core L1 data caches with a shared L2.  A
   directory-based MESI coherence controller manages cross-core
   consistency.

3. **L1 + L2 + L3** — same as above with an additional shared L3
   between L2 and main memory.

MESI coherence protocol
^^^^^^^^^^^^^^^^^^^^^^^

The coherence controller is directory-based (not snoopy-bus).  Each
cache line can be in one of four states:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - State
     - Meaning
   * - Modified
     - Line is dirty, only this core has it. Must write back before sharing.
   * - Exclusive
     - Line is clean, only this core has it. Can transition to Modified on write
       without bus traffic.
   * - Shared
     - Line is clean, multiple cores may have it. Must invalidate other copies
       before writing.
   * - Invalid
     - Line is not present in this cache.

**Coherence latency model:**

- snoop = 2 cycles
- cache-to-cache transfer = 5 cycles
- invalidation = 2 cycles
 
Platform and device MMIO
^^^^^^^^^^^^^^^^^^^^^^^^

All core models — the single-cycle CPU, the pipelined CPU, and the
multi-core system — run on the same platform, which owns RAM, the devices,
the physical memory map, and the interrupt wiring. Devices sit at fixed
addresses that follow the common SiFive / QEMU ``virt`` layout and do not
depend on RAM size:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Base
     - Size
     - Region
   * - +0x0000_0000
     - RAM size
     - RAM (demand-allocated; base is configurable)
   * - 0x0200_0000
     - 64 KiB
     - CLINT: ``msip[h]`` at +0x0 _ 4·h, ``mtimecmp[h]`` at +0x4000 + 8·h,
       ``mtime`` at +0xBFF8
   * - 0x0C00_0000
     - 64 MiB
     - PLIC: priorities at +0x0, pending at +0x1000, enables at
       +0x2000 + 0x80·h, threshold / claim at +0x200000 + 0x1000·h
   * - 0x1000_0000
     - 4 KiB
     - NIC (only when a NIC is attached)

With RAM at address 0, RAM can be at most 32 MiB before it would overlap
the CLINT; the config validator enforces this. Accesses that hit no region
fault. Device accesses bypass the caches: each hart's bus routes device
addresses straight to the device, and only the RAM range is forwarded to
the hart's L1 (or to RAM when caches are disabled).

Devices
^^^^^^^

- **PLIC (Platform-Level Interrupt Controller)** — routes up to 31
  external interrupt sources with priority-based arbitration. Each hart
  has its own M-mode context (enable bits, threshold, claim/complete), so
  external interrupts reach only the harts that enabled them.
- **CLINT (Core-Local Interruptor)** — a shared 64-bit ``mtime``
  counter, plus a per-hart ``mtimecmp`` (drives that hart's ``mip.MTIP``)
  and a per-hart ``msip`` (drivse ``mip.MSIP``, for inter-processor
  interrupts). ``mtime`` advances once per core cycle by default; a
  timebase divider can make it slower than the core clock.
- **NIC (Network Interface Controller)** — cycle-accurate DMA-based
  NIC with descriptor rings, interrupt coalescing, and RSS
  (Receive Side Scaling) for multi-queue packet distribution.
  Supports up to 8 RX/TX queues with configurable coalescing
  thresholds (packet count and timer-based). Construct it with the
  system's ``dma_memory()`` so its DMA stays coherent with the D-caches.


Assembler
---------

The two-pass assembler translates ``.s`` source files into machine
code loaded directly into the simulator's memory.

- **Pass 1:** Scan lines, record label addresses, compute instruction
  sizes (LI and LA expand to 2 instructions = 8 bytes).
- **Pass 2:** Encode instructions using the label table for branch and
  jump target resolution.

Syntax
^^^^^^

- **Labels:** ``name:`` at the start of a line
- **Comments:** ``#``, ``;``, or ``//`` to end of line
- **Register names:** ``x0``–``x31`` and ABI names (``zero``, ``ra``,
  ``sp``, ``a0``–``a7``, ``s0``–``s11``, ``t0``–``t6``, ``fp``)
- **Vector registers:** ``v0``–``v31``
- **Immediates:** decimal (``42``, ``-10``), hex (``0xFF``)
- **Memory operands:** ``offset(reg)`` or ``(reg)``
- **Directives:** ``.word``, ``.byte``, ``.zero``, ``.align``

Pseudo-instructions
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Pseudo
     - Expansion
   * - ``nop``
     - ``addi x0, x0, 0``
   * - ``mv``
     - ``addi rd, rs, 0``
   * - ``li``
     - ``addi`` (small) or ``lui+addi``
   * - ``j``
     - ``jal x0, offset``
   * - ``jr``
     - ``jalr x0, rs, 0``
   * - ``ret``
     - ``jalr x0, ra, 0``
   * - ``not``
     - ``xori rd, rs, -1``
   * - ``neg``
     - ``sub rd, x0, rs``
   * - ``beqz``
     - ``beq rs, x0, offset``
   * - ``bnez``
     - ``bne rs, x0, offset``
   * - ``seqz``
     - ``sltiu rd, rs, 1``
   * - ``snez``
     - ``sltu rd, x0, rs``
   * - ``vmnot.m``
     - ``vmnand.mm vd, vs, vs``
   * - ``vmclr.m``
     - ``vmxor.mm vd, vd, vd``
   * - ``vmset.m``
     - ``vmxnor.mm vd, vd, vd``
   * - ``vmcpy.m``
     - ``vmand.mm vd, vs, vs``

