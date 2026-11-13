==================
Configuration File
==================

The ``.cfg`` file is the location for setting architecture
parameters, memory initialization, register values, and sweep axes.

The file uses a simple INI/TOML-style syntax with dotted section names.
All parameters have sensible defaults; an empty config file (or no
config file at all) still produces a working system.


Syntax
------
::

    # Comments start with # or ;
    [section.name]
    key = value

- Blank lines are ignored.
- Comments run from ``#`` or ``;`` to end of line.
- Section headers are enclosed in brackets: ``[section.name]``.
- Key-value pairs use ``=`` as the separator.
- Size values accept K and M suffixes: ``32K`` = 32768, ``1M`` = 1048576.
- Numeric values can be decimal (``42``) or hex (``0xFF``).
- Unknown section names are rejected (catches typos like ``[cach.l1]``).


Sections Reference
------------------

[memory]
^^^^^^^^

Controls the main memory backing store.

.. list-table::
   :header-rows: 1
   :widths: 10 10 80

   * - Key
     - Default
     - Description
   * - ``size``
     - ``64K``
     - Total memory size. Accepts K/M suffixes.

**Example:**
::

    [memory]
    size = 128K

[memory.init]
^^^^^^^^^^^^^

Pre-loads data into memory before execution begins.  Each line is
an address-value pair.  The address is the key; the value format
determines how data is written.

.. list-table:: **Value formats**
   :header-rows: 1
   :widths: 50 50

   * - Format
     - Example
   * - ``u32`` array
     - ``0x8000 = [1, 2, 3, 4]``
   * - Single ``u32`` value
     - ``0x5000 = 42``
   * - Fill region
     - ``0x9000 = fill(0xFF, 256)``
   * - ASCII string (null-term)
     - ``0xA000 = "hello"``

**Hex values work in arrays:**
::

    0x1000 = [0xDEAD, 0xBEEF]

**Validation rules:**

- ``u32`` arrays must be 4-byte aligned.
- Data must not extend past the configured memory size. Devices sit at
  fixed addresses above RAM, so data that stays inside RAM can never
  overlap them.

**Example:**
::

    [memory.init]
    0x8000 = [10, 20, 30, 40, 50, 60, 70, 80]   # u32 array
    0x9000 = fill(0x00, 1024)                   # 1 KB of zeros
    0xA000 = "packet_data"                      # null-terminated string
    0xB000 = 0xDEADBEEF                         # single word

[cache.l1]
^^^^^^^^^^

Per-core L1 data cache, which also serves instruction fetches unless
``[cache.l1i]`` is enabled.  Set ``size = 0`` to disable caching
entirely (CPU reads/writes go directly to main memory).

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Default
     - Values
   * - ``size``
     - ``4K``
     - Power of 2 (or ``0`` to disable)
   * - ``assoc``
     - ``8``
     - Power of 2
   * - ``line``
     - ``64``
     - Power of 2
   * - ``replacement``
     - ``lru``
     - ``lru``, ``mru``, ``plru``, ``fifo``, ``random``
   * - ``write``
     - ``writeback``
     - ``writeback`` (``wb``), ``writethrough`` (``wt``)
   * - ``write_alloc``
     - ``allocate``
     - ``allocate``, ``no_allocate``

**Geometry constraint:**
``size`` must be divisible by ``line × assoc``.  For example:
::

    size=4K, line=64, assoc=8     → 4096 / (64 × 8) = 8 sets — valid.
    size=4K, line=64, assoc=128   → 4096 / 8192 — rejected.

[cache.l1i]
^^^^^^^^^^^

Optional L1 instruction cache. When enabled, instruction fetches stop
going through ``[cache.l1]`` and the two are separate caches.  Leave
``size = 0`` for a unified L1.  Requires ``[cache.l1]``, and shares its
line size.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Default
     - Values
   * - ``size``
     - ``0``
     - Power of 2 (or ``0`` for a unified L1)
   * - ``assoc``
     - ``8``
     - Power of 2
   * - ``replacement``
     - ``lru``
     - ``lru``, ``mru``, ``plru``, ``fifo``, ``random``

[cache.l2]
^^^^^^^^^^

Shared L2 cache.  Required when L1 is enabled in multicore mode.
Set ``size = 0`` to disable.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Default
     - Values
   * - ``size``
     - ``0``
     - Power of 2 (or ``0`` to disable)
   * - ``assoc``
     - ``8``
     - Power of 2

L2 shares the line size with L1 (set in ``[cache.l1] line``).
L2 size must be larger than both L1 sizes.

[cache.l3]
^^^^^^^^^^

Shared L3 cache.  Optional; requires L2 to be enabled.
Set ``size = 0`` to disable (default).

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Default
     - Values
   * - ``size``
     - ``0``
     - Power of 2 (or ``0`` to disable)
   * - ``assoc``
     - ``16``
     - Power of 2
 
L3 size must be ≥ L2 size.

[pipeline]
^^^^^^^^^^

Pipeline configuration. Applies to both pipelined and multicore
CPU modes (ignored for simple CPU mode).

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Default
     - Values
   * - ``forwarding``
     - ``partial``
     - ``none``, ``partial``
   * - ``branch_pred``
     - ``bimodal_2bit``
     - ``not_taken``, ``always_taken``, ``backward_taken``,
       ``bimodal_1bit``, ``bimodal_2bit``
   * - ``mispredict_penalty``
     - ``3``
     - Integer ≥ 1

[system]
^^^^^^^^

System-level configuration.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Default
     - Values
   * - ``cpu``
     - ``pipelined``
     - ``simple``, ``pipelined``, ``multicore``
   * - ``cores``
     - ``1``
     - 1-16 (must be ≥ 2 for multicore)
   * - ``max_cycles``
     - ``10000000``
     - Maximum simulation cycles

The ``cpu`` field selects the CPU model:
 
- ``simple`` — single-cycle execution, no pipeline. Currently unused.
- ``pipelined`` — 5-stage in-order pipeline with configurable
  forwarding and branch prediction. Used by the research platform.
- ``multicore`` — N pipelined cores with shared memory, cache
  coherence, and MMIO devices. Currently unused.

The research platform, as of writing this, always uses pipelined mode regardless of this
setting.

[registers]
^^^^^^^^^^^

Initial register values applied to all cores.  Register names use
either ABI names (``a0``, ``sp``, ``t0``) or numeric names
(``x1``, ``x10``).

Values can be decimal or hex.  Size suffixes are accepted
(``sp = 64K`` sets ``sp`` to ``65536``).

::

    [registers]
    a0 = 8            # number of elements
    a1 = 0x8000       # data pointer
    sp = 0xFFF0       # stack pointer

**Rules:**
 
- Cannot set ``x0`` / ``zero`` (hardwired to zero).
- Register index must be 0–31.

[cores.N]
^^^^^^^^^

Per-core configuration for multi-core programs. ``N`` is the
zero-based core index.  These sections are currently unused by
the research platform.

Each core can have its own program, starting PC, and register
overrides.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Default
     - Description
   * - ``program``
     - (none)
     - Path to ``.s`` source file for this core. If omitted, the
       core uses the main source file passed on the command line.
   * - ``pc``
     - ``0``
     - Address where the program is loaded and where the core
       begins execution.
   * - (register)
     - —
     - Any register name (e.g., ``a0 = 5``) sets that register for
       this core only.

**Register precedence:**

1. ``[registers]`` section is applied to all cores (global baseline).
2. ``[cores.N]`` register assignments override per-core.

This means you write global defaults once and specialize per-core
where needed:
::

    [registers]
    sp = 0xFFF0        # all cores
    a1 = 0x8000        # all cores: shared buffer

    [cores.0]
    program = producer.s
    pc = 0x0000
    a0 = 0             # core 0: hart_id = 0

    [cores.1]
    program = consumer.s
    pc = 0x4000
    a0 = 1             # core 1: hart_id = 1

Each core's program is assembled with its own label namespace.
Shared data is communicated via ``[memory.init]`` addresses.

[sweep]
^^^^^^^

Defines parameter axes for the research platform's sweep mode.
Each key is a qualified parameter name; the value is a bracketed
list of values to try.
::

    [sweep]
    cache.l1.size = [16K, 32K, 64K]
    pipeline.forwarding = [none, partial]
    pipeline.branch_pred = [not_taken, bimodal_2bit]

The research platform computes the Cartesian product of all axes
and runs the assembled program under each configuration. For the
example above, there's 3 × 2 × 2 = 12 configurations.

**Valid sweep parameters:**

All parameters from ``[cache.l1]``, ``[cache.l2]``, ``[cache.l3]``,
``[pipeline]``, and ``[system]`` can be swept using their qualified
names:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Qualified name
     - Corresponding section key
   * - ``cache.l1.size``
     - ``[cache.l1] size``
   * - ``cache.l1.assoc``
     - ``[cache.l1] assoc``
   * - ``cache.l1.line``
     - ``[cache.l1] line``
   * - ``cache.l1.replacement``
     - ``[cache.l1] replacement``
   * - ``cache.l1.write``
     - ``[cache.l1] write``
   * - ``cache.l1.write_alloc``
     - ``[cache.l1] write_alloc``
   * - ``cache.l1i.size``
     - ``[cache.l1i] size``
   * - ``cache.l1i.assoc``
     - ``[cache.l1i] assoc``
   * - ``cache.l1i.replacement``
     - ``[cache.l1i] replacement``
   * - ``cache.l2.size``
     - ``[cache.l2] size``
   * - ``cache.l2.assoc``
     - ``[cache.l2] assoc``
   * - ``cache.l3.size``
     - ``[cache.l3] size``
   * - ``cache.l3.assoc``
     - ``[cache.l3] assoc``
   * - ``pipeline.forwarding``
     - ``[pipeline] forwarding``
   * - ``pipeline.branch_pred``
     - ``[pipeline] branch_pred``
   * - ``pipeline.mispredict_penalty``
     - ``[pipeline] mispredict_penalty``
   * - ``system.cpu``
     - ``[system] cpu``
   * - ``system.cores``
     - ``[system] cores``
   * - ``system.max_cycles``
     - ``[system] max_cycles``

**Validation rules:**

- Parameter names must be recognized (rejects typos like
  ``cache.l1.sizee``).
- Each value must be valid for its parameter (e.g., cache sizes
  must be powers of 2).
- Duplicate values within an axis are rejected.

**Cross-axis geometry skipping:**
 
When sweeping multiple cache parameters simultaneously, some
combinations are geometrically invalid. The research platform
skips these at runtime (printing a message to stderr) rather than
rejecting them at config time.  Skipped combinations include:

- L2 size < L1 size
- L3 size < L2 size
- L1 without L2
- Cache size not divisible by ``line_size × associativity``


Validation Summary
------------------

The config parser validates the entire configuration after parsing.
Here is the complete list of checks, grouped by category.

**Memory:**

- Size must be > 0 and fit in a 32-bit address space.
- ``max_cycles`` must be > 0.
- Init data must not extend past memory size.
- ``u32`` arrays must be 4-byte aligned.

**Cache geometry (per level):**

- Size, associativity, and line size must be powers of 2.
- ``size % (line_size × associativity)`` must be 0.
- Line size must be ≤ cache size.

**Cache hierarchy:**

- L2 requires L1; L3 requires L2.
- Sizes must not decrease: L2 > L1, L3 > L2.
- Multicore with L1 requires L2.

**Enum strings:**

- Replacement, write policy, write-allocate, forwarding, branch
  predictor, and CPU type must all be recognized values.

**Pipeline:**

- ``mispredict_penalty`` must be ≥ 1.

**System:**

- CPU type must be ``simple``, ``pipelined``, or ``multicore``.
- Multicore requires ≥ 2 cores.
- Non-multicore rejects > 1 core.
- Core count must be 1–16.

**Registers:**

- Cannot set ``x0``.
- Register index must be 0–31.

**Per-core (stepper):**

- Core index must not exceed ``num_cores``.
- PC must be within memory.
- Same ``x0`` and range checks as global registers.

**Sweep:**

- Parameter names must be recognized.
- No duplicate values within an axis.
- Each value must pass the same validation as its static equivalent.

**Memory map (all CPU types):**

- RAM starts at address 0, and devices sit at fixed addresses (CLINT at
  0x0200_0000, PLIC at 0x0C00_0000, NIC at 0x1000_0000; see
  :doc:`architecture`). Memory size must therefore be at most 32 MiB so
  RAM ends before the CLINT.

Complete Example
----------------
::

    # benchmark.cfg — Full example featuring all sections used by the research platform

    [memory]
    size = 256K

    [memory.init]
    # Two 8-element vectors for dot product
    0x8000 = [1, 2, 3, 4, 5, 6, 7, 8]
    0x8100 = [8, 7, 6, 5, 4, 3, 2, 1]

    [cache.l1]
    size = 16K
    assoc = 8
    line = 64
    replacement = lru
    write = writeback
    write_alloc = allocate

    [cache.l2]
    size = 64K
    assoc = 8

    # L3 disabled

    [pipeline]
    forwarding = partial
    branch_pred = bimodal_2bit
    mispredict_penalty = 3

    [system]
    cpu = pipelined
    max_cycles = 10000000

    [registers]
    a0 = 8          # vector length
    a1 = 0x8000     # pointer to vector A
    a2 = 0x8100     # pointer to vector B

    [sweep]
    cache.l1.size = [8K, 16K, 32K, 64K]
    pipeline.forwarding = [none, partial, full]

