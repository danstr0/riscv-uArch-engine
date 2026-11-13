=================
Research Platform
=================

The research platform (``rvsim``) assembles a RISC-V program, runs
it under a configurable microarchitecture, and reports performance
metrics. When sweep axes are defined in the config file, it runs
the program under every configuration in the Cartesian product and
outputs CSV data suitable for analysis and visualization.


Usage
-----
::

    rvsim <source.s> [--config <file.cfg>] [--csv]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``source.s``
     - RISC-V assembly source file.
   * - ``--config``
     - Configuration file
   * - ``--csv``
     - Force CSV output for single-run mode.
   * - ``--version`` / ``-v``
     - Print version and exit.
   * - ``--help`` / ``-h``
     - Print usage and exit.

Modes
-----

Single run (no [sweep] section)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assembles the program, executes it once with the configured
architecture, and prints results in a human-readable box format
to stderr::

    $ rvsim programs/sum_to_n.s --config config/sum_to_n.cfg

     -- Architecture --
      L1: 32768B, 8-way, line=64, replacement=lru, writeback
      L2: 0B, 8-way (disabled)
      L3: 0B, 16-way (disabled)
      Pipeline: forwarding=partial, branch_pred=bimodal_2bit, mispredict_penalty=3

     -- Results --
      Cycles:           318
      Instructions:     304
      IPC:              0.9560
      Stalls:           9
      Flushes:          3
      L1 hit rate:      99.68%  (1 misses)
      L2 hit rate:      0.00%  (0 misses)
      L3 hit rate:      0.00%  (0 misses)
      Branches:         100 (3 mispredicts, 97.0% accuracy)

Add ``--csv`` to get a single CSV row on stdout instead.

Sweep mode (config file has [sweep] section)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The research platform computes the Cartesian product of all sweep
axes, runs the program under each configuration, and writes one
CSV row per configuration to stdout.  Informational messages
(assembly status, skip reasons, summary) go to stderr.
::

    $ rvsim programs/sum_to_n.s --config configs/sum_sweep.cfg > results/sum_to_n.csv

    # stderr shows:
    Loaded config from examples/sum_sweep.cfg
    Assembled 28 bytes (1 labels) from examples/sum_to_n.s
    Sweep complete: 6 configurations

    # stdout (results.csv) contains:
    pipeline.forwarding,pipeline.branch_pred,cycles,instructions,...
    none,not_taken,807,304,...
    none,backward_taken,519,304,...
    ...
    partial,bimodal_2bit,318,304,...

.. note::

  Always test-run a program with single run mode; an error-producing
  (e.g., misaligned load) program should never be input for sweep mode.

CSV Output Format
-----------------

The CSV has two sections: sweep parameter columns (one per axis),
followed by metric columns.

**Sweep parameter columns** appear in the order they are defined
in the ``[sweep]`` section. Their names are the qualified parameter
names (e.g., ``pipeline.forwarding``, ``cache.l1.size``).

**Metric columns** (fixed order):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``cycles``
     - Total simulation cycles.
   * - ``instructions``
     - Instructions retired.
   * - ``ipc``
     - Instructions per cycle.
   * - ``stalls``
     - Total pipeline stalls
   * - ``flushes``
     - Pipeline flushes from branch mispredictions.
   * - ``l1_hit_rate``
     - L1 data cache hit rate (0.0–1.0), or the unified L1's when
       ``[cache.l1i]`` is disabled. ``0`` if no L1.
   * - ``l1_misses``
     - L1 cache miss count.
   * - ``l1i_hit_rate``
     - L1 instruction cache hit rate. ``0`` if the L1 is unified.
   * - ``l1i_misses``
     - L1 instruction cache miss count.
   * - ``l2_hit_rate``
     - L2 cache hit rate.  0 if no L2.
   * - ``l2_misses``
     - L2 cache miss count.
   * - ``l3_hit_rate``
     - L3 cache hit rate.  0 if no L3.
   * - ``l3_misses``
     - L3 cache miss count.
   * - ``branches``
     - Total branch instructions executed.
   * - ``mispredicts``
     - Branch misprediction count.
   * - ``branch_accuracy``
     - Branch prediction accuracy (0.0-1.0)


Cross-Axis Geometry Skipping
----------------------------

When sweeping multiple cache parameters simultaneously, some
combinations of values from different axes produce invalid cache
geometries.  Rather than rejecting the entire config file, the
research platform skips invalid combinations at runtime.

Skipped conditions:

- L2 size <= L1 size
- L3 size <= L2 size
- L1 enabled without L2
- Cache ``size % (line_size × associativity) != 0``

Skipped configurations print a diagnostic to stderr::

    SKIP: L1 size not divisible by line*assoc (cache.l1.size=4K, cache.l1.assoc=16)

No CSV row is emitted for skipped configurations.  This keeps the
output clean for the visualizer.


Workflow
--------

A typical research workflow:

1. **Write the workload** — a ``.s`` assembly file implementing the
   algorithm you want to benchmark.  Keep it pure: data setup and
   register initialization go in the config file, not in the assembly.
2. **Write the config** — a ``.cfg`` file that pre-loads input data
   via ``[memory.init]``, sets initial registers via ``[registers]``,
   and defines sweep axes via ``[sweep]``.
3. **Run the sweep** — ``rvsim program.s --config sweep.cfg > results.csv``
4. **Visualize** — ``python tools/visualize.py results.csv``
5. **Interpret** — the plots show which parameters matter, where the
   diminishing returns are, and what the Pareto-optimal configurations
   look like.


Visualization
-------------

The ``tools/visualize.py`` script reads the sweep CSV and generates
PNG plots.

::

    python tools/visualize.py results.csv [options]

.. list-table:: **Options**
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--metric m1 m2 ...``
     - Which metrics to plot (y-axes on charts, axes on parallel coordinates).

       Default: ``ipc``, ``l1_hit_rate``, ``cycles``.
   * - ``--pareto m1 m2 ...``
     - Which metrics define Pareto optimality (the objectives to optimize
       simultaneously).

       Default: ``ipc``, ``l1_hit_rate``.
   * - ``--output dir``
     - Output directory for PNGs.  Default: ``plots/``.

.. list-table:: **Plot types**
   :header-rows: 1
   :widths: 10 90

   * - Swept parameters
     - Plots generated
   * - 1
     - Bar charts (one per metric, with error bars).
   * - 2
     - Heatmaps (param × param, color = metric) + bar charts.
   * - 3+
     - Pairwise scatter plots per param-metric pair.
   * - Any
     - Pareto front scatter (metric × metric, optimal points circled in red).
       Parallel coordinates (all axes, Pareto-optimal lines in red, others
       grayed out).

**Pareto front:**

A configuration is Pareto-optimal if no other configuration is
better on every objective simultaneously.  "Better" means higher
for IPC, hit rates, and accuracy; lower for cycles, misses, and
stalls.  The Pareto front is the set of non-dominated configurations
— these represent the best available tradeoffs.

The ``--metric`` flag controls what you see.  The ``--pareto`` flag
controls what gets highlighted.  They can differ: you might plot
five metrics but define optimality on only two.

