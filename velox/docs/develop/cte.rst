Explicit Common Table Expressions
=================================

A CTE defines one computation and independent readers of its rows, all inside
one Velox task. The ``exec::test::PlanBuilder`` helper exposes ``with()``,
``fromCte()``, and ``build()``. Producers and consumers may contain any relational
operators supported by Velox in the execution scope described below. There is
no CTE-specific operator whitelist. Finalization preserves relational subplans
and keeps a terminal partitioned/broadcast output above the CTE bindings.

Use ``newPlan()`` for another branch of the same query. It shares the node ID
generator and inherits a snapshot of the named definitions. Use ``planNode()``
for intermediate branches and ``build()`` only at the complete query root.
Definitions added in a branch stay local to that branch. Definitions may refer
to earlier definitions; recursive definitions are not supported.

Example: revenue and maximum revenue
-----------------------------------

Here ``input`` contains supplier IDs in ``c0`` and revenue contributions in
``c1``. This is the shared aggregation and maximum-revenue portion of TPC-H Q15;
the date filter and final supplier lookup are omitted.

.. code-block:: cpp

    PlanBuilder query;
    query.with("revenue", [&](PlanBuilder& cte) {
      cte.values({input})
          .singleAggregation({"c0"}, {"sum(c1) as revenue"});
    });

    auto maximum = query.newPlan()
        .fromCte("revenue")
        .partialAggregation({}, {"max(revenue) as max_revenue"})
        .localPartition({})
        .finalAggregation()
        .planNode();

    auto plan = query.fromCte("revenue")
        .hashJoin({"revenue"}, {"max_revenue"}, maximum, "", {"c0", "revenue"})
        .build();
    // Execute 'plan' using the normal Task API.

The producer computes revenue once, including any volatile expressions. Each
``fromCte()`` creates a distinct reader. The maximum aggregation can consume
batches as they arrive, while the probe reader may wait for the maximum to
finish. No separate task or application-managed result table is required.

``build()`` registers only definitions and readers reachable from the complete
query, including references inside other definitions. Unused definitions and
abandoned branches do not execute or pin buffered data. Multiple independent
and dependent CTEs are supported. A single-use CTE keeps its explicit producer
and reader; inlining it is a future optimizer decision.

Direct plan-node API
--------------------

Applications and plan translators can use ``exec::Cte`` from
``velox/exec/Cte.h`` without the test PlanBuilder utility:

.. code-block:: cpp

    exec::Cte revenue(ids.next(), revenueProducer);
    auto first = revenue.scan(ids.next());
    auto second = revenue.scan(ids.next());
    // Assemble ordinary Velox nodes around first and second into queryRoot.
    auto executable = exec::finalizeCtes(queryRoot);

All definition and plan-node IDs must be unique in the complete query. Each use
needs its own ``scan()`` node. A definition handle can be copied; its scans
continue to refer to the same computation. A scan keeps its definition alive.
Call ``finalizeCtes()`` once at the root. Calling ``build()`` repeatedly on the
original builder is supported and does not mutate its logical plan.

Finalization traverses references and adds producer pipelines. It neither
finds equivalent expressions nor infers sharing from repeated plan-node
pointers. It preserves producer and consumer subtrees; a terminal
``PartitionedOutput`` is copied with its source replaced to keep it at the root,
preserving its output layout, partitioning, and serialization settings. A future
optimizer or DuckDB plan translator can explicitly choose CTE boundaries without
requiring CTE-specific handling of joins, windows, sorts, or repartitioning.

This supplies a representation for non-recursive CTE producers and scans; it
is not a DuckDB plan importer. Translating DuckDB's other operators, expressions,
correlation/decorrelation machinery, types, and null semantics remains separate
work. The TPC-H benchmark includes manual translations of the six observed
SF10 CTE sharing shapes, with validated comparisons described below.

Plan serialization
------------------

Finalize the complete query with ``build()`` or ``finalizeCtes()`` before
serializing. Register ``exec::registerCteSerDe()`` alongside the usual type,
expression, core plan-node, connector, and partition-function serializers at
application startup. Then use the standard APIs:

.. code-block:: cpp

    auto serialized = executable->serialize();
    auto restored = ISerializable::deserialize<core::PlanNode>(serialized, pool);
    // Execute restored directly; it is already finalized.

Each producer is serialized once under its binding. Scans carry only their
binding ID and output type, preserving sharing and dependent CTEs without
embedding another copy of the producer. The optional retention limit is also
preserved (as a decimal string for the full unsigned 64-bit range); an absent
field means unlimited. Terminal partitioned/broadcast output stays at the root.
Local planning validates reader IDs, types, bindings, and acyclic dependencies
before starting execution. Serialization transports a plan, not a running
queue or its retained rows; execution still shares only within one task.

Streaming and retention
-----------------------

Each public CTE scan makes batches available immediately,
without waiting for full materialization. A task-owned buffer retains batches
until every open logical reader has claimed them, then drops its references.
Downstream references keep payloads alive until the last consumer releases them. Each
logical reader has its own cursor; its drivers claim disjoint batches. A driver
closing early does not retire the reader while peers remain. The last driver
closing removes that reader from the retention decision.
Cancellation wakes blocked readers through the task bridge lifecycle.
Publication wakes one waiting driver per logical reader. If that driver closes
without claiming an available batch, a waiting peer is notified. Producer
completion wakes all waiting readers.

Retained readers do not backpressure the producer. This is safe for arbitrary
acyclic consumer dependencies, including a join probe waiting for a build that
reads the same CTE. It requires no operator-specific scheduling rules or cost
estimates. In the worst case it retains the entire result, so eager streaming
does not imply bounded streaming memory.

By default there is no additional per-CTE retention limit. Payload buffers remain
charged to the Velox pools that allocated them for as long as any reader or the
queue retains them. New producer allocations use ordinary pool reservation and
memory arbitration, within the query's root-pool capacity. Sharing neither
copies nor charges the same allocation a second time. Downstream writable
copies are charged to their allocating pools as usual.

The retained CTE payload is not spillable/reclaimable. Arbitration can obtain
available capacity and reclaim eligible memory elsewhere; it cannot evict rows
that a reader still needs. If the query cannot obtain enough memory, the normal
Velox memory error fails the query. There is no separate CTE memory manager.
Application-supplied vectors (for example, Values inputs) keep their original
pool ownership; applications must allocate them under the intended query pool
if they want them included in that query's budget.

Set ``CteOptions::maxRetainedBytes`` to a positive byte count to add an optional
per-CTE guard, or leave it ``std::nullopt`` for the default. The guard counts
buffered vector storage plus estimated per-batch metadata; even zero-column
batches incur a positive charge. Exceeding it fails the query rather than
waiting on a dependent reader. This is not a total query-memory limit: operator
state, consumer metadata, and downstream references to claimed batches are
additional memory. The per-batch charge is conservative when several batches
refer to the same storage. Preserving an encoding can also retain dictionary
or array storage beyond the selected rows. Queue entries and their root vector
metadata use ordinary heap allocations, with a per-batch estimate charged to
the sink operator's query pool via ``reportExternalAllocation``. This ensures
even zero-column batches participate in query limits and arbitration. The charge
is released with the queue entry, outside the mutex. This is an estimate, not
exact heap accounting for every nested vector wrapper or downstream object.
Spill/reload is not implemented.

Allocation capacity matters even when the useful result is small. Many tiny
batches can retain mostly unused string buffers. CTE sharing uses the standard
Velox vector allocation policy and preserves existing payloads; it does not
compact or copy published data to reduce that capacity. General vector
allocation-policy optimizations are separate from CTE execution.

The CTE handoff does not copy payload buffers. It retains scalar leaves and
shares value, null, string, offset, index, and length buffers using Velox's
normal reference counts. Each reader gets separate container/wrapper metadata;
this protects mutable row caches and sequence lookup cursors without copying
the data. Producer container metadata is snapshotted as well, so advancing the
producer does not change a published batch's shape. Lazy inputs are resolved
before their backing reader advances, using
``LazyVector::ensureLoadedRowsRecursively`` to load complete selected values.
Dictionary, constant, and sequence selections and parent row null masks are
preserved, so excluded lazy rows are not evaluated. FlatMap values also respect
map nulls and per-key presence bits. Missing child slots in partially projected
ROWs are preserved. This explicit full-value loading leaves normal FlatMap
expression evaluation free to load only the requested keys.
Fully loaded vector trees are detected through metadata and their wrappers
initialized from the leaves outward, avoiding element selection bitmaps for
arrays and maps that need no lazy loading.

Read-only consumption stays zero-copy at the CTE boundary. A producer or
consumer that modifies shared data must use the normal Velox writable/reuse
APIs, such as ``BaseVector::ensureWritable``. These detach shared storage only
when writing requires it. This does not promise that every downstream operator
is allocation-free: aggregation, sorting, and other computations may create
new results. Lazy loading, metadata construction, and downstream writes run
outside the queue lock. Released queue references are also destroyed after
unlocking, allowing external buffer releasers to reenter the queue. The usual
one-batch release uses inline storage for its deferred reference.

Parallel execution
------------------

CTEs impose no driver-count limit. Producers and each logical reader use normal
Velox pipeline parallelism, including ``Task::start(64)``. All producer drivers
publish into one buffer; EOF becomes visible only after every producer driver
finishes. Each reader's drivers divide batches rather than each replaying the
whole result. Consequently every logical reference receives each produced row
exactly once. Readers need not have the same number of drivers as producers.

Driver counts come from the planned pipelines, not the number of drivers that
have started executing. A driver that finishes before its peers initialize
cannot prematurely signal EOF or close their reader. Batch publication and
claims use a short shared mutex; metadata construction and downstream computation run
outside it. Runtime metrics ``activeProducerDrivers``
and ``activeReaderDrivers`` count drivers that actually processed data.

This is unordered batch distribution, not preservation of producer partitions
or order. As with ordinary Velox physical plans, consumers must express the
exchanges and sorting their operators require. For example, use partial
aggregation, gather, and final aggregation for a global maximum; use keyed
repartitioning before grouped aggregation. A final sort or limit can still
require a single driver. Sources with only a few splits/batches cannot keep 64
drivers busy; enough work and executor threads are also required.

``CteTest.parallelScanAt64Drivers`` checks one parallel scan producer and two
parallel readers, including globally distinct rows in every split, exact row multiplicity, and
driver statistics.
The disabled ``CteTest.DISABLED_parallelScaling`` is an opt-in timing experiment
with computation in producers and readers at 1, 8, 32 and 64 drivers. The default
uses 8,388,608 rows and repeated sine evaluations; set ``VELOX_CTE_SCALING_SHA=1``
for a smaller SHA workload. It checks results, reports active driver counts,
operator CPU times (including ``isBlocked``), and median task execution time
after warming each setting.
These are workload-specific measurements, not guarantees of linear scaling:

.. code-block:: shell

    velox_exec_test_CteTest --gtest_also_run_disabled_tests \
        --gtest_filter=CteTest.DISABLED_parallelScaling

Execution scope
---------------

* Non-recursive relational subplans in one ungrouped task. Grouped execution is
  rejected during local planning, before any pipelines are created. Normal Velox
  operator contracts still apply, including required input ordering and partitioning.
* Ordinary hash joins, aggregations, and sorts may spill in producers and
  consumers using their normal Velox configuration. CTEs add no join-spill
  restriction or separate join spiller. The shared result buffer itself remains
  in memory; distributed CTE exchange and CTE spill/reload are not supported.
* Supply scan splits once to the original producer's scan IDs; CTE readers need
  no splits. Use one set of IDs across the complete query.
* No automatic common-subplan discovery, cost model, or equivalence reasoning.

The implementation uses built-in plan-node translation, custom pipeline sinks,
and task-owned bridges. The internal ``SharedResult`` operator forwards query
output and attaches a producer pipeline; ``SharedScan`` reads the shared buffer.
``CteTest`` also covers Q15, immediate consumption, arbitrary operator composition,
multiple/dependent definitions, scan split accounting, volatile expressions,
early close, Task cancellation with blocked readers, terminal output,
zero-column and nested FlatMap budgets, selective lazy loading, selective
wakeups, producer sort spilling, and forced hash-join spilling in producers
and consumers at 1 and 64 drivers. Join coverage includes null keys, inner and
outer joins, semi/anti joins, serialized plans, and spilling producers feeding
spilling consumer self-joins carrying long strings with early limits. Tests also cover
serial dependent CTEs, all readers closing early, shared payload lifetime,
nested copy-on-write isolation, independent sequence cursors, publication
overlapping cancellation/reader closure, reentrant buffer-release callbacks, and real
allocation-triggered arbitration on downstream writes, JSON plan round trips,
and query-pool arbitration/capacity failures with more than 64 MiB retained. Read-only handoff is
checked to allocate no payload memory, including in Release builds.

TPC-H plan comparisons
----------------------

``TpchQueryBuilder::getCtePlan()`` supplies hand-built relational translations
of the CTE boundaries observed in DuckDB 2.0.0 at SF10:

* Q2: qualifying part/supplier/region join, reused for minimum cost and output.
* Q11: German supplier inventory, reused for grouped value and global threshold.
* Q15: supplier revenue, reused for maximum revenue and output.
* Q17: filtered part/lineitem join, reused for correlation keys and output.
* Q20: forest-part inventory, reused for correlation keys and output.
* Q22: customers above the balance threshold, reused for order correlation and output.

These use ordinary Velox joins, exchanges and partial/final aggregates. They
preserve the sharing boundaries, not DuckDB-specific compression operators or
its internal implementation of mark/delimiter joins. TPC-H's non-null keys allow
ordinary semi/anti joins for the corresponding SQL predicates.

Passing ``share=false`` rebuilds each producer at every reference. This gives
a controlled duplicate-work comparison using the same relational shape. The
existing ``getQueryPlan()`` plans are also compared because other changes in
plan shape can matter more than sharing. All Q15 variants use decimal revenue
arithmetic before their equality join: independently summed floating-point
revenues can differ with parallel scheduling and incorrectly eliminate winners.
The final revenue column remains DOUBLE.

Build ``velox_tpch_benchmark`` with ``VELOX_ENABLE_BENCHMARKS=ON``. The six
``qN_cte`` registrations are available alongside the original queries. For
validated comparisons, prepare data from an existing DuckDB TPC-H database:

.. code-block:: shell

    python velox/benchmarks/tpch/prepare_cte_data.py \
      --database /data/sf10.duckdb --output /data/velox-sf10
    velox_tpch_benchmark --data_path=/data/velox-sf10 \
      --cte_compare --cte_reference_path=/data/velox-sf10/expected \
      --cte_samples=3 --num_drivers=64 --num_splits_per_file=64 \
      --cte_plan_directory=/data/cte-plans

The export utility requires DuckDB and its installed TPC-H extension. It stores
money/quantities as DOUBLE to match the existing Velox benchmark expressions,
then runs DuckDB SQL against those exported files for reference results. The
output directory must be new. It records the engine version, row counts and SQL.
The Q11 threshold is the SQL literal 0.0001; it is not adjusted with scale factor.

``--cte_compare`` checks all three variants on every run, uses one warmup per
variant and rotates their order across measured samples. ``--run_query_verbose``
selects one query; its default runs all six. Without ``--cte_reference_path``,
the original plan supplies expected results. ``--num_repeats`` must be one.
``--cte_max_retained_mb`` sets the benchmark's optional per-CTE cap; zero
(the default) means unlimited, matching the public API default.

``CTE_TPCH`` lines report wall time, summed operator CPU (including blocked
checks), SharedScan/SharedResultSink CPU, task-pool peak memory, peak retained
CTE bytes, pool allocation count, scanned rows and active producer/reader drivers.
The retention metric uses the same conservative charge as the cap and excludes
claimed downstream batches. Pool metrics exclude ordinary heap metadata
allocations. Plans saved during warmup include per-operator statistics. Driver
counts apply per eligible pipeline, not to the task as a whole; small tables or
few row groups may leave many drivers idle.
