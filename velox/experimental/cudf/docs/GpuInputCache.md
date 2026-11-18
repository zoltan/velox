# Elastic encoded-input cache

The GPU cache retains immutable file byte ranges above OSS `AsyncDataCache`.
It shares a custom memory resource and arbitrator with query operators. Eviction
releases GPU storage; reconstruction reads through the existing host cache. It
does not spill operator state or introduce a second host cache implementation.

## Enablement

Install the CPU `AsyncDataCache` instance and call
`registerCudfMemoryResource(deviceCapacity)` before `registerCudf()`. Registration
owns the GPU input cache when the backend enables discardable reclamation.
Set `CudfConfig::kGpuInputCacheEnabled` (`cudf.gpu_input_cache_enabled`) to `false`
for CPU-only input caching. A previously installed explicit cache takes priority.

All GPU queries sharing this backend must use `attachCudfMemoryPool(queryCtx)`
before task creation. Registration does not attach query pools. The cache and
query pools share the device budget. An application-supplied backend must enable
discardable reclamation; registration preserves its policy and capacity.
Finish all readers and queries before `unregisterCudf()`, which releases the
registration-owned cache. Host footer inspection may precede query resource
attachment; the datasource retains file identity during that preparation.

BufferedInputDataSource uses an installed cache automatically when BufferedInput
has a host cache and the file has a retained immutable identity. Disabling the
installed cache leaves the existing batched copy path. An uncacheable source or
refused admission also uses that path. A file generation lease prevents reuse of
its numeric identity while GPU bytes are retained. The byte coordinates and
Parquet decoder padding form the remaining key contract.

Each device lease pins its entry and records CUDA completion before unpinning.
An entry is eligible only after its last lease is released. Eviction detaches
idle entries under one of 16 shard locks, waits for recorded consumers outside
metadata locks, and then frees the allocations. Allocator calls share a separate
mutex. Concurrent eviction visits idle entries only; it does not scan pinned
entries. Cache fills use spare capacity and do not spill or abort query state.

Logical capacity pressure uses SharedArbitrator's discardable-first policy.
An operator allocation that encounters a physical RMM allocation failure can
also request cache-only reclamation and retry. This path drains concurrent
reclamation, recognizes that physical release may receive no capacity credit,
and bounds its retry batches. Admission does not use this recovery path.

A budget must accommodate live operator state, pinned input, and reservation
rounding. An elastic cache cannot release bytes that a decoder is still reading.
Once discardable entries are exhausted, ordinary arbitration behavior applies;
operator-state spilling requires its own implementation.

## Validation

`velox_cudf_gpu_input_cache_test` exercises device pointer reuse, padding, retained
identity, concurrent fills/evictions, logical pressure and a bounded physical RMM
pool. Set `VELOX_GPU_CACHE_FULL_CAPACITY_TEST=1` on an idle 96 GiB GPU to exercise
an actual 90 GiB shared budget. The test reports live allocation peak and device
memory delta. The opt-in test allocates tens of GiB and is unsuitable for shared
CI GPUs.

`velox_cudf_gpu_encoded_cache_exception_test` isolates allocation and CUDA event
failure injection. `velox_cudf_memory_resource_test` includes rollback, pinned
cache, concurrent reclaim and retained-capacity controls. The standalone cuDF
Parquet accepted-read cleanup fix is a dependency for cancellation safety;
`velox_cudf_read_failure_test` and `velox_cudf_copy_failure_test` reproduce the
failure on the unmodified cuDF pin. Keep that cuDF source change separate when
submitting the Velox branches.

The saved-plan benchmark accepts `--suite_passes`, `--suite_queries`, and
`--suite_output` to run queries with shared process caches. Set
`--gpu_capacity_bytes` to attach accounting and `--gpu_input_cache` to install the
cache. `--suite_memory` reports actual live RMM allocation peaks in a separate
measurement run. Query time includes cursor creation, reads, execution and
completion, and excludes plan loading and JSON result serialization. Results
include column names because stripping a PartitionedOutput root can change their
order relative to SQL output. Validate names and values against the saved output
schema, preserving row multiplicities.

## Cold reads and pressure

The datasource reserves all ranges before submitting reads. Adjacent cache
misses share a host read up to 16 MiB; a single larger requested range remains
indivisible. Independent groups are submitted to the IO executor before
returning a pending batch. Same-batch duplicates share their reservation. A
reservation still being filled by another batch is bypassed, so that producer's
read failure cannot fail an unrelated query.

Cached and uncached uploads share one completion-retirement service. Each read
records an event immediately after its copies and transfers its host buffer to
a dedicated polling thread. Neither the driver nor an IO worker waits for normal
upload retirement. The poller checks each stream's oldest pending event and drains its ready
prefix. New uploads join the next 1 ms poll instead of waking an active poller.
Each owner is checked independently because concurrent submissions can enqueue
owners in a different order from event recording. An incomplete stream
does not prevent observing completion on another stream. Buffer destruction runs
outside the queue lock and outside CUDA callbacks; destructors may call CUDA.
Completed events return to the submitting thread's pool, which retains at most
64 idle handles. In-flight uploads keep that pool alive after its thread exits.
An event is reused only after completion has been established.

Host allocations must remain valid until copy completion whether their memory is
pageable or pinned. This contract also supports readers that retain pinned cache
leases; it does not require another staging copy or change the host cache. On
exceptional handoff failure the owner drains synchronously. If CUDA cannot
establish completion, cleanup retains the host allocation rather than freeing
potential DMA input. `unregisterCudf()` drains pending uploads before releasing
cuDF resources. Reader submissions must already have stopped, as required for
resource teardown. The worker starts lazily on the next read after registration;
its static destructor is only a fallback for callers that omit unregister.

The batch drains every accepted read on failure or abandonment. A rejected
submission discards unpublished destinations after accepted work completes.
Successful groups remain cached for retry.

When admission is refused, the cache may evict its own idle entries and retry
once. Global query waiters take priority over new cache admissions. Pinned
entries remain valid. Discardable pools are never arbitration abort victims;
if an explicit abort or a reclaim exception aborts the cache, admission is
disabled, idle entries are released, and future reads use the normal copy path.
Physical allocation recovery has separate request, duration and capacity
counters in `SharedArbitrator::discardableRecoveryStats()`.

## cuDF failure regression dependency

`VELOX_CUDF_TEST_READ_COMPLETION_FIX` defaults to `OFF`. Enable it only when
building against cuDF with the accepted-read completion fix; it registers the
isolated read/copy failure regression executables. The current pinned cuDF fails
these tests. The gate keeps default CI honest; it does not remove the cache's
cuDF dependency for cancellation and exceptional read completion. Production
cache deployment must include that fix until the dependency pin contains it.

## Relationship to the host cache

This is a device-resident reuse layer above OSS `AsyncDataCache`. Device input
needs contiguous allocations and CUDA stream completion to govern reclamation.
The host cache keeps its own statistics, TTL and SSD policy. Evicting a GPU entry
only releases its device allocation; the existing BufferedInput path reconstructs
it from host cache or storage. The cache does not implement operator spilling.
The memory library depends on `velox_exec` for driver suspension hooks; it sits
below cuDF readers and operators, not below all of Velox execution.

### Host upload storage

The OSS datasource accounts contiguous staging buffers through BufferedInput's
external-allocation API and retains that pool with each buffer. The existing
vector allocation path is preserved. Both pinned buffers supplied
by another datasource and these pageable buffers remain alive until the recorded
copy event completes. CUDA's asynchronous pageable-copy contract does not promise
source completion at API return. This is independent of GPU input caching.
Already-accounted host cache pins returned by an override incur no second charge.

CUDA 12.8 introduces `cudaMemcpyBatchAsync` with
`cudaMemcpySrcAccessOrderDuringApiCall`; its signature changes in CUDA 13. An
RTX PRO 6000/CUDA 13 experiment found explicit source completion slower for
1 MiB and 16 MiB transfers, so the production path retains event-based completion.
This also preserves asynchronous copies from internal pinned host caches.
