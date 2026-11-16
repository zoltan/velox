# Discardable memory reclamation

A cache can occupy a root memory pool alongside query pools. When memory is
scarce, dropping clean cache entries can satisfy an allocation without spilling
operator state. `SharedArbitrator` supports this policy with the opt-in extra
configuration `bounded-discardable-reclaim=true`. It defaults to false.

## Reclaimer contract

A root pool opts into the policy by using a reclaimer derived from
`DiscardableMemoryReclaimer`. Its storage must be reconstructible after eviction.
The reclaimer implements the existing `reclaimableBytes`, `reclaim`, and `abort`
interfaces. It must preserve pinned and in-flight users and report only eligible
bytes. A stale eligibility estimate may produce zero reclaimed bytes; ordinary
arbitration then provides a fallback. The marker supplies no storage or eviction
algorithm.

Reclaimers record released storage in `MemoryReclaimer::Stats::reclaimedBytes`.
Another allocator may return a freed reservation before the controller's own
shrink. These statistics let the controller recognize progress and reconsider
current demand without unnecessarily falling back to operator reclamation.

Cache admission first uses its own available capacity, then unassigned capacity
or unused reservations in other pools. If these cannot satisfy the request,
admission fails with `MEM_POOL_CAP_EXCEEDED`. A cache adapter should treat this as
an opportunity to bypass admission and continue the read. Filling a speculative
cache does not justify reclaiming live operator state or existing cache entries.

## Pressure handling

With global arbitration enabled, the controller tries discardable pools before
choosing an operator to spill or abort. This also applies when ordinary operator
spilling is disabled. Before each cache participant, it sums the outstanding
allocation requests and subtracts free non-reserved capacity. The target adds
16 MiB of slack to amortize small allocation bursts and is capped at the
candidate's reported reclaimable bytes.

Targets use allocation demand rather than the global percentage-based reclaim
floor or speculative pool growth. They are not hard limits on bytes freed:
existing participant minimum-reclaim settings and cache entry granularity still
apply. Successful reclamation returns capacity through the existing arbitrator
and wakes waiting requests. If discardable pools make no progress, ordinary
spill/abort selection resumes. Exceptions from a cache reclaimer retain the
existing participant-abort behavior.

The policy reuses the controller, participant synchronization, and capacity
accounting. It adds no lock to allocations that fit in their current reservation.
Demand snapshots use the existing arbitrator state mutex. Global arbitration
remains serialized; this policy does not promise lock-free reclamation.

## Scope and validation

This policy applies to any reconstructible root-pool storage. It does not depend
on a file format, AsyncDataCache implementation, GPU library, or query plan.
Disabling or omitting the configuration retains ordinary arbitration selection.

`velox_discardable_memory_test` uses actual CPU pool allocations to exercise
admission, cache-first selection, fallback, and concurrent requesters. GPU memory
accounting, reader cache adapters, and physical-device allocation failure retries
are separate integrations. This change alone does not install a GPU cache or
enable operator spilling into a CPU cache.
