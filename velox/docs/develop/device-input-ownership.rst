Datasource-owned device input
=============================

A reader can consume bytes that already reside on the GPU when the datasource
returns their address and retains their storage. A destination-only read API
requires another allocation and copy, even when the bytes are already resident.

``gpu::DeviceReadSource`` is an optional interface alongside the cuDF datasource.
It returns a ``DeviceReadBatch``: immutable views, a storage owner, pending host
submission and an optional device-readiness hook. The contract uses CUDA/RMM but
has no cuDF, file-cache or format dependencies. Providers can allocate transient
storage or retain existing storage. The interface does not implement a cache.

The cuDF adapter detects this interface explicitly. The existing cuDF
``supports_device_read()`` flag cannot distinguish owning reads from reads into
caller-provided storage. Sources that do not implement the optional interface
continue through ``fetch_byte_ranges_to_device_async``. Errors from an owning
provider propagate; they do not silently trigger another read.

``BufferedInputDataSource`` implements the interface by calling the same batched
copy routine through ``copyByteRangesAsync``. This preserves cuDF's allocation,
coalescing, padding and I/O-submission policy. The datasource now returns the
resulting allocation with its views. BufferedInput and the CPU cache interfaces
are unchanged. Host-to-device transfers still occur on this path.

For resident bytes, the adapter retains the provider's owner and passes the
original addresses to the reader. It allocates no device payload and issues no
payload copy. Host allocations for descriptors and ownership handles can still
occur. This is zero-copy input reuse, not zero allocation throughout the query.

Lifetime and execution
----------------------

The batch contains one view per requested range, in request order. View sizes
are logical byte lengths; providers also satisfy the consumer's decoder-padding
requirements. Parquet storage must meet cuDF's allocation-tail padding contract.
The adapter checks range bounds and rejects missing owners or malformed views.

The adapter retains the datasource until the batch is released. ``wait()`` finishes
host submission and reports failures. ``waitOn(stream)`` additionally invokes the
provider's readiness hook before the first consumer. Without that hook,
production and consumption use the stream supplied to the read. The first
consumer hook is not a general cross-stream ownership-transfer mechanism.

Consumers retain the batch until their last GPU use has been submitted.
Providers must free or unpin storage only after that queued use is safe, using
stream-ordered deallocation or an appropriate completion event. Memory resources
and streams outlive the batch. Destruction and replacement finish any pending
host submission before releasing its owner; an abandoned future's error is not
rethrown by the destructor. Consumers call ``wait()`` to observe read errors.

An owner marks the batch as started, even when it has no pending future. The
split reader uses this state to avoid fetching an immediately ready result twice.
It converts the generic views to cuDF descriptors at the decoder boundary.

Tests
-----

``CudfSplitReaderByteFetchTest`` checks pointer identity, exact bytes, allocation
and copy counts, ownership after datasource release, legacy copy-only sources,
coalescing parity, empty batches, malformed results, range bounds, completion
ordering and deferred-work cleanup. The resident fixture explicitly implements
the new interface and uses cuDF's existing owning-buffer API for its storage.

These are focused interface tests. A production GPU cache, eviction policy,
full-query throughput and host-cache streaming performance require separate
integration measurements. Reusing the same copy algorithm alone does not prove
that the extra host-side ownership and descriptor work has no performance cost.
