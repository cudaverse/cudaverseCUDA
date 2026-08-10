# cudaverseCUDA 0.4.0.9000

- Added capability-gated native automatic selection with a cached runtime
  self-test covering transfer, float32/float64 matmul and reductions, sparse
  normalization, arithmetic, reshape, broadcasting, and transpose.
- Added native float32 cuBLAS matrix multiplication plus device-native
  arithmetic, reshape, trailing-dimension broadcasting, and transpose so the
  full public tensor surface can pass the automatic-selection gate.
- Extended diagnostics with cuSOLVER state, runtime completeness, self-test
  evidence, and automatic-selection eligibility.
- Added the native PCA prediction adapter required for full automatic backend
  compatibility.
- Published the RTX 2000 Phase 4 report with automatic-selection evidence,
  float32/float64 tensor parity, dense and sparse 1,000-cycle lifecycle checks,
  and checksum-pinned Phase 3 benchmark-regression gates.

# cudaverseCUDA 0.3.0.9000

- Added shared-ownership device COO/CSR storage with Matrix-compatible
  round trips.
- Added native sparse matrix-vector/matrix multiplication, row/column
  reductions, and sparse-preserving row/column normalization.
- Added sparse-input PCA and kNN adapters that expand on the GPU and reuse the
  validated dense resident pipeline without a full host materialization.
- Added Phase 3 sparse parity, structured-error, interruption, lifetime, and
  benchmark contracts.
- Published the RTX 2000 Phase 3 raw report and rendered summary, including
  CPU/native/torch timings, operation-owned peak VRAM, and provenance.

# cudaverseCUDA 0.2.0.9000

- Added reproducible CUDA 12.8.1 PTX kernels for casts and dense reductions.
- Added dynamically loaded cuSOLVER 11 SVD and PCA for tall and wide matrices.
- Added exact Euclidean and cosine distance blocks plus deterministic stable
  top-k/kNN selection on the device.
- Kept `PCA -> distance -> top-k` intermediates device-resident while
  preserving the `cudaverse-stage/1` provenance schema.
- Added structured failure, interruption recovery, repeated-pipeline, and
  operation-owned peak-VRAM tests.
- Added a reproducible Phase 2 benchmark/report contract.

# cudaverseCUDA 0.1.0.9000

- Added the first native CUDA backend contract implementation.
- Added dynamic CUDA Driver and cuBLAS 12 discovery for Windows and Linux.
- Added a macOS unsupported-platform stub.
- Added shared external-pointer ownership for device allocations.
- Added float64 host/device transfer and cuBLAS matrix multiplication.
- Added RTX hardware parity and 1,000-cycle lifetime tests.
- Added SBOM and third-party redistribution gates.
