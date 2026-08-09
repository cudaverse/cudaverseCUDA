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
