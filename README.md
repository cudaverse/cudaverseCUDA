# cudaverseCUDA

`cudaverseCUDA` is the optional lightweight native CUDA backend for
[`cudaverse`](https://github.com/cudaverse/cudaverse). It is an implementation
extension, not a second user-facing numerical API: users continue to call
`cuda_tensor()`, `tensor_matmul()`, and the other functions exported by
`cudaverse`.

## Why this backend exists

- It does not install the multi-gigabyte LibTorch distribution.
- It uses registered `.Call` routines and the R C API, without Rcpp.
- Tensor allocation, transfer, lifetime, and provenance remain under
  cudaverse control.
- Dense operations can remain device-resident between calls.
- The public R API can remain stable while other backends are added later.

The sparse Phase 3 milestone now implements:

- CUDA Driver detection, allocation/free, transfer, synchronization, casts,
  reductions, and shared external-pointer ownership;
- float64 cuBLAS matrix multiplication;
- cuSOLVER SVD and PCA for tall and wide matrices;
- device-resident PCA scores, exact Euclidean/cosine distance blocks, and
  deterministic top-k/kNN selection; and
- shared-ownership device COO/CSR storage, Matrix-compatible conversion,
  sparse multiplication and reductions, and sparse-preserving normalization;
- sparse-input PCA/kNN that expands on the device and reuses the resident
  dense continuation; and
- structured backend errors, interruption recovery, allocation high-water
  telemetry, and stage-level provenance.

For the native path,
`sparse normalization -> PCA -> distance -> top-k` keeps its compute
intermediates on the device. Sparse objects retain a compact host metadata
mirror for R compatibility; only PCA's ordinary R result fields and the final
`n x k` neighbour output are materialized as ordinary R results.

## Runtime model

The package installs without a local CUDA Toolkit. Its small, package-owned
kernel PTX is reproducibly generated from
[`tools/cuda/dense_kernels.cu`](tools/cuda/dense_kernels.cu) in a pinned CUDA
12.8.1 CI container, while runtime libraries are dynamically discovered. On
Windows, `CUDAVERSE_CUBLAS_PATH` and `CUDAVERSE_CUSOLVER_PATH` can point to
`cublas64_12.dll` and `cusolver64_11.dll`; on Linux they can point to
`libcublas.so.12` and `libcusolver.so.11`. The directory containing an
explicitly selected Windows DLL is used to resolve that library's dependencies.

No NVIDIA binary is bundled in the source package or current CI artifacts.
macOS builds a clear unsupported-platform stub. An explicit CUDA request never
silently falls back when the driver, cuBLAS, cuSOLVER, a required symbol, or
memory is unavailable. Native remains explicitly opt-in in 0.3. Sparse Phase 3
closes the sparse workflow gate, but global `device = "auto"` preference still
waits for the element-wise arithmetic and broadcasting surface to pass the
same full compatibility contract.

For development hardware tests:

```r
Sys.setenv(CUDAVERSE_NATIVE_TESTS = "true")
options(cudaverse.cuda_backends = "native")
testthat::test_local()
```

See [`inst/reports/THIRD_PARTY_LICENSES.md`](inst/reports/THIRD_PARTY_LICENSES.md)
and the generated CycloneDX SBOM before distributing any binary artifact.

The machine-readable Phase 3 benchmark contract separates complete sparse
workflow timing from the device-resident PCA/kNN continuation and reports cold
time, median/p95, operation-owned peak VRAM, numerical error, installation
size, density, and provenance.

See the completed [RTX 2000 Phase 3 report](inst/reports/STAGE3.md) and its
[machine-readable evidence](inst/reports/phase3-rtx2000.json). The largest
`10000 x 128` case measured a 0.21-second native median versus 13.40 seconds
for base R and 5.46 seconds for torch, with numerical parity and zero tracked
bytes remaining after cleanup. These measurements describe this machine and
contract; they are not a universal speed claim.

## Status

Development version `0.3.0.9000`. This extension is not being submitted to
CRAN in the cudaverse 0.1 release cycle.
