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

The first development milestone intentionally implements only CUDA Driver
detection, allocation/free, host-device transfer, synchronization, shared
external-pointer ownership, and float64 cuBLAS matrix multiplication. PCA,
distance, top-k, sparse operations, and automatic preference for native CUDA
are later milestones.

## Runtime model

The package compiles without a local CUDA Toolkit because it dynamically loads
the NVIDIA CUDA Driver and cuBLAS 12 libraries at runtime. On Windows,
`CUDAVERSE_CUBLAS_PATH` can point to `cublas64_12.dll`; on Linux it can point to
`libcublas.so.12`. The directory containing an explicitly selected Windows DLL
is used to resolve that library's dependencies.

No NVIDIA binary is bundled in the source package or current CI artifacts.
macOS builds a clear unsupported-platform stub. An explicit CUDA request never
silently falls back when the driver, cuBLAS, a required symbol, or memory is
unavailable.

For development hardware tests:

```r
Sys.setenv(CUDAVERSE_NATIVE_TESTS = "true")
options(cudaverse.cuda_backends = "native")
testthat::test_local()
```

See [`inst/reports/THIRD_PARTY_LICENSES.md`](inst/reports/THIRD_PARTY_LICENSES.md)
and the generated CycloneDX SBOM before distributing any binary artifact.

## Status

Development version `0.1.0.9000`. This extension is not being submitted to
CRAN in the cudaverse 0.1 release cycle.
