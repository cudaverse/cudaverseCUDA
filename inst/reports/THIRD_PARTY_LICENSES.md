# Third-party software and redistribution status

This source package and its current CI artifacts do **not** bundle NVIDIA
libraries or other third-party binary payloads.

| Component | How it is used | Bundled | License action |
|---|---|---:|---|
| NVIDIA CUDA Driver | Dynamically loaded from the user's system | No | Supplied by the NVIDIA driver installation |
| NVIDIA cuBLAS 12 | Dynamically loaded from a user-provided/system path | No | Do not redistribute until the exact files are checked against the current CUDA EULA Attachment A |
| R | C API and package build system | No | Runtime/build dependency |
| cudaverse | Imported R package | No | MIT |

The release gate fails if a CUDA, cuBLAS, cuSPARSE, cuSOLVER, cuDNN, NVRTC, or
LibTorch binary appears in a built package. A future binary distribution may
only include exact files permitted by the current
[NVIDIA CUDA EULA Attachment A](https://docs.nvidia.com/cuda/eula/) and must
carry all notices required there.
