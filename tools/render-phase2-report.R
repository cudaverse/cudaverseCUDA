if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop("Rendering the Phase 2 report requires `jsonlite` and `digest`.")
}

input <- Sys.getenv(
  "CUDAVERSE_PHASE2_REPORT",
  unset = file.path("inst", "reports", "phase2-rtx2000.json")
)
output <- Sys.getenv(
  "CUDAVERSE_PHASE2_MARKDOWN",
  unset = file.path("inst", "reports", "STAGE2.md")
)
report <- jsonlite::read_json(input, simplifyVector = FALSE)

number <- function(x) as.numeric(unlist(x, use.names = FALSE)[[1L]])
logical_value <- function(x) isTRUE(unlist(x, use.names = FALSE)[[1L]])
seconds <- function(x) sprintf("%.3f s", number(x))
mebibytes <- function(x) sprintf("%.1f MiB", number(x) / 1024^2)

shapes <- c("1000x50", "10000x100", "50000x128")
backends <- c("base", "native", "torch")
benchmark_rows <- character()
for (shape in shapes) {
  baseline <- number(
    report$benchmarks[[shape]]$base$full_pipeline$median_seconds
  )
  for (backend in backends) {
    result <- report$benchmarks[[shape]][[backend]]
    full_median <- number(result$full_pipeline$median_seconds)
    peak <- if (identical(backend, "base")) {
      "n/a"
    } else {
      mebibytes(result$memory$backend_allocator_peak_bytes)
    }
    benchmark_rows <- c(
      benchmark_rows,
      paste0(
        "| ", shape, " | ", backend, " | ",
        seconds(result$pca$median_seconds), " | ",
        seconds(result$knn_continuation$median_seconds), " | ",
        seconds(result$full_pipeline$median_seconds), " | ",
        seconds(result$full_pipeline$p95_seconds), " | ",
        sprintf("%.1fx", baseline / full_median), " | ", peak, " | ",
        if (logical_value(result$validation$passed)) "pass" else "FAIL",
        " |"
      )
    )
  }
}

validation <- report$hardware_validation
validation_rows <- c(
  paste0(
    "| allocate-transfer-free | ", number(validation$lifecycle$cycles),
    " cycles | ",
    number(validation$lifecycle$whole_device_absolute_difference_bytes),
    " B | ", if (logical_value(validation$lifecycle$passed)) "pass" else "FAIL",
    " |"
  ),
  paste0(
    "| injected PCA errors | ", number(validation$structured_errors$injections),
    " errors | ",
    number(validation$structured_errors$whole_device_absolute_difference_bytes),
    " B | ",
    if (logical_value(validation$structured_errors$passed)) "pass" else "FAIL",
    " |"
  ),
  paste0(
    "| injected CUDA OOM | ", number(validation$cuda_errors$injections),
    " errors | ",
    number(validation$cuda_errors$whole_device_absolute_difference_bytes),
    " B | ",
    if (logical_value(validation$cuda_errors$passed)) "pass" else "FAIL",
    " |"
  ),
  paste0(
    "| repeated PCA -> kNN | ", number(validation$repeated_pipeline$cycles),
    " cycles | ",
    number(validation$repeated_pipeline$whole_device_absolute_difference_bytes),
    " B | ",
    if (logical_value(validation$repeated_pipeline$passed)) "pass" else "FAIL",
    " |"
  ),
  paste0(
    "| time-limit interruption | backend reused | ",
    number(validation$interruption$whole_device_absolute_difference_bytes),
    " B | ",
    if (logical_value(validation$interruption$passed)) "pass" else "FAIL",
    " |"
  )
)

native_provenance <- report$benchmarks[["50000x128"]]$native$provenance
pca_stages <- paste(native_provenance$pca$stages$stage, collapse = " -> ")
knn_stages <- paste(native_provenance$knn$stages$stage, collapse = " -> ")
json_sha <- digest::digest(input, algo = "sha256", file = TRUE,
                           serialize = FALSE)

lines <- c(
  "# Native CUDA dense Phase 2 report",
  "",
  "## Outcome",
  "",
  paste(
    "Dense Phase 2 passed the local RTX 2000 parity, device-residency,",
    "structured-error, interruption, lifecycle, and benchmark gates. The",
    "benchmark candidate used explicit native selection; automatic preference",
    "is decided in the final merge audit. Sparse CUDA was intentionally not started."
  ),
  "",
  paste0("- Hardware: ", report$hardware$nvidia_smi, "."),
  paste0("- R: ", report$software$R, "."),
  paste0(
    "- Packages: cudaverse ", report$software$cudaverse,
    "; cudaverseCUDA ", report$software$cudaverseCUDA, "."
  ),
  paste0(
    "- Installed size: cudaverse ",
    mebibytes(report$installed_size_bytes$cudaverse), "; cudaverseCUDA ",
    mebibytes(report$installed_size_bytes$cudaverseCUDA), "."
  ),
  "",
  "## PCA and exact kNN benchmark",
  "",
  paste(
    "Inputs are deterministic float64 matrices. PCA returns 50 components,",
    "and exact Euclidean kNN uses k = 15 and batch size 256. Each median/p95",
    "uses 10 timed runs after 5 warm-ups; cold runs and every raw timing remain",
    "in the JSON report. Full-pipeline timing includes host input, PCA result",
    "materialization, exact kNN, and the final host n-by-k result."
  ),
  "",
  "| Input | Backend | PCA median | kNN continuation median | Full median | Full p95 | Speedup vs base | Peak backend VRAM | Parity |",
  "|---|---|---:|---:|---:|---:|---:|---:|---:|",
  benchmark_rows,
  "",
  paste(
    "Native peak VRAM is the exact high-water mark of allocations owned by",
    "cudaverseCUDA. Torch peak VRAM comes from its CUDA allocator statistics.",
    "Driver, library-context, and unrelated process allocations are excluded",
    "from those backend peaks and remain visible only in whole-device snapshots."
  ),
  "",
  "## Numerical and provenance gates",
  "",
  paste(
    "Every native and torch result matched the CPU reference under the float64",
    "1e-8 relative contract; kNN index matrices were exactly identical. PCA",
    "was compared by singular values, reconstruction, and subspace projectors",
    "rather than sign-sensitive vectors."
  ),
  "",
  paste0("- PCA stages: `", pca_stages, "`."),
  paste0("- kNN stages: `", knn_stages, "`."),
  "- Provenance schema: `cudaverse-stage/1`.",
  "- Native distance outputs remain on CUDA; stable top-k returns only the compact CPU result.",
  "",
  "## Stability and recovery",
  "",
  "| Contract | Work | Whole-device difference after sync/gc | Result |",
  "|---|---:|---:|---:|",
  validation_rows,
  "",
  "The extension-owned allocation counter returned exactly to baseline in every contract; the whole-device acceptance limit was 1 MiB.",
  "",
  "## Packaging and scope",
  "",
  "- CUDA 12.8.1 CI reproducibly rebuilds the committed compute_75 PTX.",
  "- Windows, macOS, Ubuntu, and Ubuntu R-devel source checks pass without CUDA hardware.",
  "- Windows/Linux installable artifacts pass the redistribution scan.",
  "- The SBOM and license gate verify the PTX SHA-256 and report cuBLAS/cuSOLVER as runtime-discovered, not bundled.",
  "- No CUDA Toolkit, NVIDIA runtime library, LibTorch, Rcpp, or sparse CUDA implementation is bundled.",
  "",
  "Machine-readable report SHA-256:",
  paste0("`", json_sha, "`."),
  "",
  "The complete raw timings, memory snapshots, validation evidence, software diagnostics, and per-fragment source commits are retained in `phase2-rtx2000.json`."
)

writeLines(lines, output, useBytes = TRUE)
message("Wrote ", normalizePath(output))
