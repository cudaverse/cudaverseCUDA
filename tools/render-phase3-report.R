if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE)) {
  stop("Rendering the Phase 3 report requires `jsonlite` and `digest`.")
}

input <- Sys.getenv(
  "CUDAVERSE_PHASE3_REPORT",
  unset = file.path("inst", "reports", "phase3-rtx2000.json")
)
output <- Sys.getenv(
  "CUDAVERSE_PHASE3_MARKDOWN",
  unset = file.path("inst", "reports", "STAGE3.md")
)
report <- jsonlite::read_json(input, simplifyVector = FALSE)

scalar <- function(x) unlist(x, recursive = TRUE, use.names = FALSE)[[1L]]
number <- function(x) as.numeric(scalar(x))
logical_value <- function(x) isTRUE(as.logical(scalar(x)))
characters <- function(x) {
  as.character(unlist(x, recursive = TRUE, use.names = FALSE))
}
seconds <- function(x) sprintf("%.3f s", number(x))
mebibytes <- function(x) sprintf("%.1f MiB", number(x) / 1024^2)

cases <- names(report$benchmarks)
backends <- c("base", "native", "torch")
benchmark_rows <- character()
for (case_name in cases) {
  baseline <- number(
    report$benchmarks[[case_name]]$base$full_pipeline$median_seconds
  )
  for (backend in backends) {
    result <- report$benchmarks[[case_name]][[backend]]
    full_median <- number(result$full_pipeline$median_seconds)
    peak <- if (identical(backend, "base")) {
      "n/a"
    } else {
      mebibytes(result$memory$backend_allocator_peak_bytes)
    }
    benchmark_rows <- c(
      benchmark_rows,
      paste0(
        "| ", case_name, " | ", backend, " | ",
        seconds(result$transfer$median_seconds), " | ",
        seconds(result$normalization$median_seconds), " | ",
        seconds(result$pca$median_seconds), " | ",
        seconds(result$knn$median_seconds), " | ",
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
    "| sparse allocate-normalize-free | ",
    number(validation$lifecycle$cycles), " cycles | ",
    number(validation$lifecycle$whole_device_absolute_difference_bytes),
    " B | ",
    if (logical_value(validation$lifecycle$passed)) "pass" else "FAIL",
    " |"
  ),
  paste0(
    "| shared COO/CSR ownership | release first owner, use second | n/a | ",
    if (logical_value(validation$shared_ownership$passed)) "pass" else "FAIL",
    " |"
  ),
  paste0(
    "| structured sparse error | backend reusable | n/a | ",
    if (logical_value(validation$structured_error$passed)) "pass" else "FAIL",
    " |"
  ),
  paste0(
    "| injected CUDA error | tracked cleanup | ",
    number(
      validation$injected_cuda_error$tracked_current_difference_bytes
    ), " B | ",
    if (logical_value(validation$injected_cuda_error$passed)) "pass" else
      "FAIL",
    " |"
  ),
  paste0(
    "| time-limit interruption | backend reusable | ",
    number(validation$interruption$tracked_current_difference_bytes),
    " B | ",
    if (logical_value(validation$interruption$passed)) "pass" else "FAIL",
    " |"
  ),
  paste0(
    "| resident sparse pipeline | stage/1 provenance | n/a | ",
    if (logical_value(validation$resident_pipeline$passed)) "pass" else
      "FAIL",
    " |"
  ),
  paste0(
    "| exact-distance ties | original row order | n/a | ",
    if (logical_value(validation$stable_ties$passed)) "pass" else "FAIL",
    " |"
  )
)

largest <- tail(cases, 1L)
native_provenance <- report$benchmarks[[largest]]$native$provenance
normalized_stages <- paste(
  characters(native_provenance$normalized$stages$stage), collapse = " -> "
)
pca_stages <- paste(
  characters(native_provenance$pca$stages$stage), collapse = " -> "
)
knn_stages <- paste(
  characters(native_provenance$knn$stages$stage), collapse = " -> "
)
json_bytes <- readChar(
  input,
  nchars = file.info(input)$size,
  useBytes = TRUE
)
json_bytes <- gsub("\r\n", "\n", json_bytes, fixed = TRUE)
json_sha <- digest::digest(
  charToRaw(json_bytes), algo = "sha256", serialize = FALSE
)

lines <- c(
  "# Native CUDA sparse Phase 3 report",
  "",
  "## Outcome",
  "",
  paste(
    "Sparse Phase 3 passed the local RTX 2000 numerical-parity,",
    "device-residency, shared-ownership, structured-error, interruption,",
    "1,000-cycle lifecycle, and benchmark gates. The completed path is",
    "sparse normalization -> PCA -> exact distance -> stable top-k/kNN."
  ),
  "",
  paste(
    "Native selection remains explicit. Global `auto` preference is",
    "unchanged until the remaining element-wise and broadcasting contract is",
    "native-complete; this report does not widen the public API or start graph",
    "or embedding work."
  ),
  "",
  paste0("- Hardware: ", paste(characters(report$hardware$nvidia_smi),
                               collapse = "; "), "."),
  paste0("- R: ", scalar(report$software$R), "."),
  paste0(
    "- Packages: cudaverse ", scalar(report$software$cudaverse),
    "; cudaverseCUDA ", scalar(report$software$cudaverseCUDA), "."
  ),
  paste0(
    "- Source commits: cudaverse `",
    scalar(report$source$cudaverse$commit), "`; cudaverseCUDA `",
    scalar(report$source$cudaverseCUDA$commit), "`."
  ),
  paste0(
    "- Installed size: cudaverse ",
    mebibytes(report$installed_size_bytes$cudaverse),
    "; cudaverseCUDA ",
    mebibytes(report$installed_size_bytes$cudaverseCUDA), "."
  ),
  "",
  "## Sparse normalization, PCA, and exact kNN benchmark",
  "",
  paste(
    "Inputs are deterministic non-negative float64 `dgCMatrix` objects with",
    "at least two non-zero values per row and no empty columns. Rows are",
    "sum-normalized to 1,000 and transformed",
    "with `log1p()`; PCA returns 20 components and exact Euclidean kNN uses",
    "k = 15 with batch size 256. Each median/p95 uses five timed runs after",
    "two warm-ups. Full timing starts from the host sparse matrix and includes",
    "transfer, normalization, PCA, exact distance/top-k, and the final host",
    "n-by-k result."
  ),
  "",
  paste(
    "Torch peak memory is its allocator session high-water mark. Native peak",
    "memory is the operation-owned cudaverseCUDA high-water mark; neither",
    "metric claims ownership of driver context memory."
  ),
  "",
  paste(
    "| Input (requested density) | Backend | Transfer | Normalize | PCA |",
    "kNN | Full median | Full p95 | Speedup vs base | Peak backend allocation |",
    "Parity |"
  ),
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  benchmark_rows,
  "",
  "## RTX 2000 validation",
  "",
  "| Gate | Contract | Post-cleanup difference | Result |",
  "|---|---:|---:|---:|",
  validation_rows,
  "",
  "The lifecycle ceiling is 1 MiB for whole-device used memory and exactly zero bytes for cudaverseCUDA-tracked allocations after synchronization and `gc()`. The interruption and injected-error checks also prove that the same backend instance remains reusable.",
  "",
  "## Device residency and provenance",
  "",
  paste0("Largest benchmark: `", largest, "`."),
  "",
  paste0("- Normalization: `", normalized_stages, "`."),
  paste0("- PCA: `", pca_stages, "`."),
  paste0("- kNN: `", knn_stages, "`."),
  "- Every recorded provenance object uses `cudaverse-stage/1`.",
  "- PCA parity compares the numerically identifiable subspace plus full truncated reconstruction, so a zero singular-value tie cannot create a false failure.",
  "- Normalized CSR storage and PCA scores are external pointers with shared ownership; PCA scores feed distance/top-k without a host round trip.",
  "- Stable top-k resolves equal distances by original one-based row index.",
  "",
  "## Packaging boundary",
  "",
  "- The extension is optional and is not a second user-facing compute API.",
  "- No CUDA Toolkit, LibTorch, Rcpp, or NVIDIA runtime DLL is bundled.",
  "- CUDA 12.8.1 CI builds the package-owned PTX artifact; runtime libraries are discovered dynamically.",
  "- The SBOM and redistributable-license gate remain part of CI.",
  "",
  paste0("Canonical-LF report JSON SHA-256: `", json_sha, "`."),
  "",
  "The complete raw timings, numerical errors, memory snapshots, diagnostics, and validation evidence are retained in `phase3-rtx2000.json`."
)

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
writeLines(lines, output, useBytes = TRUE)
message("Rendered Phase 3 report to ", normalizePath(output))
