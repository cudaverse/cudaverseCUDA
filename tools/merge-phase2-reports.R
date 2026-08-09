if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The `jsonlite` package is required to merge phase 2 reports.")
}

files_setting <- Sys.getenv("CUDAVERSE_PHASE2_FRAGMENTS", unset = "")
files <- if (nzchar(files_setting)) {
  strsplit(files_setting, ";", fixed = TRUE)[[1L]]
} else {
  list.files(
    file.path("inst", "reports", "phase2-fragments"),
    pattern = "[.]json$", full.names = TRUE
  )
}
if (!length(files) || any(!file.exists(files))) {
  stop("No complete phase 2 fragment set was found.")
}

fragments <- lapply(files, jsonlite::read_json, simplifyVector = FALSE)
first <- fragments[[1L]]
report <- first
report$schema <- "cudaverse-native-phase2/1"
report$source_fragments <- basename(files)
report$fragment_sources <- stats::setNames(
  lapply(fragments, `[[`, "source"), basename(files)
)
latest_fragment <- fragments[[length(fragments)]]
report$hardware <- latest_fragment$hardware
report$software <- latest_fragment$software
report$installed_size_bytes <- latest_fragment$installed_size_bytes
report$benchmarks <- list()
report$contract$peak_vram_definition <- paste(
  "cudaverseCUDA allocation high-water bytes for native; torch CUDA allocator",
  "allocated/reserved byte peaks for torch; external library/context memory is",
  "shown only by whole-device snapshots"
)

for (fragment in fragments) {
  for (shape in names(fragment$benchmarks)) {
    if (is.null(report$benchmarks[[shape]])) report$benchmarks[[shape]] <- list()
    for (backend in names(fragment$benchmarks[[shape]])) {
      if (!is.null(report$benchmarks[[shape]][[backend]])) {
        stop("Duplicate phase 2 result for ", shape, " / ", backend, ".")
      }
      report$benchmarks[[shape]][[backend]] <-
        fragment$benchmarks[[shape]][[backend]]
    }
  }
}

required_shapes <- c("1000x50", "10000x100", "50000x128")
required_backends <- c("base", "native", "torch")
missing <- character()
for (shape in required_shapes) {
  for (backend in required_backends) {
    if (is.null(report$benchmarks[[shape]][[backend]])) {
      missing <- c(missing, paste(shape, backend, sep = "/"))
    }
  }
}
if (length(missing)) {
  stop("Missing required phase 2 results: ", paste(missing, collapse = ", "))
}

values <- function(x) unname(unlist(x, use.names = FALSE))
integer_value <- function(x) as.integer(values(x)[[1L]])
for (index in seq_along(fragments)) {
  contract <- fragments[[index]]$contract
  valid_contract <- identical(integer_value(contract$warmup_runs), 5L) &&
    identical(integer_value(contract$timed_runs), 10L) &&
    identical(integer_value(contract$k), 15L) &&
    identical(integer_value(contract$requested_components), 50L) &&
    identical(integer_value(contract$batch_size), 256L) &&
    identical(contract$dtype, "float64") &&
    identical(contract$metric, "euclidean") &&
    isTRUE(contract$center) && isTRUE(contract$scale)
  if (!valid_contract) {
    stop("Phase 2 benchmark contract mismatch: ", basename(files[[index]]), ".")
  }
}

for (shape in required_shapes) {
  for (backend in required_backends) {
    result <- report$benchmarks[[shape]][[backend]]
    label <- paste(shape, backend, sep = "/")
    if (!identical(result$status, "complete")) {
      stop("Phase 2 result is not complete: ", label, ".")
    }
    if (!isTRUE(result$validation$passed) ||
        !isTRUE(result$validation$knn_indices_identical)) {
      stop("Phase 2 numerical parity failed: ", label, ".")
    }
  }
}

for (shape in required_shapes) {
  native <- report$benchmarks[[shape]]$native
  pca <- native$provenance$pca
  knn <- native$provenance$knn
  if (!isTRUE(native$provenance$pca_scores_device_resident) ||
      !identical(pca$schema, "cudaverse-stage/1") ||
      !identical(knn$schema, "cudaverse-stage/1") ||
      !identical(
        values(pca$stages$stage),
        c("preprocessing", "decomposition", "scores_resident")
      ) ||
      !identical(
        values(pca$stages$output_device),
        c("cuda", "cpu", "cuda")
      ) ||
      !identical(
        values(knn$stages$stage),
        c("input_materialization", "distance", "neighbor_selection")
      ) ||
      !identical(values(knn$stages$backend), rep("native", 3L)) ||
      !identical(
        values(knn$stages$output_device),
        c("cuda", "cuda", "cpu")
      )) {
    stop("Native Phase 2 residency/provenance failed: ", shape, ".")
  }
}

dirty_sources <- vapply(
  report$fragment_sources,
  function(source) {
    isTRUE(source$cudaverse$dirty) || isTRUE(source$cudaverseCUDA$dirty)
  },
  logical(1)
)
if (any(dirty_sources)) {
  stop(
    "Phase 2 fragments were generated from dirty source trees: ",
    paste(names(dirty_sources)[dirty_sources], collapse = ", "),
    "."
  )
}

report$generated_at_utc <- format(Sys.time(), tz = "UTC", usetz = TRUE)
validation_file <- Sys.getenv("CUDAVERSE_PHASE2_VALIDATION", unset = "")
if (nzchar(validation_file)) {
  if (!file.exists(validation_file)) {
    stop("The requested Phase 2 validation file does not exist.")
  }
  report$hardware_validation <- jsonlite::read_json(
    validation_file, simplifyVector = FALSE
  )
  validation <- report$hardware_validation
  validation_sections <- list(
    validation$lifecycle,
    validation$structured_errors,
    validation$cuda_errors,
    validation$repeated_pipeline,
    validation$interruption
  )
  valid_validation <- identical(
    validation$schema, "cudaverse-native-phase2-validation/1"
  ) &&
    identical(integer_value(validation$lifecycle$cycles), 1000L) &&
    identical(integer_value(validation$structured_errors$injections), 100L) &&
    identical(integer_value(validation$cuda_errors$injections), 100L) &&
    identical(integer_value(validation$repeated_pipeline$cycles), 100L) &&
    all(vapply(validation_sections, function(x) isTRUE(x$passed), logical(1))) &&
    isTRUE(validation$interruption$interrupted) &&
    isTRUE(validation$interruption$backend_reusable) &&
    !isTRUE(validation$source$cudaverse$dirty) &&
    !isTRUE(validation$source$cudaverseCUDA$dirty)
  if (!isTRUE(validation$passed) || !valid_validation) {
    stop("RTX 2000 Phase 2 stability validation failed.")
  }
}
output <- Sys.getenv(
  "CUDAVERSE_REPORT_FILE",
  unset = file.path("inst", "reports", "phase2-rtx2000.json")
)
jsonlite::write_json(
  report, output, auto_unbox = TRUE, pretty = TRUE, digits = 16,
  null = "null", na = "null"
)
message("Wrote ", normalizePath(output))
