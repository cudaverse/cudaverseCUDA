if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The `jsonlite` package is required to write the phase 2 report.")
}
if (!requireNamespace("cudaverse", quietly = TRUE) ||
    !requireNamespace("cudaverseCUDA", quietly = TRUE)) {
  stop("Install the development cudaverse and cudaverseCUDA packages first.")
}

integer_setting <- function(name, default, minimum = 1L) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, unset = default)))
  if (length(value) != 1L || is.na(value) || value < minimum) {
    stop("Invalid `", name, "` setting.")
  }
  value
}

parse_shapes <- function(value) {
  fields <- strsplit(value, ",", fixed = TRUE)[[1L]]
  shapes <- lapply(fields, function(field) {
    parts <- suppressWarnings(as.integer(
      strsplit(tolower(trimws(field)), "x", fixed = TRUE)[[1L]]
    ))
    if (length(parts) != 2L || anyNA(parts) || any(parts < 2L)) {
      stop("Invalid benchmark shape `", field, "`; use rowsxcolumns.")
    }
    parts
  })
  names(shapes) <- vapply(shapes, function(x) paste(x, collapse = "x"), "")
  shapes
}

summary_times <- function(values) {
  list(
    median_seconds = unname(stats::median(values)),
    p95_seconds = unname(stats::quantile(values, 0.95, type = 8)),
    runs_seconds = unname(values)
  )
}

installed_size <- function(package) {
  path <- find.package(package)
  files <- list.files(
    path, recursive = TRUE, full.names = TRUE, all.files = TRUE, no.. = TRUE
  )
  sum(file.info(files)$size, na.rm = TRUE)
}

gpu_line <- function() {
  tryCatch(
    system2(
      "nvidia-smi",
      c(
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits"
      ),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(error) conditionMessage(error)
  )
}

source_state <- function(path = ".") {
  commit <- tryCatch(
    system2("git", c("-C", path, "rev-parse", "HEAD"), stdout = TRUE),
    error = function(error) "unavailable"
  )
  dirty <- tryCatch(
    length(system2(
      "git", c("-C", path, "status", "--porcelain"), stdout = TRUE
    )) > 0L,
    error = function(error) NA
  )
  list(commit = unname(commit[[1L]]), dirty = dirty)
}

provenance_payload <- function(x) {
  value <- cudaverse::cuda_provenance(x)
  list(
    schema = attr(value, "schema", exact = TRUE),
    compute_device = attr(value, "compute_device", exact = TRUE),
    stages = lapply(value, unname)
  )
}

strip_matrix <- function(x) {
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
}

reference_payload <- function(value) {
  list(
    pca = list(
      sdev = unname(value$pca$sdev),
      rotation = strip_matrix(value$pca$rotation),
      x = strip_matrix(value$pca$x)
    ),
    knn = list(
      index = unname(value$knn$index),
      distance = unname(value$knn$distance)
    )
  )
}

validation_payload <- function(value, reference) {
  actual_projector <- value$pca$rotation %*% t(value$pca$rotation)
  reference_projector <- reference$pca$rotation %*%
    t(reference$pca$rotation)
  actual_reconstruction <- value$pca$x %*% t(value$pca$rotation)
  reference_reconstruction <- reference$pca$x %*%
    t(reference$pca$rotation)
  reconstruction_error <- max(abs(
    actual_reconstruction - reference_reconstruction
  ))
  reconstruction_scale <- max(1, max(abs(reference_reconstruction)))
  distance_error <- max(abs(
    unname(value$knn$distance) - reference$knn$distance
  ))
  distance_scale <- max(1, max(abs(reference$knn$distance)))
  list(
    pca_sdev_max_absolute_error = max(abs(
      unname(value$pca$sdev) - reference$pca$sdev
    )),
    pca_subspace_projector_max_absolute_error = max(abs(
      actual_projector - reference_projector
    )),
    pca_reconstruction_max_absolute_error = reconstruction_error,
    pca_reconstruction_max_relative_error =
      reconstruction_error / reconstruction_scale,
    knn_indices_identical = identical(
      unname(value$knn$index), reference$knn$index
    ),
    knn_distance_max_absolute_error = distance_error,
    knn_distance_max_relative_error = distance_error / distance_scale,
    passed = identical(unname(value$knn$index), reference$knn$index) &&
      reconstruction_error / reconstruction_scale <= 1e-8 &&
      distance_error / distance_scale <= 1e-8
  )
}

factory <- cudaverseCUDA::cudaverse_cuda_backend_factory()
native_diagnostics <- factory$diagnostics()
warmup_runs <- integer_setting("CUDAVERSE_WARMUP_RUNS", "5")
timed_runs <- integer_setting("CUDAVERSE_TIMED_RUNS", "10")
k <- integer_setting("CUDAVERSE_PHASE2_K", "15")
batch_size <- integer_setting("CUDAVERSE_PHASE2_BATCH_SIZE", "256")
components <- integer_setting("CUDAVERSE_PHASE2_COMPONENTS", "50")
shapes <- parse_shapes(Sys.getenv(
  "CUDAVERSE_PHASE2_SIZES", unset = "1000x50,10000x100,50000x128"
))
backends <- strsplit(Sys.getenv(
  "CUDAVERSE_PHASE2_BACKENDS", unset = "base,native,torch"
), ",", fixed = TRUE)[[1L]]
backends <- trimws(backends)
if (any(!backends %in% c("base", "native", "torch"))) {
  stop("Backends must be selected from base, native, and torch.")
}

reference_directory <- Sys.getenv(
  "CUDAVERSE_PHASE2_REFERENCE_DIR",
  unset = file.path(tempdir(), "cudaverse-phase2-reference")
)
dir.create(reference_directory, recursive = TRUE, showWarnings = FALSE)
output <- Sys.getenv(
  "CUDAVERSE_REPORT_FILE",
  unset = file.path("inst", "reports", "phase2-fragment.json")
)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)

report <- list(
  schema = "cudaverse-native-phase2-fragment/1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  source = list(
    cudaverseCUDA = source_state("."),
    cudaverse = source_state(file.path("..", "cudaverse"))
  ),
  hardware = list(nvidia_smi = gpu_line()),
  software = list(
    R = R.version.string,
    cudaverse = as.character(utils::packageVersion("cudaverse")),
    cudaverseCUDA = as.character(utils::packageVersion("cudaverseCUDA")),
    native_diagnostics = native_diagnostics
  ),
  contract = list(
    warmup_runs = warmup_runs,
    timed_runs = timed_runs,
    k = k,
    requested_components = components,
    batch_size = batch_size,
    dtype = "float64",
    metric = "euclidean",
    center = TRUE,
    scale = TRUE,
    cold_definition = paste(
      "first workload after packages and diagnostics are loaded, before",
      "workload warm-up"
    ),
    full_pipeline_boundary = paste(
      "host input -> PCA -> host PCA metadata/score mirror -> exact kNN ->",
      "host n-by-k result"
    ),
    knn_continuation_boundary = paste(
      "PCA completion -> exact kNN -> host n-by-k result; native reuses the",
      "device-resident score allocation and never materializes a dense",
      "distance matrix on the host"
    ),
    peak_vram_definition = paste(
      "high-water mark of allocations owned by cudaverseCUDA; external CUDA",
      "library/context allocations are excluded"
    )
  ),
  installed_size_bytes = list(
    cudaverse = installed_size("cudaverse"),
    cudaverseCUDA = installed_size("cudaverseCUDA")
  ),
  benchmarks = list()
)

write_report <- function() {
  report$generated_at_utc <<- format(Sys.time(), tz = "UTC", usetz = TRUE)
  jsonlite::write_json(
    report, output, auto_unbox = TRUE, pretty = TRUE, digits = 16,
    null = "null", na = "null"
  )
}

for (shape_name in names(shapes)) {
  shape <- shapes[[shape_name]]
  rows <- shape[[1L]]
  columns <- shape[[2L]]
  if (k >= rows) stop("`k` must be smaller than every row count.")
  n_components <- min(components, columns, rows - 1L)
  set.seed(20260809L + rows + columns)
  values <- matrix(stats::rnorm(rows * columns), rows, columns)
  report$benchmarks[[shape_name]] <- list()

  reference_file <- file.path(
    reference_directory,
    sprintf("reference-%sx%s-c%s-k%s.rds", rows, columns, n_components, k)
  )

  for (backend in backends) {
    message("Benchmarking ", shape_name, " with ", backend, " backend")
    if (identical(backend, "native") && !isTRUE(native_diagnostics$available)) {
      report$benchmarks[[shape_name]][[backend]] <- list(
        status = "skipped", reason = native_diagnostics$detection_error
      )
      write_report()
      next
    }

    old_options <- options(cudaverse.cuda_backends = backend)
    device <- if (identical(backend, "base")) "cpu" else "cuda"
    selected <- tryCatch(
      cudaverse::cuda_diagnostics()$backend_diagnostics[[backend]],
      error = function(error) NULL
    )
    if (!identical(backend, "base") && !isTRUE(selected$available)) {
      options(old_options)
      report$benchmarks[[shape_name]][[backend]] <- list(
        status = "skipped",
        reason = if (is.null(selected)) "backend diagnostics unavailable" else
          selected$detection_error
      )
      write_report()
      next
    }

    synchronize <- if (identical(backend, "native")) {
      factory$synchronize
    } else if (identical(backend, "torch")) {
      function() torch::cuda_synchronize()
    } else {
      function() invisible(NULL)
    }
    run_once <- function() {
      total_start <- proc.time()[["elapsed"]]
      pca_start <- total_start
      fit <- cudaverse::cuda_pca(
        values,
        n_components = n_components,
        center = TRUE,
        scale. = TRUE,
        device = device
      )
      synchronize()
      pca_end <- proc.time()[["elapsed"]]
      neighbors <- cudaverse::cuda_knn(
        fit$x,
        k = k,
        metric = "euclidean",
        device = device,
        batch_size = batch_size
      )
      synchronize()
      end <- proc.time()[["elapsed"]]
      list(
        value = list(pca = fit, knn = neighbors),
        seconds = c(
          pca = pca_end - pca_start,
          knn_continuation = end - pca_end,
          full_pipeline = end - total_start
        )
      )
    }
    cold <- run_once()
    cold_seconds <- cold$seconds
    cold$value <- NULL
    cold <- NULL
    gc(FALSE)
    for (index in seq_len(warmup_runs)) {
      warm <- run_once()
      warm$value <- NULL
      warm <- NULL
      gc(FALSE)
    }

    measurements <- matrix(
      NA_real_, nrow = timed_runs, ncol = 3L,
      dimnames = list(NULL, c("pca", "knn_continuation", "full_pipeline"))
    )
    last <- NULL
    for (index in seq_len(timed_runs)) {
      timed <- run_once()
      measurements[index, ] <- timed$seconds
      if (index == timed_runs) {
        last <- timed$value
      } else {
        timed$value <- NULL
      }
      timed <- NULL
      if (index != timed_runs) gc(FALSE)
    }

    if (identical(backend, "base")) {
      saveRDS(reference_payload(last), reference_file, compress = FALSE)
    }
    if (!file.exists(reference_file)) {
      stop(
        "The CPU reference is missing for ", shape_name,
        ". Run the base backend first or reuse its reference directory."
      )
    }
    reference <- readRDS(reference_file)
    validation <- validation_payload(last, reference)
    provenance <- list(
      pca = provenance_payload(last$pca),
      knn = provenance_payload(last$knn),
      pca_scores_device_resident = is.list(attr(
        last$pca$x, "cudaverse_native_state", exact = TRUE
      ))
    )
    if (identical(backend, "native")) {
      pca_stages <- provenance$pca$stages
      knn_stages <- provenance$knn$stages
      expected_pca <- all(c(
        "preprocessing", "decomposition", "scores_resident"
      ) %in% pca_stages$stage) &&
        identical(
          pca_stages$output_device[pca_stages$stage == "scores_resident"],
          "cuda"
        )
      expected_knn <- all(c(
        "input_materialization", "distance", "neighbor_selection"
      ) %in% knn_stages$stage) &&
        identical(
          knn_stages$output_device[knn_stages$stage == "distance"], "cuda"
        ) &&
        identical(
          knn_stages$backend[knn_stages$stage == "neighbor_selection"],
          "native"
        )
      if (!expected_pca || !expected_knn) {
        stop(
          "Native phase 2 provenance was not observed. Reinstall both ",
          "development packages before benchmarking."
        )
      }
    }
    last <- NULL
    gc(FALSE)

    factory$synchronize()
    gc()
    tracker_before <- cudaverseCUDA:::.native_memory_tracker(reset = TRUE)
    device_before <- cudaverseCUDA:::.native_memory_info()
    memory_run <- run_once()
    tracker_during <- cudaverseCUDA:::.native_memory_tracker()
    device_during <- cudaverseCUDA:::.native_memory_info()
    memory_run$value <- NULL
    memory_run <- NULL
    factory$synchronize()
    gc()
    tracker_after <- cudaverseCUDA:::.native_memory_tracker()
    device_after <- cudaverseCUDA:::.native_memory_info()

    report$benchmarks[[shape_name]][[backend]] <- list(
      status = "complete",
      rows = rows,
      columns = columns,
      n_components = n_components,
      cold_seconds = as.list(cold_seconds),
      pca = summary_times(measurements[, "pca"]),
      knn_continuation = summary_times(
        measurements[, "knn_continuation"]
      ),
      full_pipeline = summary_times(measurements[, "full_pipeline"]),
      validation = validation,
      provenance = provenance,
      memory = list(
        operation_owned_current_before_bytes = tracker_before$current,
        operation_owned_peak_bytes = tracker_during$peak -
          tracker_before$current,
        operation_owned_result_bytes = tracker_during$current -
          tracker_before$current,
        operation_owned_post_cleanup_difference_bytes =
          tracker_after$current - tracker_before$current,
        whole_device_used_before_bytes = device_before$used,
        whole_device_used_with_result_bytes = device_during$used,
        whole_device_used_after_cleanup_bytes = device_after$used,
        whole_device_post_cleanup_absolute_difference_bytes =
          abs(device_after$used - device_before$used)
      )
    )
    options(old_options)
    write_report()
  }
  rm(values)
  gc()
}

message("Wrote ", normalizePath(output))
