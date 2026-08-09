if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("Matrix", quietly = TRUE)) {
  stop("The `jsonlite` and `Matrix` packages are required for Phase 3.")
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

parse_cases <- function(value) {
  fields <- strsplit(value, ",", fixed = TRUE)[[1L]]
  cases <- lapply(fields, function(field) {
    parts <- strsplit(tolower(trimws(field)), "x", fixed = TRUE)[[1L]]
    if (length(parts) != 3L) {
      stop("Invalid case `", field, "`; use rowsxcolumnsxdensity.")
    }
    rows <- suppressWarnings(as.integer(parts[[1L]]))
    columns <- suppressWarnings(as.integer(parts[[2L]]))
    density <- suppressWarnings(as.numeric(parts[[3L]]))
    if (anyNA(c(rows, columns, density)) || rows < 2L || columns < 2L ||
        density <= 0 || density > 1) {
      stop("Invalid case `", field, "`; density must be in (0, 1].")
    }
    list(rows = rows, columns = columns, density = density)
  })
  names(cases) <- vapply(
    cases,
    function(x) sprintf("%sx%s@%s", x$rows, x$columns, x$density),
    character(1)
  )
  cases
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
  tracked_changes <- tryCatch(
    system2(
      "git",
      c("-C", path, "status", "--porcelain", "--untracked-files=no"),
      stdout = TRUE
    ),
    error = function(error) "unavailable"
  )
  list(
    commit = unname(commit[[1L]]),
    tracked_dirty = length(tracked_changes) > 0L
  )
}

provenance_payload <- function(x) {
  value <- cudaverse::cuda_provenance(x)
  list(
    schema = attr(value, "schema", exact = TRUE),
    compute_device = attr(value, "compute_device", exact = TRUE),
    stages = lapply(value, unname)
  )
}

make_sparse_input <- function(rows, columns, density, seed) {
  set.seed(seed)
  target <- max(2L * rows + columns, ceiling(rows * columns * density))
  remaining <- max(0L, target - 2L * rows - columns)
  i <- c(
    seq_len(rows),
    seq_len(rows),
    rep.int(1L, columns),
    sample.int(rows, remaining, replace = TRUE)
  )
  j <- c(
    rep.int(1L, rows),
    rep.int(2L, rows),
    seq_len(columns),
    sample.int(columns, remaining, replace = TRUE)
  )
  values <- stats::rexp(length(i), rate = 1) + 0.05
  result <- Matrix::sparseMatrix(
    i = i,
    j = j,
    x = values,
    dims = c(rows, columns),
    dimnames = list(
      paste0("sample_", seq_len(rows)),
      paste0("feature_", seq_len(columns))
    ),
    giveCsparse = TRUE
  )
  methods::as(Matrix::drop0(result), "dgCMatrix")
}

strip_matrix <- function(x) {
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
}

reference_payload <- function(value) {
  list(
    normalized = strip_matrix(as.matrix(
      cudaverse::to_dgCMatrix(value$normalized)
    )),
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
  normalized <- strip_matrix(as.matrix(
    cudaverse::to_dgCMatrix(value$normalized)
  ))
  normalized_error <- max(abs(normalized - reference$normalized))
  normalized_scale <- max(1, max(abs(reference$normalized)))

  actual_rotation <- strip_matrix(value$pca$rotation)
  actual_scores <- strip_matrix(value$pca$x)
  actual_projector <- tcrossprod(actual_rotation)
  reference_projector <- tcrossprod(reference$pca$rotation)
  actual_reconstruction <- actual_scores %*% t(actual_rotation)
  reference_reconstruction <- reference$pca$x %*%
    t(reference$pca$rotation)
  reconstruction_error <- max(abs(
    actual_reconstruction - reference_reconstruction
  ))
  reconstruction_scale <- max(1, max(abs(reference_reconstruction)))
  projector_error <- max(abs(actual_projector - reference_projector))

  distance <- unname(value$knn$distance)
  distance_error <- max(abs(distance - reference$knn$distance))
  distance_scale <- max(1, max(abs(reference$knn$distance)))
  indices_identical <- identical(
    unname(value$knn$index), reference$knn$index
  )
  passed <- normalized_error / normalized_scale <= 1e-10 &&
    projector_error <= 1e-8 &&
    reconstruction_error / reconstruction_scale <= 1e-8 &&
    indices_identical &&
    distance_error / distance_scale <= 1e-8

  list(
    normalized_max_absolute_error = normalized_error,
    normalized_max_relative_error = normalized_error / normalized_scale,
    pca_sdev_max_absolute_error = max(abs(
      unname(value$pca$sdev) - reference$pca$sdev
    )),
    pca_subspace_projector_max_absolute_error = projector_error,
    pca_reconstruction_max_absolute_error = reconstruction_error,
    pca_reconstruction_max_relative_error =
      reconstruction_error / reconstruction_scale,
    knn_indices_identical = indices_identical,
    knn_distance_max_absolute_error = distance_error,
    knn_distance_max_relative_error = distance_error / distance_scale,
    passed = passed
  )
}

factory <- cudaverseCUDA::cudaverse_cuda_backend_factory()
native_diagnostics <- factory$diagnostics()
if (!isTRUE(native_diagnostics$available)) {
  stop(
    "The native CUDA backend is unavailable: ",
    native_diagnostics$detection_error
  )
}

warmup_runs <- integer_setting("CUDAVERSE_PHASE3_WARMUPS", "2")
timed_runs <- integer_setting("CUDAVERSE_PHASE3_TIMED_RUNS", "5")
k <- integer_setting("CUDAVERSE_PHASE3_K", "15")
batch_size <- integer_setting("CUDAVERSE_PHASE3_BATCH_SIZE", "256")
components <- integer_setting("CUDAVERSE_PHASE3_COMPONENTS", "20")
cases <- parse_cases(Sys.getenv(
  "CUDAVERSE_PHASE3_CASES",
  unset = "1000x50x0.10,5000x100x0.03,10000x128x0.01"
))
backends <- trimws(strsplit(Sys.getenv(
  "CUDAVERSE_PHASE3_BACKENDS", unset = "base,native,torch"
), ",", fixed = TRUE)[[1L]])
if (length(backends) < 1L || any(!backends %in% c("base", "native", "torch"))) {
  stop("Backends must be selected from base, native, and torch.")
}

output <- Sys.getenv(
  "CUDAVERSE_PHASE3_REPORT",
  unset = file.path("inst", "reports", "phase3-rtx2000.json")
)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)

report <- list(
  schema = "cudaverse-native-phase3/1",
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
    normalization = "row sum to 1000 followed by log1p",
    full_pipeline_boundary = paste(
      "host dgCMatrix -> backend COO/CSR transfer -> sparse normalization ->",
      "PCA -> device-resident scores -> exact distance -> stable top-k ->",
      "host n-by-k result"
    ),
    native_selection = "explicit; global auto preference is unchanged",
    tolerances = list(
      normalization_relative = 1e-10,
      pca_projector_absolute = 1e-8,
      pca_reconstruction_relative = 1e-8,
      knn_distance_relative = 1e-8,
      knn_indices = "identical with original-row stable tie ordering"
    ),
    cases = cases,
    backends = backends
  ),
  installed_size_bytes = list(
    cudaverse = installed_size("cudaverse"),
    cudaverseCUDA = installed_size("cudaverseCUDA")
  ),
  benchmarks = list(),
  hardware_validation = NULL,
  overall_pass = FALSE
)

write_report <- function() {
  report$generated_at_utc <<- format(Sys.time(), tz = "UTC", usetz = TRUE)
  jsonlite::write_json(
    report, output, auto_unbox = TRUE, pretty = TRUE, digits = 16,
    null = "null", na = "null"
  )
}

backend_diagnostics <- function(backend) {
  if (identical(backend, "base")) return(list(available = TRUE))
  tryCatch(
    cudaverse::cuda_diagnostics()$backend_diagnostics[[backend]],
    error = function(error) list(
      available = FALSE,
      detection_error = conditionMessage(error)
    )
  )
}

backend_synchronize <- function(backend) {
  if (identical(backend, "native")) {
    factory$synchronize()
  } else if (identical(backend, "torch")) {
    torch::cuda_synchronize()
  }
  invisible(NULL)
}

pipeline_once <- function(backend, source, n_components) {
  old_options <- options(cudaverse.cuda_backends = backend)
  on.exit(options(old_options), add = TRUE)
  device <- if (identical(backend, "base")) "cpu" else "cuda"
  start <- proc.time()[["elapsed"]]
  sparse <- cudaverse::cuda_sparse(
    source, format = "csr", device = device
  )
  backend_synchronize(backend)
  transfer_end <- proc.time()[["elapsed"]]
  normalized <- cudaverse::sparse_normalize(
    sparse, margin = "rows", scale_factor = 1000, log1p = TRUE
  )
  backend_synchronize(backend)
  normalization_end <- proc.time()[["elapsed"]]
  fit <- cudaverse::cuda_pca(
    normalized,
    n_components = n_components,
    center = TRUE,
    scale. = FALSE,
    device = device
  )
  backend_synchronize(backend)
  pca_end <- proc.time()[["elapsed"]]
  neighbors <- cudaverse::cuda_knn(
    fit$x,
    k = k,
    metric = "euclidean",
    device = device,
    batch_size = batch_size
  )
  backend_synchronize(backend)
  end <- proc.time()[["elapsed"]]
  list(
    value = list(
      sparse = sparse,
      normalized = normalized,
      pca = fit,
      knn = neighbors
    ),
    seconds = c(
      transfer = transfer_end - start,
      normalization = normalization_end - transfer_end,
      pca = pca_end - normalization_end,
      knn = end - pca_end,
      full_pipeline = end - start
    )
  )
}

benchmark_backend <- function(backend, source, n_components) {

  cold <- pipeline_once(backend, source, n_components)
  cold_seconds <- cold$seconds
  cold$value <- NULL
  rm(cold)
  gc(FALSE)
  for (index in seq_len(warmup_runs)) {
    warm <- pipeline_once(backend, source, n_components)
    warm$value <- NULL
    rm(warm)
    gc(FALSE)
  }

  measurements <- matrix(
    NA_real_, nrow = timed_runs, ncol = 5L,
    dimnames = list(
      NULL,
      c("transfer", "normalization", "pca", "knn", "full_pipeline")
    )
  )
  last <- NULL
  for (index in seq_len(timed_runs)) {
    timed <- pipeline_once(backend, source, n_components)
    measurements[index, ] <- timed$seconds
    if (index == timed_runs) {
      last <- timed$value
    } else {
      timed$value <- NULL
    }
    rm(timed)
    if (index != timed_runs) gc(FALSE)
  }

  list(
    cold_seconds = cold_seconds,
    measurements = measurements,
    last = last
  )
}

for (case_name in names(cases)) {
  case <- cases[[case_name]]
  if (k >= case$rows) stop("`k` must be smaller than every row count.")
  n_components <- min(components, case$columns, case$rows - 1L)
  source <- make_sparse_input(
    case$rows,
    case$columns,
    case$density,
    seed = 20260809L + case$rows + case$columns
  )
  observed_density <- Matrix::nnzero(source) / prod(dim(source))
  report$benchmarks[[case_name]] <- list()
  reference <- NULL

  for (backend in backends) {
    message("Benchmarking ", case_name, " with ", backend, " backend")
    diagnostics <- backend_diagnostics(backend)
    if (!isTRUE(diagnostics$available)) {
      report$benchmarks[[case_name]][[backend]] <- list(
        status = "skipped",
        reason = diagnostics$detection_error
      )
      write_report()
      next
    }

    result <- benchmark_backend(backend, source, n_components)
    if (identical(backend, "base")) {
      reference <- reference_payload(result$last)
    }
    if (is.null(reference)) {
      stop("The base backend must run before GPU backends for each case.")
    }
    validation <- validation_payload(result$last, reference)
    provenance <- list(
      normalized = provenance_payload(result$last$normalized),
      pca = provenance_payload(result$last$pca),
      knn = provenance_payload(result$last$knn),
      sparse_storage_device_resident = identical(
        result$last$normalized$device, "cuda"
      ) && typeof(result$last$normalized$storage) == "externalptr",
      pca_scores_device_resident = local({
        score_state <- attr(
          result$last$pca$x,
          "cudaverse_native_state",
          exact = TRUE
        )
        is.list(score_state) && typeof(score_state$storage) == "externalptr"
      })
    )

    cold_seconds <- result$cold_seconds
    measurements <- result$measurements

    result$last <- NULL
    gc(FALSE)
    backend_synchronize(backend)

    tracker_before <- cudaverseCUDA:::.native_memory_tracker(reset = TRUE)
    device_before <- cudaverseCUDA:::.native_memory_info()
    memory_run <- pipeline_once(backend, source, n_components)
    tracker_during <- cudaverseCUDA:::.native_memory_tracker()
    device_during <- cudaverseCUDA:::.native_memory_info()
    torch_memory <- if (identical(backend, "torch")) {
      torch::cuda_memory_stats()
    } else {
      NULL
    }
    memory_run$value <- NULL
    rm(memory_run)
    rm(result)
    gc(FALSE)
    factory$synchronize()
    if (identical(backend, "torch")) torch::cuda_synchronize()
    tracker_after <- cudaverseCUDA:::.native_memory_tracker()
    device_after <- cudaverseCUDA:::.native_memory_info()

    memory <- list(
      native_tracker_current_before_bytes = tracker_before$current,
      native_tracker_peak_delta_bytes = tracker_during$peak -
        tracker_before$current,
      native_tracker_current_with_result_delta_bytes =
        tracker_during$current - tracker_before$current,
      native_tracker_post_cleanup_difference_bytes =
        tracker_after$current - tracker_before$current,
      whole_device_used_before_bytes = device_before$used,
      whole_device_used_with_result_bytes = device_during$used,
      whole_device_used_after_cleanup_bytes = device_after$used,
      whole_device_post_cleanup_absolute_difference_bytes =
        abs(device_after$used - device_before$used),
      backend_allocator_peak_bytes = if (identical(backend, "native")) {
        tracker_during$peak - tracker_before$current
      } else if (identical(backend, "torch")) {
        unname(torch_memory$allocated_bytes$all$peak)
      } else {
        0
      },
      backend_allocator_reserved_peak_bytes = if (identical(backend, "torch")) {
        unname(torch_memory$reserved_bytes$all$peak)
      } else {
        NA_real_
      },
      backend_allocator_peak_source = if (identical(backend, "native")) {
        "cudaverseCUDA operation-owned allocation tracker"
      } else if (identical(backend, "torch")) {
        "torch CUDA allocator session high-water mark"
      } else {
        "not applicable (CPU)"
      }
    )

    report$benchmarks[[case_name]][[backend]] <- list(
      status = "complete",
      rows = case$rows,
      columns = case$columns,
      requested_density = case$density,
      observed_density = observed_density,
      nnz = Matrix::nnzero(source),
      n_components = n_components,
      cold_seconds = as.list(cold_seconds),
      transfer = summary_times(measurements[, "transfer"]),
      normalization = summary_times(measurements[, "normalization"]),
      pca = summary_times(measurements[, "pca"]),
      knn = summary_times(measurements[, "knn"]),
      full_pipeline = summary_times(
        measurements[, "full_pipeline"]
      ),
      validation = validation,
      provenance = provenance,
      memory = memory
    )
    write_report()
  }

  rm(source, reference)
  gc(FALSE)
}

message("Running Phase 3 RTX lifecycle and recovery validation")
old_options <- options(cudaverse.cuda_backends = "native")

factory$synchronize()
invisible(gc())
lifecycle_device_before <- cudaverseCUDA:::.native_memory_info()$used
lifecycle_tracker_before <-
  cudaverseCUDA:::.native_memory_tracker(reset = TRUE)$current
i <- rep(seq_len(16L), each = 3L)
j <- rep(c(1L, 4L, 8L), 16L)
values <- rep(c(1, 2, 3), 16L)
for (iteration in seq_len(1000L)) {
  storage <- factory$sparse_from_coo(i, j, values, c(16L, 8L), "csr")
  normalized <- factory$sparse_normalize(storage, 0L, 100, TRUE)
  factory$sparse_release(normalized$storage)
  factory$sparse_release(storage)
}
factory$synchronize()
invisible(gc())
lifecycle_device_after <- cudaverseCUDA:::.native_memory_info()$used
lifecycle_tracker_after <- cudaverseCUDA:::.native_memory_tracker()
lifecycle <- list(
  cycles = 1000L,
  whole_device_absolute_difference_bytes = abs(
    lifecycle_device_after - lifecycle_device_before
  ),
  tracked_current_difference_bytes =
    lifecycle_tracker_after$current - lifecycle_tracker_before,
  tracked_peak_delta_bytes =
    lifecycle_tracker_after$peak - lifecycle_tracker_before
)
lifecycle$passed <- lifecycle$whole_device_absolute_difference_bytes <= 1024^2 &&
  identical(lifecycle$tracked_current_difference_bytes, 0)

shared_source <- Matrix::sparseMatrix(
  i = c(1L, 1L, 2L, 4L),
  j = c(1L, 3L, 2L, 3L),
  x = c(2, 1, 4, 5),
  dims = c(4L, 3L)
)
shared_csr <- cudaverse::cuda_sparse(
  shared_source, format = "csr", device = "cuda"
)
shared_coo <- cudaverse::as_coo(shared_csr)
factory$sparse_release(shared_csr$storage)
shared_roundtrip <- isTRUE(all.equal(
  as.matrix(cudaverse::to_dgCMatrix(shared_coo)),
  as.matrix(shared_source),
  tolerance = 0
))
shared_final_release <- factory$sparse_release(shared_coo$storage)
shared_released_error <- inherits(tryCatch(
  factory$sparse_to_host(shared_coo$storage), error = identity
), "error")
shared_ownership <- list(
  roundtrip_after_first_owner_release = shared_roundtrip,
  final_release_succeeded = isTRUE(shared_final_release),
  use_after_final_release_errors = shared_released_error,
  passed = shared_roundtrip && isTRUE(shared_final_release) &&
    shared_released_error
)
rm(shared_csr, shared_coo)
gc(FALSE)

error_source <- Matrix::Diagonal(4L, x = 1:4)
error_sparse <- cudaverse::cuda_sparse(error_source, device = "cuda")
broken <- error_sparse
broken$shape <- c(4L, 3L)
condition <- tryCatch(
  cudaverse::sparse_matmul_dense(broken, matrix(1, 3, 2)),
  error = identity
)
backend_reusable <- isTRUE(all.equal(
  as.numeric(cudaverse::sparse_row_sums(error_sparse)),
  as.numeric(1:4),
  tolerance = 0
))
structured_error <- list(
  classes = class(condition),
  backend = condition$backend,
  operation = condition$operation,
  message = conditionMessage(condition),
  backend_reusable = backend_reusable
)
structured_error$passed <- inherits(condition, "cudaverse_native_error") &&
  identical(condition$backend, "native") &&
  identical(condition$operation, "sparse_matmul_dense") &&
  backend_reusable

factory$synchronize()
gc(FALSE)
injected_before <- cudaverseCUDA:::.native_memory_tracker(reset = TRUE)$current
injected <- tryCatch(
  cudaverse:::.backend_call(
    "native", "test_inject_cuda_error", 4096L
  ),
  error = identity
)
factory$synchronize()
gc(FALSE)
injected_after <- cudaverseCUDA:::.native_memory_tracker()
injected_error <- list(
  classes = class(injected),
  backend = injected$backend,
  operation = injected$operation,
  message = conditionMessage(injected),
  tracked_current_difference_bytes = injected_after$current - injected_before
)
injected_error$passed <- inherits(injected, "cudaverse_native_error") &&
  identical(injected$backend, "native") &&
  identical(injected$operation, "test_inject_cuda_error") &&
  identical(injected_error$tracked_current_difference_bytes, 0)

set.seed(20260809L)
interrupt_source <- abs(Matrix::rsparsematrix(512, 256, density = 0.05))
interrupt_sparse <- cudaverse::cuda_sparse(
  interrupt_source, device = "cuda"
)
interrupt_dense <- matrix(stats::rnorm(256 * 64), 256, 64)
factory$synchronize()
gc(FALSE)
interrupt_before <- cudaverseCUDA:::.native_memory_tracker(reset = TRUE)$current
interrupt_condition <- tryCatch({
  setTimeLimit(elapsed = 0.01, transient = TRUE)
  repeat {
    interrupt_product <- cudaverse::sparse_matmul_dense(
      interrupt_sparse, interrupt_dense
    )
    cudaverse::to_cpu(interrupt_product)
  }
}, error = identity, finally = setTimeLimit(
  cpu = Inf, elapsed = Inf, transient = FALSE
))
if (exists("interrupt_product", inherits = FALSE)) rm(interrupt_product)
gc(FALSE)
factory$synchronize()
interrupt_after <- cudaverseCUDA:::.native_memory_tracker()$current
interrupt_reusable <- isTRUE(all.equal(
  as.numeric(cudaverse::sparse_row_sums(interrupt_sparse)),
  as.numeric(Matrix::rowSums(interrupt_source)),
  tolerance = 1e-10
))
interruption <- list(
  condition_message = conditionMessage(interrupt_condition),
  interrupted = inherits(interrupt_condition, "error") && grepl(
    "time limit|elapsed",
    conditionMessage(interrupt_condition),
    ignore.case = TRUE
  ),
  tracked_current_difference_bytes = interrupt_after - interrupt_before,
  backend_reusable = interrupt_reusable
)
interruption$passed <- isTRUE(interruption$interrupted) &&
  identical(interruption$tracked_current_difference_bytes, 0) &&
  interrupt_reusable

resident_source <- make_sparse_input(256L, 48L, 0.08, 20260810L)
resident_sparse <- cudaverse::cuda_sparse(resident_source, device = "cuda")
resident_normalized <- cudaverse::sparse_normalize(
  resident_sparse, margin = "rows", scale_factor = 1000, log1p = TRUE
)
resident_pca <- cudaverse::cuda_pca(
  resident_normalized,
  n_components = 12L,
  center = TRUE,
  scale. = FALSE,
  device = "cuda"
)
resident_knn <- cudaverse::cuda_knn(
  resident_pca$x,
  k = 15L,
  device = "cuda",
  batch_size = 64L
)
resident_normalized_provenance <- provenance_payload(resident_normalized)
resident_pca_provenance <- provenance_payload(resident_pca)
resident_knn_provenance <- provenance_payload(resident_knn)
score_state <- attr(
  resident_pca$x, "cudaverse_native_state", exact = TRUE
)
resident_pipeline <- list(
  provenance_schema = c(
    normalized = resident_normalized_provenance$schema,
    pca = resident_pca_provenance$schema,
    knn = resident_knn_provenance$schema
  ),
  normalized_stages = resident_normalized_provenance$stages$stage,
  pca_stages = resident_pca_provenance$stages$stage,
  knn_stages = resident_knn_provenance$stages$stage,
  normalized_storage_external_pointer =
    typeof(resident_normalized$storage) == "externalptr",
  pca_scores_storage_external_pointer =
    is.list(score_state) && typeof(score_state$storage) == "externalptr",
  normalized_compute_device = resident_normalized_provenance$compute_device,
  pca_compute_device = resident_pca_provenance$compute_device,
  knn_compute_device = resident_knn_provenance$compute_device
)
resident_pipeline$passed <- all(
  resident_pipeline$provenance_schema == "cudaverse-stage/1"
) && isTRUE(resident_pipeline$normalized_storage_external_pointer) &&
  isTRUE(resident_pipeline$pca_scores_storage_external_pointer) &&
  all(c(
    "normalization", "sparse_to_dense", "preprocessing",
    "decomposition", "scores_resident"
  ) %in% resident_pipeline$pca_stages) &&
  all(c("distance", "neighbor_selection") %in%
        resident_pipeline$knn_stages) &&
  identical(resident_pipeline$normalized_compute_device, "cuda") &&
  identical(resident_pipeline$pca_compute_device, "cuda") &&
  identical(resident_pipeline$knn_compute_device, "cuda")

tie_values <- rbind(
  c(0, 0), c(1, 0), c(-1, 0), c(0, 1), c(0, -1), c(1, 0)
)
tie_sparse <- cudaverse::cuda_sparse(
  Matrix::Matrix(tie_values, sparse = TRUE), device = "cuda"
)
tie_native <- cudaverse::cuda_knn(
  tie_sparse, k = 4L, device = "cuda", batch_size = 2L
)
tie_cpu <- cudaverse::cuda_knn(
  tie_sparse, k = 4L, device = "cpu", batch_size = 2L
)
stable_ties <- list(
  native_index = unname(tie_native$index),
  cpu_index = unname(tie_cpu$index),
  indices_identical = identical(
    unname(tie_native$index), unname(tie_cpu$index)
  ),
  distance_max_absolute_error = max(abs(
    unname(tie_native$distance) - unname(tie_cpu$distance)
  )),
  rule = "equal distances are ordered by original one-based row index"
)
stable_ties$passed <- stable_ties$indices_identical &&
  stable_ties$distance_max_absolute_error <= 1e-10

report$hardware_validation <- list(
  schema = "cudaverse-native-phase3-validation/1",
  lifecycle = lifecycle,
  shared_ownership = shared_ownership,
  structured_error = structured_error,
  injected_cuda_error = injected_error,
  interruption = interruption,
  resident_pipeline = resident_pipeline,
  stable_ties = stable_ties
)

benchmark_pass <- all(vapply(
  report$benchmarks,
  function(case) all(vapply(
    case,
    function(result) identical(result$status, "complete") &&
      isTRUE(result$validation$passed),
    logical(1)
  )),
  logical(1)
))
validation_pass <- all(vapply(
  report$hardware_validation[-1L],
  function(value) isTRUE(value$passed),
  logical(1)
))
report$overall_pass <- benchmark_pass && validation_pass
write_report()
options(old_options)

if (!isTRUE(report$overall_pass)) {
  stop("Phase 3 report completed, but one or more gates failed.")
}
message("Wrote passing Phase 3 report to ", normalizePath(output))
