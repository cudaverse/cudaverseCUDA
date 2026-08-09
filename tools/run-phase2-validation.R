if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The `jsonlite` package is required to write validation evidence.")
}
if (!requireNamespace("cudaverse", quietly = TRUE) ||
    !requireNamespace("cudaverseCUDA", quietly = TRUE)) {
  stop("Install the development cudaverse and cudaverseCUDA packages first.")
}

factory <- cudaverseCUDA::cudaverse_cuda_backend_factory()
diagnostics <- factory$diagnostics()
if (!isTRUE(diagnostics$available)) {
  stop("Native CUDA is unavailable: ", diagnostics$detection_error)
}

old_options <- options(cudaverse.cuda_backends = "native")
on.exit(options(old_options), add = TRUE)

source_state <- function(path) {
  list(
    commit = unname(system2(
      "git", c("-C", path, "rev-parse", "HEAD"), stdout = TRUE
    )[[1L]]),
    dirty = length(system2(
      "git", c("-C", path, "status", "--porcelain"), stdout = TRUE
    )) > 0L
  )
}

snapshot <- function(reset = FALSE) {
  factory$synchronize()
  gc()
  list(
    tracker = cudaverseCUDA:::.native_memory_tracker(reset = reset),
    device = cudaverseCUDA:::.native_memory_info()
  )
}

memory_result <- function(before, after, threshold = 1024^2) {
  tracker_difference <- after$tracker$current - before$tracker$current
  device_difference <- abs(after$device$used - before$device$used)
  list(
    tracked_bytes_before = before$tracker$current,
    tracked_bytes_after = after$tracker$current,
    tracked_difference_bytes = tracker_difference,
    whole_device_used_before_bytes = before$device$used,
    whole_device_used_after_bytes = after$device$used,
    whole_device_absolute_difference_bytes = device_difference,
    acceptance_bytes = threshold,
    passed = identical(tracker_difference, 0) &&
      device_difference <= threshold
  )
}

values <- matrix(seq_len(64) / 13, 8, 8)
warm <- factory$from_host(values, "float64", dim(values))
factory$release(warm)
before <- snapshot(reset = TRUE)
for (iteration in seq_len(1000L)) {
  pointer <- factory$from_host(values, "float64", dim(values))
  returned <- factory$to_host(pointer)
  if (!identical(returned, as.vector(values))) {
    stop("Lifecycle transfer parity failed at iteration ", iteration, ".")
  }
  factory$release(pointer)
}
after <- snapshot()
lifecycle <- c(
  list(cycles = 1000L, transfer_parity = TRUE),
  memory_result(before, after)
)

set.seed(20260809)
warmup <- matrix(stats::rnorm(24), 8, 3)
cudaverse::cuda_pca(warmup, 2, device = "cuda")
constant <- matrix(1, 8, 3)
before <- snapshot(reset = TRUE)
condition <- tryCatch(
  cudaverse:::.backend_call(
    "native", "algorithm_pca", constant, 2L, TRUE, TRUE
  ),
  error = identity
)
for (iteration in seq_len(99L)) {
  try(
    cudaverse:::.backend_call(
      "native", "algorithm_pca", constant, 2L, TRUE, TRUE
    ),
    silent = TRUE
  )
}
rm(warmup, constant)
after <- snapshot()
structured_errors <- c(
  list(
    injections = 100L,
    condition_classes = class(condition),
    backend = condition$backend,
    operation = condition$operation,
    structured = inherits(condition, "cudaverse_native_error") &&
      inherits(condition, "cudaverse_backend_error") &&
      identical(condition$backend, "native") &&
      identical(condition$operation, "algorithm_pca")
  ),
  memory_result(before, after)
)
structured_errors$passed <- isTRUE(structured_errors$structured) &&
  isTRUE(structured_errors$passed)

before <- snapshot(reset = TRUE)
condition <- tryCatch(
  cudaverse:::.backend_call(
    "native", "test_inject_cuda_error", 4096L
  ),
  error = identity
)
for (iteration in seq_len(99L)) {
  try(
    cudaverse:::.backend_call(
      "native", "test_inject_cuda_error", 4096L
    ),
    silent = TRUE
  )
}
after <- snapshot()
cuda_errors <- c(
  list(
    injections = 100L,
    temporary_allocation_bytes = 4096L,
    condition_classes = class(condition),
    backend = condition$backend,
    operation = condition$operation,
    message = conditionMessage(condition),
    structured = inherits(condition, "cudaverse_native_error") &&
      inherits(condition, "cudaverse_backend_error") &&
      identical(condition$backend, "native") &&
      identical(condition$operation, "test_inject_cuda_error") &&
      grepl("OUT_OF_MEMORY|out of memory", conditionMessage(condition))
  ),
  memory_result(before, after)
)
cuda_errors$passed <- isTRUE(cuda_errors$structured) &&
  isTRUE(cuda_errors$passed)

pipeline_values <- matrix(stats::rnorm(400), 50, 8)
fit <- factory$algorithm_pca(pipeline_values, 4L, TRUE, FALSE)
state <- factory$algorithm_knn_prepare(fit$x)
factory$algorithm_knn_select(state, fit$x, 5L, "euclidean", 13L)
rm(fit, state)
before <- snapshot(reset = TRUE)
result <- NULL
for (iteration in seq_len(100L)) {
  fit <- factory$algorithm_pca(pipeline_values, 4L, TRUE, FALSE)
  state <- factory$algorithm_knn_prepare(fit$x)
  result <- factory$algorithm_knn_select(
    state, fit$x, 5L, "euclidean", 13L
  )
}
shape_ok <- identical(dim(result$index), c(50L, 5L))
rm(fit, state, result, pipeline_values)
after <- snapshot()
repeated_pipeline <- c(
  list(cycles = 100L, result_shape_valid = shape_ok),
  memory_result(before, after)
)
repeated_pipeline$passed <- isTRUE(shape_ok) &&
  isTRUE(repeated_pipeline$passed)

interrupt_values <- matrix(stats::rnorm(40000), 2000, 20)
before <- snapshot(reset = TRUE)
state <- factory$algorithm_knn_prepare(interrupt_values)
condition <- tryCatch(
  {
    setTimeLimit(elapsed = 0.02, transient = TRUE)
    factory$algorithm_knn_select(
      state, interrupt_values, 15L, "euclidean", 1L
    )
  },
  error = identity,
  finally = setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
)
interrupt_message <- if (inherits(condition, "condition")) {
  conditionMessage(condition)
} else {
  "The operation completed before the time limit."
}
probe <- factory$from_host(matrix(1:4, 2), "float64", c(2L, 2L))
probe_ok <- identical(factory$to_host(probe), as.numeric(1:4))
factory$release(probe)
rm(state, interrupt_values)
after <- snapshot()
interruption <- c(
  list(
    condition_classes = class(condition),
    condition_message = interrupt_message,
    interrupted = inherits(condition, "error") && grepl(
      "time limit|elapsed", interrupt_message, ignore.case = TRUE
    ),
    backend_reusable = probe_ok
  ),
  memory_result(before, after)
)
interruption$passed <- isTRUE(interruption$interrupted) &&
  isTRUE(interruption$backend_reusable) && isTRUE(interruption$passed)

report <- list(
  schema = "cudaverse-native-phase2-validation/1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  source = list(
    cudaverseCUDA = source_state("."),
    cudaverse = source_state(file.path("..", "cudaverse"))
  ),
  software = list(
    R = R.version.string,
    cudaverse = as.character(utils::packageVersion("cudaverse")),
    cudaverseCUDA = as.character(utils::packageVersion("cudaverseCUDA")),
    diagnostics = diagnostics
  ),
  lifecycle = lifecycle,
  structured_errors = structured_errors,
  cuda_errors = cuda_errors,
  repeated_pipeline = repeated_pipeline,
  interruption = interruption,
  passed = all(vapply(
    list(
      lifecycle, structured_errors, cuda_errors, repeated_pipeline,
      interruption
    ),
    function(x) isTRUE(x$passed), logical(1)
  ))
)

output <- Sys.getenv(
  "CUDAVERSE_VALIDATION_FILE",
  unset = file.path("inst", "reports", "phase2-rtx2000-validation.json")
)
jsonlite::write_json(
  report, output, auto_unbox = TRUE, pretty = TRUE, digits = 16,
  null = "null", na = "null"
)
message("Wrote ", normalizePath(output))
