if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The `jsonlite` package is required to write the stage report.")
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

summary_times <- function(values) {
  list(
    median_seconds = unname(stats::median(values)),
    p95_seconds = unname(stats::quantile(values, 0.95, type = 8)),
    runs_seconds = unname(values)
  )
}

time_call <- function(code, synchronize = function() NULL) {
  start <- proc.time()[["elapsed"]]
  value <- force(code)
  synchronize()
  list(value = value, seconds = proc.time()[["elapsed"]] - start)
}

benchmark_backend <- function(left, right, backend, expected) {
  timed_runs <- as.integer(Sys.getenv("CUDAVERSE_TIMED_RUNS", unset = "10"))
  if (is.na(timed_runs) || timed_runs < 1L) stop("Invalid timed run count.")
  if (identical(backend, "base")) {
    compute <- function() left %*% right
    synchronize <- function() NULL
    resident_left <- resident_right <- NULL
  } else {
    old <- options(cudaverse.cuda_backends = backend)
    on.exit(options(old), add = TRUE)
    resident_left <- cudaverse::cuda_tensor(
      left, device = "cuda", dtype = "float64"
    )
    resident_right <- cudaverse::cuda_tensor(
      right, device = "cuda", dtype = "float64"
    )
    compute <- function() cudaverse::tensor_matmul(
      resident_left, resident_right
    )
    synchronize <- if (identical(backend, "native")) {
      factory$synchronize
    } else {
      function() torch::cuda_synchronize()
    }
  }

  cold <- time_call(compute(), synchronize)
  for (index in seq_len(5L)) time_call(compute(), synchronize)
  resident <- numeric(timed_runs)
  last <- NULL
  for (index in seq_len(timed_runs)) {
    timing <- time_call(compute(), synchronize)
    resident[[index]] <- timing$seconds
    last <- timing$value
  }

  if (identical(backend, "base")) {
    included <- resident
    actual <- last
    provenance <- list(backend = "base", output_device = "cpu")
  } else {
    actual <- cudaverse::to_cpu(last)
    provenance_frame <- cudaverse::cuda_provenance(last)
    provenance <- lapply(provenance_frame, unname)
    for (index in seq_len(5L)) {
      temporary <- cudaverse::tensor_matmul(
        cudaverse::cuda_tensor(left, device = "cuda", dtype = "float64"),
        cudaverse::cuda_tensor(right, device = "cuda", dtype = "float64")
      )
      cudaverse::to_cpu(temporary)
      synchronize()
    }
    included <- numeric(timed_runs)
    for (index in seq_len(timed_runs)) {
      start <- proc.time()[["elapsed"]]
      temporary <- cudaverse::tensor_matmul(
        cudaverse::cuda_tensor(left, device = "cuda", dtype = "float64"),
        cudaverse::cuda_tensor(right, device = "cuda", dtype = "float64")
      )
      actual <- cudaverse::to_cpu(temporary)
      synchronize()
      included[[index]] <- proc.time()[["elapsed"]] - start
    }
  }

  error <- max(abs(actual - expected))
  relative_error <- error / max(1, max(abs(expected)))
  list(
    cold_seconds = cold$seconds,
    resident = summary_times(resident),
    transfer_included = summary_times(included),
    max_absolute_error = error,
    max_relative_error = relative_error,
    provenance = provenance
  )
}

gpu_line <- tryCatch(
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

set.seed(20260806)
size_setting <- Sys.getenv("CUDAVERSE_BENCH_SIZES", unset = "256,1024,4096")
sizes <- as.integer(strsplit(size_setting, ",", fixed = TRUE)[[1L]])
backend_setting <- Sys.getenv(
  "CUDAVERSE_BENCH_BACKENDS",
  unset = "base,native,torch"
)
requested_backends <- strsplit(backend_setting, ",", fixed = TRUE)[[1L]]
benchmarks <- list()
for (size in sizes) {
  left <- matrix(stats::rnorm(size * size), size, size)
  right <- matrix(stats::rnorm(size * size), size, size)
  expected <- left %*% right
  size_results <- list()
  if ("base" %in% requested_backends) {
    size_results$base <- benchmark_backend(left, right, "base", expected)
  }
  if ("native" %in% requested_backends) {
    size_results$native <- benchmark_backend(left, right, "native", expected)
  }
  torch_diagnostics <- cudaverse::cuda_diagnostics()$backend_diagnostics$torch
  if ("torch" %in% requested_backends && isTRUE(torch_diagnostics$available)) {
    size_results$torch <- benchmark_backend(left, right, "torch", expected)
  }
  benchmarks[[as.character(size)]] <- size_results
  rm(left, right, expected, size_results)
  gc()
}

run_lifecycle <- !identical(
  Sys.getenv("CUDAVERSE_RUN_LIFECYCLE", unset = "true"),
  "false"
)
lifecycle <- NULL
if (run_lifecycle) {
  factory$synchronize()
  gc()
  before <- cudaverseCUDA:::.native_memory_info()$used
  values <- matrix(seq_len(64) / 13, 8, 8)
  for (iteration in seq_len(1000L)) {
    pointer <- factory$from_host(values, "float64", dim(values))
    returned <- factory$to_host(pointer)
    stopifnot(identical(returned, as.vector(values)))
    factory$release(pointer)
  }
  factory$synchronize()
  gc()
  after <- cudaverseCUDA:::.native_memory_info()$used
  lifecycle <- list(
    cycles = 1000L,
    used_bytes_before = before,
    used_bytes_after = after,
    absolute_difference_bytes = abs(after - before),
    acceptance_bytes = 1024^2,
    passed = abs(after - before) <= 1024^2
  )
}

installed_path <- find.package("cudaverseCUDA")
installed_files <- list.files(installed_path, recursive = TRUE,
                              full.names = TRUE, all.files = TRUE, no.. = TRUE)
installed_bytes <- sum(file.info(installed_files)$size, na.rm = TRUE)

report <- list(
  schema = "cudaverse-native-stage1/1",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  source_commit = Sys.getenv("GITHUB_SHA", unset = "local-working-tree"),
  hardware = list(nvidia_smi = gpu_line),
  software = list(
    R = R.version.string,
    cudaverse = as.character(utils::packageVersion("cudaverse")),
    cudaverseCUDA = as.character(utils::packageVersion("cudaverseCUDA")),
    native_diagnostics = diagnostics
  ),
  tolerances = list(float64_matmul_rtol = 1e-8),
  lifecycle = lifecycle,
  installed_size_bytes = installed_bytes,
  benchmarks = benchmarks
)

output <- Sys.getenv(
  "CUDAVERSE_REPORT_FILE",
  unset = file.path("inst", "reports", "stage1-rtx2000.json")
)
jsonlite::write_json(report, output, auto_unbox = TRUE, pretty = TRUE,
                     digits = 16, null = "null")
message("Wrote ", normalizePath(output))
