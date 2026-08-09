test_that("native diagnostics and factory follow the extension contract", {
  diagnostics <- cudaverseCUDA:::.native_diagnostics()
  factory <- cudaverse_cuda_backend_factory()

  expect_named(
    diagnostics,
    c(
      "installed", "available", "device_count", "version", "reason",
      "detection_error", "driver_version", "cublas_loaded",
      "kernels_loaded"
    )
  )
  expect_true(diagnostics$installed)
  expect_identical(factory$name, "native")
  expect_identical(factory$device, "cuda")
  expect_true(all(c(
    "diagnostics", "capabilities", "from_host", "to_host", "cast",
    "matmul", "reduce", "synchronize", "release", "error_translate"
  ) %in% names(factory)))
})

test_that("native reductions match R across dimensions and dtypes", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  values <- array(seq_len(24) / 7, dim = c(2, 3, 4))
  dimnames(values) <- list(
    sample = c("sample_a", "sample_b"),
    feature = paste0("feature_", 1:3),
    batch = paste0("batch_", 1:4)
  )
  tensor <- cudaverse::cuda_tensor(
    values, device = "cuda", dtype = "float64"
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_sum(tensor)),
    array(sum(values), dim = 1L),
    tolerance = 1e-10
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_sum(tensor, dim = c(1, 3))),
    array(
      apply(values, 2, sum),
      dim = 3L,
      dimnames = list(feature = paste0("feature_", 1:3))
    ),
    tolerance = 1e-10
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_mean(tensor, dim = 2, keepdim = TRUE)),
    array(
      apply(values, c(1, 3), mean),
      dim = c(2, 1, 4),
      dimnames = list(
        sample = c("sample_a", "sample_b"),
        feature = NULL,
        batch = paste0("batch_", 1:4)
      )
    ),
    tolerance = 1e-10
  )
  reduced <- cudaverse::tensor_sum(tensor, dim = 2, keepdim = TRUE)
  expect_identical(
    dimnames(cudaverse::to_cpu(reduced)),
    list(
      sample = c("sample_a", "sample_b"),
      feature = NULL,
      batch = paste0("batch_", 1:4)
    )
  )

  float_values <- matrix(seq_len(20) / 11, 4, 5)
  float_tensor <- cudaverse::cuda_tensor(
    float_values, device = "cuda", dtype = "float32"
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_sum(float_tensor, dim = 1)),
    array(colSums(float_values), dim = 5L),
    tolerance = 1e-5
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_mean(float_tensor, dim = 2)),
    array(rowMeans(float_values), dim = 4L),
    tolerance = 1e-5
  )

  integer_tensor <- cudaverse::cuda_tensor(
    array(1:12, c(3, 4)), device = "cuda", dtype = "integer"
  )
  integer_sum <- cudaverse::tensor_sum(integer_tensor, dim = 1)
  expect_identical(integer_sum$dtype, "float64")
  expect_equal(
    cudaverse::to_cpu(integer_sum),
    array(colSums(matrix(1:12, 3, 4)), dim = 4L),
    tolerance = 0
  )
})

test_that("native cast and integer reduction kernels preserve values", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  factory <- cudaverse_cuda_backend_factory()

  integer_storage <- factory$from_host(1:12, "integer", c(3L, 4L))
  on.exit(factory$release(integer_storage), add = TRUE)
  integer_reduction <- factory$reduce(
    integer_storage, 1L, FALSE, "sum"
  )
  on.exit(factory$release(integer_reduction$storage), add = TRUE)
  expect_identical(
    factory$to_host(integer_reduction$storage),
    as.integer(c(6, 15, 24, 33))
  )

  float_storage <- factory$cast(integer_storage, "float32")
  on.exit(factory$release(float_storage), add = TRUE)
  double_storage <- factory$cast(float_storage, "float64")
  on.exit(factory$release(double_storage), add = TRUE)
  expect_equal(factory$to_host(double_storage), as.numeric(1:12), tolerance = 0)

  non_finite <- factory$from_host(
    c(1, Inf, NaN), "float64", 3L
  )
  on.exit(factory$release(non_finite), add = TRUE)
  non_finite_sum <- factory$reduce(non_finite, NULL, FALSE, "sum")
  on.exit(factory$release(non_finite_sum$storage), add = TRUE)
  expect_true(is.nan(factory$to_host(non_finite_sum$storage)))
})

test_that("native cuSOLVER SVD matches reconstruction for tall and wide data", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  skip_if_not(nzchar(Sys.getenv("CUDAVERSE_CUSOLVER_PATH")))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  set.seed(20260809)
  for (shape in list(c(12L, 5L), c(5L, 12L), c(5L, 1L))) {
    values <- matrix(rnorm(prod(shape)), nrow = shape[[1L]])
    actual <- cudaverse::cuda_svd(values, device = "cuda")
    expected <- base::svd(values)
    reconstruction <- actual$u %*% diag(actual$d, nrow = length(actual$d)) %*%
      t(actual$v)

    expect_equal(unname(actual$d), expected$d, tolerance = 1e-8)
    expect_equal(reconstruction, values, tolerance = 1e-8)
  }
})

test_that("native PCA matches R subspaces and retains device scores", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  skip_if_not(nzchar(Sys.getenv("CUDAVERSE_CUSOLVER_PATH")))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  set.seed(2718)
  values <- matrix(rnorm(200), 40, 5)
  for (center in c(FALSE, TRUE)) {
    for (scale in c(FALSE, TRUE)) {
      actual <- cudaverse::cuda_pca(
        values, n_components = 3, center = center, scale. = scale,
        device = "cuda"
      )
      expected <- stats::prcomp(
        values, rank. = 3, center = center, scale. = scale
      )
      actual_reconstruction <- actual$x %*% t(actual$rotation)
      expected_reconstruction <- expected$x[, 1:3, drop = FALSE] %*%
        t(expected$rotation[, 1:3, drop = FALSE])

      expect_equal(unname(actual$sdev), expected$sdev[1:3], tolerance = 1e-8)
      expect_equal(actual_reconstruction, expected_reconstruction,
                   tolerance = 1e-8)
      expect_equal(actual$center, expected$center, tolerance = 1e-10)
      expect_equal(actual$scale, expected$scale, tolerance = 1e-10)
      state <- attr(actual$x, "cudaverse_native_state", exact = TRUE)
      expect_identical(state$backend, "native")
      expect_identical(state$shape, c(40L, 3L))
      expect_type(state$storage, "externalptr")
    }
  }
})

test_that("native distances and stable device top-k match CPU ordering", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  values <- rbind(
    c(0, 0), c(1, 0), c(-1, 0), c(0, 1), c(0, -1), c(1, 0)
  )
  cpu_distance <- cudaverse::cuda_distance(values, device = "cpu")
  native_distance <- cudaverse::cuda_distance(values, device = "cuda")
  expect_equal(
    as.vector(native_distance), as.vector(cpu_distance), tolerance = 1e-10
  )

  factory <- cudaverse_cuda_backend_factory()
  state <- factory$algorithm_knn_prepare(values)
  native <- factory$algorithm_knn_select(
    state, values, 4L, "euclidean", 2L
  )
  cpu <- cudaverse::cuda_knn(
    values, k = 4, device = "cpu", batch_size = 2
  )
  expect_identical(native$index, cpu$index)
  expect_equal(native$distance, cpu$distance, tolerance = 1e-10)

  cosine_values <- rbind(
    c(1, 0), c(2, 0), c(0, 1), c(0, -1), c(-1, 0), c(1, 1)
  )
  unit_values <- cosine_values / sqrt(rowSums(cosine_values^2))
  cosine_state <- factory$algorithm_knn_prepare(unit_values)
  native_cosine <- factory$algorithm_knn_select(
    cosine_state, unit_values, 3L, "cosine", 3L
  )
  cpu_cosine <- cudaverse::cuda_knn(
    cosine_values, k = 3, metric = "cosine", device = "cpu", batch_size = 3
  )
  expect_identical(native_cosine$index, cpu_cosine$index)
  expect_equal(native_cosine$distance, cpu_cosine$distance, tolerance = 1e-10)

  set.seed(33)
  general_values <- matrix(rnorm(135), 45, 3)
  general_state <- factory$algorithm_knn_prepare(general_values)
  native_general <- factory$algorithm_knn_select(
    general_state, general_values, 40L, "euclidean", 11L
  )
  cpu_general <- cudaverse::cuda_knn(
    general_values, k = 40, device = "cpu", batch_size = 11
  )
  expect_identical(native_general$index, cpu_general$index)
  expect_equal(native_general$distance, cpu_general$distance, tolerance = 1e-10)
})

test_that("native algorithm failures are structured and release temporaries", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  skip_if_not(nzchar(Sys.getenv("CUDAVERSE_CUSOLVER_PATH")))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  warmup <- matrix(rnorm(24), 8, 3)
  cudaverse::cuda_pca(warmup, 2, device = "cuda")
  factory <- cudaverse_cuda_backend_factory()
  factory$synchronize()
  gc()
  baseline <- cudaverseCUDA:::.native_memory_info()$used

  constant <- matrix(1, 8, 3)
  condition <- tryCatch(
    cudaverse:::.backend_call(
      "native", "algorithm_pca", constant, 2L, TRUE, TRUE
    ),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_native_error")
  expect_s3_class(condition, "cudaverse_backend_error")
  expect_identical(condition$backend, "native")
  expect_identical(condition$operation, "algorithm_pca")

  for (iteration in seq_len(100L)) {
    try(
      cudaverse:::.backend_call(
        "native", "algorithm_pca", constant, 2L, TRUE, TRUE
      ),
      silent = TRUE
    )
  }
  factory$synchronize()
  gc()
  final <- cudaverseCUDA:::.native_memory_info()$used
  expect_lte(abs(final - baseline), 1024^2)
})

test_that("injected CUDA failures unwind temporary allocations", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  factory <- cudaverse_cuda_backend_factory()
  factory$synchronize()
  gc()
  baseline <- cudaverseCUDA:::.native_memory_tracker(reset = TRUE)$current

  condition <- tryCatch(
    cudaverse:::.backend_call(
      "native", "test_inject_cuda_error", 4096L
    ),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_native_error")
  expect_s3_class(condition, "cudaverse_backend_error")
  expect_identical(condition$backend, "native")
  expect_identical(condition$operation, "test_inject_cuda_error")
  expect_match(conditionMessage(condition), "OUT_OF_MEMORY|out of memory")

  for (iteration in seq_len(100L)) {
    try(
      cudaverse:::.backend_call(
        "native", "test_inject_cuda_error", 4096L
      ),
      silent = TRUE
    )
  }
  factory$synchronize()
  gc()
  final <- cudaverseCUDA:::.native_memory_tracker()
  expect_identical(final$current, baseline)
  expect_gte(final$peak - baseline, 4096)
})

test_that("repeated dense pipelines release all operation-owned VRAM", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  skip_if_not(nzchar(Sys.getenv("CUDAVERSE_CUSOLVER_PATH")))
  factory <- cudaverse_cuda_backend_factory()
  values <- matrix(rnorm(400), 50, 8)

  warmup <- factory$algorithm_pca(values, 4L, TRUE, FALSE)
  state <- factory$algorithm_knn_prepare(warmup$x)
  factory$algorithm_knn_select(state, warmup$x, 5L, "euclidean", 13L)
  rm(warmup, state)
  factory$synchronize()
  gc()
  baseline <- cudaverseCUDA:::.native_memory_info()$used

  for (iteration in seq_len(100L)) {
    fit <- factory$algorithm_pca(values, 4L, TRUE, FALSE)
    state <- factory$algorithm_knn_prepare(fit$x)
    result <- factory$algorithm_knn_select(
      state, fit$x, 5L, "euclidean", 13L
    )
    expect_identical(dim(result$index), c(50L, 5L))
  }
  rm(fit, state, result)
  factory$synchronize()
  gc()
  final <- cudaverseCUDA:::.native_memory_info()$used
  expect_lte(abs(final - baseline), 1024^2)
})

test_that("R time-limit interruption leaves the native backend reusable", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  factory <- cudaverse_cuda_backend_factory()
  values <- matrix(rnorm(40000), 2000, 20)
  factory$synchronize()
  gc()
  baseline <- cudaverseCUDA:::.native_memory_info()$used

  state <- factory$algorithm_knn_prepare(values)
  on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE), add = TRUE)
  setTimeLimit(elapsed = 0.02, transient = TRUE)
  condition <- tryCatch(
    factory$algorithm_knn_select(
      state, values, 15L, "euclidean", 1L
    ),
    error = identity
  )
  setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
  expect_s3_class(condition, "error")
  expect_match(conditionMessage(condition), "time limit|elapsed")

  factory$synchronize()
  probe <- factory$from_host(matrix(1:4, 2), "float64", c(2L, 2L))
  expect_equal(factory$to_host(probe), as.numeric(1:4), tolerance = 0)
  factory$release(probe)
  gc()
  final <- cudaverseCUDA:::.native_memory_info()$used
  expect_lte(abs(final - baseline), 1024^2)
})
test_that("native errors translate into structured cudaverse conditions", {
  factory <- cudaverse_cuda_backend_factory()
  raw <- simpleError("injected CUDA failure")
  condition <- factory$error_translate(raw, "matmul")

  expect_s3_class(condition, "cudaverse_native_error")
  expect_s3_class(condition, "cudaverse_backend_error")
  expect_identical(condition$backend, "native")
  expect_identical(condition$operation, "matmul")
  expect_identical(condition$parent, raw)
})

test_that("native transfer and cuBLAS matmul match base R", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  old <- options(cudaverse.cuda_backends = "native")
  on.exit(options(old), add = TRUE)

  left <- matrix(seq_len(15) / 7, 5, 3)
  right <- matrix(seq_len(12) / 11, 3, 4)
  actual <- cudaverse::tensor_matmul(
    cudaverse::cuda_tensor(left, device = "cuda", dtype = "float64"),
    cudaverse::cuda_tensor(right, device = "cuda", dtype = "float64")
  )

  expect_identical(
    cudaverse::tensor_device(actual),
    c(device = "cuda", backend = "native")
  )
  expect_equal(cudaverse::to_cpu(actual), left %*% right, tolerance = 1e-10)
  provenance <- cudaverse::cuda_provenance(actual)
  expect_identical(provenance$backend, "native")
  expect_identical(provenance$output_device, "cuda")
})

test_that("shared native ownership frees an allocation exactly once", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  factory <- cudaverse_cuda_backend_factory()

  original <- factory$from_host(as.double(1:8), "float64", 8L)
  shared <- cudaverseCUDA:::.native_share(original)
  factory$release(original)
  expect_equal(factory$to_host(shared), as.double(1:8))
  expect_true(factory$release(shared))
  expect_error(factory$to_host(shared), "released")
})

test_that("operation-owned allocation telemetry reports a high-water mark", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  factory <- cudaverse_cuda_backend_factory()
  gc()
  baseline <- cudaverseCUDA:::.native_memory_tracker(reset = TRUE)$current

  first <- factory$from_host(double(128), "float64", 128L)
  second <- factory$from_host(double(256), "float64", 256L)
  during <- cudaverseCUDA:::.native_memory_tracker()
  expect_identical(during$current - baseline, 384 * 8)
  expect_gte(during$peak - baseline, 384 * 8)

  factory$release(first)
  factory$release(second)
  factory$synchronize()
  gc()
  final <- cudaverseCUDA:::.native_memory_tracker()
  expect_identical(final$current, baseline)
  expect_gte(final$peak - baseline, 384 * 8)
})

test_that("one thousand allocate-transfer-free cycles do not leak VRAM", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  factory <- cudaverse_cuda_backend_factory()
  factory$synchronize()
  gc()
  baseline <- cudaverseCUDA:::.native_memory_info()$used
  tracked_baseline <- cudaverseCUDA:::.native_memory_tracker(reset = TRUE)$current

  values <- matrix(seq_len(64) / 13, 8, 8)
  for (iteration in seq_len(1000L)) {
    pointer <- factory$from_host(values, "float64", dim(values))
    expect_equal(factory$to_host(pointer), as.vector(values), tolerance = 0)
    factory$release(pointer)
  }
  factory$synchronize()
  gc()
  final <- cudaverseCUDA:::.native_memory_info()$used
  tracked_final <- cudaverseCUDA:::.native_memory_tracker()

  expect_lte(abs(final - baseline), 1024^2)
  expect_identical(tracked_final$current, tracked_baseline)
  expect_gte(tracked_final$peak - tracked_baseline, length(values) * 8)
})
