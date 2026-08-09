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
    array(apply(values, 2, sum), dim = 3L),
    tolerance = 1e-10
  )
  expect_equal(
    cudaverse::to_cpu(cudaverse::tensor_mean(tensor, dim = 2, keepdim = TRUE)),
    array(apply(values, c(1, 3), mean), dim = c(2, 1, 4)),
    tolerance = 1e-10
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

test_that("one thousand allocate-transfer-free cycles do not leak VRAM", {
  skip_if_not(identical(Sys.getenv("CUDAVERSE_NATIVE_TESTS"), "true"))
  skip_if_not(isTRUE(cudaverseCUDA:::.native_diagnostics()$available))
  factory <- cudaverse_cuda_backend_factory()
  factory$synchronize()
  gc()
  baseline <- cudaverseCUDA:::.native_memory_info()$used

  values <- matrix(seq_len(64) / 13, 8, 8)
  for (iteration in seq_len(1000L)) {
    pointer <- factory$from_host(values, "float64", dim(values))
    expect_equal(factory$to_host(pointer), as.vector(values), tolerance = 0)
    factory$release(pointer)
  }
  factory$synchronize()
  gc()
  final <- cudaverseCUDA:::.native_memory_info()$used

  expect_lte(abs(final - baseline), 1024^2)
})
