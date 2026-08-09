.native_kernel_state <- new.env(parent = emptyenv())
.native_kernel_state$loaded <- FALSE

.native_kernel_path <- function() {
  system.file(
    "kernels",
    "cudaverse_dense_kernels.ptx",
    package = "cudaverseCUDA",
    mustWork = FALSE
  )
}

.native_ensure_kernels <- function() {
  if (isTRUE(.native_kernel_state$loaded)) return(invisible(TRUE))
  path <- .native_kernel_path()
  if (!nzchar(path) || !file.exists(path)) {
    stop(
      "The CUDA 12.8.1 dense-kernel PTX artifact is missing.",
      call. = FALSE
    )
  }
  .Call(C_cudaverse_cuda_load_kernels, normalizePath(path, mustWork = TRUE))
  .native_kernel_state$loaded <- TRUE
  invisible(TRUE)
}

.native_diagnostics <- function() {
  diagnostics <- .Call(C_cudaverse_cuda_diagnostics)
  kernel_error <- NULL
  if (isTRUE(diagnostics$available)) {
    tryCatch(
      .native_ensure_kernels(),
      error = function(error) kernel_error <<- conditionMessage(error)
    )
  }
  diagnostics$kernels_loaded <- isTRUE(.native_kernel_state$loaded)
  if (!is.null(kernel_error)) {
    diagnostics$available <- FALSE
    diagnostics$reason <- "kernel_unavailable"
    diagnostics$detection_error <- kernel_error
  }
  diagnostics
}

.native_from_host <- function(x, dtype, shape, dimnames = NULL) {
  .Call(
    C_cudaverse_cuda_from_host,
    x,
    as.character(dtype),
    as.integer(shape)
  )
}

.native_to_host <- function(storage) {
  .Call(C_cudaverse_cuda_to_host, storage)
}

.native_cast <- function(storage, dtype) {
  .native_ensure_kernels()
  .Call(C_cudaverse_cuda_cast, storage, as.character(dtype))
}

.native_reduce <- function(storage, dim, keepdim, method) {
  .native_ensure_kernels()
  .Call(
    C_cudaverse_cuda_reduce,
    storage,
    if (is.null(dim)) integer() else as.integer(dim),
    as.logical(keepdim),
    as.character(method)
  )
}

.native_algorithm_svd <- function(x, nu, nv) {
  .native_ensure_kernels()
  storage <- .native_from_host(x, "float64", dim(x))
  on.exit(.native_release(storage), add = TRUE)
  result <- .Call(
    C_cudaverse_cuda_svd,
    storage,
    as.integer(nu),
    as.integer(nv)
  )
  result$u <- matrix(result$u, nrow = nrow(x), ncol = nu)
  result$v <- matrix(result$v, nrow = ncol(x), ncol = nv)
  result
}

.native_algorithm_pca <- function(x, n_components, center, scale) {
  .native_ensure_kernels()
  storage <- .native_from_host(x, "float64", dim(x))
  on.exit(.native_release(storage), add = TRUE)
  result <- .Call(
    C_cudaverse_cuda_pca,
    storage,
    as.integer(n_components),
    as.logical(center),
    as.logical(scale)
  )
  result$rotation <- matrix(
    result$rotation,
    nrow = ncol(x),
    ncol = n_components
  )
  result$x <- matrix(result$x, nrow = nrow(x), ncol = n_components)
  attr(result$x, "cudaverse_native_state") <- list(
    storage = result$scores_storage,
    shape = as.integer(c(nrow(x), n_components)),
    dtype = "float64",
    backend = "native"
  )
  result$scores_storage <- NULL
  result$center <- if (isTRUE(center)) result$center else FALSE
  result$scale <- if (isTRUE(scale)) result$scale else FALSE
  result
}

.native_matrix_storage <- function(values) {
  state <- attr(values, "cudaverse_native_state", exact = TRUE)
  valid_state <- is.list(state) && identical(state$backend, "native") &&
    identical(state$dtype, "float64") &&
    identical(state$shape, as.integer(dim(values))) &&
    typeof(state$storage) == "externalptr"
  if (valid_state) {
    return(.native_share(state$storage))
  }
  .native_from_host(values, "float64", dim(values))
}

.native_distance_storage <- function(values, source_values, metric) {
  source_state <- attr(
    source_values, "cudaverse_native_state", exact = TRUE
  )
  resident_source <- is.list(source_state) &&
    identical(source_state$backend, "native") &&
    typeof(source_state$storage) == "externalptr"
  if (identical(metric, "cosine") && resident_source) {
    raw <- .native_matrix_storage(source_values)
    on.exit(.native_release(raw), add = TRUE)
    return(.Call(C_cudaverse_cuda_normalize_rows, raw))
  }
  .native_matrix_storage(values)
}

.native_algorithm_distance <- function(x, y, metric,
                                       source_x = x, source_y = y) {
  .native_ensure_kernels()
  self <- identical(source_x, source_y)
  x_storage <- .native_distance_storage(x, source_x, metric)
  on.exit(.native_release(x_storage), add = TRUE)
  y_storage <- if (self) {
    x_storage
  } else {
    .native_distance_storage(y, source_y, metric)
  }
  if (!self) on.exit(.native_release(y_storage), add = TRUE)
  distance_storage <- .Call(
    C_cudaverse_cuda_distance,
    x_storage,
    y_storage,
    as.character(metric),
    self
  )
  on.exit(.native_release(distance_storage), add = TRUE)
  matrix(
    .native_to_host(distance_storage),
    nrow = nrow(x),
    ncol = nrow(y)
  )
}

.native_knn_prepare <- function(values, metric = "euclidean",
                                source_values = values) {
  .native_ensure_kernels()
  storage <- .native_distance_storage(values, source_values, metric)
  norms <- .Call(C_cudaverse_cuda_row_norms, storage)
  list(
    storage = storage,
    norms = norms,
    rows = nrow(values),
    columns = ncol(values)
  )
}

.native_knn_block_compat <- function(storage, values, rows, metric) {
  .native_algorithm_distance(
    values[rows, , drop = FALSE],
    values,
    metric
  )
}

.native_knn_select <- function(storage, values, k, metric, batch_size) {
  on.exit(.native_release(storage$storage), add = TRUE)
  on.exit(.native_release(storage$norms), add = TRUE)
  starts <- seq.int(1L, storage$rows, by = batch_size)
  index <- matrix(NA_integer_, storage$rows, k)
  distance <- matrix(NA_real_, storage$rows, k)
  for (start in starts) {
    count <- min(batch_size, storage$rows - start + 1L)
    block <- .Call(
      C_cudaverse_cuda_knn_block,
      storage$storage,
      storage$norms,
      as.integer(start - 1L),
      as.integer(count),
      as.integer(k),
      as.character(metric)
    )
    rows <- seq.int(start, length.out = count)
    index[rows, ] <- matrix(block$index, nrow = count, ncol = k)
    distance[rows, ] <- matrix(block$distance, nrow = count, ncol = k)
  }
  list(index = index, distance = distance)
}

.native_matmul <- function(x, y) {
  .Call(C_cudaverse_cuda_matmul, x, y)
}

.native_synchronize <- function() {
  invisible(.Call(C_cudaverse_cuda_synchronize))
}

.native_release <- function(storage) {
  invisible(.Call(C_cudaverse_cuda_release, storage))
}

.native_error_translate <- function(error, operation) {
  structure(
    list(
      message = sprintf(
        "Native CUDA backend failed during `%s`: %s",
        operation,
        conditionMessage(error)
      ),
      call = NULL,
      backend = "native",
      operation = operation,
      parent = error
    ),
    class = c(
      "cudaverse_native_error",
      "cudaverse_backend_operation_error",
      "cudaverse_backend_error",
      "error",
      "condition"
    )
  )
}

#' Construct the native CUDA backend adapter
#'
#' This developer-facing factory is discovered lazily by `cudaverse`. Users
#' continue to call `cudaverse::cuda_tensor()` and related functions rather
#' than calling the extension directly.
#'
#' @return A backend contract list consumed by `cudaverse`.
#' @export
cudaverse_cuda_backend_factory <- function() {
  list(
    name = "native",
    device = "cuda",
    diagnostics = .native_diagnostics,
    capabilities = function() c(
      "driver-detection",
      "allocation",
      "transfer",
      "cast",
      "matmul",
      "reduce",
      "svd",
      "pca",
      "distance",
      "knn",
      "stable-topk",
      "synchronize",
      "shared-ownership"
    ),
    from_host = .native_from_host,
    to_host = .native_to_host,
    cast = .native_cast,
    matmul = .native_matmul,
    reduce = .native_reduce,
    algorithm_svd = .native_algorithm_svd,
    algorithm_pca = .native_algorithm_pca,
    algorithm_distance = .native_algorithm_distance,
    algorithm_knn_prepare = .native_knn_prepare,
    algorithm_knn_block = .native_knn_block_compat,
    algorithm_knn_select = .native_knn_select,
    synchronize = .native_synchronize,
    release = .native_release,
    test_inject_cuda_error = .native_test_inject_cuda_error,
    error_translate = .native_error_translate
  )
}

.native_memory_info <- function() {
  .Call(C_cudaverse_cuda_memory_info)
}

.native_memory_tracker <- function(reset = FALSE) {
  .Call(C_cudaverse_cuda_memory_tracker, reset)
}

.native_test_inject_cuda_error <- function(bytes = 4096L) {
  .Call(C_cudaverse_cuda_test_inject_error, as.integer(bytes))
}

.native_share <- function(storage) {
  .Call(C_cudaverse_cuda_share, storage)
}
