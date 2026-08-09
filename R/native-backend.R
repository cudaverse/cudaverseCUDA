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
      "synchronize",
      "shared-ownership"
    ),
    from_host = .native_from_host,
    to_host = .native_to_host,
    cast = .native_cast,
    matmul = .native_matmul,
    reduce = .native_reduce,
    synchronize = .native_synchronize,
    release = .native_release,
    error_translate = .native_error_translate
  )
}

.native_memory_info <- function() {
  .Call(C_cudaverse_cuda_memory_info)
}

.native_share <- function(storage) {
  .Call(C_cudaverse_cuda_share, storage)
}
