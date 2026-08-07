.native_diagnostics <- function() {
  .Call(C_cudaverse_cuda_diagnostics)
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
      "matmul",
      "synchronize",
      "shared-ownership"
    ),
    from_host = .native_from_host,
    to_host = .native_to_host,
    matmul = .native_matmul,
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
