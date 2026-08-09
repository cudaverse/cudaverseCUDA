arguments <- commandArgs(trailingOnly = TRUE)
root <- if (length(arguments)) normalizePath(arguments[[1L]]) else getwd()
package_files <- list.files(root, recursive = TRUE, full.names = TRUE,
                            all.files = TRUE, no.. = TRUE)
binary_pattern <- paste0(
  "(nvcuda|libcuda|cublas|cudart|cusparse|cusolver|cudnn|nvrtc|nvjitlink|",
  "torch|libtorch).*(\\.dll|\\.so(\\.[0-9]+)*|\\.dylib|\\.cubin|\\.fatbin)$"
)
bundled <- package_files[grepl(
  binary_pattern,
  basename(package_files),
  ignore.case = TRUE,
  perl = TRUE
)]
if (length(bundled)) {
  message("Unapproved redistributable candidate(s):")
  message(paste0("- ", bundled, collapse = "\n"))
  quit(status = 1L)
}
device_code <- package_files[grepl(
  "\\.(ptx|cubin|fatbin)$", package_files, ignore.case = TRUE, perl = TRUE
)]
allowed_ptx <- normalizePath(
  file.path(root, "inst", "kernels", "cudaverse_dense_kernels.ptx"),
  mustWork = FALSE
)
unexpected_device_code <- setdiff(
  normalizePath(device_code, mustWork = FALSE), allowed_ptx
)
if (length(unexpected_device_code)) {
  message("Unexpected CUDA device-code payload(s):")
  message(paste0("- ", unexpected_device_code, collapse = "\n"))
  quit(status = 1L)
}
message(paste(
  "Redistribution gate passed: no NVIDIA or LibTorch runtime binary is",
  "bundled; only the package-owned, checksum-pinned dense-kernel PTX is present."
))
