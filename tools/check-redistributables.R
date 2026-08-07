arguments <- commandArgs(trailingOnly = TRUE)
root <- if (length(arguments)) normalizePath(arguments[[1L]]) else getwd()
package_files <- list.files(root, recursive = TRUE, full.names = TRUE,
                            all.files = TRUE, no.. = TRUE)
binary_pattern <- paste0(
  "(cublas|cudart|cusparse|cusolver|cudnn|nvrtc|nvjitlink|torch|libtorch)",
  ".*(\\.dll|\\.so(\\.[0-9]+)*|\\.dylib)$"
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
message("Redistribution gate passed: no NVIDIA or LibTorch binaries bundled.")
