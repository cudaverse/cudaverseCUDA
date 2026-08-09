if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The `jsonlite` package is required to merge phase 2 reports.")
}

files_setting <- Sys.getenv("CUDAVERSE_PHASE2_FRAGMENTS", unset = "")
files <- if (nzchar(files_setting)) {
  strsplit(files_setting, ";", fixed = TRUE)[[1L]]
} else {
  list.files(
    file.path("inst", "reports", "phase2-fragments"),
    pattern = "[.]json$", full.names = TRUE
  )
}
if (!length(files) || any(!file.exists(files))) {
  stop("No complete phase 2 fragment set was found.")
}

fragments <- lapply(files, jsonlite::read_json, simplifyVector = FALSE)
first <- fragments[[1L]]
report <- first
report$schema <- "cudaverse-native-phase2/1"
report$source_fragments <- basename(files)
report$benchmarks <- list()

for (fragment in fragments) {
  for (shape in names(fragment$benchmarks)) {
    if (is.null(report$benchmarks[[shape]])) report$benchmarks[[shape]] <- list()
    for (backend in names(fragment$benchmarks[[shape]])) {
      if (!is.null(report$benchmarks[[shape]][[backend]])) {
        stop("Duplicate phase 2 result for ", shape, " / ", backend, ".")
      }
      report$benchmarks[[shape]][[backend]] <-
        fragment$benchmarks[[shape]][[backend]]
    }
  }
}

required_shapes <- c("1000x50", "10000x100", "50000x128")
required_backends <- c("base", "native", "torch")
missing <- character()
for (shape in required_shapes) {
  for (backend in required_backends) {
    if (is.null(report$benchmarks[[shape]][[backend]])) {
      missing <- c(missing, paste(shape, backend, sep = "/"))
    }
  }
}
if (length(missing)) {
  stop("Missing required phase 2 results: ", paste(missing, collapse = ", "))
}

report$generated_at_utc <- format(Sys.time(), tz = "UTC", usetz = TRUE)
output <- Sys.getenv(
  "CUDAVERSE_REPORT_FILE",
  unset = file.path("inst", "reports", "phase2-rtx2000.json")
)
jsonlite::write_json(
  report, output, auto_unbox = TRUE, pretty = TRUE, digits = 16,
  null = "null", na = "null"
)
message("Wrote ", normalizePath(output))
