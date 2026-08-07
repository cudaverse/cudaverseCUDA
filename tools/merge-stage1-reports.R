if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The `jsonlite` package is required.")
}
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 2L) {
  stop("Pass one or more input reports followed by the output report.")
}
inputs <- arguments[-length(arguments)]
output <- arguments[[length(arguments)]]
reports <- lapply(inputs, jsonlite::read_json, simplifyVector = FALSE)
merged <- reports[[1L]]
merged$benchmarks <- list()
merged$lifecycle <- NULL
for (report in reports) {
  if (!is.null(report$lifecycle)) merged$lifecycle <- report$lifecycle
  for (size in names(report$benchmarks)) {
    if (is.null(merged$benchmarks[[size]])) merged$benchmarks[[size]] <- list()
    for (backend in names(report$benchmarks[[size]])) {
      incoming <- report$benchmarks[[size]][[backend]]
      existing <- merged$benchmarks[[size]][[backend]]
      if (is.null(existing)) {
        merged$benchmarks[[size]][[backend]] <- incoming
      } else {
        for (field in c("resident", "transfer_included")) {
          runs <- as.numeric(unlist(c(
            existing[[field]]$runs_seconds,
            incoming[[field]]$runs_seconds
          )))
          existing[[field]]$runs_seconds <- runs
          existing[[field]]$median_seconds <- unname(stats::median(runs))
          existing[[field]]$p95_seconds <- unname(
            stats::quantile(runs, 0.95, type = 8)
          )
        }
        existing$max_absolute_error <- max(as.numeric(unlist(c(
          existing$max_absolute_error,
          incoming$max_absolute_error
        ))))
        existing$max_relative_error <- max(as.numeric(unlist(c(
          existing$max_relative_error,
          incoming$max_relative_error
        ))))
        merged$benchmarks[[size]][[backend]] <- existing
      }
    }
  }
}
jsonlite::write_json(merged, output, auto_unbox = TRUE, pretty = TRUE,
                     digits = 16, null = "null")
message("Wrote merged report ", normalizePath(output))
