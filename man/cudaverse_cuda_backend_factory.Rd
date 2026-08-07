\name{cudaverse_cuda_backend_factory}
\alias{cudaverse_cuda_backend_factory}
\title{Construct the Native CUDA Backend Adapter}
\usage{
cudaverse_cuda_backend_factory()
}
\value{
A backend contract list consumed internally by \pkg{cudaverse}.
}
\description{
Constructs the developer-facing backend factory that \pkg{cudaverse}
discovers lazily. Users continue to call the public \pkg{cudaverse} API.
}
