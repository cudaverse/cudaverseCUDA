#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Include the C++ standard library before R headers. Rinternals.h defines a
// legacy length() macro that otherwise collides with libc++ locale methods on
// macOS.
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Utils.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUdeviceptr = std::uint64_t;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using cublasHandle_t = void*;
using cublasStatus_t = int;
using cusolverDnHandle_t = void*;
using cusolverStatus_t = int;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
constexpr cusolverStatus_t CUSOLVER_STATUS_SUCCESS = 0;
constexpr int CUBLAS_OP_N = 0;
constexpr int CUBLAS_OP_T = 1;

using cuInit_t = CUresult (*)(unsigned int);
using cuDriverGetVersion_t = CUresult (*)(int*);
using cuDeviceGetCount_t = CUresult (*)(int*);
using cuDeviceGet_t = CUresult (*)(CUdevice*, int);
using cuDevicePrimaryCtxRetain_t = CUresult (*)(CUcontext*, CUdevice);
using cuCtxSetCurrent_t = CUresult (*)(CUcontext);
using cuMemAlloc_t = CUresult (*)(CUdeviceptr*, std::size_t);
using cuMemFree_t = CUresult (*)(CUdeviceptr);
using cuMemcpyHtoD_t = CUresult (*)(CUdeviceptr, const void*, std::size_t);
using cuMemcpyDtoH_t = CUresult (*)(void*, CUdeviceptr, std::size_t);
using cuMemcpyDtoD_t = CUresult (*)(CUdeviceptr, CUdeviceptr, std::size_t);
using cuCtxSynchronize_t = CUresult (*)();
using cuMemGetInfo_t = CUresult (*)(std::size_t*, std::size_t*);
using cuGetErrorName_t = CUresult (*)(CUresult, const char**);
using cuGetErrorString_t = CUresult (*)(CUresult, const char**);
using cuModuleLoad_t = CUresult (*)(CUmodule*, const char*);
using cuModuleUnload_t = CUresult (*)(CUmodule);
using cuModuleGetFunction_t = CUresult (*)(CUfunction*, CUmodule, const char*);
using cuLaunchKernel_t = CUresult (*)(
    CUfunction, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, unsigned int, CUstream,
    void**, void**);

using cublasCreate_t = cublasStatus_t (*)(cublasHandle_t*);
using cublasDgemm_t = cublasStatus_t (*)(
    cublasHandle_t, int, int, int, int, int, const double*,
    const double*, int, const double*, int, const double*, double*, int);
using cusolverDnCreate_t = cusolverStatus_t (*)(cusolverDnHandle_t*);
using cusolverDnDgesvd_bufferSize_t = cusolverStatus_t (*)(
    cusolverDnHandle_t, int, int, int*);
using cusolverDnDgesvd_t = cusolverStatus_t (*)(
    cusolverDnHandle_t, signed char, signed char, int, int, double*, int,
    double*, double*, int, double*, int, double*, int, double*, int*);

struct BackendApi {
  void* driver = nullptr;
  void* cublas = nullptr;
  void* cusolver = nullptr;
  CUcontext context = nullptr;
  CUmodule kernels = nullptr;
  cublasHandle_t cublas_handle = nullptr;
  cusolverDnHandle_t cusolver_handle = nullptr;
  int driver_version = 0;
  int device_count = 0;

  cuInit_t cuInit = nullptr;
  cuDriverGetVersion_t cuDriverGetVersion = nullptr;
  cuDeviceGetCount_t cuDeviceGetCount = nullptr;
  cuDeviceGet_t cuDeviceGet = nullptr;
  cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain = nullptr;
  cuCtxSetCurrent_t cuCtxSetCurrent = nullptr;
  cuMemAlloc_t cuMemAlloc = nullptr;
  cuMemFree_t cuMemFree = nullptr;
  cuMemcpyHtoD_t cuMemcpyHtoD = nullptr;
  cuMemcpyDtoH_t cuMemcpyDtoH = nullptr;
  cuMemcpyDtoD_t cuMemcpyDtoD = nullptr;
  cuCtxSynchronize_t cuCtxSynchronize = nullptr;
  cuMemGetInfo_t cuMemGetInfo = nullptr;
  cuGetErrorName_t cuGetErrorName = nullptr;
  cuGetErrorString_t cuGetErrorString = nullptr;
  cuModuleLoad_t cuModuleLoad = nullptr;
  cuModuleUnload_t cuModuleUnload = nullptr;
  cuModuleGetFunction_t cuModuleGetFunction = nullptr;
  cuLaunchKernel_t cuLaunchKernel = nullptr;
  cublasCreate_t cublasCreate = nullptr;
  cublasDgemm_t cublasDgemm = nullptr;
  cusolverDnCreate_t cusolverDnCreate = nullptr;
  cusolverDnDgesvd_bufferSize_t cusolverDnDgesvd_bufferSize = nullptr;
  cusolverDnDgesvd_t cusolverDnDgesvd = nullptr;
};

BackendApi api;

enum class DType { Float64, Float32, Integer };

constexpr int CUDAVERSE_MAX_RANK = 8;

struct ReductionMeta {
  int rank;
  int output_rank;
  int keepdim;
  int operation;
  int shape[CUDAVERSE_MAX_RANK];
  int reduced[CUDAVERSE_MAX_RANK];
  int output_shape[CUDAVERSE_MAX_RANK];
  int output_to_input[CUDAVERSE_MAX_RANK];
};

struct SharedBuffer {
  CUdeviceptr pointer;
  std::size_t bytes;
  std::size_t elements;
  DType dtype;
  std::vector<int> shape;
  std::atomic<unsigned int> references;
};

void* open_library(const char* environment_name,
                   const std::vector<const char*>& candidates) {
  const char* explicit_path = std::getenv(environment_name);
#ifdef _WIN32
  if (explicit_path != nullptr && std::strlen(explicit_path) > 0) {
    HMODULE handle = LoadLibraryExA(
        explicit_path,
        nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (handle == nullptr) handle = LoadLibraryA(explicit_path);
    if (handle != nullptr) return reinterpret_cast<void*>(handle);
  }
  for (const char* candidate : candidates) {
    HMODULE handle = LoadLibraryA(candidate);
    if (handle != nullptr) return reinterpret_cast<void*>(handle);
  }
#else
  if (explicit_path != nullptr && std::strlen(explicit_path) > 0) {
    void* handle = dlopen(explicit_path, RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) return handle;
  }
  for (const char* candidate : candidates) {
    void* handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) return handle;
  }
#endif
  return nullptr;
}

void* lookup(void* library, const char* name) {
  if (library == nullptr) return nullptr;
#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(
      reinterpret_cast<HMODULE>(library), name));
#else
  return dlsym(library, name);
#endif
}

template <typename T>
bool bind_symbol(T& target, void* library, const char* name) {
  target = reinterpret_cast<T>(lookup(library, name));
  return target != nullptr;
}

std::string cuda_error(CUresult status) {
  const char* name = nullptr;
  const char* description = nullptr;
  if (api.cuGetErrorName != nullptr) api.cuGetErrorName(status, &name);
  if (api.cuGetErrorString != nullptr) api.cuGetErrorString(status, &description);
  std::string result = "CUDA error " + std::to_string(status);
  if (name != nullptr) result += " (" + std::string(name) + ")";
  if (description != nullptr) result += ": " + std::string(description);
  return result;
}

std::string cublas_error(cublasStatus_t status) {
  return "cuBLAS error " + std::to_string(status);
}

std::string cusolver_error(cusolverStatus_t status) {
  return "cuSOLVER error " + std::to_string(status);
}

bool load_driver(std::string& reason) {
#ifdef __APPLE__
  reason = "Native CUDA is not supported on macOS.";
  return false;
#else
  if (api.driver == nullptr) {
#ifdef _WIN32
    api.driver = open_library("CUDAVERSE_CUDA_DRIVER_PATH", {"nvcuda.dll"});
#else
    api.driver = open_library("CUDAVERSE_CUDA_DRIVER_PATH",
                              {"libcuda.so.1", "libcuda.so"});
#endif
  }
  if (api.driver == nullptr) {
    reason = "The NVIDIA CUDA Driver library could not be loaded.";
    return false;
  }
  bool bound =
      bind_symbol(api.cuInit, api.driver, "cuInit") &&
      bind_symbol(api.cuDriverGetVersion, api.driver, "cuDriverGetVersion") &&
      bind_symbol(api.cuDeviceGetCount, api.driver, "cuDeviceGetCount") &&
      bind_symbol(api.cuDeviceGet, api.driver, "cuDeviceGet") &&
      bind_symbol(api.cuDevicePrimaryCtxRetain, api.driver,
                  "cuDevicePrimaryCtxRetain") &&
      bind_symbol(api.cuCtxSetCurrent, api.driver, "cuCtxSetCurrent") &&
      bind_symbol(api.cuMemAlloc, api.driver, "cuMemAlloc_v2") &&
      bind_symbol(api.cuMemFree, api.driver, "cuMemFree_v2") &&
      bind_symbol(api.cuMemcpyHtoD, api.driver, "cuMemcpyHtoD_v2") &&
      bind_symbol(api.cuMemcpyDtoH, api.driver, "cuMemcpyDtoH_v2") &&
      bind_symbol(api.cuMemcpyDtoD, api.driver, "cuMemcpyDtoD_v2") &&
      bind_symbol(api.cuCtxSynchronize, api.driver, "cuCtxSynchronize") &&
      bind_symbol(api.cuMemGetInfo, api.driver, "cuMemGetInfo_v2") &&
      bind_symbol(api.cuGetErrorName, api.driver, "cuGetErrorName") &&
      bind_symbol(api.cuGetErrorString, api.driver, "cuGetErrorString") &&
      bind_symbol(api.cuModuleLoad, api.driver, "cuModuleLoad") &&
      bind_symbol(api.cuModuleUnload, api.driver, "cuModuleUnload") &&
      bind_symbol(api.cuModuleGetFunction, api.driver,
                  "cuModuleGetFunction") &&
      bind_symbol(api.cuLaunchKernel, api.driver, "cuLaunchKernel");
  if (!bound) {
    reason = "The CUDA Driver library is missing required symbols.";
    return false;
  }
  CUresult status = api.cuInit(0);
  if (status != CUDA_SUCCESS) {
    reason = cuda_error(status);
    return false;
  }
  status = api.cuDriverGetVersion(&api.driver_version);
  if (status != CUDA_SUCCESS) {
    reason = cuda_error(status);
    return false;
  }
  status = api.cuDeviceGetCount(&api.device_count);
  if (status != CUDA_SUCCESS) {
    reason = cuda_error(status);
    return false;
  }
  if (api.device_count < 1) {
    reason = "The CUDA driver reported no devices.";
    return false;
  }
  if (api.context == nullptr) {
    CUdevice device = 0;
    status = api.cuDeviceGet(&device, 0);
    if (status == CUDA_SUCCESS) {
      status = api.cuDevicePrimaryCtxRetain(&api.context, device);
    }
    if (status != CUDA_SUCCESS) {
      reason = cuda_error(status);
      return false;
    }
  }
  status = api.cuCtxSetCurrent(api.context);
  if (status != CUDA_SUCCESS) {
    reason = cuda_error(status);
    return false;
  }
  return true;
#endif
}

bool load_cublas(std::string& reason) {
  if (api.cublas == nullptr) {
#ifdef _WIN32
    api.cublas = open_library("CUDAVERSE_CUBLAS_PATH", {"cublas64_12.dll"});
#else
    api.cublas = open_library("CUDAVERSE_CUBLAS_PATH",
                              {"libcublas.so.12", "libcublas.so"});
#endif
  }
  if (api.cublas == nullptr) {
    reason = "The NVIDIA cuBLAS 12 library could not be loaded.";
    return false;
  }
  bool bound = bind_symbol(api.cublasCreate, api.cublas, "cublasCreate_v2") &&
               bind_symbol(api.cublasDgemm, api.cublas, "cublasDgemm_v2");
  if (!bound) {
    reason = "The cuBLAS library is missing required symbols.";
    return false;
  }
  if (api.cublas_handle == nullptr) {
    cublasStatus_t status = api.cublasCreate(&api.cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      reason = cublas_error(status);
      return false;
    }
  }
  return true;
}

bool ensure_backend(std::string& reason) {
  return load_driver(reason) && load_cublas(reason);
}

void require_backend() {
  std::string reason;
  if (!ensure_backend(reason)) Rf_error("%s", reason.c_str());
}

void check_cuda(CUresult status, const char* operation) {
  if (status != CUDA_SUCCESS) {
    std::string message = std::string(operation) + ": " + cuda_error(status);
    Rf_error("%s", message.c_str());
  }
}

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::string message = std::string(operation) + ": " + cublas_error(status);
    Rf_error("%s", message.c_str());
  }
}

bool load_cusolver(std::string& reason) {
  if (!load_driver(reason)) return false;
  if (api.cusolver == nullptr) {
#ifdef _WIN32
    api.cusolver = open_library(
        "CUDAVERSE_CUSOLVER_PATH", {"cusolver64_11.dll"});
#elif defined(__APPLE__)
    reason = "Native CUDA is not supported on macOS.";
    return false;
#else
    api.cusolver = open_library(
        "CUDAVERSE_CUSOLVER_PATH",
        {"libcusolver.so.11", "libcusolver.so"});
#endif
  }
  if (api.cusolver == nullptr) {
    reason = "The NVIDIA cuSOLVER 11 library could not be loaded.";
    return false;
  }
  bool bound =
      bind_symbol(api.cusolverDnCreate, api.cusolver,
                  "cusolverDnCreate") &&
      bind_symbol(api.cusolverDnDgesvd_bufferSize, api.cusolver,
                  "cusolverDnDgesvd_bufferSize") &&
      bind_symbol(api.cusolverDnDgesvd, api.cusolver,
                  "cusolverDnDgesvd");
  if (!bound) {
    reason = "The cuSOLVER library is missing required SVD symbols.";
    return false;
  }
  if (api.cusolver_handle == nullptr) {
    cusolverStatus_t status = api.cusolverDnCreate(&api.cusolver_handle);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      reason = cusolver_error(status);
      return false;
    }
  }
  return true;
}

void require_kernels() {
  if (api.kernels == nullptr) {
    Rf_error("The cudaverseCUDA PTX module has not been loaded.");
  }
}

CUfunction get_kernel(const char* name) {
  require_kernels();
  CUfunction function = nullptr;
  check_cuda(api.cuModuleGetFunction(&function, api.kernels, name),
             "cuModuleGetFunction");
  return function;
}

CUresult launch_1d(CUfunction function, std::size_t elements,
                   void** parameters) {
  if (elements == 0) return CUDA_SUCCESS;
  constexpr unsigned int threads = 256;
  unsigned long long blocks =
      (static_cast<unsigned long long>(elements) + threads - 1) / threads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    Rf_error("Native CUDA kernel launch is too large.");
  }
  return api.cuLaunchKernel(
      function, static_cast<unsigned int>(blocks), 1, 1,
      threads, 1, 1, 0, nullptr, parameters, nullptr);
}

SEXP buffer_tag() {
  static SEXP tag = R_NilValue;
  if (tag == R_NilValue) tag = Rf_install("cudaverseCUDA_buffer");
  return tag;
}

SharedBuffer* get_buffer(SEXP pointer) {
  if (TYPEOF(pointer) != EXTPTRSXP ||
      R_ExternalPtrTag(pointer) != buffer_tag()) {
    Rf_error("Expected a live cudaverseCUDA external pointer.");
  }
  auto* buffer = static_cast<SharedBuffer*>(R_ExternalPtrAddr(pointer));
  if (buffer == nullptr) Rf_error("The native CUDA allocation was released.");
  return buffer;
}

void release_reference(SEXP pointer) {
  auto* buffer = static_cast<SharedBuffer*>(R_ExternalPtrAddr(pointer));
  if (buffer == nullptr) return;
  R_ClearExternalPtr(pointer);
  if (buffer->references.fetch_sub(1) == 1) {
    if (buffer->pointer != 0 && api.cuMemFree != nullptr) {
      if (api.context != nullptr && api.cuCtxSetCurrent != nullptr) {
        api.cuCtxSetCurrent(api.context);
      }
      api.cuMemFree(buffer->pointer);
    }
    delete buffer;
  }
}

void buffer_finalizer(SEXP pointer) {
  release_reference(pointer);
}

SEXP make_pointer(SharedBuffer* buffer, bool add_reference) {
  if (add_reference) buffer->references.fetch_add(1);
  SEXP pointer = PROTECT(R_MakeExternalPtr(buffer, buffer_tag(), R_NilValue));
  R_RegisterCFinalizerEx(pointer, buffer_finalizer, static_cast<Rboolean>(1));
  UNPROTECT(1);
  return pointer;
}

DType parse_dtype(SEXP dtype) {
  if (TYPEOF(dtype) != STRSXP || XLENGTH(dtype) != 1) {
    Rf_error("`dtype` must be one character string.");
  }
  std::string value = CHAR(STRING_ELT(dtype, 0));
  if (value == "float64") return DType::Float64;
  if (value == "float32") return DType::Float32;
  if (value == "integer") return DType::Integer;
  Rf_error("Unsupported native CUDA dtype `%s`.", value.c_str());
  return DType::Float64;
}

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::Float64: return sizeof(double);
    case DType::Float32: return sizeof(float);
    case DType::Integer: return sizeof(int);
  }
  return 0;
}

std::vector<int> parse_shape(SEXP shape, std::size_t& elements) {
  if (TYPEOF(shape) != INTSXP || XLENGTH(shape) < 1) {
    Rf_error("`shape` must be a non-empty integer vector.");
  }
  std::vector<int> result(static_cast<std::size_t>(XLENGTH(shape)));
  elements = 1;
  for (R_xlen_t index = 0; index < XLENGTH(shape); ++index) {
    int extent = INTEGER(shape)[index];
    if (extent < 1) Rf_error("Native CUDA shapes must be positive.");
    if (elements > std::numeric_limits<std::size_t>::max() /
                       static_cast<std::size_t>(extent)) {
      Rf_error("Native CUDA shape is too large.");
    }
    elements *= static_cast<std::size_t>(extent);
    result[static_cast<std::size_t>(index)] = extent;
  }
  return result;
}

SharedBuffer* allocate_buffer(DType dtype, const std::vector<int>& shape,
                              std::size_t elements) {
  auto* buffer = new SharedBuffer{
      0, elements * dtype_size(dtype), elements, dtype, shape, 1};
  CUresult status = api.cuMemAlloc(&buffer->pointer, buffer->bytes);
  if (status != CUDA_SUCCESS) {
    delete buffer;
    check_cuda(status, "cuMemAlloc");
  }
  return buffer;
}

class DeviceMemory {
 public:
  DeviceMemory() = default;
  explicit DeviceMemory(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ == 0) return;
    CUresult status = api.cuMemAlloc(&pointer_, bytes_);
    if (status != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemAlloc: " + cuda_error(status));
    }
  }
  ~DeviceMemory() {
    if (pointer_ != 0 && api.cuMemFree != nullptr) api.cuMemFree(pointer_);
  }
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;
  DeviceMemory(DeviceMemory&& other) noexcept
      : pointer_(other.pointer_), bytes_(other.bytes_) {
    other.pointer_ = 0;
    other.bytes_ = 0;
  }
  DeviceMemory& operator=(DeviceMemory&& other) noexcept {
    if (this != &other) {
      if (pointer_ != 0 && api.cuMemFree != nullptr) api.cuMemFree(pointer_);
      pointer_ = other.pointer_;
      bytes_ = other.bytes_;
      other.pointer_ = 0;
      other.bytes_ = 0;
    }
    return *this;
  }
  CUdeviceptr pointer() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }
  CUdeviceptr release() {
    CUdeviceptr result = pointer_;
    pointer_ = 0;
    bytes_ = 0;
    return result;
  }

 private:
  CUdeviceptr pointer_ = 0;
  std::size_t bytes_ = 0;
};

void cuda_or_throw(CUresult status, const char* operation) {
  if (status != CUDA_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + ": " + cuda_error(status));
  }
}

void cusolver_or_throw(cusolverStatus_t status, const char* operation) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + ": " + cusolver_error(status));
  }
}

void launch_or_throw(CUfunction function, const char* name,
                     std::size_t elements, void** parameters) {
  cuda_or_throw(launch_1d(function, elements, parameters), name);
}

struct DeviceSvd {
  int rank = 0;
  DeviceMemory values;
  DeviceMemory left;
  DeviceMemory right;
};

DeviceSvd run_device_svd(CUdeviceptr input, int rows, int columns) {
  std::string reason;
  if (!load_cusolver(reason)) throw std::runtime_error(reason);
  require_kernels();
  CUfunction transpose = get_kernel("cudaverse_transpose_f64");

  bool transposed = rows < columns;
  int solver_rows = transposed ? columns : rows;
  int solver_columns = transposed ? rows : columns;
  int rank = solver_columns;
  std::size_t matrix_elements =
      static_cast<std::size_t>(rows) * columns;
  DeviceMemory matrix(matrix_elements * sizeof(double));
  if (transposed) {
    CUdeviceptr output = matrix.pointer();
    void* parameters[] = {&input, &output, &rows, &columns};
    launch_or_throw(transpose, "cudaverse_transpose_f64",
                    matrix_elements, parameters);
  } else {
    cuda_or_throw(api.cuMemcpyDtoD(
                      matrix.pointer(), input,
                      matrix_elements * sizeof(double)),
                  "cuMemcpyDtoD");
  }

  int workspace_elements = 0;
  cusolver_or_throw(
      api.cusolverDnDgesvd_bufferSize(
          api.cusolver_handle, solver_rows, solver_columns,
          &workspace_elements),
      "cusolverDnDgesvd_bufferSize");
  if (workspace_elements < 1) {
    throw std::runtime_error("cuSOLVER returned an invalid SVD workspace.");
  }

  DeviceMemory singular_values(static_cast<std::size_t>(rank) *
                               sizeof(double));
  DeviceMemory solver_left(
      static_cast<std::size_t>(solver_rows) * rank * sizeof(double));
  DeviceMemory solver_right_transposed(
      static_cast<std::size_t>(rank) * solver_columns * sizeof(double));
  DeviceMemory workspace(
      static_cast<std::size_t>(workspace_elements) * sizeof(double));
  DeviceMemory rwork(
      static_cast<std::size_t>(rank > 1 ? rank - 1 : 1) * sizeof(double));
  DeviceMemory device_info(sizeof(int));

  cusolver_or_throw(
      api.cusolverDnDgesvd(
          api.cusolver_handle, static_cast<signed char>('S'),
          static_cast<signed char>('S'), solver_rows, solver_columns,
          reinterpret_cast<double*>(matrix.pointer()), solver_rows,
          reinterpret_cast<double*>(singular_values.pointer()),
          reinterpret_cast<double*>(solver_left.pointer()), solver_rows,
          reinterpret_cast<double*>(solver_right_transposed.pointer()), rank,
          reinterpret_cast<double*>(workspace.pointer()), workspace_elements,
          reinterpret_cast<double*>(rwork.pointer()),
          reinterpret_cast<int*>(device_info.pointer())),
      "cusolverDnDgesvd");
  int info = 0;
  cuda_or_throw(api.cuMemcpyDtoH(&info, device_info.pointer(), sizeof(int)),
                "cuMemcpyDtoH(SVD info)");
  if (info != 0) {
    throw std::runtime_error(
        "cuSOLVER SVD failed to converge (info=" +
        std::to_string(info) + ").");
  }

  DeviceMemory original_left;
  DeviceMemory original_right;
  if (!transposed) {
    original_left = std::move(solver_left);
    original_right = DeviceMemory(
        static_cast<std::size_t>(columns) * rank * sizeof(double));
    CUdeviceptr source = solver_right_transposed.pointer();
    CUdeviceptr destination = original_right.pointer();
    int transpose_rows = rank;
    int transpose_columns = columns;
    void* parameters[] = {
        &source, &destination, &transpose_rows, &transpose_columns};
    launch_or_throw(
        transpose, "cudaverse_transpose_f64",
        static_cast<std::size_t>(rank) * columns, parameters);
  } else {
    original_right = std::move(solver_left);
    original_left = DeviceMemory(
        static_cast<std::size_t>(rows) * rank * sizeof(double));
    CUdeviceptr source = solver_right_transposed.pointer();
    CUdeviceptr destination = original_left.pointer();
    int transpose_rows = rank;
    int transpose_columns = rows;
    void* parameters[] = {
        &source, &destination, &transpose_rows, &transpose_columns};
    launch_or_throw(
        transpose, "cudaverse_transpose_f64",
        static_cast<std::size_t>(rank) * rows, parameters);
  }

  DeviceSvd result;
  result.rank = rank;
  result.values = std::move(singular_values);
  result.left = std::move(original_left);
  result.right = std::move(original_right);
  return result;
}

std::vector<double> copy_double_to_host(CUdeviceptr pointer,
                                        std::size_t elements,
                                        const char* operation) {
  std::vector<double> result(elements);
  cuda_or_throw(api.cuMemcpyDtoH(
                    result.data(), pointer, elements * sizeof(double)),
                operation);
  return result;
}

SEXP numeric_vector(const std::vector<double>& values) {
  SEXP result = PROTECT(Rf_allocVector(REALSXP, values.size()));
  if (!values.empty()) {
    std::memcpy(REAL(result), values.data(), values.size() * sizeof(double));
  }
  UNPROTECT(1);
  return result;
}

DeviceMemory row_norms_squared(CUdeviceptr input, int rows, int columns) {
  CUfunction function = get_kernel("cudaverse_row_norms_squared_f64");
  DeviceMemory output(static_cast<std::size_t>(rows) * sizeof(double));
  CUdeviceptr output_pointer = output.pointer();
  void* parameters[] = {&input, &output_pointer, &rows, &columns};
  launch_or_throw(function, "cudaverse_row_norms_squared_f64", rows,
                  parameters);
  return output;
}

DeviceMemory distance_device(CUdeviceptr query, int query_rows,
                             CUdeviceptr reference, int reference_rows,
                             int columns, int metric, int query_offset,
                             int self, CUdeviceptr cached_reference_norms = 0) {
  CUfunction finalize = get_kernel("cudaverse_distance_from_gram_f64");
  DeviceMemory query_norms;
  DeviceMemory reference_norms;
  CUdeviceptr query_norm_pointer = 0;
  CUdeviceptr reference_norm_pointer = cached_reference_norms;
  if (metric == 0) {
    query_norms = row_norms_squared(query, query_rows, columns);
    query_norm_pointer = query_norms.pointer();
    if (reference_norm_pointer == 0) {
      reference_norms = row_norms_squared(
          reference, reference_rows, columns);
      reference_norm_pointer = reference_norms.pointer();
    }
  }

  std::size_t output_elements =
      static_cast<std::size_t>(query_rows) * reference_rows;
  DeviceMemory gram(output_elements * sizeof(double));
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasStatus_t blas_status = api.cublasDgemm(
      api.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
      query_rows, reference_rows, columns, &alpha,
      reinterpret_cast<const double*>(query), query_rows,
      reinterpret_cast<const double*>(reference), reference_rows, &beta,
      reinterpret_cast<double*>(gram.pointer()), query_rows);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublasDgemm(distance): " +
                             cublas_error(blas_status));
  }

  DeviceMemory output(output_elements * sizeof(double));
  CUdeviceptr gram_pointer = gram.pointer();
  CUdeviceptr output_pointer = output.pointer();
  void* parameters[] = {
      &gram_pointer, &query, &reference, &query_norm_pointer,
      &reference_norm_pointer, &output_pointer, &query_rows,
      &reference_rows, &columns, &metric, &query_offset, &self};
  launch_or_throw(finalize, "cudaverse_distance_from_gram_f64",
                  output_elements, parameters);
  return output;
}

SEXP scalar_string_or_na(const std::string& value) {
  return value.empty() ? Rf_ScalarString(NA_STRING) : Rf_mkString(value.c_str());
}

SEXP named_list(const std::vector<const char*>& names) {
  SEXP result = PROTECT(Rf_allocVector(VECSXP, names.size()));
  SEXP result_names = PROTECT(Rf_allocVector(STRSXP, names.size()));
  for (std::size_t index = 0; index < names.size(); ++index) {
    SET_STRING_ELT(result_names, index, Rf_mkChar(names[index]));
  }
  Rf_setAttrib(result, R_NamesSymbol, result_names);
  UNPROTECT(2);
  return result;
}

}  // namespace

extern "C" SEXP C_cudaverse_cuda_load_kernels(SEXP path_sexp) {
  if (TYPEOF(path_sexp) != STRSXP || XLENGTH(path_sexp) != 1 ||
      STRING_ELT(path_sexp, 0) == NA_STRING) {
    Rf_error("`path` must be one PTX file path.");
  }
  std::string reason;
  if (!load_driver(reason)) Rf_error("%s", reason.c_str());
  if (api.kernels != nullptr) return Rf_ScalarLogical(1);
  const char* path = Rf_translateCharUTF8(STRING_ELT(path_sexp, 0));
  check_cuda(api.cuModuleLoad(&api.kernels, path), "cuModuleLoad");
  return Rf_ScalarLogical(1);
}

extern "C" SEXP C_cudaverse_cuda_diagnostics() {
  std::string reason;
  bool available = ensure_backend(reason);
  SEXP result = PROTECT(named_list({
      "installed", "available", "device_count", "version", "reason",
      "detection_error", "driver_version", "cublas_loaded"}));
  SET_VECTOR_ELT(result, 0, Rf_ScalarLogical(1));
  SET_VECTOR_ELT(result, 1, Rf_ScalarLogical(available));
  SET_VECTOR_ELT(result, 2, Rf_ScalarInteger(api.device_count));
  std::string version = api.driver_version > 0
      ? std::to_string(api.driver_version)
      : std::string();
  SET_VECTOR_ELT(result, 3, scalar_string_or_na(version));
  SET_VECTOR_ELT(result, 4, Rf_mkString(available ? "cuda_available" :
#ifdef __APPLE__
                                       "unsupported_platform"));
#else
                                       (api.driver == nullptr
                                            ? "driver_unavailable"
                                            : "backend_error")));
#endif
  SET_VECTOR_ELT(result, 5,
                 available ? R_NilValue : Rf_mkString(reason.c_str()));
  SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(api.driver_version));
  SET_VECTOR_ELT(result, 7, Rf_ScalarLogical(api.cublas_handle != nullptr));
  UNPROTECT(1);
  return result;
}

extern "C" SEXP C_cudaverse_cuda_from_host(SEXP x, SEXP dtype_sexp,
                                             SEXP shape_sexp) {
  require_backend();
  DType dtype = parse_dtype(dtype_sexp);
  std::size_t elements = 0;
  std::vector<int> shape = parse_shape(shape_sexp, elements);
  if (static_cast<std::size_t>(XLENGTH(x)) != elements) {
    Rf_error("Host data length does not match `shape`.");
  }
  SharedBuffer* buffer = allocate_buffer(dtype, shape, elements);
  CUresult status = CUDA_SUCCESS;
  if (dtype == DType::Float64) {
    SEXP values = PROTECT(Rf_coerceVector(x, REALSXP));
    status = api.cuMemcpyHtoD(buffer->pointer, REAL(values), buffer->bytes);
    UNPROTECT(1);
  } else if (dtype == DType::Float32) {
    SEXP values = PROTECT(Rf_coerceVector(x, REALSXP));
    std::vector<float> converted(elements);
    for (std::size_t index = 0; index < elements; ++index) {
      converted[index] = static_cast<float>(REAL(values)[index]);
    }
    status = api.cuMemcpyHtoD(buffer->pointer, converted.data(), buffer->bytes);
    UNPROTECT(1);
  } else {
    SEXP values = PROTECT(Rf_coerceVector(x, INTSXP));
    status = api.cuMemcpyHtoD(buffer->pointer, INTEGER(values), buffer->bytes);
    UNPROTECT(1);
  }
  if (status != CUDA_SUCCESS) {
    api.cuMemFree(buffer->pointer);
    delete buffer;
    check_cuda(status, "cuMemcpyHtoD");
  }
  return make_pointer(buffer, false);
}

extern "C" SEXP C_cudaverse_cuda_to_host(SEXP pointer) {
  require_backend();
  SharedBuffer* buffer = get_buffer(pointer);
  if (buffer->dtype == DType::Float64) {
    SEXP result = PROTECT(Rf_allocVector(REALSXP, buffer->elements));
    check_cuda(api.cuMemcpyDtoH(REAL(result), buffer->pointer, buffer->bytes),
               "cuMemcpyDtoH");
    UNPROTECT(1);
    return result;
  }
  if (buffer->dtype == DType::Float32) {
    std::vector<float> values(buffer->elements);
    check_cuda(api.cuMemcpyDtoH(values.data(), buffer->pointer, buffer->bytes),
               "cuMemcpyDtoH");
    SEXP result = PROTECT(Rf_allocVector(REALSXP, buffer->elements));
    for (std::size_t index = 0; index < buffer->elements; ++index) {
      REAL(result)[index] = static_cast<double>(values[index]);
    }
    UNPROTECT(1);
    return result;
  }
  SEXP result = PROTECT(Rf_allocVector(INTSXP, buffer->elements));
  check_cuda(api.cuMemcpyDtoH(INTEGER(result), buffer->pointer, buffer->bytes),
             "cuMemcpyDtoH");
  UNPROTECT(1);
  return result;
}

extern "C" SEXP C_cudaverse_cuda_cast(SEXP pointer, SEXP dtype_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  DType target = parse_dtype(dtype_sexp);
  if (input->dtype == target) return make_pointer(input, true);

  const char* kernel = nullptr;
  if (input->dtype == DType::Integer && target == DType::Float64) {
    kernel = "cudaverse_cast_i32_f64";
  } else if (input->dtype == DType::Float32 && target == DType::Float64) {
    kernel = "cudaverse_cast_f32_f64";
  } else if (input->dtype == DType::Float64 && target == DType::Float32) {
    kernel = "cudaverse_cast_f64_f32";
  } else if (input->dtype == DType::Integer && target == DType::Float32) {
    kernel = "cudaverse_cast_i32_f32";
  } else if (input->dtype == DType::Float64 && target == DType::Integer) {
    kernel = "cudaverse_cast_f64_i32";
  } else if (input->dtype == DType::Float32 && target == DType::Integer) {
    kernel = "cudaverse_cast_f32_i32";
  }
  if (kernel == nullptr) Rf_error("Unsupported native CUDA cast.");

  R_CheckUserInterrupt();
  CUfunction function = get_kernel(kernel);
  SharedBuffer* output = allocate_buffer(
      target, input->shape, input->elements);
  CUdeviceptr input_pointer = input->pointer;
  CUdeviceptr output_pointer = output->pointer;
  unsigned long long elements = input->elements;
  void* parameters[] = {&input_pointer, &output_pointer, &elements};
  CUresult status = launch_1d(function, input->elements, parameters);
  if (status != CUDA_SUCCESS) {
    api.cuMemFree(output->pointer);
    delete output;
    check_cuda(status, kernel);
  }
  return make_pointer(output, false);
}

extern "C" SEXP C_cudaverse_cuda_reduce(SEXP pointer, SEXP dim_sexp,
                                           SEXP keepdim_sexp,
                                           SEXP method_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  if (input->shape.size() > CUDAVERSE_MAX_RANK) {
    Rf_error("Native CUDA reductions support at most %d dimensions.",
             CUDAVERSE_MAX_RANK);
  }
  if (TYPEOF(dim_sexp) != INTSXP) {
    Rf_error("`dim` must be an integer vector.");
  }
  if (TYPEOF(keepdim_sexp) != LGLSXP || XLENGTH(keepdim_sexp) != 1 ||
      LOGICAL(keepdim_sexp)[0] == NA_LOGICAL) {
    Rf_error("`keepdim` must be TRUE or FALSE.");
  }
  if (TYPEOF(method_sexp) != STRSXP || XLENGTH(method_sexp) != 1) {
    Rf_error("`method` must be one character string.");
  }
  std::string method = CHAR(STRING_ELT(method_sexp, 0));
  if (method != "sum" && method != "mean") {
    Rf_error("Native CUDA reduction method must be `sum` or `mean`.");
  }

  ReductionMeta meta{};
  meta.rank = static_cast<int>(input->shape.size());
  meta.keepdim = LOGICAL(keepdim_sexp)[0] != 0;
  meta.operation = method == "mean" ? 1 : 0;
  for (int dimension = 0; dimension < meta.rank; ++dimension) {
    meta.shape[dimension] = input->shape[dimension];
  }
  if (XLENGTH(dim_sexp) == 0) {
    for (int dimension = 0; dimension < meta.rank; ++dimension) {
      meta.reduced[dimension] = 1;
    }
  } else {
    for (R_xlen_t index = 0; index < XLENGTH(dim_sexp); ++index) {
      int dimension = INTEGER(dim_sexp)[index];
      if (dimension < 1 || dimension > meta.rank) {
        Rf_error("Native CUDA reduction dimension is out of range.");
      }
      meta.reduced[dimension - 1] = 1;
    }
  }

  std::vector<int> output_shape;
  if (meta.keepdim) {
    for (int dimension = 0; dimension < meta.rank; ++dimension) {
      int output_dimension = static_cast<int>(output_shape.size());
      output_shape.push_back(meta.reduced[dimension]
                                 ? 1
                                 : input->shape[dimension]);
      meta.output_to_input[output_dimension] =
          meta.reduced[dimension] ? -1 : dimension;
    }
  } else {
    for (int dimension = 0; dimension < meta.rank; ++dimension) {
      if (!meta.reduced[dimension]) {
        int output_dimension = static_cast<int>(output_shape.size());
        output_shape.push_back(input->shape[dimension]);
        meta.output_to_input[output_dimension] = dimension;
      }
    }
    if (output_shape.empty()) {
      output_shape.push_back(1);
      meta.output_to_input[0] = -1;
    }
  }
  meta.output_rank = static_cast<int>(output_shape.size());
  std::size_t output_elements = 1;
  for (int dimension = 0; dimension < meta.output_rank; ++dimension) {
    meta.output_shape[dimension] = output_shape[dimension];
    output_elements *= static_cast<std::size_t>(output_shape[dimension]);
  }

  const char* kernel = input->dtype == DType::Float64
      ? "cudaverse_reduce_f64"
      : (input->dtype == DType::Float32
             ? "cudaverse_reduce_f32"
             : "cudaverse_reduce_i32");
  R_CheckUserInterrupt();
  CUfunction function = get_kernel(kernel);
  SharedBuffer* output = allocate_buffer(
      input->dtype, output_shape, output_elements);
  CUdeviceptr input_pointer = input->pointer;
  CUdeviceptr output_pointer = output->pointer;
  unsigned long long kernel_elements = output_elements;
  void* parameters[] = {
      &input_pointer, &output_pointer, &meta, &kernel_elements};
  CUresult status = launch_1d(function, output_elements, parameters);
  if (status != CUDA_SUCCESS) {
    api.cuMemFree(output->pointer);
    delete output;
    check_cuda(status, kernel);
  }

  SEXP result = PROTECT(named_list({"storage", "shape"}));
  SEXP result_pointer = PROTECT(make_pointer(output, false));
  SEXP result_shape = PROTECT(Rf_allocVector(INTSXP, output_shape.size()));
  for (std::size_t index = 0; index < output_shape.size(); ++index) {
    INTEGER(result_shape)[index] = output_shape[index];
  }
  SET_VECTOR_ELT(result, 0, result_pointer);
  SET_VECTOR_ELT(result, 1, result_shape);
  UNPROTECT(3);
  return result;
}

extern "C" SEXP C_cudaverse_cuda_svd(SEXP pointer, SEXP nu_sexp,
                                        SEXP nv_sexp) {
  SharedBuffer* input = get_buffer(pointer);
  if (input->dtype != DType::Float64 || input->shape.size() != 2) {
    Rf_error("Native CUDA SVD requires one float64 matrix.");
  }
  int rows = input->shape[0];
  int columns = input->shape[1];
  int rank = rows < columns ? rows : columns;
  int nu = Rf_asInteger(nu_sexp);
  int nv = Rf_asInteger(nv_sexp);
  if (nu == NA_INTEGER || nv == NA_INTEGER ||
      nu < 0 || nu > rank || nv < 0 || nv > rank) {
    Rf_error("Native CUDA SVD vector counts are out of range.");
  }
  R_CheckUserInterrupt();
  try {
    DeviceSvd decomposition = run_device_svd(input->pointer, rows, columns);
    std::vector<double> values = copy_double_to_host(
        decomposition.values.pointer(), decomposition.rank,
        "cuMemcpyDtoH(SVD values)");
    std::vector<double> left = copy_double_to_host(
        decomposition.left.pointer(), static_cast<std::size_t>(rows) * nu,
        "cuMemcpyDtoH(SVD left vectors)");
    std::vector<double> right = copy_double_to_host(
        decomposition.right.pointer(), static_cast<std::size_t>(columns) * nv,
        "cuMemcpyDtoH(SVD right vectors)");

    SEXP result = PROTECT(named_list({"d", "u", "v"}));
    SEXP d = PROTECT(numeric_vector(values));
    SEXP u = PROTECT(numeric_vector(left));
    SEXP v = PROTECT(numeric_vector(right));
    SET_VECTOR_ELT(result, 0, d);
    SET_VECTOR_ELT(result, 1, u);
    SET_VECTOR_ELT(result, 2, v);
    UNPROTECT(4);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_pca(SEXP pointer,
                                        SEXP components_sexp,
                                        SEXP center_sexp,
                                        SEXP scale_sexp) {
  SharedBuffer* input = get_buffer(pointer);
  if (input->dtype != DType::Float64 || input->shape.size() != 2) {
    Rf_error("Native CUDA PCA requires one float64 matrix.");
  }
  int rows = input->shape[0];
  int columns = input->shape[1];
  int components = Rf_asInteger(components_sexp);
  int max_components = (rows - 1) < columns ? (rows - 1) : columns;
  if (components == NA_INTEGER || components < 1 ||
      components > max_components) {
    Rf_error("Native CUDA PCA component count is out of range.");
  }
  int use_center = Rf_asLogical(center_sexp);
  int use_scale = Rf_asLogical(scale_sexp);
  if (use_center == NA_LOGICAL || use_scale == NA_LOGICAL) {
    Rf_error("Native CUDA PCA flags must be TRUE or FALSE.");
  }
  R_CheckUserInterrupt();
  try {
    std::string reason;
    if (!load_cusolver(reason)) throw std::runtime_error(reason);
    require_kernels();
    CUfunction statistics = get_kernel("cudaverse_column_stats_f64");
    CUfunction transform = get_kernel("cudaverse_center_scale_f64");
    CUfunction scale_columns = get_kernel("cudaverse_scale_columns_f64");

    DeviceMemory centers(static_cast<std::size_t>(columns) * sizeof(double));
    DeviceMemory scales(static_cast<std::size_t>(columns) * sizeof(double));
    DeviceMemory transformed(
        static_cast<std::size_t>(rows) * columns * sizeof(double));
    CUdeviceptr input_pointer = input->pointer;
    CUdeviceptr center_pointer = centers.pointer();
    CUdeviceptr scale_pointer = scales.pointer();
    void* statistics_parameters[] = {
        &input_pointer, &center_pointer, &scale_pointer, &rows, &columns,
        &use_center, &use_scale};
    launch_or_throw(statistics, "cudaverse_column_stats_f64", columns,
                    statistics_parameters);

    std::vector<double> host_centers = copy_double_to_host(
        centers.pointer(), columns, "cuMemcpyDtoH(PCA center)");
    std::vector<double> host_scales = copy_double_to_host(
        scales.pointer(), columns, "cuMemcpyDtoH(PCA scale)");
    if (use_scale) {
      for (double value : host_scales) {
        if (!std::isfinite(value) || value <= 0.0) {
          throw std::runtime_error(
              "Cannot scale constant or non-finite features.");
        }
      }
    }

    CUdeviceptr transformed_pointer = transformed.pointer();
    void* transform_parameters[] = {
        &input_pointer, &transformed_pointer, &center_pointer,
        &scale_pointer, &rows, &columns};
    launch_or_throw(
        transform, "cudaverse_center_scale_f64",
        static_cast<std::size_t>(rows) * columns, transform_parameters);

    DeviceSvd decomposition = run_device_svd(
        transformed.pointer(), rows, columns);
    DeviceMemory scores(
        static_cast<std::size_t>(rows) * components * sizeof(double));
    CUdeviceptr left_pointer = decomposition.left.pointer();
    CUdeviceptr scores_pointer = scores.pointer();
    CUdeviceptr singular_pointer = decomposition.values.pointer();
    void* score_parameters[] = {
        &left_pointer, &scores_pointer, &singular_pointer,
        &rows, &components};
    launch_or_throw(
        scale_columns, "cudaverse_scale_columns_f64",
        static_cast<std::size_t>(rows) * components, score_parameters);

    std::vector<double> host_singular = copy_double_to_host(
        decomposition.values.pointer(), components,
        "cuMemcpyDtoH(PCA singular values)");
    std::vector<double> host_rotation = copy_double_to_host(
        decomposition.right.pointer(),
        static_cast<std::size_t>(columns) * components,
        "cuMemcpyDtoH(PCA rotation)");
    std::vector<double> host_scores = copy_double_to_host(
        scores.pointer(), static_cast<std::size_t>(rows) * components,
        "cuMemcpyDtoH(PCA scores)");
    double denominator = std::sqrt(static_cast<double>(rows - 1));
    for (double& value : host_singular) value /= denominator;

    auto* score_buffer = new SharedBuffer{
        scores.release(),
        static_cast<std::size_t>(rows) * components * sizeof(double),
        static_cast<std::size_t>(rows) * components,
        DType::Float64,
        {rows, components},
        1};
    SEXP score_storage = PROTECT(make_pointer(score_buffer, false));
    SEXP result = PROTECT(named_list({
        "sdev", "rotation", "x", "center", "scale", "scores_storage"}));
    SEXP sdev = PROTECT(numeric_vector(host_singular));
    SEXP rotation = PROTECT(numeric_vector(host_rotation));
    SEXP host_x = PROTECT(numeric_vector(host_scores));
    SEXP host_center = PROTECT(numeric_vector(host_centers));
    SEXP host_scale = PROTECT(numeric_vector(host_scales));
    SET_VECTOR_ELT(result, 0, sdev);
    SET_VECTOR_ELT(result, 1, rotation);
    SET_VECTOR_ELT(result, 2, host_x);
    SET_VECTOR_ELT(result, 3, host_center);
    SET_VECTOR_ELT(result, 4, host_scale);
    SET_VECTOR_ELT(result, 5, score_storage);
    UNPROTECT(7);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_row_norms(SEXP pointer) {
  require_backend();
  require_kernels();
  SharedBuffer* input = get_buffer(pointer);
  if (input->dtype != DType::Float64 || input->shape.size() != 2) {
    Rf_error("Native CUDA row norms require one float64 matrix.");
  }
  try {
    DeviceMemory norms = row_norms_squared(
        input->pointer, input->shape[0], input->shape[1]);
    auto* result = new SharedBuffer{
        norms.release(),
        static_cast<std::size_t>(input->shape[0]) * sizeof(double),
        static_cast<std::size_t>(input->shape[0]),
        DType::Float64,
        {input->shape[0]},
        1};
    return make_pointer(result, false);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_distance(SEXP query_pointer,
                                             SEXP reference_pointer,
                                             SEXP metric_sexp,
                                             SEXP self_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* query = get_buffer(query_pointer);
  SharedBuffer* reference = get_buffer(reference_pointer);
  if (query->dtype != DType::Float64 ||
      reference->dtype != DType::Float64 ||
      query->shape.size() != 2 || reference->shape.size() != 2 ||
      query->shape[1] != reference->shape[1]) {
    Rf_error("Native CUDA distance requires conformable float64 matrices.");
  }
  std::string metric_name = CHAR(STRING_ELT(metric_sexp, 0));
  int metric = metric_name == "cosine" ? 1 : 0;
  int self = Rf_asLogical(self_sexp) == 1 ? 1 : 0;
  try {
    DeviceMemory distance = distance_device(
        query->pointer, query->shape[0], reference->pointer,
        reference->shape[0], query->shape[1], metric, 0, self);
    std::size_t elements = static_cast<std::size_t>(query->shape[0]) *
                           reference->shape[0];
    auto* result = new SharedBuffer{
        distance.release(), elements * sizeof(double), elements,
        DType::Float64, {query->shape[0], reference->shape[0]}, 1};
    return make_pointer(result, false);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_knn_block(
    SEXP reference_pointer, SEXP reference_norms_pointer,
    SEXP first_sexp, SEXP count_sexp, SEXP k_sexp, SEXP metric_sexp) {
  require_backend();
  require_kernels();
  SharedBuffer* reference = get_buffer(reference_pointer);
  if (reference->dtype != DType::Float64 || reference->shape.size() != 2) {
    Rf_error("Native CUDA kNN requires one float64 reference matrix.");
  }
  int reference_rows = reference->shape[0];
  int columns = reference->shape[1];
  int first = Rf_asInteger(first_sexp);
  int count = Rf_asInteger(count_sexp);
  int k = Rf_asInteger(k_sexp);
  if (first == NA_INTEGER || count == NA_INTEGER || k == NA_INTEGER ||
      first < 0 || count < 1 || first + count > reference_rows ||
      k < 1 || k >= reference_rows) {
    Rf_error("Native CUDA kNN block parameters are out of range.");
  }
  std::string metric_name = CHAR(STRING_ELT(metric_sexp, 0));
  int metric = metric_name == "cosine" ? 1 : 0;
  CUdeviceptr cached_norms = 0;
  if (metric == 0) {
    SharedBuffer* norms = get_buffer(reference_norms_pointer);
    if (norms->dtype != DType::Float64 || norms->elements !=
        static_cast<std::size_t>(reference_rows)) {
      Rf_error("Native CUDA kNN reference norms are invalid.");
    }
    cached_norms = norms->pointer;
  }

  try {
    CUfunction gather = get_kernel("cudaverse_gather_rows_f64");
    DeviceMemory query(
        static_cast<std::size_t>(count) * columns * sizeof(double));
    CUdeviceptr reference_device = reference->pointer;
    CUdeviceptr query_device = query.pointer();
    void* gather_parameters[] = {
        &reference_device, &query_device, &reference_rows, &columns,
        &first, &count};
    launch_or_throw(gather, "cudaverse_gather_rows_f64",
                    static_cast<std::size_t>(count) * columns,
                    gather_parameters);

    DeviceMemory distances = distance_device(
        query.pointer(), count, reference->pointer, reference_rows,
        columns, metric, first, 1, cached_norms);
    std::size_t output_elements = static_cast<std::size_t>(count) * k;
    DeviceMemory output_index(output_elements * sizeof(int));
    DeviceMemory output_distance(output_elements * sizeof(double));
    CUdeviceptr distance_pointer = distances.pointer();
    CUdeviceptr index_pointer = output_index.pointer();
    CUdeviceptr output_distance_pointer = output_distance.pointer();
    int self = 1;
    void* topk_parameters[] = {
        &distance_pointer, &index_pointer, &output_distance_pointer,
        &count, &reference_rows, &k, &first, &self};
    if (k <= 32) {
      constexpr unsigned int threads = 128;
      unsigned int shared_bytes = static_cast<unsigned int>(
          static_cast<std::size_t>(threads) * k *
          (sizeof(double) + sizeof(int)));
      cuda_or_throw(
          api.cuLaunchKernel(
              get_kernel("cudaverse_topk_stable_f64"), count, 1, 1,
              threads, 1, 1, shared_bytes, nullptr, topk_parameters, nullptr),
          "cudaverse_topk_stable_f64");
    } else {
      launch_or_throw(
          get_kernel("cudaverse_topk_stable_general_f64"),
          "cudaverse_topk_stable_general_f64", count, topk_parameters);
    }

    std::vector<int> host_index(output_elements);
    std::vector<double> host_distance(output_elements);
    cuda_or_throw(api.cuMemcpyDtoH(
                      host_index.data(), output_index.pointer(),
                      output_elements * sizeof(int)),
                  "cuMemcpyDtoH(kNN index)");
    cuda_or_throw(api.cuMemcpyDtoH(
                      host_distance.data(), output_distance.pointer(),
                      output_elements * sizeof(double)),
                  "cuMemcpyDtoH(kNN distance)");

    SEXP result = PROTECT(named_list({"index", "distance"}));
    SEXP index = PROTECT(Rf_allocVector(INTSXP, output_elements));
    SEXP distance = PROTECT(Rf_allocVector(REALSXP, output_elements));
    std::memcpy(INTEGER(index), host_index.data(),
                output_elements * sizeof(int));
    std::memcpy(REAL(distance), host_distance.data(),
                output_elements * sizeof(double));
    SET_VECTOR_ELT(result, 0, index);
    SET_VECTOR_ELT(result, 1, distance);
    UNPROTECT(3);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP C_cudaverse_cuda_matmul(SEXP left_pointer,
                                          SEXP right_pointer) {
  require_backend();
  SharedBuffer* left = get_buffer(left_pointer);
  SharedBuffer* right = get_buffer(right_pointer);
  if (left->dtype != DType::Float64 || right->dtype != DType::Float64) {
    Rf_error("Native CUDA matmul currently requires float64 tensors.");
  }
  if (left->shape.size() != 2 || right->shape.size() != 2 ||
      left->shape[1] != right->shape[0]) {
    Rf_error("Native CUDA matrix dimensions are not conformable.");
  }
  int m = left->shape[0];
  int k = left->shape[1];
  int n = right->shape[1];
  SharedBuffer* output = allocate_buffer(
      DType::Float64, {m, n}, static_cast<std::size_t>(m) * n);
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasStatus_t status = api.cublasDgemm(
      api.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
      reinterpret_cast<const double*>(left->pointer), m,
      reinterpret_cast<const double*>(right->pointer), k, &beta,
      reinterpret_cast<double*>(output->pointer), m);
  if (status != CUBLAS_STATUS_SUCCESS) {
    api.cuMemFree(output->pointer);
    delete output;
    check_cublas(status, "cublasDgemm");
  }
  return make_pointer(output, false);
}

extern "C" SEXP C_cudaverse_cuda_synchronize() {
  require_backend();
  check_cuda(api.cuCtxSynchronize(), "cuCtxSynchronize");
  return Rf_ScalarLogical(1);
}

extern "C" SEXP C_cudaverse_cuda_release(SEXP pointer) {
  get_buffer(pointer);
  release_reference(pointer);
  return Rf_ScalarLogical(1);
}

extern "C" SEXP C_cudaverse_cuda_share(SEXP pointer) {
  SharedBuffer* buffer = get_buffer(pointer);
  return make_pointer(buffer, true);
}

extern "C" SEXP C_cudaverse_cuda_memory_info() {
  std::string reason;
  if (!load_driver(reason)) Rf_error("%s", reason.c_str());
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check_cuda(api.cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo");
  SEXP result = PROTECT(named_list({"free", "total", "used"}));
  SET_VECTOR_ELT(result, 0, Rf_ScalarReal(static_cast<double>(free_bytes)));
  SET_VECTOR_ELT(result, 1, Rf_ScalarReal(static_cast<double>(total_bytes)));
  SET_VECTOR_ELT(result, 2,
                 Rf_ScalarReal(static_cast<double>(total_bytes - free_bytes)));
  UNPROTECT(1);
  return result;
}

static const R_CallMethodDef call_methods[] = {
    {"C_cudaverse_cuda_load_kernels",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_load_kernels), 1},
    {"C_cudaverse_cuda_diagnostics",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_diagnostics), 0},
    {"C_cudaverse_cuda_from_host",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_from_host), 3},
    {"C_cudaverse_cuda_to_host",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_to_host), 1},
    {"C_cudaverse_cuda_cast",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_cast), 2},
    {"C_cudaverse_cuda_reduce",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_reduce), 4},
    {"C_cudaverse_cuda_svd",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_svd), 3},
    {"C_cudaverse_cuda_pca",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_pca), 4},
    {"C_cudaverse_cuda_row_norms",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_row_norms), 1},
    {"C_cudaverse_cuda_distance",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_distance), 4},
    {"C_cudaverse_cuda_knn_block",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_knn_block), 6},
    {"C_cudaverse_cuda_matmul",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_matmul), 2},
    {"C_cudaverse_cuda_synchronize",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_synchronize), 0},
    {"C_cudaverse_cuda_release",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_release), 1},
    {"C_cudaverse_cuda_share",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_share), 1},
    {"C_cudaverse_cuda_memory_info",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_memory_info), 0},
    {nullptr, nullptr, 0}};

extern "C" void R_init_cudaverseCUDA(DllInfo* dll) {
  R_registerRoutines(dll, nullptr, call_methods, nullptr, nullptr);
  R_useDynamicSymbols(dll, static_cast<Rboolean>(0));
  R_forceSymbols(dll, static_cast<Rboolean>(1));
}
