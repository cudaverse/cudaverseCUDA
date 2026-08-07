#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

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
using cublasHandle_t = void*;
using cublasStatus_t = int;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
constexpr int CUBLAS_OP_N = 0;

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
using cuCtxSynchronize_t = CUresult (*)();
using cuMemGetInfo_t = CUresult (*)(std::size_t*, std::size_t*);
using cuGetErrorName_t = CUresult (*)(CUresult, const char**);
using cuGetErrorString_t = CUresult (*)(CUresult, const char**);

using cublasCreate_t = cublasStatus_t (*)(cublasHandle_t*);
using cublasDgemm_t = cublasStatus_t (*)(
    cublasHandle_t, int, int, int, int, int, const double*,
    const double*, int, const double*, int, const double*, double*, int);

struct BackendApi {
  void* driver = nullptr;
  void* cublas = nullptr;
  CUcontext context = nullptr;
  cublasHandle_t cublas_handle = nullptr;
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
  cuCtxSynchronize_t cuCtxSynchronize = nullptr;
  cuMemGetInfo_t cuMemGetInfo = nullptr;
  cuGetErrorName_t cuGetErrorName = nullptr;
  cuGetErrorString_t cuGetErrorString = nullptr;
  cublasCreate_t cublasCreate = nullptr;
  cublasDgemm_t cublasDgemm = nullptr;
};

BackendApi api;

enum class DType { Float64, Float32, Integer };

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
      bind_symbol(api.cuCtxSynchronize, api.driver, "cuCtxSynchronize") &&
      bind_symbol(api.cuMemGetInfo, api.driver, "cuMemGetInfo_v2") &&
      bind_symbol(api.cuGetErrorName, api.driver, "cuGetErrorName") &&
      bind_symbol(api.cuGetErrorString, api.driver, "cuGetErrorString");
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
    {"C_cudaverse_cuda_diagnostics",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_diagnostics), 0},
    {"C_cudaverse_cuda_from_host",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_from_host), 3},
    {"C_cudaverse_cuda_to_host",
     reinterpret_cast<DL_FUNC>(&C_cudaverse_cuda_to_host), 1},
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
