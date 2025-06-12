
// 检查是否支持 CUDA 和 FAISS GPU
#ifdef __has_include
  #if __has_include(<cuda_runtime.h>) && __has_include(<faiss/gpu/StandardGpuResources.h>)
    #define FAISS_GPU_AVAILABLE 1
  #else
    #define FAISS_GPU_AVAILABLE 0
  #endif
#else
  // 对于不支持 __has_include 的编译器，使用预定义宏
  #ifdef CUDA_VERSION
    #define FAISS_GPU_AVAILABLE 1
  #else
    #define FAISS_GPU_AVAILABLE 0
  #endif
#endif

// 如果没有 GPU 支持，则不编译此文件的内容
#if FAISS_GPU_AVAILABLE
#include <cuda_runtime.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <memory>
#include <stdexcept>

extern "C" {

// FAISS GPU 资源包装器结构
struct FaissGpuWrapper {
    std::unique_ptr<faiss::gpu::StandardGpuResources> resources;
    std::unique_ptr<faiss::gpu::GpuIndexFlat> index_flat;
    std::unique_ptr<faiss::gpu::GpuIndexIVF> index_ivf;
    // CPU 索引作为备用
    std::unique_ptr<faiss::IndexFlat> cpu_index_flat;
    std::unique_ptr<faiss::IndexIVF> cpu_index_ivf;
    int device_id;
    bool initialized;
    bool use_gpu;
    int dimension;
    std::string index_type;
};

// 创建新的 FAISS GPU 包装器
FaissGpuWrapper* faiss_gpu_wrapper_new(int device_id) {
    try {
        auto wrapper = new FaissGpuWrapper();
        wrapper->device_id = device_id;
        wrapper->initialized = false;
        wrapper->dimension = 0;
        wrapper->use_gpu = true;
        return wrapper;
    } catch (...) {
        return nullptr;
    }
}

// 初始化 FAISS 资源
int faiss_gpu_wrapper_init(FaissGpuWrapper* wrapper, int dimension, const char* index_type) {
    if (!wrapper) return -1;

    try {
        wrapper->dimension = dimension;
        wrapper->index_type = std::string(index_type);

        if (wrapper->use_gpu) {
            // 尝试初始化 GPU 资源
            try {
                // 设置 CUDA 设备
                cudaSetDevice(wrapper->device_id);

                // 创建 GPU 资源
                wrapper->resources = std::make_unique<faiss::gpu::StandardGpuResources>();
                wrapper->resources->setDefaultDevice(wrapper->device_id);
                wrapper->resources->setMemoryFraction(0.8f);
                wrapper->resources->setTempMemory(512 * 1024 * 1024); // 512MB

                // 根据索引类型创建相应的 GPU 索引
                if (wrapper->index_type == "Flat") {
                    faiss::gpu::GpuIndexFlatConfig config;
                    config.device = wrapper->device_id;
                    config.useFloat16 = false;

                    wrapper->index_flat = std::make_unique<faiss::gpu::GpuIndexFlat>(
                        wrapper->resources.get(), dimension, faiss::METRIC_INNER_PRODUCT, config);
                } else if (wrapper->index_type == "IVF") {
                    // 创建 CPU 索引作为量化器
                    auto cpu_quantizer = std::make_unique<faiss::IndexFlat>(dimension, faiss::METRIC_INNER_PRODUCT);

                    faiss::gpu::GpuIndexIVFConfig config;
                    config.device = wrapper->device_id;
                    config.useFloat16 = false;

                    int nlist = std::min(4096, std::max(1, dimension / 4));
                    wrapper->index_ivf = std::make_unique<faiss::gpu::GpuIndexIVF>(
                        wrapper->resources.get(), cpu_quantizer.release(), dimension, nlist, faiss::METRIC_INNER_PRODUCT, config);
                }

                wrapper->initialized = true;
                return 0;
            } catch (const std::exception& e) {
                // GPU 初始化失败，回退到 CPU
                wrapper->use_gpu = false;
            }
        }

        // 使用 CPU 索引作为备用或默认选项
        if (wrapper->index_type == "Flat") {
            wrapper->cpu_index_flat = std::make_unique<faiss::IndexFlat>(dimension, faiss::METRIC_INNER_PRODUCT);
        } else if (wrapper->index_type == "IVF") {
            auto cpu_quantizer = std::make_unique<faiss::IndexFlat>(dimension, faiss::METRIC_INNER_PRODUCT);
            int nlist = std::min(4096, std::max(1, dimension / 4));
            wrapper->cpu_index_ivf = std::make_unique<faiss::IndexIVF>(cpu_quantizer.release(), dimension, nlist, faiss::METRIC_INNER_PRODUCT);
        }

        wrapper->initialized = true;
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

// 添加向量到索引
int faiss_gpu_wrapper_add_vectors(FaissGpuWrapper* wrapper, int n, const float* vectors) {
    if (!wrapper || !wrapper->initialized) return -1;

    try {
        if (wrapper->use_gpu) {
            if (wrapper->index_type == "Flat" && wrapper->index_flat) {
                wrapper->index_flat->add(n, vectors);
                return 0;
            } else if (wrapper->index_type == "IVF" && wrapper->index_ivf) {
                wrapper->index_ivf->train(n, vectors);
                wrapper->index_ivf->add(n, vectors);
                return 0;
            }
        }

        // 使用 CPU 索引
        if (wrapper->index_type == "Flat" && wrapper->cpu_index_flat) {
            wrapper->cpu_index_flat->add(n, vectors);
        } else if (wrapper->index_type == "IVF" && wrapper->cpu_index_ivf) {
            wrapper->cpu_index_ivf->train(n, vectors);
            wrapper->cpu_index_ivf->add(n, vectors);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

// 执行搜索
int faiss_gpu_wrapper_search(FaissGpuWrapper* wrapper, int n_queries, const float* queries, int k, float* distances, long* labels) {
    if (!wrapper || !wrapper->initialized) return -1;

    try {
        if (wrapper->use_gpu) {
            if (wrapper->index_type == "Flat" && wrapper->index_flat) {
                wrapper->index_flat->search(n_queries, queries, k, distances, labels);
                return 0;
            } else if (wrapper->index_type == "IVF" && wrapper->index_ivf) {
                wrapper->index_ivf->search(n_queries, queries, k, distances, labels);
                return 0;
            }
        }

        // 使用 CPU 索引
        if (wrapper->index_type == "Flat" && wrapper->cpu_index_flat) {
            wrapper->cpu_index_flat->search(n_queries, queries, k, distances, labels);
        } else if (wrapper->index_type == "IVF" && wrapper->cpu_index_ivf) {
            wrapper->cpu_index_ivf->search(n_queries, queries, k, distances, labels);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

// 释放资源
void faiss_gpu_wrapper_free(FaissGpuWrapper* wrapper) {
    if (wrapper) {
        delete wrapper;
    }
}

// 获取 GPU 设备数量
int faiss_gpu_get_device_count() {
    int count = 0;
    cudaError_t result = cudaGetDeviceCount(&count);
    return (result == cudaSuccess) ? count : 0;
}

// 设置 GPU 设备
int faiss_gpu_set_device(int device_id) {
    cudaError_t result = cudaSetDevice(device_id);
    return (result == cudaSuccess) ? 0 : -1;
}

// 检查是否支持 GPU
int faiss_gpu_is_available() {
    return 1;
}

} // extern "C"

#else // FAISS_GPU_AVAILABLE == 0

// 当没有 GPU 支持时，提供空的实现或错误返回
#include <memory>
#include <stdexcept>

extern "C" {

// 空的包装器结构
struct FaissGpuWrapper {
    int device_id;
    bool initialized;
    int dimension;
    std::string index_type;
};

// 创建包装器（但不支持 GPU）
FaissGpuWrapper* faiss_gpu_wrapper_new(int device_id) {
    auto wrapper = new FaissGpuWrapper();
    wrapper->device_id = device_id;
    wrapper->initialized = false;
    wrapper->dimension = 0;
    return wrapper;
}

// 初始化（返回错误，因为不支持 GPU）
int faiss_gpu_wrapper_init(FaissGpuWrapper* wrapper, int dimension, const char* index_type) {
    return -1; // 不支持 GPU
}

// 添加向量（返回错误）
int faiss_gpu_wrapper_add_vectors(FaissGpuWrapper* wrapper, int n, const float* vectors) {
    return -1; // 不支持 GPU
}

// 搜索（返回错误）
int faiss_gpu_wrapper_search(FaissGpuWrapper* wrapper, int n_queries, const float* queries, int k, float* distances, long* labels) {
    return -1; // 不支持 GPU
}

// 释放资源
void faiss_gpu_wrapper_free(FaissGpuWrapper* wrapper) {
    if (wrapper) {
        delete wrapper;
    }
}

// 获取 GPU 设备数量（返回 0）
int faiss_gpu_get_device_count() {
    return 0; // 没有 GPU 支持
}

// 设置 GPU 设备（返回错误）
int faiss_gpu_set_device(int device_id) {
    return -1; // 没有 GPU 支持
}

// 检查是否支持 GPU
int faiss_gpu_is_available() {
    return 0;
}

} // extern "C"

#endif // FAISS_GPU_AVAILABLE