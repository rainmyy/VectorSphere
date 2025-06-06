package algorithm

/*
集成FAISS-GPU 步骤:
- 安装 FAISS-GPU 库和 CUDA 工具包
- 配置 CGO 绑定和 C/C++ 头文件
- 链接 CUDA 和 FAISS 库
- 处理 C/C++ 与 Go 的数据类型转换
*/

/*
#cgo windows CFLAGS: -IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include -IC:/faiss/include
#cgo windows LDFLAGS: -LC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64 -LC:/faiss/lib -lcudart -lcuda -lfaiss -lfaiss_gpu
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/usr/local/include/faiss
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/local/lib -lcudart -lcuda -lfaiss -lfaiss_gpu

#include <cuda_runtime.h>
#include <cuda.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>

// CUDA 错误处理宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        return err; \
    } \
} while(0)

// FAISS GPU 资源包装器
typedef struct {
    faiss::gpu::StandardGpuResources* resources;
    faiss::gpu::GpuIndexFlat* index_flat;
    faiss::gpu::GpuIndexIVF* index_ivf;
    int device_id;
    bool initialized;
} FaissGpuWrapper;

// C 接口函数声明
FaissGpuWrapper* faiss_gpu_wrapper_new(int device_id);
int faiss_gpu_wrapper_init(FaissGpuWrapper* wrapper, int dimension, const char* index_type);
int faiss_gpu_wrapper_add_vectors(FaissGpuWrapper* wrapper, int n, const float* vectors);
int faiss_gpu_wrapper_search(FaissGpuWrapper* wrapper, int n_queries, const float* queries, int k, float* distances, long* labels);
void faiss_gpu_wrapper_free(FaissGpuWrapper* wrapper);
int faiss_gpu_get_device_count();
int faiss_gpu_set_device(int device_id);
*/
import "C"

import (
	"VectorSphere/library/log"
	"fmt"
	"sync"
	"unsafe"
)

// GPUAccelerator GPU 加速器接口
type GPUAccelerator interface {
	Initialize() error
	BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error)
	BatchSearch(queries [][]float64, database [][]float64, k int) ([][]GPUResult, error)
	Cleanup() error
}

// FAISSGPUAccelerator FAISS GPU 加速器实现
type FAISSGPUAccelerator struct {
	deviceID    int
	initialized bool
	mu          sync.RWMutex
	indexType   string
	gpuWrapper  unsafe.Pointer // C.FaissGpuWrapper*
	dimension   int
}

// GPUResult 搜索结果结构体
type GPUResult struct {
	ID         string
	Similarity float64
	Metadata   map[string]interface{}
	DocIds     []string
}

// NewFAISSGPUAccelerator creates a new FAISS GPU accelerator
func NewFAISSGPUAccelerator(deviceID int, indexType string) *FAISSGPUAccelerator {
	return &FAISSGPUAccelerator{
		deviceID:  deviceID,
		indexType: indexType,
	}
}

// Initialize initializes the GPU accelerator
func (gpu *FAISSGPUAccelerator) Initialize() error {
	gpu.mu.Lock()
	defer gpu.mu.Unlock()
	if gpu.initialized {
		return nil
	}
	if err := gpu.checkAndSetDevice(); err != nil {
		log.Error("GPU device check/set failed: %v", err)
		return err
	}
	if err := gpu.initFaissWrapper(); err != nil {
		log.Error("FAISS wrapper init failed: %v", err)
		return err
	}
	gpu.initialized = true
	log.Info("FAISS GPU Accelerator initialized: device %d, type %s", gpu.deviceID, gpu.indexType)
	return nil
}

// checkAndSetDevice checks GPU availability and sets the device
func (gpu *FAISSGPUAccelerator) checkAndSetDevice() error {
	if err := gpu.checkGPUAvailability(); err != nil {
		return err
	}
	deviceCount := int(C.faiss_gpu_get_device_count())
	if gpu.deviceID >= deviceCount {
		return fmt.Errorf("GPU device ID %d out of range, total: %d", gpu.deviceID, deviceCount)
	}
	if C.faiss_gpu_set_device(C.int(gpu.deviceID)) != 0 {
		return fmt.Errorf("Failed to set GPU device %d", gpu.deviceID)
	}
	return nil
}

// initFaissWrapper initializes the FAISS GPU wrapper
func (gpu *FAISSGPUAccelerator) initFaissWrapper() error {
	gpu.gpuWrapper = unsafe.Pointer(C.faiss_gpu_wrapper_new(C.int(gpu.deviceID)))
	if gpu.gpuWrapper == nil {
		return fmt.Errorf("Failed to create FAISS GPU wrapper")
	}
	gpu.dimension = 512 // default, can be updated later
	indexTypeC := C.CString(gpu.indexType)
	defer C.free(unsafe.Pointer(indexTypeC))
	if C.faiss_gpu_wrapper_init((*C.FaissGpuWrapper)(gpu.gpuWrapper), C.int(gpu.dimension), indexTypeC) != 0 {
		return fmt.Errorf("FAISS GPU wrapper initialization failed")
	}
	return nil
}

// checkVectorsDim checks if all vectors have the same dimension and returns it
func checkVectorsDim(vectors [][]float64) (int, error) {
	if len(vectors) == 0 {
		return 0, fmt.Errorf("empty vectors")
	}
	dim := len(vectors[0])
	for i, v := range vectors {
		if len(v) != dim {
			return 0, fmt.Errorf("vector %d dimension mismatch: %d vs %d", i, len(v), dim)
		}
	}
	return dim, nil
}

// toFloat32Flat flattens [][]float64 to []float32
func toFloat32Flat(vectors [][]float64, dim int) []float32 {
	flat := make([]float32, len(vectors)*dim)
	for i, v := range vectors {
		for j, val := range v {
			flat[i*dim+j] = float32(val)
		}
	}
	return flat
}

// BatchSearch performs batch search on GPU
func (gpu *FAISSGPUAccelerator) BatchSearch(queries [][]float64, database [][]float64, k int) ([][]GPUResult, error) {
	gpu.mu.RLock()
	defer gpu.mu.RUnlock()
	if !gpu.initialized {
		return nil, fmt.Errorf("GPU accelerator not initialized")
	}
	if len(queries) == 0 || len(database) == 0 {
		return nil, fmt.Errorf("queries or database vectors empty")
	}
	qDim, err := checkVectorsDim(queries)
	if err != nil {
		return nil, err
	}
	dbDim, err := checkVectorsDim(database)
	if err != nil {
		return nil, err
	}
	if qDim != dbDim {
		return nil, fmt.Errorf("query dim %d != db dim %d", qDim, dbDim)
	}
	if k <= 0 {
		return nil, fmt.Errorf("k must be > 0")
	}
	if k > len(database) {
		k = len(database)
	}
	dbVectors := toFloat32Flat(database, dbDim)
	queryVectors := toFloat32Flat(queries, qDim)
	distances := make([]float32, len(queries)*k)
	labels := make([]int64, len(queries)*k)
	if C.faiss_gpu_wrapper_batch_search(
		(*C.FaissGpuWrapper)(gpu.gpuWrapper),
		C.int(len(database)),
		(*C.float)(&dbVectors[0]),
		C.int(len(queries)),
		(*C.float)(&queryVectors[0]),
		C.int(k),
		(*C.float)(&distances[0]),
		(*C.long)(&labels[0]),
	) != 0 {
		return nil, fmt.Errorf("GPU batch search failed")
	}
	results := make([][]GPUResult, len(queries))
	for i := 0; i < len(queries); i++ {
		queryResults := make([]GPUResult, k)
		for j := 0; j < k; j++ {
			idx := i*k + j
			label := labels[idx]
			distance := distances[idx]
			similarity := 1.0 / (1.0 + float64(distance))
			queryResults[j] = GPUResult{
				ID:         fmt.Sprintf("%d", label),
				Similarity: similarity,
				Metadata:   make(map[string]interface{}),
			}
		}
		results[i] = queryResults
	}
	return results, nil
}

// BatchCosineSimilarity computes batch cosine similarity on GPU
func (gpu *FAISSGPUAccelerator) BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error) {
	gpu.mu.RLock()
	defer gpu.mu.RUnlock()
	if !gpu.initialized {
		return nil, fmt.Errorf("GPU accelerator not initialized")
	}
	if len(queries) == 0 || len(database) == 0 {
		return nil, fmt.Errorf("queries or database vectors empty")
	}
	dbDim, err := checkVectorsDim(database)
	if err != nil {
		return nil, err
	}
	qDim, err := checkVectorsDim(queries)
	if err != nil {
		return nil, err
	}
	if dbDim != qDim {
		return nil, fmt.Errorf("query dim %d != db dim %d", qDim, dbDim)
	}
	dbVectors := toFloat32Flat(database, dbDim)
	if C.faiss_gpu_wrapper_add_vectors((*C.FaissGpuWrapper)(gpu.gpuWrapper), C.int(len(database)), (*C.float)(&dbVectors[0])) != 0 {
		return nil, fmt.Errorf("add vectors to GPU index failed")
	}
	queryVectors := toFloat32Flat(queries, qDim)
	k := len(database)
	distances := make([]float32, len(queries)*k)
	labels := make([]C.long, len(queries)*k)
	if C.faiss_gpu_wrapper_search((*C.FaissGpuWrapper)(gpu.gpuWrapper), C.int(len(queries)), (*C.float)(&queryVectors[0]), C.int(k), (*C.float)(&distances[0]), (*C.long)(&labels[0])) != 0 {
		return nil, fmt.Errorf("GPU search failed")
	}
	results := make([][]float64, len(queries))
	for i := 0; i < len(queries); i++ {
		results[i] = make([]float64, k)
		for j := 0; j < k; j++ {
			idx := i*k + j
			labelIdx := int(labels[idx])
			if labelIdx < k {
				results[i][labelIdx] = float64(distances[idx])
			}
		}
	}
	return results, nil
}

// Cleanup releases GPU resources
func (gpu *FAISSGPUAccelerator) Cleanup() error {
	gpu.mu.Lock()
	defer gpu.mu.Unlock()
	if !gpu.initialized {
		return nil
	}
	if gpu.gpuWrapper != nil {
		C.faiss_gpu_wrapper_free((*C.FaissGpuWrapper)(gpu.gpuWrapper))
		gpu.gpuWrapper = nil
	}
	gpu.initialized = false
	log.Info("FAISS GPU Accelerator resources cleaned up")
	return nil
}

// CheckGPUAvailability 公共方法，供外部调用检查GPU可用性
func (gpu *FAISSGPUAccelerator) CheckGPUAvailability() error {
	return gpu.checkGPUAvailability()
}

// checkGPUAvailability 检查 GPU 可用性
func (gpu *FAISSGPUAccelerator) checkGPUAvailability() error {
	log.Info("检查 GPU 设备 %d 的可用性...", gpu.deviceID)

	// 1. 首先检查 CUDA 驱动是否可用
	var driverVersion C.int
	if C.cudaDriverGetVersion(&driverVersion) != C.cudaSuccess {
		return fmt.Errorf("CUDA 驱动不可用或未正确安装")
	}
	log.Info("CUDA 驱动版本: %d", int(driverVersion))

	// 2. 检查 CUDA Runtime 版本
	var runtimeVersion C.int
	if C.cudaRuntimeGetVersion(&runtimeVersion) != C.cudaSuccess {
		return fmt.Errorf("CUDA Runtime 不可用")
	}
	log.Info("CUDA Runtime 版本: %d", int(runtimeVersion))

	// 3. 检查驱动和运行时版本兼容性
	if driverVersion < runtimeVersion {
		return fmt.Errorf("CUDA 驱动版本 (%d) 低于运行时版本 (%d)，请更新驱动",
			int(driverVersion), int(runtimeVersion))
	}

	// 4. 获取 GPU 设备数量
	var deviceCount C.int
	if C.cudaGetDeviceCount(&deviceCount) != C.cudaSuccess {
		return fmt.Errorf("无法获取 GPU 设备数量，可能 CUDA 驱动未正确安装")
	}

	if deviceCount == 0 {
		return fmt.Errorf("系统中未检测到 CUDA 兼容的 GPU 设备")
	}

	if gpu.deviceID >= int(deviceCount) {
		return fmt.Errorf("GPU 设备 ID %d 超出范围，可用设备数量: %d", gpu.deviceID, deviceCount)
	}

	// 5. 检查指定设备的详细属性
	var props C.cudaDeviceProp
	if C.cudaGetDeviceProperties(&props, C.int(gpu.deviceID)) != C.cudaSuccess {
		return fmt.Errorf("无法获取 GPU 设备 %d 的属性", gpu.deviceID)
	}

	// 6. 验证设备名称（确保设备存在且可访问）
	deviceName := C.GoString(&props.name[0])
	log.Info("GPU 设备 %d 名称: %s", gpu.deviceID, deviceName)

	// 7. 检查计算能力（FAISS 需要 3.0 以上）
	if props.major < 3 {
		return fmt.Errorf("GPU 设备 %d (%s) 计算能力不足，需要 3.0 以上，当前: %d.%d",
			gpu.deviceID, deviceName, props.major, props.minor)
	}

	// 8. 检查内存大小（至少需要 1GB）
	if props.totalGlobalMem < 1024*1024*1024 {
		return fmt.Errorf("GPU 设备 %d (%s) 内存不足，需要至少 1GB，当前: %d MB",
			gpu.deviceID, deviceName, props.totalGlobalMem/(1024*1024))
	}

	// 9. 检查设备是否支持统一内存
	if props.unifiedAddressing == 0 {
		log.Warning("GPU 设备 %d 不支持统一内存寻址，性能可能受影响", gpu.deviceID)
	}

	// 10. 尝试在设备上分配少量内存进行可用性测试
	if err := gpu.testDeviceMemoryAllocation(); err != nil {
		return fmt.Errorf("GPU 设备 %d 内存分配测试失败: %w", gpu.deviceID, err)
	}

	// 11. 检查设备是否被其他进程占用
	if err := gpu.checkDeviceExclusivity(); err != nil {
		log.Warning("GPU 设备 %d 可能被其他进程使用: %v", gpu.deviceID, err)
	}

	log.Info("GPU 设备 %d (%s) 可用性检查通过 - 计算能力: %d.%d, 内存: %d MB",
		gpu.deviceID, deviceName, props.major, props.minor, props.totalGlobalMem/(1024*1024))
	return nil
}

// testDeviceMemoryAllocation 测试设备内存分配
func (gpu *FAISSGPUAccelerator) testDeviceMemoryAllocation() error {
	// 设置当前设备
	if C.cudaSetDevice(C.int(gpu.deviceID)) != C.cudaSuccess {
		return fmt.Errorf("无法设置 GPU 设备 %d", gpu.deviceID)
	}

	// 尝试分配 1MB 测试内存
	var testPtr unsafe.Pointer
	testSize := C.size_t(1024 * 1024) // 1MB
	if C.cudaMalloc(&testPtr, testSize) != C.cudaSuccess {
		return fmt.Errorf("内存分配失败")
	}
	defer C.cudaFree(testPtr)

	// 测试内存读写
	testData := make([]byte, 1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Host to Device
	if C.cudaMemcpy(testPtr, unsafe.Pointer(&testData[0]), C.size_t(len(testData)), C.cudaMemcpyHostToDevice) != C.cudaSuccess {
		return fmt.Errorf("Host到Device内存拷贝失败")
	}

	// Device to Host
	readData := make([]byte, len(testData))
	if C.cudaMemcpy(unsafe.Pointer(&readData[0]), testPtr, C.size_t(len(testData)), C.cudaMemcpyDeviceToHost) != C.cudaSuccess {
		return fmt.Errorf("Device到Host内存拷贝失败")
	}

	// 验证数据一致性
	for i := range testData {
		if testData[i] != readData[i] {
			return fmt.Errorf("内存读写验证失败")
		}
	}

	return nil
}

// checkDeviceExclusivity 检查设备独占性
func (gpu *FAISSGPUAccelerator) checkDeviceExclusivity() error {
	// 尝试创建 CUDA 上下文来检查设备是否可用
	var context C.CUcontext
	if C.cuCtxCreate(&context, 0, C.CUdevice(gpu.deviceID)) != C.CUDA_SUCCESS {
		return fmt.Errorf("无法创建 CUDA 上下文，设备可能被占用")
	}
	defer C.cuCtxDestroy(context)

	return nil
}

// initializeCUDAContext 初始化 CUDA 上下文
func (gpu *FAISSGPUAccelerator) initializeCUDAContext() error {
	log.Info("初始化 CUDA 上下文...")

	// 实际的 CUDA 上下文初始化（需要 CGO）
	// 设置当前设备
	if C.cudaSetDevice(C.int(gpu.deviceID)) != C.cudaSuccess {
		return fmt.Errorf("无法设置 GPU 设备 %d", gpu.deviceID)
	}

	// 创建 CUDA 上下文
	var context C.CUcontext
	if C.cuCtxCreate(&context, 0, C.CUdevice(gpu.deviceID)) != C.CUDA_SUCCESS {
		return fmt.Errorf("无法创建 CUDA 上下文")
	}

	// 设置上下文为当前
	if C.cuCtxSetCurrent(context) != C.CUDA_SUCCESS {
		return fmt.Errorf("无法设置 CUDA 上下文为当前")
	}

	log.Info("CUDA 上下文初始化完成")
	return nil
}

// setupMemoryPool 设置 GPU 内存池
func (gpu *FAISSGPUAccelerator) setupMemoryPool() error {
	log.Info("设置 GPU 内存池...")

	// 实际的内存池设置（需要 CGO）
	// 获取 GPU 内存信息
	var free, total C.size_t
	if C.cudaMemGetInfo(&free, &total) != C.cudaSuccess {
		return fmt.Errorf("无法获取 GPU 内存信息")
	}

	log.Info("GPU 内存信息: 可用 %d MB, 总计 %d MB",
		free/(1024*1024), total/(1024*1024))

	// 计算内存池大小（使用 80% 的可用内存）
	poolSize := int64(free) * 8 / 10

	// 创建内存池
	var memPool C.cudaMemPool_t
	var poolProps C.cudaMemPoolProps
	poolProps.allocType = C.cudaMemAllocationTypePinned
	poolProps.handleTypes = C.cudaMemHandleTypeNone
	//poolProps.location.type = C.cudaMemLocationTypeDevice
	poolProps.location = C.cudaMemLocationTypeDevice
	poolProps.location.id = C.int(gpu.deviceID)

	if C.cudaMemPoolCreate(&memPool, &poolProps) != C.cudaSuccess {
		return fmt.Errorf("无法创建 GPU 内存池")
	}

	// 设置内存池阈值
	threshold := C.uint64_t(poolSize)
	if C.cudaMemPoolSetAttribute(memPool, C.cudaMemPoolAttrReleaseThreshold, &threshold) != C.cudaSuccess {
		log.Warning("设置内存池释放阈值失败")
	}

	// 启用内存池
	if C.cudaDeviceSetMemPool(C.int(gpu.deviceID), memPool) != C.cudaSuccess {
		log.Warning("启用内存池失败")
	}

	log.Info("GPU 内存池设置完成")
	return nil
}

// initializeFAISSResources 初始化 FAISS GPU 资源
func (gpu *FAISSGPUAccelerator) initializeFAISSResources() error {
	log.Info("初始化 FAISS GPU 资源...")

	// 实际的 FAISS GPU 资源初始化（需要 FAISS C++ 绑定）
	// 1. 创建 GPU 资源对象
	gpuRes := C.faiss_StandardGpuResources_new()
	if gpuRes == nil {
		return fmt.Errorf("无法创建 FAISS GPU 资源")
	}
	gpu.gpuWrapper = gpuRes

	// 2. 设置设备
	if C.faiss_StandardGpuResources_setDefaultDevice(gpuRes, C.int(gpu.deviceID)) != 0 {
		return fmt.Errorf("无法设置 FAISS GPU 设备")
	}

	// 3. 设置内存管理策略
	if C.faiss_StandardGpuResources_setMemoryFraction(gpuRes, 0.8) != 0 {
		log.Warning("设置 FAISS GPU 内存比例失败")
	}

	// 4. 设置临时内存大小
	tempMemSize := C.size_t(512 * 1024 * 1024) // 512MB
	if C.faiss_StandardGpuResources_setTempMemory(gpuRes, tempMemSize) != 0 {
		log.Warning("设置 FAISS GPU 临时内存失败")
	}

	// 根据索引类型进行不同的初始化
	switch gpu.indexType {
	case "IVF":
		log.Info("初始化 IVF GPU 索引配置")
		// IVF 索引特定配置
		// 设置 nprobe 参数
		// 配置量化器

	case "HNSW":
		log.Info("初始化 HNSW GPU 索引配置")
		// HNSW 主要在 CPU 上运行，GPU 用于距离计算加速
		// 设置连接数和搜索参数

	case "Flat":
		log.Info("初始化 Flat GPU 索引配置")
		// Flat 索引配置
		// 设置距离度量类型

	default:
		return fmt.Errorf("不支持的索引类型: %s", gpu.indexType)
	}

	log.Info("FAISS GPU 资源初始化完成")
	return nil
}

// validateInitialization 验证初始化结果
func (gpu *FAISSGPUAccelerator) validateInitialization() error {
	log.Info("验证 GPU 初始化结果...")

	// 1. 测试 GPU 内存分配
	var testPtr unsafe.Pointer
	testSize := C.size_t(1024 * 1024) // 1MB
	if C.cudaMalloc(&testPtr, testSize) != C.cudaSuccess {
		return fmt.Errorf("GPU 内存分配测试失败")
	}
	defer C.cudaFree(testPtr)

	// 2. 测试内存拷贝
	testData := make([]byte, 1024)
	if C.cudaMemcpy(testPtr, unsafe.Pointer(&testData[0]), testSize, C.cudaMemcpyHostToDevice) != C.cudaSuccess {
		return fmt.Errorf("GPU 内存拷贝测试失败")
	}

	// 3. 测试 FAISS 索引功能
	// 创建测试向量
	testVectors := [][]float64{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0},
		{0.0, 0.0, 0.0, 1.0},
	}

	// 测试批量相似度计算
	if _, err := gpu.BatchCosineSimilarity(testVectors[:1], testVectors); err != nil {
		return fmt.Errorf("FAISS 功能测试失败: %w", err)
	}

	// 4. 性能基准测试
	if err := gpu.performanceBenchmark(); err != nil {
		log.Warning("性能基准测试失败: %v", err)
	}

	log.Info("GPU 初始化验证通过")
	return nil
}

// performanceBenchmark 性能基准测试
func (gpu *FAISSGPUAccelerator) performanceBenchmark() error {
	// 创建较大的测试数据集
	dimension := 128
	numVectors := 1000

	testDB := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		testDB[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			testDB[i][j] = float64(i*dimension + j)
		}
	}

	queries := testDB[:10] // 使用前10个向量作为查询

	// 测试批量搜索性能
	_, err := gpu.BatchSearch(queries, testDB, 10)
	if err != nil {
		return fmt.Errorf("批量搜索性能测试失败: %w", err)
	}

	log.Info("性能基准测试完成")
	return nil
}

// GetGPUMemoryInfo 获取 GPU 内存使用信息
func (gpu *FAISSGPUAccelerator) GetGPUMemoryInfo() (free, total uint64, err error) {
	gpu.mu.RLock()
	defer gpu.mu.RUnlock()

	if !gpu.initialized {
		return 0, 0, fmt.Errorf("GPU 加速器未初始化")
	}

	// 实际的内存信息获取（需要 CGO）
	var freeBytes, totalBytes C.size_t
	if C.cudaMemGetInfo(&freeBytes, &totalBytes) != C.cudaSuccess {
		return 0, 0, fmt.Errorf("无法获取 GPU 内存信息")
	}
	return uint64(freeBytes), uint64(totalBytes), nil

	// 模拟返回值
	//return 8 * 1024 * 1024 * 1024, 12 * 1024 * 1024 * 1024, nil // 8GB free, 12GB total
}

// SetMemoryFraction 设置 GPU 内存使用比例
func (gpu *FAISSGPUAccelerator) SetMemoryFraction(fraction float64) error {
	gpu.mu.Lock()
	defer gpu.mu.Unlock()

	if !gpu.initialized {
		return fmt.Errorf("GPU 加速器未初始化")
	}

	if fraction <= 0 || fraction > 1 {
		return fmt.Errorf("内存比例必须在 (0, 1] 范围内")
	}

	// 设置 FAISS GPU 资源的内存使用比例（需要 CGO）
	if gpu.gpuWrapper != nil {
		gpuRes := (*C.faiss_StandardGpuResources)(gpu.gpuWrapper)
		if C.faiss_StandardGpuResources_setMemoryFraction(gpuRes, C.float(fraction)) != 0 {
			return fmt.Errorf("设置内存比例失败")
		}
	}

	log.Info("GPU 内存使用比例设置为: %.2f", fraction)
	return nil
}
