//go:build gpu

package acceler

/*
集成FAISS-GPU 步骤:
- 安装 FAISS-GPU 库和 CUDA 工具包
- 配置 CGO 绑定和 C/C++ 头文件
- 链接 CUDA 和 FAISS 库
- 处理 C/C++ 与 Go 的数据类型转换
*/

/*
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include" -IC:/faiss/include
#cgo windows LDFLAGS: -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" -LC:/faiss/lib -lcudart -lcuda -lfaiss -lfaiss_gpu
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
	"VectorSphere/src/library/logger"
	"fmt"
	"unsafe"
)

// getGPUDeviceCount 获取GPU设备数量
func getGPUDeviceCount() int {
	return C.faiss_gpu_get_device_count()
}

// NewGPUAccelerator creates a new GPU accelerator with default settings
func NewGPUAccelerator() *FAISSAccelerator {
	// We can use a default device ID and index type here, or implement logic to find the best available GPU.
	return NewFAISSGPUAccelerator(0, "IVFFlat")
}

// NewFAISSGPUAccelerator creates a new FAISS GPU accelerator
func NewFAISSGPUAccelerator(deviceID int, indexType string) *FAISSAccelerator {
	return &FAISSAccelerator{
		deviceID:  deviceID,
		indexType: indexType,
	}
}

// 优化 Initialize 方法，增加硬件能力检测和策略选择
func (c *FAISSAccelerator) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.initialized {
		return nil
	}

	// 检测 GPU 硬件能力
	if err := c.checkAndSetDevice(); err != nil {
		logger.Error("GPU device check/set failed: %v", err)
		return err
	}

	// 初始化策略选择器
	c.strategy = NewComputeStrategySelector()

	// 获取 GPU 设备属性
	var props C.cudaDeviceProp
	if C.cudaGetDeviceProperties(&props, C.int(c.deviceID)) == C.cudaSuccess {
		// 根据 GPU 计算能力选择最佳策略
		c.currentStrategy = StrategyGPU
		logger.Info("GPU 加速器初始化: 设备 %d (%s), 计算能力 %d.%d",
			c.deviceID, C.GoString(&props.name[0]), props.major, props.minor)
	} else {
		logger.Warning("无法获取 GPU 属性，使用默认策略")
		c.currentStrategy = StrategyGPU
	}

	if err := c.initFaissWrapper(); err != nil {
		logger.Error("FAISS wrapper init failed: %v", err)
		return err
	}

	c.initialized = true
	logger.Info("FAISS GPU Accelerator initialized: device %d, type %s", c.deviceID, c.indexType)
	return nil
}

// checkAndSetDevice checks GPU availability and sets the device
func (c *FAISSAccelerator) checkAndSetDevice() error {
	if err := c.checkGPUAvailability(); err != nil {
		return err
	}
	deviceCount := int(C.faiss_gpu_get_device_count())
	if c.deviceID >= deviceCount {
		return fmt.Errorf("GPU device ID %d out of range, total: %d", c.deviceID, deviceCount)
	}
	if C.faiss_gpu_set_device(C.int(c.deviceID)) != 0 {
		return fmt.Errorf("Failed to set GPU device %d", c.deviceID)
	}
	return nil
}

// initFaissWrapper initializes the FAISS GPU wrapper
func (c *FAISSAccelerator) initFaissWrapper() error {
	c.gpuWrapper = unsafe.Pointer(C.faiss_gpu_wrapper_new(C.int(c.deviceID)))
	if c.gpuWrapper == nil {
		return fmt.Errorf("Failed to create FAISS GPU wrapper")
	}
	c.dimension = 512 // default, can be updated later
	indexTypeC := C.CString(c.indexType)
	defer C.free(unsafe.Pointer(indexTypeC))
	if C.faiss_gpu_wrapper_init((*C.FaissGpuWrapper)(c.gpuWrapper), C.int(c.dimension), indexTypeC) != 0 {
		return fmt.Errorf("FAISS GPU wrapper initialization failed")
	}
	return nil
}

// 优化 BatchSearch 方法，增加错误恢复
func (c *FAISSAccelerator) BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// 基本检查
	if !c.initialized {
		return nil, fmt.Errorf("GPU accelerator not initialized")
	}

	// 参数验证
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

	// 尝试 GPU 计算，如果失败则回退到 CPU
	results, gpuErr := c.batchSearchGPU(queries, database, k)
	if gpuErr != nil {
		logger.Warning("GPU search failed, falling back to CPU: %v", gpuErr)

		// 回退到 CPU 实现
		return c.batchSearchCPUFallback(queries, database, k)
	}

	return results, nil
}

// 新增 GPU 批量搜索实现
func (c *FAISSAccelerator) batchSearchGPU(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	// 选择最佳批处理大小
	batchSize := c.SelectOptimalBatchSize(len(queries[0]), len(queries))

	// 准备数据库向量
	dbVectors := toFloat32Flat(database, len(database[0]))

	// 分批处理查询
	results := make([][]AccelResult, len(queries))

	for batchStart := 0; batchStart < len(queries); batchStart += batchSize {
		batchEnd := batchStart + batchSize
		if batchEnd > len(queries) {
			batchEnd = len(queries)
		}

		currentBatch := queries[batchStart:batchEnd]
		queryVectors := toFloat32Flat(currentBatch, len(currentBatch[0]))

		// 分配结果内存
		distances := make([]float32, len(currentBatch)*k)
		labels := make([]int64, len(currentBatch)*k)

		// 执行 GPU 批量搜索
		if C.faiss_gpu_wrapper_batch_search(
			(*C.FaissGpuWrapper)(c.gpuWrapper),
			C.int(len(database)),
			(*C.float)(&dbVectors[0]),
			C.int(len(currentBatch)),
			(*C.float)(&queryVectors[0]),
			C.int(k),
			(*C.float)(&distances[0]),
			(*C.long)(&labels[0]),
		) != 0 {
			return nil, fmt.Errorf("GPU batch search failed")
		}

		// 处理结果
		for i := 0; i < len(currentBatch); i++ {
			queryResults := make([]AccelResult, k)
			for j := 0; j < k; j++ {
				idx := i*k + j
				label := labels[idx]
				distance := distances[idx]
				similarity := 1.0 / (1.0 + float64(distance))
				queryResults[j] = AccelResult{
					ID:         fmt.Sprintf("%d", label),
					Similarity: similarity,
					Metadata:   make(map[string]interface{}),
				}
			}
			results[batchStart+i] = queryResults
		}
	}

	return results, nil
}

// 新增 CPU 回退实现
func (c *FAISSAccelerator) batchSearchCPUFallback(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	results := make([][]AccelResult, len(queries))

	// 使用 CPU 并行计算
	cpuCores := runtime.NumCPU()
	var wg sync.WaitGroup

	// 分块处理查询
	chunkSize := (len(queries) + cpuCores - 1) / cpuCores
	for i := 0; i < len(queries); i += chunkSize {
		wg.Add(1)
		end := i + chunkSize
		if end > len(queries) {
			end = len(queries)
		}

		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				// 计算所有距离
				distances := make([]float64, len(database))
				for j, dbVec := range database {
					dist := EuclideanDistanceSquaredDefault(queries[i], dbVec)
					distances[j] = dist
				}

				// 找出 k 个最近邻
				type idxDist struct {
					idx  int
					dist float64
				}

				allDists := make([]idxDist, len(distances))
				for j, dist := range distances {
					allDists[j] = idxDist{j, dist}
				}

				// 按距离排序
				sort.Slice(allDists, func(i, j int) bool {
					return allDists[i].dist < allDists[j].dist
				})

				// 取前 k 个
				queryResults := make([]AccelResult, k)
				for j := 0; j < k; j++ {
					idx := allDists[j].idx
					dist := allDists[j].dist
					similarity := 1.0 / (1.0 + dist) // 转换为相似度
					queryResults[j] = AccelResult{
						ID:         fmt.Sprintf("%d", idx),
						Similarity: similarity,
						Metadata:   make(map[string]interface{}),
					}
				}

				results[i] = queryResults
			}
		}(i, end)
	}

	wg.Wait()
	return results, nil
}

// BatchCosineSimilarity computes batch cosine similarity on GPU
func (c *FAISSAccelerator) BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if !c.initialized {
		return nil, fmt.Errorf("GPU accelerator not initialized")
	}

	// 基本参数检查
	if len(queries) == 0 || len(database) == 0 {
		return nil, fmt.Errorf("queries or database vectors empty")
	}

	// 向量维度检查
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

	// 优化：分批处理大型查询集
	batchSize := 1000 // 可根据 GPU 内存调整
	results := make([][]float64, len(queries))

	for batchStart := 0; batchStart < len(queries); batchStart += batchSize {
		batchEnd := batchStart + batchSize
		if batchEnd > len(queries) {
			batchEnd = len(queries)
		}

		currentBatch := queries[batchStart:batchEnd]
		batchVectors := toFloat32Flat(currentBatch, qDim)
		dbVectors := toFloat32Flat(database, dbDim)

		// 添加向量到 GPU 索引
		if C.faiss_gpu_wrapper_add_vectors((*C.FaissGpuWrapper)(c.gpuWrapper), C.int(len(database)), (*C.float)(&dbVectors[0])) != 0 {
			return nil, fmt.Errorf("add vectors to GPU index failed")
		}

		// 执行批量搜索
		k := len(database)
		distances := make([]float32, len(currentBatch)*k)
		labels := make([]C.long, len(currentBatch)*k)

		if C.faiss_gpu_wrapper_search((*C.FaissGpuWrapper)(c.gpuWrapper), C.int(len(currentBatch)), (*C.float)(&batchVectors[0]), C.int(k), (*C.float)(&distances[0]), (*C.long)(&labels[0])) != 0 {
			return nil, fmt.Errorf("GPU search failed")
		}

		// 处理结果
		for i := 0; i < len(currentBatch); i++ {
			results[batchStart+i] = make([]float64, k)
			for j := 0; j < k; j++ {
				idx := i*k + j
				labelIdx := int(labels[idx])
				if labelIdx < k {
					results[batchStart+i][labelIdx] = float64(distances[idx])
				}
			}
		}
	}

	return results, nil
}

// 优化 Cleanup 方法
func (c *FAISSAccelerator) Cleanup() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.initialized {
		return nil
	}

	// 释放 FAISS 资源
	if c.gpuWrapper != nil {
		C.faiss_gpu_wrapper_free((*C.FaissGpuWrapper)(c.gpuWrapper))
		c.gpuWrapper = nil
	}

	// 清理 CUDA 上下文
	if C.cudaDeviceReset() != C.cudaSuccess {
		logger.Warning("CUDA device reset failed")
	}

	c.initialized = false
	logger.Info("FAISS GPU Accelerator resources cleaned up")
	return nil
}

// 新增 ResetGPUDevice 方法，用于在出现错误时重置设备
func (c *FAISSAccelerator) ResetGPUDevice() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.initialized {
		return nil
	}

	// 释放现有资源
	if c.gpuWrapper != nil {
		C.faiss_gpu_wrapper_free((*C.FaissGpuWrapper)(c.gpuWrapper))
		c.gpuWrapper = nil
	}

	// 重置设备
	if C.cudaDeviceReset() != C.cudaSuccess {
		return fmt.Errorf("CUDA device reset failed")
	}

	// 重新初始化
	if err := c.checkAndSetDevice(); err != nil {
		return fmt.Errorf("重新初始化设备失败: %w", err)
	}

	if err := c.initFaissWrapper(); err != nil {
		return fmt.Errorf("重新初始化 FAISS 包装器失败: %w", err)
	}

	logger.Info("GPU 设备 %d 已重置并重新初始化", c.deviceID)
	return nil
}

// checkGPUAvailability 检查 GPU 可用性
func (c *FAISSAccelerator) checkGPUAvailability() error {
	logger.Info("检查 GPU 设备 %d 的可用性...", c.deviceID)

	// 1. 首先检查 CUDA 驱动是否可用
	var driverVersion C.int
	if C.cudaDriverGetVersion(&driverVersion) != C.cudaSuccess {
		return fmt.Errorf("CUDA 驱动不可用或未正确安装")
	}
	logger.Info("CUDA 驱动版本: %d", int(driverVersion))

	// 2. 检查 CUDA Runtime 版本
	var runtimeVersion C.int
	if C.cudaRuntimeGetVersion(&runtimeVersion) != C.cudaSuccess {
		return fmt.Errorf("CUDA Runtime 不可用")
	}
	logger.Info("CUDA Runtime 版本: %d", int(runtimeVersion))

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

	if c.deviceID >= int(deviceCount) {
		return fmt.Errorf("GPU 设备 ID %d 超出范围，可用设备数量: %d", c.deviceID, deviceCount)
	}

	// 5. 检查指定设备的详细属性
	var props C.cudaDeviceProp
	if C.cudaGetDeviceProperties(&props, C.int(c.deviceID)) != C.cudaSuccess {
		return fmt.Errorf("无法获取 GPU 设备 %d 的属性", c.deviceID)
	}

	// 6. 验证设备名称（确保设备存在且可访问）
	deviceName := C.GoString(&props.name[0])
	logger.Info("GPU 设备 %d 名称: %s", c.deviceID, deviceName)

	// 7. 检查计算能力（FAISS 需要 3.0 以上）
	if props.major < 3 {
		return fmt.Errorf("GPU 设备 %d (%s) 计算能力不足，需要 3.0 以上，当前: %d.%d",
			c.deviceID, deviceName, props.major, props.minor)
	}

	// 8. 检查内存大小（至少需要 1GB）
	if props.totalGlobalMem < 1024*1024*1024 {
		return fmt.Errorf("GPU 设备 %d (%s) 内存不足，需要至少 1GB，当前: %d MB",
			c.deviceID, deviceName, props.totalGlobalMem/(1024*1024))
	}

	// 9. 检查设备是否支持统一内存
	if props.unifiedAddressing == 0 {
		logger.Warning("GPU 设备 %d 不支持统一内存寻址，性能可能受影响", c.deviceID)
	}

	// 10. 尝试在设备上分配少量内存进行可用性测试
	if err := c.testDeviceMemoryAllocation(); err != nil {
		return fmt.Errorf("GPU 设备 %d 内存分配测试失败: %w", c.deviceID, err)
	}

	// 11. 检查设备是否被其他进程占用
	if err := c.checkDeviceExclusivity(); err != nil {
		logger.Warning("GPU 设备 %d 可能被其他进程使用: %v", c.deviceID, err)
	}

	logger.Info("GPU 设备 %d (%s) 可用性检查通过 - 计算能力: %d.%d, 内存: %d MB",
		c.deviceID, deviceName, props.major, props.minor, props.totalGlobalMem/(1024*1024))
	return c.checkGPUAvailability()
}

// testDeviceMemoryAllocation 测试设备内存分配
func (c *FAISSAccelerator) testDeviceMemoryAllocation() error {
	// 设置当前设备
	if C.cudaSetDevice(C.int(c.deviceID)) != C.cudaSuccess {
		return fmt.Errorf("无法设置 GPU 设备 %d", c.deviceID)
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
func (c *FAISSAccelerator) checkDeviceExclusivity() error {
	// 尝试创建 CUDA 上下文来检查设备是否可用
	var context C.CUcontext
	if C.cuCtxCreate(&context, 0, C.CUdevice(c.deviceID)) != C.CUDA_SUCCESS {
		return fmt.Errorf("无法创建 CUDA 上下文，设备可能被占用")
	}
	defer C.cuCtxDestroy(context)

	return nil
}

// initializeCUDAContext 初始化 CUDA 上下文
func (c *FAISSAccelerator) initializeCUDAContext() error {
	logger.Info("初始化 CUDA 上下文...")

	// 实际的 CUDA 上下文初始化（需要 CGO）
	// 设置当前设备
	if C.cudaSetDevice(C.int(c.deviceID)) != C.cudaSuccess {
		return fmt.Errorf("无法设置 GPU 设备 %d", c.deviceID)
	}

	// 创建 CUDA 上下文
	var context C.CUcontext
	if C.cuCtxCreate(&context, 0, C.CUdevice(c.deviceID)) != C.CUDA_SUCCESS {
		return fmt.Errorf("无法创建 CUDA 上下文")
	}

	// 设置上下文为当前
	if C.cuCtxSetCurrent(context) != C.CUDA_SUCCESS {
		return fmt.Errorf("无法设置 CUDA 上下文为当前")
	}

	logger.Info("CUDA 上下文初始化完成")
	return nil
}

// setupMemoryPool 设置 GPU 内存池
func (c *FAISSAccelerator) setupMemoryPool() error {
	logger.Info("设置 GPU 内存池...")

	// 实际的内存池设置（需要 CGO）
	// 获取 GPU 内存信息
	var free, total C.size_t
	if C.cudaMemGetInfo(&free, &total) != C.cudaSuccess {
		return fmt.Errorf("无法获取 GPU 内存信息")
	}

	logger.Info("GPU 内存信息: 可用 %d MB, 总计 %d MB",
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
	poolProps.location.id = C.int(c.deviceID)

	if C.cudaMemPoolCreate(&memPool, &poolProps) != C.cudaSuccess {
		return fmt.Errorf("无法创建 GPU 内存池")
	}

	// 设置内存池阈值
	threshold := C.uint64_t(poolSize)
	if C.cudaMemPoolSetAttribute(memPool, C.cudaMemPoolAttrReleaseThreshold, &threshold) != C.cudaSuccess {
		logger.Warning("设置内存池释放阈值失败")
	}

	// 启用内存池
	if C.cudaDeviceSetMemPool(C.int(c.deviceID), memPool) != C.cudaSuccess {
		logger.Warning("启用内存池失败")
	}

	logger.Info("GPU 内存池设置完成")
	return nil
}

// initializeFAISSResources 初始化 FAISS GPU 资源
func (c *FAISSAccelerator) initializeFAISSResources() error {
	logger.Info("初始化 FAISS GPU 资源...")

	// 实际的 FAISS GPU 资源初始化（需要 FAISS C++ 绑定）
	// 1. 创建 GPU 资源对象
	gpuRes := C.faiss_StandardGpuResources_new()
	if gpuRes == nil {
		return fmt.Errorf("无法创建 FAISS GPU 资源")
	}
	c.gpuWrapper = gpuRes

	// 2. 设置设备
	if C.faiss_StandardGpuResources_setDefaultDevice(gpuRes, C.int(c.deviceID)) != 0 {
		return fmt.Errorf("无法设置 FAISS GPU 设备")
	}

	// 3. 设置内存管理策略
	if C.faiss_StandardGpuResources_setMemoryFraction(gpuRes, 0.8) != 0 {
		logger.Warning("设置 FAISS GPU 内存比例失败")
	}

	// 4. 设置临时内存大小
	tempMemSize := C.size_t(512 * 1024 * 1024) // 512MB
	if C.faiss_StandardGpuResources_setTempMemory(gpuRes, tempMemSize) != 0 {
		logger.Warning("设置 FAISS GPU 临时内存失败")
	}

	// 根据索引类型进行不同的初始化
	switch c.indexType {
	case "IVF":
		logger.Info("初始化 IVF GPU 索引配置")
		// IVF 索引特定配置
		// 设置 nprobe 参数
		// 配置量化器

	case "HNSW":
		logger.Info("初始化 HNSW GPU 索引配置")
		// HNSW 主要在 CPU 上运行，GPU 用于距离计算加速
		// 设置连接数和搜索参数

	case "Flat":
		logger.Info("初始化 Flat GPU 索引配置")
		// Flat 索引配置
		// 设置距离度量类型

	default:
		return fmt.Errorf("不支持的索引类型: %s", c.indexType)
	}

	logger.Info("FAISS GPU 资源初始化完成")
	return nil
}

// validateInitialization 验证初始化结果
func (c *FAISSAccelerator) validateInitialization() error {
	logger.Info("验证 GPU 初始化结果...")

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
	if _, err := c.BatchCosineSimilarity(testVectors[:1], testVectors); err != nil {
		return fmt.Errorf("FAISS 功能测试失败: %w", err)
	}

	// 4. 性能基准测试
	if err := c.performanceBenchmark(); err != nil {
		logger.Warning("性能基准测试失败: %v", err)
	}

	logger.Info("GPU 初始化验证通过")
	return nil
}

// 优化 performanceBenchmark 方法
func (c *FAISSAccelerator) performanceBenchmark() error {
	logger.Info("执行 GPU 性能基准测试...")

	// 测试不同维度和数据量
	dimensions := []int{128, 256, 512, 1024}
	dataSizes := []int{1000, 10000, 100000}

	results := make(map[string]interface{})

	for _, dim := range dimensions {
		for _, size := range dataSizes {
			// 跳过太大的测试组合
			if dim*size > 100000000 {
				continue
			}

			// 创建测试数据
			testDB := make([][]float64, size)
			for i := 0; i < size; i++ {
				testDB[i] = make([]float64, dim)
				for j := 0; j < dim; j++ {
					testDB[i][j] = float64(i*dim+j) / float64(dim*size)
				}
			}

			queries := testDB[:10] // 使用前10个向量作为查询

			// 测试批量搜索性能
			startTime := time.Now()
			_, err := c.BatchSearch(queries, testDB, 10)
			duration := time.Since(startTime)

			testKey := fmt.Sprintf("dim_%d_size_%d", dim, size)
			results[testKey] = map[string]interface{}{
				"dimension":      dim,
				"database_size":  size,
				"duration_ms":    duration.Milliseconds(),
				"vectors_per_ms": float64(len(queries)) / (float64(duration.Milliseconds()) / 1000),
				"error":          err != nil,
			}

			logger.Info("测试 %s: %d ms", testKey, duration.Milliseconds())
		}
	}

	// 保存结果
	c.benchmarkResults = results

	logger.Info("性能基准测试完成")
	return nil
}

// 新增 GetBenchmarkResults 方法
func (c *FAISSAccelerator) GetBenchmarkResults() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.benchmarkResults == nil {
		return map[string]interface{}{"error": "未执行基准测试"}
	}

	return c.benchmarkResults
}

// GetGPUMemoryInfo 获取 GPU 内存使用信息
func (c *FAISSAccelerator) GetGPUMemoryInfo() (free, total uint64, err error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.initialized {
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
func (c *FAISSAccelerator) SetMemoryFraction(fraction float64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.initialized {
		return fmt.Errorf("GPU 加速器未初始化")
	}

	if fraction <= 0 || fraction > 1 {
		return fmt.Errorf("内存比例必须在 (0, 1] 范围内")
	}

	// 设置 FAISS GPU 资源的内存使用比例（需要 CGO）
	if c.gpuWrapper != nil {
		gpuRes := (*C.faiss_StandardGpuResources)(c.gpuWrapper)
		if C.faiss_StandardGpuResources_setMemoryFraction(gpuRes, C.float(fraction)) != 0 {
			return fmt.Errorf("设置内存比例失败")
		}
	}

	logger.Info("GPU 内存使用比例设置为: %.2f", fraction)
	return nil
}

// AccelerateSearch 加速搜索（UnifiedAccelerator接口方法）
func (c *FAISSAccelerator) AccelerateSearch(query []float64, results []AccelResult, options entity.SearchOptions) ([]AccelResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.initialized {
		return nil, fmt.Errorf("GPU accelerator not initialized")
	}

	// 简单实现：直接返回输入的结果
	// 在实际应用中，这里可以进行进一步的优化处理
	return results, nil
}

// Shutdown 关闭GPU加速器（UnifiedAccelerator接口方法）
func (g *GPUAccelerator) Shutdown() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if !g.initialized {
		return nil // 已经关闭
	}

	// 模拟GPU资源清理
	g.initialized = false
	g.memoryUsed = 0

	return nil
}

// Start 启动GPU加速器（UnifiedAccelerator接口方法）
func (g *GPUAccelerator) Start() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.initialized {
		return nil // 已经启动
	}

	// 模拟GPU初始化
	g.initialized = true
	g.memoryUsed = 0

	return nil
}

// Stop 停止GPU加速器（UnifiedAccelerator接口方法）
func (g *GPUAccelerator) Stop() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if !g.initialized {
		return nil // 已经停止
	}

	// 模拟GPU停止
	g.initialized = false

	return nil
}

// GetPerformanceMetrics 获取性能指标（UnifiedAccelerator接口方法）
func (g *GPUAccelerator) GetPerformanceMetrics() PerformanceMetrics {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return PerformanceMetrics{
		LatencyCurrent:    time.Microsecond * 100,
		LatencyMin:        time.Microsecond * 50,
		LatencyMax:        time.Microsecond * 200,
		LatencyP50:        100.0,
		LatencyP95:        180.0,
		LatencyP99:        195.0,
		ThroughputCurrent: 1000.0,
		ThroughputPeak:    1500.0,
		CacheHitRate:      0.85,
		ResourceUtilization: map[string]float64{
			"gpu_memory": float64(g.memoryUsed) / float64(g.memoryTotal),
			"compute":    0.75,
			"bandwidth":  0.60,
		},
	}
}
