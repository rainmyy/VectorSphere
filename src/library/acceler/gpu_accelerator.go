//go:build gpu

package acceler

/*
#cgo CFLAGS: -I/usr/local/cuda/include -I/usr/local/include/faiss
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/local/lib -lcudart -lcublas -lfaiss_gpu -lfaiss -lstdc++

#include <cuda_runtime.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/Index.h>
#include <stdlib.h>

// CUDA错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        return -1; \
    } \
} while(0)

// FAISS GPU包装器结构体
typedef struct {
    faiss::gpu::StandardGpuResources* resources;
    faiss::Index* index;
    int device_id;
    int dimension;
    bool initialized;
} FaissGpuWrapper;

// StandardGpuResources类型定义
typedef faiss::gpu::StandardGpuResources StandardGpuResources;

// C接口函数声明
extern "C" {
    FaissGpuWrapper* faiss_gpu_wrapper_new(int device_id);
    int faiss_gpu_wrapper_init(FaissGpuWrapper* wrapper, int dimension, const char* index_type);
    int faiss_gpu_wrapper_add_vectors(FaissGpuWrapper* wrapper, int n, const float* vectors);
    int faiss_gpu_wrapper_search(FaissGpuWrapper* wrapper, int n, const float* queries, int k, float* distances, long* labels);
    int faiss_gpu_wrapper_batch_search(FaissGpuWrapper* wrapper, int db_size, const float* database, int query_size, const float* queries, int k, float* distances, long* labels);
    void faiss_gpu_wrapper_free(FaissGpuWrapper* wrapper);

    // GPU内存管理
    int cudaMemGetInfo(size_t* free, size_t* total);
    int cudaSetDevice(int device);
    int cudaGetDeviceCount(int* count);
    int cudaGetDeviceProperties(void* prop, int device);

    // FAISS GPU资源管理
    StandardGpuResources* faiss_StandardGpuResources_new();
    void faiss_StandardGpuResources_free(StandardGpuResources* res);
    int faiss_StandardGpuResources_setTempMemory(StandardGpuResources* res, size_t size);
    int faiss_StandardGpuResources_setMemoryFraction(StandardGpuResources* res, float fraction);
}
*/
import "C"

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
	"time"
	"unsafe"
)

// NewGPUAccelerator 创建新的GPU加速器实例
func NewGPUAccelerator(deviceID int) *GPUAccelerator {
	return &GPUAccelerator{
		deviceID:    deviceID,
		indexType:   "IVF",
		batchSize:   1000,
		streamCount: 4,
		strategy:    NewComputeStrategySelector(),
	}
}

// GetType 返回加速器类型
func (g *GPUAccelerator) GetType() string {
	return AcceleratorGPU
}

// IsAvailable 检查GPU是否可用
func (g *GPUAccelerator) IsAvailable() bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.available
}

// Initialize 初始化GPU加速器
func (g *GPUAccelerator) Initialize() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.initialized {
		return nil
	}

	// 检查GPU可用性
	if err := g.CheckGPUAvailability(); err != nil {
		return fmt.Errorf("GPU不可用: %v", err)
	}

	// 设置GPU设备
	if err := g.checkAndSetDevice(); err != nil {
		return fmt.Errorf("设置GPU设备失败: %v", err)
	}

	// 初始化FAISS GPU资源
	if err := g.initializeFAISSResources(); err != nil {
		return fmt.Errorf("初始化FAISS GPU资源失败: %v", err)
	}

	// 初始化FAISS包装器
	if err := g.initFaissWrapper(); err != nil {
		return fmt.Errorf("初始化FAISS包装器失败: %v", err)
	}

	// 验证初始化
	if err := g.validateInitialization(); err != nil {
		return fmt.Errorf("GPU初始化验证失败: %v", err)
	}

	g.initialized = true
	g.available = true
	g.stats.LastUsed = time.Now()

	logger.Info("GPU加速器初始化成功，设备ID: %d", g.deviceID)
	return nil
}

// CheckGPUAvailability 检查GPU可用性
func (g *GPUAccelerator) CheckGPUAvailability() error {
	// 获取GPU设备数量
	var deviceCount C.int
	if C.cudaGetDeviceCount(&deviceCount) != C.cudaSuccess {
		return fmt.Errorf("无法获取GPU设备数量")
	}

	g.deviceCount = int(deviceCount)
	if g.deviceCount == 0 {
		return fmt.Errorf("未检测到GPU设备")
	}

	// 检查设备ID是否有效
	if g.deviceID >= g.deviceCount {
		return fmt.Errorf("GPU设备ID %d 超出范围 [0, %d)", g.deviceID, g.deviceCount)
	}

	// 获取设备属性
	var prop C.struct_cudaDeviceProp
	if C.cudaGetDeviceProperties(unsafe.Pointer(&prop), C.int(g.deviceID)) != C.cudaSuccess {
		return fmt.Errorf("无法获取GPU设备 %d 的属性", g.deviceID)
	}

	// 验证设备名称
	deviceName := C.GoString(&prop.name[0])
	if len(deviceName) == 0 {
		return fmt.Errorf("GPU设备 %d 名称为空", g.deviceID)
	}

	// 检查计算能力
	computeCapability := float64(prop.major) + float64(prop.minor)/10.0
	if computeCapability < 3.0 {
		return fmt.Errorf("GPU设备 %d 计算能力 %.1f 过低，需要至少 3.0", g.deviceID, computeCapability)
	}

	// 测试内存分配
	if err := g.testDeviceMemoryAllocation(); err != nil {
		return fmt.Errorf("GPU内存分配测试失败: %v", err)
	}

	// 检查设备排他性
	if prop.computeMode == C.cudaComputeModeExclusive {
		logger.Warning("GPU设备 %d 处于排他模式", g.deviceID)
	}

	return nil
}

// testDeviceMemoryAllocation 测试设备内存分配
func (g *GPUAccelerator) testDeviceMemoryAllocation() error {
	// 尝试分配小块内存进行测试
	testSize := 1024 * 1024 // 1MB
	var ptr unsafe.Pointer

	if C.cudaMalloc(&ptr, C.size_t(testSize)) != C.cudaSuccess {
		return fmt.Errorf("无法分配GPU内存")
	}

	// 立即释放测试内存
	C.cudaFree(ptr)
	return nil
}

// checkAndSetDevice 检查并设置GPU设备
func (g *GPUAccelerator) checkAndSetDevice() error {
	if C.cudaSetDevice(C.int(g.deviceID)) != C.cudaSuccess {
		return fmt.Errorf("无法设置GPU设备 %d", g.deviceID)
	}
	return nil
}

// initializeFAISSResources 初始化FAISS GPU资源
func (g *GPUAccelerator) initializeFAISSResources() error {
	// 创建FAISS GPU资源
	g.gpuResources = unsafe.Pointer(C.faiss_StandardGpuResources_new())
	if g.gpuResources == nil {
		return fmt.Errorf("无法创建FAISS GPU资源")
	}

	// 设置FAISS GPU临时内存
	tempMemSize := 512 * 1024 * 1024 // 512MB
	if C.faiss_StandardGpuResources_setTempMemory((*C.StandardGpuResources)(g.gpuResources), C.size_t(tempMemSize)) != 0 {
		return fmt.Errorf("设置FAISS GPU临时内存失败")
	}

	// 根据索引类型进行不同的初始化
	switch g.indexType {
	case "IVF":
		// IVF索引特定初始化
		logger.Info("初始化IVF索引")
	case "HNSW":
		// HNSW索引特定初始化
		logger.Info("初始化HNSW索引")
	case "Flat":
		// Flat索引特定初始化
		logger.Info("初始化Flat索引")
	default:
		logger.Warning("未知索引类型 %s，使用默认IVF", g.indexType)
		g.indexType = "IVF"
	}

	return nil
}

// initFaissWrapper 初始化FAISS包装器
func (g *GPUAccelerator) initFaissWrapper() error {
	g.gpuWrapper = unsafe.Pointer(C.faiss_gpu_wrapper_new(C.int(g.deviceID)))
	if g.gpuWrapper == nil {
		return fmt.Errorf("无法创建FAISS GPU包装器")
	}
	g.dimension = 512 // 默认维度，可以后续更新
	indexTypeC := C.CString(g.indexType)
	defer C.free(unsafe.Pointer(indexTypeC))
	if C.faiss_gpu_wrapper_init((*C.FaissGpuWrapper)(g.gpuWrapper), C.int(g.dimension), indexTypeC) != 0 {
		return fmt.Errorf("FAISS GPU包装器初始化失败")
	}
	return nil
}

// validateInitialization 验证初始化
func (g *GPUAccelerator) validateInitialization() error {
	// 测试GPU内存分配
	free, total, err := g.GetGPUMemoryInfo()
	if err != nil {
		return fmt.Errorf("获取GPU内存信息失败: %v", err)
	}

	g.memoryTotal = int64(total)
	g.memoryUsed = int64(total - free)

	// 测试GPU内存拷贝
	testData := make([]float32, 1000)
	for i := range testData {
		testData[i] = float32(i)
	}

	var devicePtr unsafe.Pointer
	if C.cudaMalloc(&devicePtr, C.size_t(len(testData)*4)) != C.cudaSuccess {
		return fmt.Errorf("GPU内存分配测试失败")
	}
	defer C.cudaFree(devicePtr)

	if C.cudaMemcpy(devicePtr, unsafe.Pointer(&testData[0]), C.size_t(len(testData)*4), C.cudaMemcpyHostToDevice) != C.cudaSuccess {
		return fmt.Errorf("GPU内存拷贝测试失败")
	}

	// 测试FAISS索引功能（批量相似度计算）
	if err := g.testFAISSIndexFunctionality(); err != nil {
		return fmt.Errorf("FAISS索引功能测试失败: %v", err)
	}

	// 执行性能基准测试
	if err := g.performBenchmark(); err != nil {
		logger.Warning("性能基准测试失败: %v", err)
		// 不将基准测试失败视为致命错误
	}

	return nil
}

// testFAISSIndexFunctionality 测试FAISS索引功能
func (g *GPUAccelerator) testFAISSIndexFunctionality() error {
	// 创建测试向量
	testVectors := make([]float32, 10*128) // 10个128维向量
	for i := range testVectors {
		testVectors[i] = float32(i%100) / 100.0
	}

	// 测试向量添加
	if C.faiss_gpu_wrapper_add_vectors((*C.FaissGpuWrapper)(g.gpuWrapper), 10, &testVectors[0]) != 0 {
		return fmt.Errorf("FAISS向量添加测试失败")
	}

	// 测试搜索
	queryVector := testVectors[:128] // 使用第一个向量作为查询
	distances := make([]float32, 5)
	labels := make([]int64, 5)

	if C.faiss_gpu_wrapper_search((*C.FaissGpuWrapper)(g.gpuWrapper), 1, &queryVector[0], 5, &distances[0], &labels[0]) != 0 {
		return fmt.Errorf("FAISS搜索测试失败")
	}

	return nil
}

// performBenchmark 执行性能基准测试
func (g *GPUAccelerator) performBenchmark() error {
	// 创建基准测试数据
	numVectors := 1000
	dimension := 128
	testData := make([]float32, numVectors*dimension)
	for i := range testData {
		testData[i] = float32(i%100) / 100.0
	}

	// 测量向量添加性能
	start := time.Now()
	if C.faiss_gpu_wrapper_add_vectors((*C.FaissGpuWrapper)(g.gpuWrapper), C.int(numVectors), &testData[0]) != 0 {
		return fmt.Errorf("基准测试向量添加失败")
	}
	addTime := time.Since(start)

	// 测量搜索性能
	queryVector := testData[:dimension]
	distances := make([]float32, 10)
	labels := make([]int64, 10)

	start = time.Now()
	for i := 0; i < 100; i++ { // 执行100次搜索
		if C.faiss_gpu_wrapper_search((*C.FaissGpuWrapper)(g.gpuWrapper), 1, &queryVector[0], 10, &distances[0], &labels[0]) != 0 {
			return fmt.Errorf("基准测试搜索失败")
		}
	}
	searchTime := time.Since(start)

	// 记录性能指标
	g.performanceMetrics.ThroughputCurrent = float64(numVectors) / addTime.Seconds()
	g.performanceMetrics.LatencyCurrent = searchTime / 100

	logger.Info("GPU性能基准测试完成 - 添加吞吐量: %.2f vectors/sec, 平均搜索延迟: %v",
		g.performanceMetrics.ThroughputCurrent, g.performanceMetrics.LatencyCurrent)

	return nil
}

// GetGPUMemoryInfo 获取GPU内存信息
func (g *GPUAccelerator) GetGPUMemoryInfo() (free uint64, total uint64, err error) {
	var freeBytes, totalBytes C.size_t
	if C.cudaMemGetInfo(&freeBytes, &totalBytes) != C.cudaSuccess {
		return 0, 0, fmt.Errorf("获取GPU内存信息失败")
	}
	return uint64(freeBytes), uint64(totalBytes), nil
}

// SetMemoryFraction 设置FAISS GPU资源的内存使用比例
func (g *GPUAccelerator) SetMemoryFraction(fraction float32) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.gpuResources == nil {
		return fmt.Errorf("GPU资源未初始化")
	}

	if C.faiss_StandardGpuResources_setMemoryFraction((*C.StandardGpuResources)(g.gpuResources), C.float(fraction)) != 0 {
		return fmt.Errorf("设置内存比例失败")
	}

	return nil
}

// ComputeDistance 计算单个查询向量与多个向量的距离
func (g *GPUAccelerator) ComputeDistance(query []float64, vectors [][]float64) ([]float64, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if !g.initialized {
		return nil, fmt.Errorf("GPU加速器未初始化")
	}

	start := time.Now()
	defer func() {
		g.stats.ComputeTime += time.Since(start)
		g.stats.TotalOperations++
	}()

	// 转换为float32并执行GPU计算
	queryF32 := make([]float32, len(query))
	for i, v := range query {
		queryF32[i] = float32(v)
	}

	vectorsF32 := make([]float32, len(vectors)*len(vectors[0]))
	for i, vec := range vectors {
		for j, v := range vec {
			vectorsF32[i*len(vec)+j] = float32(v)
		}
	}

	// 执行GPU距离计算
	distances := make([]float32, len(vectors))
	labels := make([]int64, len(vectors))

	if C.faiss_gpu_wrapper_search((*C.FaissGpuWrapper)(g.gpuWrapper), 1, &queryF32[0], C.int(len(vectors)), &distances[0], &labels[0]) != 0 {
		g.stats.FailedOps++
		return nil, fmt.Errorf("GPU距离计算失败")
	}

	// 转换回float64
	result := make([]float64, len(distances))
	for i, d := range distances {
		result[i] = float64(d)
	}

	g.stats.SuccessfulOps++
	return result, nil
}

// BatchComputeDistance 批量计算向量距离
func (g *GPUAccelerator) BatchComputeDistance(queries [][]float64, vectors [][]float64) ([][]float64, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if !g.initialized {
		return nil, fmt.Errorf("GPU加速器未初始化")
	}

	start := time.Now()
	defer func() {
		g.stats.ComputeTime += time.Since(start)
		g.stats.KernelLaunches++
		g.stats.MemoryTransfers += int64(len(queries) * len(vectors))
	}()

	results := make([][]float64, len(queries))
	for i, query := range queries {
		dist, err := g.ComputeDistance(query, vectors)
		if err != nil {
			return nil, err
		}
		results[i] = dist
	}

	return results, nil
}

// BatchCosineSimilarity 批量计算余弦相似度
func (g *GPUAccelerator) BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if !g.initialized {
		return nil, fmt.Errorf("GPU加速器未初始化")
	}

	// 基本参数检查
	if len(queries) == 0 || len(database) == 0 {
		return nil, fmt.Errorf("查询向量或数据库向量为空")
	}

	start := time.Now()
	defer func() {
		g.stats.ComputeTime += time.Since(start)
		g.stats.KernelLaunches++
		g.stats.MemoryTransfers += int64(len(queries) * len(database))
	}()

	results := make([][]float64, len(queries))
	for i, query := range queries {
		results[i] = make([]float64, len(database))
		for j, vector := range database {
			// 计算余弦相似度
			dotProduct := 0.0
			normA := 0.0
			normB := 0.0
			for k := 0; k < len(query) && k < len(vector); k++ {
				dotProduct += query[k] * vector[k]
				normA += query[k] * query[k]
				normB += vector[k] * vector[k]
			}
			if normA > 0 && normB > 0 {
				results[i][j] = dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
			} else {
				results[i][j] = 0.0
			}
		}
	}

	return results, nil
}

// BatchSearch 批量搜索
func (g *GPUAccelerator) BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// 基本检查
	if !g.initialized {
		return nil, fmt.Errorf("GPU加速器未初始化")
	}

	// 参数验证
	if len(queries) == 0 || len(database) == 0 {
		return nil, fmt.Errorf("查询向量或数据库向量为空")
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
		return nil, fmt.Errorf("查询向量维度 %d != 数据库向量维度 %d", qDim, dbDim)
	}

	if k <= 0 {
		return nil, fmt.Errorf("k必须大于0")
	}

	if k > len(database) {
		k = len(database)
	}

	// 尝试GPU计算，如果失败则回退到CPU
	results, gpuErr := g.batchSearchGPU(queries, database, k)
	if gpuErr != nil {
		logger.Warning("GPU搜索失败，回退到CPU: %v", gpuErr)
		return g.batchSearchCPUFallback(queries, database, k)
	}

	return results, nil
}

// batchSearchGPU GPU批量搜索实现
func (g *GPUAccelerator) batchSearchGPU(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	// 选择最佳批处理大小
	batchSize := g.SelectOptimalBatchSize(len(queries[0]), len(queries))

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

		// 执行GPU批量搜索
		if C.faiss_gpu_wrapper_batch_search(
			(*C.FaissGpuWrapper)(g.gpuWrapper),
			C.int(len(database)),
			(*C.float)(&dbVectors[0]),
			C.int(len(currentBatch)),
			(*C.float)(&queryVectors[0]),
			C.int(k),
			(*C.float)(&distances[0]),
			(*C.long)(&labels[0]),
		) != 0 {
			return nil, fmt.Errorf("GPU批量搜索失败")
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
					Distance:   float64(distance),
					Metadata:   make(map[string]interface{}),
				}
			}
			results[batchStart+i] = queryResults
		}
	}

	return results, nil
}

// batchSearchCPUFallback CPU回退实现
func (g *GPUAccelerator) batchSearchCPUFallback(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	results := make([][]AccelResult, len(queries))

	// 使用CPU并行计算
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

				// 找出k个最近邻
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

				// 取前k个
				queryResults := make([]AccelResult, k)
				for j := 0; j < k; j++ {
					idx := allDists[j].idx
					dist := allDists[j].dist
					similarity := 1.0 / (1.0 + dist) // 转换为相似度
					queryResults[j] = AccelResult{
						ID:         fmt.Sprintf("%d", idx),
						Similarity: similarity,
						Distance:   dist,
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

// AccelerateSearch 加速搜索
func (g *GPUAccelerator) AccelerateSearch(query []float64, database [][]float64, options entity.SearchOptions) ([]AccelResult, error) {
	if !g.initialized {
		return nil, fmt.Errorf("GPU加速器未初始化")
	}

	// 使用BatchSearch实现单个查询
	results, err := g.BatchSearch([][]float64{query}, database, options.K)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return []AccelResult{}, nil
	}

	return results[0], nil
}

// GetCapabilities 获取硬件能力
func (g *GPUAccelerator) GetCapabilities() HardwareCapabilities {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return HardwareCapabilities{
		Type:              "gpu",
		GPUDevices:        g.deviceCount,
		MemorySize:        g.memoryTotal,
		ComputeUnits:      g.deviceCount * 2048, // 估算
		MaxBatchSize:      g.batchSize,
		SupportedOps:      []string{"distance", "similarity", "search", "batch_search"},
		PerformanceRating: 9.0,                      // GPU通常有很高的性能评级
		Bandwidth:         500 * 1024 * 1024 * 1024, // 500GB/s估算
		Latency:           time.Microsecond * 100,
		PowerConsumption:  250.0, // 250W估算
		SpecialFeatures:   []string{"CUDA", "FAISS", "parallel_processing"},
	}
}

// GetStats 获取硬件统计信息
func (g *GPUAccelerator) GetStats() HardwareStats {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// 获取GPU内存利用率
	free, total, err := g.GetGPUMemoryInfo()
	memoryUtilization := 0.0
	if err == nil && total > 0 {
		memoryUtilization = float64(total-free) / float64(total)
	}

	// 计算错误率
	errorRate := 0.0
	if g.stats.TotalOperations > 0 {
		errorRate = float64(g.stats.FailedOps) / float64(g.stats.TotalOperations)
	}

	// 计算平均延迟
	averageLatency := time.Duration(0)
	if g.stats.SuccessfulOps > 0 {
		averageLatency = g.stats.ComputeTime / time.Duration(g.stats.SuccessfulOps)
	}

	// 计算吞吐量
	throughput := 0.0
	if g.stats.ComputeTime > 0 {
		throughput = float64(g.stats.SuccessfulOps) / g.stats.ComputeTime.Seconds()
	}

	return HardwareStats{
		TotalOperations:   g.stats.TotalOperations,
		SuccessfulOps:     g.stats.SuccessfulOps,
		FailedOps:         g.stats.FailedOps,
		AverageLatency:    averageLatency,
		Throughput:        throughput,
		MemoryUtilization: memoryUtilization,
		Temperature:       0.0, // 需要特殊API获取
		PowerConsumption:  0.0, // 需要特殊API获取
		ErrorRate:         errorRate,
		LastUsed:          g.stats.LastUsed,
	}
}

// GetPerformanceMetrics 获取性能指标
func (g *GPUAccelerator) GetPerformanceMetrics() PerformanceMetrics {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return g.performanceMetrics
}

// AutoTune 自动调优
func (g *GPUAccelerator) AutoTune(workload WorkloadProfile) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if !g.initialized {
		return fmt.Errorf("GPU加速器未初始化")
	}

	// 根据工作负载调整参数
	switch workload.Type {
	case "low_latency":
		g.batchSize = 100
		g.streamCount = 8
	case "high_throughput":
		g.batchSize = 2000
		g.streamCount = 16
	case "balanced":
		g.batchSize = 1000
		g.streamCount = 8
	case "memory_efficient":
		g.batchSize = 500
		g.streamCount = 4
	default:
		return fmt.Errorf("未知的工作负载类型: %s", workload.Type)
	}

	// 根据向量维度调整
	if workload.VectorDimension > 0 {
		g.dimension = workload.VectorDimension
		if workload.VectorDimension > 1024 {
			g.batchSize = g.batchSize / 2 // 高维向量减少批处理大小
		}
	}

	// 根据数据集大小调整
	if workload.BatchSize > 0 {
		if workload.BatchSize > 1000000 { // 大数据集
			g.streamCount = g.streamCount * 2
		}
	}

	logger.Info("GPU加速器自动调优完成 - 批处理大小: %d, 流数量: %d", g.batchSize, g.streamCount)
	return nil
}

// Shutdown 关闭GPU加速器
func (g *GPUAccelerator) Shutdown() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if !g.initialized {
		return nil
	}

	// 清理FAISS包装器
	if g.gpuWrapper != nil {
		C.faiss_gpu_wrapper_free((*C.FaissGpuWrapper)(g.gpuWrapper))
		g.gpuWrapper = nil
	}

	// 清理FAISS GPU资源
	if g.gpuResources != nil {
		C.faiss_StandardGpuResources_free((*C.StandardGpuResources)(g.gpuResources))
		g.gpuResources = nil
	}

	// 重置CUDA设备
	C.cudaDeviceReset()

	g.initialized = false
	g.available = false

	logger.Info("GPU加速器已关闭")
	return nil
}

// SelectOptimalBatchSize 选择最佳批处理大小
func (g *GPUAccelerator) SelectOptimalBatchSize(vectorDim, numQueries int) int {
	// 基于可用GPU内存和向量维度计算最佳批处理大小
	free, _, err := g.GetGPUMemoryInfo()
	if err != nil {
		return g.batchSize // 使用默认值
	}

	// 估算每个向量需要的内存（float32）
	bytesPerVector := vectorDim * 4
	// 为安全起见，只使用50%的可用内存
	availableMemory := int64(free) / 2

	// 计算可以处理的最大向量数
	maxVectors := int(availableMemory / int64(bytesPerVector))

	// 选择合适的批处理大小
	optimalBatch := maxVectors / 4 // 保守估计
	if optimalBatch < 100 {
		optimalBatch = 100
	} else if optimalBatch > 2000 {
		optimalBatch = 2000
	}

	// 不超过查询数量
	if optimalBatch > numQueries {
		optimalBatch = numQueries
	}

	return optimalBatch
}
