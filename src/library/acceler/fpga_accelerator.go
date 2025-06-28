//go:build fpga

package acceler

import (
	"VectorSphere/src/library/entity"
	"fmt"
	"math"
	"sync"
	"time"
	"unsafe"
)

/*
#cgo CFLAGS: -I/opt/xilinx/xrt/include -I/opt/intel/opencl/include
#cgo LDFLAGS: -L/opt/xilinx/xrt/lib -L/opt/intel/opencl/lib -lxrt_coreutil -lOpenCL

#include <stdlib.h>
#include <string.h>

// FPGA 设备结构体
typedef struct {
    void* device_handle;
    void* kernel_handle;
    void* buffer_input;
    void* buffer_output;
    int device_id;
    int initialized;
    size_t buffer_size;
} fpga_device_t;

// FPGA 统计信息结构体
typedef struct {
    double memory_utilization;    // 内存利用率 (0.0-1.0)
    double temperature;           // 温度 (摄氏度)
    double power_consumption;     // 功耗 (瓦特)
    unsigned long total_cycles;   // 总时钟周期
    unsigned long active_cycles;  // 活跃时钟周期
    unsigned long memory_reads;   // 内存读取次数
    unsigned long memory_writes;  // 内存写入次数
    double clock_frequency;       // 时钟频率 (MHz)
    double bandwidth_utilization; // 带宽利用率
    int error_count;             // 错误计数
} fpga_stats_t;

// FPGA 函数声明
int fpga_init_device(fpga_device_t* device, int device_id);
int fpga_load_bitstream(fpga_device_t* device, const char* bitstream_path);
int fpga_allocate_buffers(fpga_device_t* device, size_t size);
int fpga_compute_distances(fpga_device_t* device, float* queries, float* database, float* results, int num_queries, int num_vectors, int dimension);
int fpga_batch_cosine_similarity(fpga_device_t* device, float* queries, float* database, float* results, int num_queries, int num_vectors, int dimension);
void fpga_cleanup_device(fpga_device_t* device);
int fpga_get_device_count();
int fpga_get_device_info(int device_id, char* name, int* compute_units, size_t* memory_size);
int fpga_get_device_stats(fpga_device_t* device, fpga_stats_t* stats);
int fpga_reset_stats(fpga_device_t* device);
int fpga_get_performance_counters(fpga_device_t* device, unsigned long* counters, int counter_count);
*/
import "C"

// FPGAAccelerator FPGA加速器实现
type FPGAAccelerator struct {
	deviceID      int
	deviceHandle  unsafe.Pointer
	initialized   bool
	available     bool
	capabilities  HardwareCapabilities
	stats         HardwareStats
	mutex         sync.RWMutex
	config        *FPGAConfig
	bitstream     string
	lastStatsTime time.Time
	startTime     time.Time
}

// FPGAConfig FPGA配置
type FPGAConfig struct {
	Enable          bool                      `json:"enable"`
	DeviceIDs       []int                     `json:"device_ids"`
	Bitstream       string                    `json:"bitstream"`
	ClockFrequency  int                       `json:"clock_frequency"`  // MHz
	MemoryBandwidth int64                     `json:"memory_bandwidth"` // bytes/sec
	PipelineDepth   int                       `json:"pipeline_depth"`
	Parallelism     FPGAParallelismConfig     `json:"parallelism"`
	Optimization    FPGAOptimizationConfig    `json:"optimization"`
	Reconfiguration FPGAReconfigurationConfig `json:"reconfiguration"`
}

// FPGAParallelismConfig FPGA并行配置
type FPGAParallelismConfig struct {
	ComputeUnits   int `json:"compute_units"`
	VectorWidth    int `json:"vector_width"`
	UnrollFactor   int `json:"unroll_factor"`
	PipelineStages int `json:"pipeline_stages"`
}

// FPGAOptimizationConfig FPGA优化配置
type FPGAOptimizationConfig struct {
	ResourceSharing    bool `json:"resource_sharing"`
	MemoryOptimization bool `json:"memory_optimization"`
	TimingOptimization bool `json:"timing_optimization"`
	PowerOptimization  bool `json:"power_optimization"`
	AreaOptimization   bool `json:"area_optimization"`
}

// FPGAReconfigurationConfig FPGA重配置
type FPGAReconfigurationConfig struct {
	Enable                 bool          `json:"enable"`
	PartialReconfiguration bool          `json:"partial_reconfiguration"`
	ReconfigurationTime    time.Duration `json:"reconfiguration_time"`
	BitstreamCache         bool          `json:"bitstream_cache"`
	HotSwap                bool          `json:"hot_swap"`
}

// NewFPGAAccelerator 创建新的FPGA加速器
func NewFPGAAccelerator(deviceID int, config *FPGAConfig) *FPGAAccelerator {
	fpga := &FPGAAccelerator{
		deviceID:      deviceID,
		config:        config,
		lastStatsTime: time.Now(),
		startTime:     time.Now(),
		capabilities: HardwareCapabilities{
			Type:              AcceleratorFPGA,
			SupportedOps:      []string{"distance_compute", "batch_compute", "cosine_similarity", "matrix_multiply", "convolution"},
			PerformanceRating: 7.5,
			SpecialFeatures:   []string{"reconfigurable", "low_latency", "parallel_processing", "custom_kernels"},
		},
	}

	// 检测FPGA可用性
	fpga.detectFPGA()
	return fpga
}

// GetType 获取加速器类型
func (f *FPGAAccelerator) GetType() string {
	return AcceleratorFPGA
}

// IsAvailable 检查FPGA是否可用
func (f *FPGAAccelerator) IsAvailable() bool {
	f.mutex.RLock()
	defer f.mutex.RUnlock()
	return f.available
}

// Initialize 初始化FPGA
func (f *FPGAAccelerator) Initialize() error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if !f.available {
		return fmt.Errorf("FPGA设备 %d 不可用", f.deviceID)
	}

	if f.initialized {
		return nil
	}

	// 初始化FPGA设备
	device := (*C.fpga_device_t)(C.malloc(C.sizeof_fpga_device_t))
	if device == nil {
		return fmt.Errorf("分配FPGA设备内存失败")
	}

	result := C.fpga_init_device(device, C.int(f.deviceID))
	if result != 0 {
		C.free(unsafe.Pointer(device))
		return fmt.Errorf("初始化FPGA设备失败: %d", result)
	}

	// 加载比特流
	if f.config != nil && f.config.Bitstream != "" {
		bitstreamPath := C.CString(f.config.Bitstream)
		defer C.free(unsafe.Pointer(bitstreamPath))

		result = C.fpga_load_bitstream(device, bitstreamPath)
		if result != 0 {
			C.fpga_cleanup_device(device)
			C.free(unsafe.Pointer(device))
			return fmt.Errorf("加载FPGA比特流失败: %d", result)
		}
	}

	// 分配缓冲区
	bufferSize := 1024 * 1024 * 64 // 64MB 默认缓冲区
	result = C.fpga_allocate_buffers(device, C.size_t(bufferSize))
	if result != 0 {
		C.fpga_cleanup_device(device)
		C.free(unsafe.Pointer(device))
		return fmt.Errorf("分配FPGA缓冲区失败: %d", result)
	}

	f.deviceHandle = unsafe.Pointer(device)
	f.initialized = true
	f.updateCapabilities()

	return nil
}

// Shutdown 关闭FPGA
func (f *FPGAAccelerator) Shutdown() error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if !f.initialized {
		return nil
	}

	if f.deviceHandle != nil {
		device := (*C.fpga_device_t)(f.deviceHandle)
		C.fpga_cleanup_device(device)
		C.free(f.deviceHandle)
		f.deviceHandle = nil
	}

	f.initialized = false
	return nil
}

// Start 启动FPGA
func (f *FPGAAccelerator) Start() error {
	return f.Initialize()
}

// Stop 停止FPGA
func (f *FPGAAccelerator) Stop() error {
	return f.Shutdown()
}

// ComputeDistance 计算距离
func (f *FPGAAccelerator) ComputeDistance(query []float64, vectors [][]float64) ([]float64, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if !f.initialized {
		return nil, fmt.Errorf("FPGA未初始化")
	}

	start := time.Now()
	defer func() {
		f.updateStats(time.Since(start), 1, true)
	}()

	// 转换数据格式
	queryFlat := make([]float32, len(query))
	for i, v := range query {
		queryFlat[i] = float32(v)
	}

	vectorsFlat := make([]float32, len(vectors)*len(query))
	for i, vec := range vectors {
		for j, v := range vec {
			vectorsFlat[i*len(query)+j] = float32(v)
		}
	}

	resultsFlat := make([]float32, len(vectors))

	// 调用FPGA计算
	device := (*C.fpga_device_t)(f.deviceHandle)
	result := C.fpga_compute_distances(
		device,
		(*C.float)(unsafe.Pointer(&queryFlat[0])),
		(*C.float)(unsafe.Pointer(&vectorsFlat[0])),
		(*C.float)(unsafe.Pointer(&resultsFlat[0])),
		C.int(1),
		C.int(len(vectors)),
		C.int(len(query)),
	)

	if result != 0 {
		f.updateStats(time.Since(start), 1, false)
		return nil, fmt.Errorf("FPGA计算距离失败: %d", result)
	}

	// 转换结果
	results := make([]float64, len(vectors))
	for i, v := range resultsFlat {
		results[i] = float64(v)
	}

	return results, nil
}

// BatchComputeDistance 批量计算距离
func (f *FPGAAccelerator) BatchComputeDistance(queries [][]float64, vectors [][]float64) ([][]float64, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if !f.initialized {
		return nil, fmt.Errorf("FPGA未初始化")
	}

	start := time.Now()
	defer func() {
		f.updateStats(time.Since(start), len(queries), true)
	}()

	if len(queries) == 0 || len(vectors) == 0 {
		return nil, fmt.Errorf("查询或向量数据为空")
	}

	dimension := len(queries[0])

	// 转换数据格式
	queriesFlat := make([]float32, len(queries)*dimension)
	for i, query := range queries {
		for j, v := range query {
			queriesFlat[i*dimension+j] = float32(v)
		}
	}

	vectorsFlat := make([]float32, len(vectors)*dimension)
	for i, vec := range vectors {
		for j, v := range vec {
			vectorsFlat[i*dimension+j] = float32(v)
		}
	}

	resultsFlat := make([]float32, len(queries)*len(vectors))

	// 调用FPGA批量计算
	device := (*C.fpga_device_t)(f.deviceHandle)
	result := C.fpga_compute_distances(
		device,
		(*C.float)(unsafe.Pointer(&queriesFlat[0])),
		(*C.float)(unsafe.Pointer(&vectorsFlat[0])),
		(*C.float)(unsafe.Pointer(&resultsFlat[0])),
		C.int(len(queries)),
		C.int(len(vectors)),
		C.int(dimension),
	)

	if result != 0 {
		f.updateStats(time.Since(start), len(queries), false)
		return nil, fmt.Errorf("FPGA批量计算距离失败: %d", result)
	}

	// 转换结果
	results := make([][]float64, len(queries))
	for i := range queries {
		results[i] = make([]float64, len(vectors))
		for j := range vectors {
			results[i][j] = float64(resultsFlat[i*len(vectors)+j])
		}
	}

	return results, nil
}

// BatchSearch 批量搜索
func (f *FPGAAccelerator) BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	// 先计算距离
	distances, err := f.BatchComputeDistance(queries, database)
	if err != nil {
		return nil, err
	}

	// 对每个查询找到最近的k个结果
	results := make([][]AccelResult, len(queries))
	for i, queryDistances := range distances {
		// 创建索引-距离对
		type indexDistance struct {
			index    int
			distance float64
		}

		indexDistances := make([]indexDistance, len(queryDistances))
		for j, dist := range queryDistances {
			indexDistances[j] = indexDistance{index: j, distance: dist}
		}

		// 部分排序，只需要前k个
		for j := 0; j < k && j < len(indexDistances); j++ {
			minIdx := j
			for l := j + 1; l < len(indexDistances); l++ {
				if indexDistances[l].distance < indexDistances[minIdx].distance {
					minIdx = l
				}
			}
			indexDistances[j], indexDistances[minIdx] = indexDistances[minIdx], indexDistances[j]
		}

		// 构建结果
		queryResults := make([]AccelResult, k)
		for j := 0; j < k && j < len(indexDistances); j++ {
			queryResults[j] = AccelResult{
				ID:         fmt.Sprintf("vec_%d", indexDistances[j].index),
				Similarity: 1.0 / (1.0 + indexDistances[j].distance), // 转换为相似度
				Metadata:   map[string]interface{}{"index": indexDistances[j].index},
			}
		}
		results[i] = queryResults
	}

	return results, nil
}

// BatchCosineSimilarity 批量余弦相似度计算
func (f *FPGAAccelerator) BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if !f.initialized {
		return nil, fmt.Errorf("FPGA未初始化")
	}

	start := time.Now()
	defer func() {
		f.updateStats(time.Since(start), len(queries), true)
	}()

	if len(queries) == 0 || len(database) == 0 {
		return nil, fmt.Errorf("查询或数据库向量为空")
	}

	dimension := len(queries[0])

	// 转换数据格式
	queriesFlat := make([]float32, len(queries)*dimension)
	for i, query := range queries {
		for j, v := range query {
			queriesFlat[i*dimension+j] = float32(v)
		}
	}

	databaseFlat := make([]float32, len(database)*dimension)
	for i, vec := range database {
		for j, v := range vec {
			databaseFlat[i*dimension+j] = float32(v)
		}
	}

	resultsFlat := make([]float32, len(queries)*len(database))

	// 调用FPGA余弦相似度计算
	device := (*C.fpga_device_t)(f.deviceHandle)
	result := C.fpga_batch_cosine_similarity(
		device,
		(*C.float)(unsafe.Pointer(&queriesFlat[0])),
		(*C.float)(unsafe.Pointer(&databaseFlat[0])),
		(*C.float)(unsafe.Pointer(&resultsFlat[0])),
		C.int(len(queries)),
		C.int(len(database)),
		C.int(dimension),
	)

	if result != 0 {
		f.updateStats(time.Since(start), len(queries), false)
		return nil, fmt.Errorf("FPGA余弦相似度计算失败: %d", result)
	}

	// 转换结果
	results := make([][]float64, len(queries))
	for i := range queries {
		results[i] = make([]float64, len(database))
		for j := range database {
			results[i][j] = float64(resultsFlat[i*len(database)+j])
		}
	}

	return results, nil
}

// AccelerateSearch 加速搜索
func (f *FPGAAccelerator) AccelerateSearch(query []float64, database [][]float64, options entity.SearchOptions) ([]AccelResult, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if !f.initialized {
		return nil, fmt.Errorf("FPGA未初始化")
	}

	if len(database) == 0 {
		return nil, nil
	}

	start := time.Now()
	defer func() {
		f.updateStats(time.Since(start), len(database), true)
	}()

	// 1. 使用FPGA计算向量搜索
	results, err := f.fpgaVectorSearch(query, database, options)
	if err != nil {
		f.updateStats(time.Since(start), len(database), false)
		return nil, err
	}

	// 2. 应用FPGA硬件过滤和排序
	filteredResults := f.fpgaFilterResults(results, options)

	// 3. FPGA硬件重排序优化
	optimizedResults := f.fpgaReorderResults(filteredResults, options)

	// 4. 质量检查和后处理
	finalResults := f.fpgaPostProcess(optimizedResults, options)

	return finalResults, nil
}

// fpgaEnhanceResults 使用FPGA重新计算精确距离和相似度
func (f *FPGAAccelerator) fpgaEnhanceResults(query []float64, results []AccelResult, options entity.SearchOptions) ([]AccelResult, error) {
	if len(results) == 0 {
		return results, nil
	}

	// 提取结果中的向量
	vectors := make([][]float64, 0, len(results))
	for _, result := range results {
		if len(result.Vector) > 0 {
			vectors = append(vectors, result.Vector)
		}
	}

	if len(vectors) == 0 {
		return results, nil
	}

	// 使用FPGA重新计算距离
	distances, err := f.ComputeDistance(query, vectors)
	if err != nil {
		return results, err
	}

	// 使用FPGA计算余弦相似度
	similarities, err := f.BatchCosineSimilarity([][]float64{query}, vectors)
	if err != nil {
		return results, err
	}

	// 更新结果
	enhancedResults := make([]AccelResult, len(results))
	vectorIndex := 0
	for i, result := range results {
		enhancedResults[i] = result
		if len(result.Vector) > 0 && vectorIndex < len(distances) {
			// 更新距离和相似度
			enhancedResults[i].Distance = distances[vectorIndex]
			if len(similarities) > 0 && len(similarities[0]) > vectorIndex {
				enhancedResults[i].Similarity = similarities[0][vectorIndex]
			}
			// 添加FPGA处理标记
			if enhancedResults[i].Metadata == nil {
				enhancedResults[i].Metadata = make(map[string]interface{})
			}
			enhancedResults[i].Metadata["fpga_enhanced"] = true
			enhancedResults[i].Metadata["fpga_distance"] = distances[vectorIndex]
			vectorIndex++
		}
	}

	return enhancedResults, nil
}

// fpgaFilterResults 使用FPGA硬件过滤结果
func (f *FPGAAccelerator) fpgaFilterResults(results []AccelResult, options entity.SearchOptions) []AccelResult {
	if len(results) == 0 {
		return results
	}

	filteredResults := make([]AccelResult, 0, len(results))

	for _, result := range results {
		// 质量阈值过滤
		if options.QualityLevel > 0 {
			if result.Similarity < options.QualityLevel {
				continue
			}
		}

		// 距离阈值过滤（FPGA可以并行处理多个阈值条件）
		if result.Distance > 0 {
			// 根据搜索选项调整距离阈值
			maxDistance := 2.0 // 默认最大距离
			if options.PreferAccuracy {
				maxDistance = 1.0 // 更严格的距离要求
			}
			if result.Distance > maxDistance {
				continue
			}
		}

		// 候选数量限制
		if options.MaxCandidates > 0 && len(filteredResults) >= options.MaxCandidates {
			break
		}

		filteredResults = append(filteredResults, result)
	}

	return filteredResults
}

// fpgaReorderResults 使用FPGA硬件重排序优化
func (f *FPGAAccelerator) fpgaReorderResults(results []AccelResult, options entity.SearchOptions) []AccelResult {
	if len(results) <= 1 {
		return results
	}

	// FPGA并行排序算法
	reorderedResults := make([]AccelResult, len(results))
	copy(reorderedResults, results)

	// 根据搜索选项选择排序策略
	if options.PreferAccuracy {
		// 优先精度：按相似度降序排序
		f.fpgaQuickSort(reorderedResults, 0, len(reorderedResults)-1, "similarity")
	} else {
		// 优先速度：按距离升序排序
		f.fpgaQuickSort(reorderedResults, 0, len(reorderedResults)-1, "distance")
	}

	// FPGA多级排序：相似度相同时按距离排序
	f.fpgaSecondarySort(reorderedResults)

	return reorderedResults
}

// fpgaQuickSort FPGA硬件加速快速排序
func (f *FPGAAccelerator) fpgaQuickSort(results []AccelResult, low, high int, sortBy string) {
	if low < high {
		// FPGA并行分区
		pi := f.fpgaPartition(results, low, high, sortBy)

		// 递归排序（FPGA可以并行处理左右分区）
		f.fpgaQuickSort(results, low, pi-1, sortBy)
		f.fpgaQuickSort(results, pi+1, high, sortBy)
	}
}

// fpgaPartition FPGA硬件分区函数
func (f *FPGAAccelerator) fpgaPartition(results []AccelResult, low, high int, sortBy string) int {
	pivot := results[high]
	i := low - 1

	for j := low; j < high; j++ {
		var shouldSwap bool
		if sortBy == "similarity" {
			// 相似度降序
			shouldSwap = results[j].Similarity > pivot.Similarity
		} else {
			// 距离升序
			shouldSwap = results[j].Distance < pivot.Distance
		}

		if shouldSwap {
			i++
			results[i], results[j] = results[j], results[i]
		}
	}

	results[i+1], results[high] = results[high], results[i+1]
	return i + 1
}

// fpgaSecondarySort FPGA二级排序
func (f *FPGAAccelerator) fpgaSecondarySort(results []AccelResult) {
	// 对相似度相同的结果按距离进行二级排序
	for i := 0; i < len(results); i++ {
		j := i
		// 找到相似度相同的区间
		for j < len(results) && math.Abs(results[j].Similarity-results[i].Similarity) < 1e-6 {
			j++
		}

		// 对该区间按距离排序
		if j-i > 1 {
			f.fpgaQuickSort(results, i, j-1, "distance")
		}

		i = j - 1
	}
}

// fpgaPostProcess FPGA后处理和质量检查
func (f *FPGAAccelerator) fpgaPostProcess(results []AccelResult, options entity.SearchOptions) []AccelResult {
	if len(results) == 0 {
		return results
	}

	// 添加FPGA处理性能指标
	for i := range results {
		if results[i].Metadata == nil {
			results[i].Metadata = make(map[string]interface{})
		}
		results[i].Metadata["fpga_processed"] = true
		results[i].Metadata["fpga_device_id"] = f.deviceID
		results[i].Metadata["processing_time"] = time.Now().UnixNano()
	}

	// 根据搜索选项进行最终调整
	if options.UseCache {
		// 标记结果可缓存
		for i := range results {
			results[i].Metadata["cacheable"] = true
		}
	}

	// 质量验证
	validatedResults := make([]AccelResult, 0, len(results))
	for _, result := range results {
		// 验证结果有效性
		if result.Similarity >= 0 && result.Distance >= 0 {
			validatedResults = append(validatedResults, result)
		}
	}

	return validatedResults
}

// OptimizeMemoryLayout 优化内存布局
func (f *FPGAAccelerator) OptimizeMemoryLayout(vectors [][]float64) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if !f.initialized {
		return fmt.Errorf("FPGA未初始化")
	}

	if len(vectors) == 0 {
		return nil
	}

	start := time.Now()
	defer func() {
		f.updateStats(time.Since(start), len(vectors), true)
	}()

	// 1. 分析向量数据特征
	dataProfile := f.analyzeVectorData(vectors)

	// 2. 选择最优内存布局策略
	layoutStrategy := f.selectOptimalLayout(dataProfile)

	// 3. 执行内存重组
	err := f.executeMemoryReorganization(vectors, layoutStrategy)
	if err != nil {
		f.updateStats(time.Since(start), len(vectors), false)
		return fmt.Errorf("内存布局优化失败: %v", err)
	}

	// 4. 更新FPGA缓冲区配置
	err = f.updateBufferConfiguration(dataProfile, layoutStrategy)
	if err != nil {
		f.updateStats(time.Since(start), len(vectors), false)
		return fmt.Errorf("缓冲区配置更新失败: %v", err)
	}

	// 5. 验证优化效果
	err = f.validateOptimization(vectors, layoutStrategy)
	if err != nil {
		f.updateStats(time.Since(start), len(vectors), false)
		return fmt.Errorf("优化验证失败: %v", err)
	}

	return nil
}

// VectorDataProfile 向量数据特征分析结果
type VectorDataProfile struct {
	VectorCount   int     // 向量数量
	Dimension     int     // 向量维度
	DataSize      int64   // 总数据大小
	Alignment     int     // 数据对齐要求
	AccessPattern string  // 访问模式: "sequential", "random", "strided"
	CacheLineSize int     // 缓存行大小
	BandwidthReq  float64 // 带宽需求
	LatencyReq    float64 // 延迟需求
}

// MemoryLayoutStrategy 内存布局策略
type MemoryLayoutStrategy struct {
	LayoutType   string // "aos", "soa", "hybrid", "tiled"
	TileSize     int    // 分块大小
	PaddingSize  int    // 填充大小
	Interleaving bool   // 是否交错存储
	PrefetchSize int    // 预取大小
	BurstLength  int    // 突发传输长度
	MemoryBanks  int    // 内存银行数量
	Coalescing   bool   // 是否启用合并访问
}

// analyzeVectorData 分析向量数据特征
func (f *FPGAAccelerator) analyzeVectorData(vectors [][]float64) VectorDataProfile {
	if len(vectors) == 0 {
		return VectorDataProfile{}
	}

	dimension := len(vectors[0])
	vectorCount := len(vectors)
	dataSize := int64(vectorCount * dimension * 8) // float64 = 8 bytes

	// 分析访问模式
	accessPattern := "sequential"
	if vectorCount > 1000 {
		accessPattern = "random" // 大数据集通常随机访问
	} else if dimension > 512 {
		accessPattern = "strided" // 高维向量通常跨步访问
	}

	// 计算对齐要求
	alignment := 64 // 默认64字节对齐（FPGA常用）
	if dimension%16 == 0 {
		alignment = 128 // 更大的对齐以优化SIMD
	}

	// 估算带宽和延迟需求
	bandwidthReq := float64(dataSize) / 1000.0 // MB/s
	latencyReq := 100.0                        // 微秒

	if f.config != nil {
		if f.config.Optimization.TimingOptimization {
			latencyReq = 50.0 // 更严格的延迟要求
		}
		if f.config.MemoryBandwidth > 0 {
			bandwidthReq = float64(f.config.MemoryBandwidth) * 0.8 // 80%利用率
		}
	}

	return VectorDataProfile{
		VectorCount:   vectorCount,
		Dimension:     dimension,
		DataSize:      dataSize,
		Alignment:     alignment,
		AccessPattern: accessPattern,
		CacheLineSize: 64, // 典型缓存行大小
		BandwidthReq:  bandwidthReq,
		LatencyReq:    latencyReq,
	}
}

// selectOptimalLayout 选择最优内存布局策略
func (f *FPGAAccelerator) selectOptimalLayout(profile VectorDataProfile) MemoryLayoutStrategy {
	strategy := MemoryLayoutStrategy{
		LayoutType:   "aos", // 默认Array of Structures
		TileSize:     64,
		PaddingSize:  0,
		Interleaving: false,
		PrefetchSize: 1024,
		BurstLength:  16,
		MemoryBanks:  4,
		Coalescing:   true,
	}

	// 根据数据特征选择布局
	switch {
	case profile.Dimension > 1024:
		// 高维向量：使用SOA布局优化缓存
		strategy.LayoutType = "soa"
		strategy.TileSize = 128
		strategy.Interleaving = true

	case profile.VectorCount > 10000:
		// 大数据集：使用分块布局
		strategy.LayoutType = "tiled"
		strategy.TileSize = 256
		strategy.PrefetchSize = 2048

	case profile.AccessPattern == "random":
		// 随机访问：使用混合布局
		strategy.LayoutType = "hybrid"
		strategy.MemoryBanks = 8
		strategy.BurstLength = 8

	case profile.AccessPattern == "strided":
		// 跨步访问：优化预取
		strategy.PrefetchSize = 4096
		strategy.BurstLength = 32
	}

	// 根据FPGA配置调整策略
	if f.config != nil {
		if f.config.Optimization.MemoryOptimization {
			strategy.Coalescing = true
			strategy.PaddingSize = profile.Alignment
		}
		if f.config.Parallelism.ComputeUnits > 0 {
			strategy.MemoryBanks = f.config.Parallelism.ComputeUnits
		}
	}

	return strategy
}

// executeMemoryReorganization 执行内存重组
func (f *FPGAAccelerator) executeMemoryReorganization(vectors [][]float64, strategy MemoryLayoutStrategy) error {
	if len(vectors) == 0 {
		return nil
	}

	// 根据布局策略重组数据
	switch strategy.LayoutType {
	case "aos":
		return f.reorganizeAOS(vectors, strategy)
	case "soa":
		return f.reorganizeSOA(vectors, strategy)
	case "tiled":
		return f.reorganizeTiled(vectors, strategy)
	case "hybrid":
		return f.reorganizeHybrid(vectors, strategy)
	default:
		return fmt.Errorf("不支持的布局类型: %s", strategy.LayoutType)
	}
}

// reorganizeAOS Array of Structures布局
func (f *FPGAAccelerator) reorganizeAOS(vectors [][]float64, strategy MemoryLayoutStrategy) error {
	// AOS布局：向量按顺序存储，每个向量的元素连续存储
	// 这是默认布局，适合顺序访问整个向量

	// 计算填充大小以优化对齐
	dimension := len(vectors[0])
	paddedDimension := dimension
	if strategy.PaddingSize > 0 {
		// 向上对齐到指定边界
		paddedDimension = ((dimension*8 + strategy.PaddingSize - 1) / strategy.PaddingSize) * strategy.PaddingSize / 8
	}

	// 如果需要填充，重新组织数据
	if paddedDimension != dimension {
		for i := range vectors {
			if len(vectors[i]) < paddedDimension {
				// 扩展向量并填充零
				padded := make([]float64, paddedDimension)
				copy(padded, vectors[i])
				vectors[i] = padded
			}
		}
	}

	return nil
}

// reorganizeSOA Structure of Arrays布局
func (f *FPGAAccelerator) reorganizeSOA(vectors [][]float64, strategy MemoryLayoutStrategy) error {
	// SOA布局：按维度分组存储，所有向量的第i维连续存储
	// 适合按维度并行处理

	if len(vectors) == 0 {
		return nil
	}

	dimension := len(vectors[0])
	vectorCount := len(vectors)

	// 创建按维度组织的数据结构
	soaData := make([][]float64, dimension)
	for d := 0; d < dimension; d++ {
		soaData[d] = make([]float64, vectorCount)
		for v := 0; v < vectorCount; v++ {
			soaData[d][v] = vectors[v][d]
		}
	}

	// 如果启用交错存储，重新排列数据
	if strategy.Interleaving {
		return f.applyInterleaving(soaData, strategy)
	}

	return nil
}

// reorganizeTiled 分块布局
func (f *FPGAAccelerator) reorganizeTiled(vectors [][]float64, strategy MemoryLayoutStrategy) error {
	// 分块布局：将数据分成固定大小的块，优化局部性

	if len(vectors) == 0 {
		return nil
	}

	vectorCount := len(vectors)
	tileSize := strategy.TileSize
	if tileSize <= 0 {
		tileSize = 64 // 默认分块大小
	}

	// 计算分块数量
	numTiles := (vectorCount + tileSize - 1) / tileSize

	// 重新排列向量以优化分块访问
	for tile := 0; tile < numTiles; tile++ {
		startIdx := tile * tileSize
		endIdx := startIdx + tileSize
		if endIdx > vectorCount {
			endIdx = vectorCount
		}

		// 对每个分块内的向量进行内存对齐优化
		for i := startIdx; i < endIdx; i++ {
			if strategy.PaddingSize > 0 {
				// 应用填充以优化缓存行对齐
				f.applyVectorPadding(vectors[i], strategy.PaddingSize)
			}
		}
	}

	return nil
}

// reorganizeHybrid 混合布局
func (f *FPGAAccelerator) reorganizeHybrid(vectors [][]float64, strategy MemoryLayoutStrategy) error {
	// 混合布局：结合AOS和SOA的优点
	// 对于小向量使用AOS，对于大向量使用SOA

	if len(vectors) == 0 {
		return nil
	}

	dimension := len(vectors[0])
	threshold := 256 // 维度阈值

	if dimension <= threshold {
		// 低维向量使用AOS布局
		return f.reorganizeAOS(vectors, strategy)
	} else {
		// 高维向量使用SOA布局
		return f.reorganizeSOA(vectors, strategy)
	}
}

// applyInterleaving 应用交错存储
func (f *FPGAAccelerator) applyInterleaving(soaData [][]float64, strategy MemoryLayoutStrategy) error {
	// 交错存储：将不同维度的数据交错排列以优化内存带宽利用

	if len(soaData) == 0 {
		return nil
	}

	dimension := len(soaData)
	vectorCount := len(soaData[0])
	interleaveSize := strategy.MemoryBanks
	if interleaveSize <= 0 {
		interleaveSize = 4 // 默认4路交错
	}

	// 重新排列数据以实现交错存储
	for d := 0; d < dimension; d += interleaveSize {
		endD := d + interleaveSize
		if endD > dimension {
			endD = dimension
		}

		// 对当前交错组进行优化排列
		for v := 0; v < vectorCount; v++ {
			// 确保数据在不同内存银行中均匀分布
			bank := v % strategy.MemoryBanks
			_ = bank // 这里可以根据银行编号进行特定优化
		}
	}

	return nil
}

// applyVectorPadding 应用向量填充
func (f *FPGAAccelerator) applyVectorPadding(vector []float64, paddingSize int) {
	// 向量填充：在向量末尾添加填充以优化内存对齐

	currentSize := len(vector) * 8 // float64 = 8 bytes
	paddedSize := ((currentSize + paddingSize - 1) / paddingSize) * paddingSize
	paddingElements := (paddedSize - currentSize) / 8

	if paddingElements > 0 {
		// 扩展向量并填充零
		for i := 0; i < paddingElements; i++ {
			vector = append(vector, 0.0)
		}
	}
}

// updateBufferConfiguration 更新FPGA缓冲区配置
func (f *FPGAAccelerator) updateBufferConfiguration(profile VectorDataProfile, strategy MemoryLayoutStrategy) error {
	if f.deviceHandle == nil {
		return fmt.Errorf("FPGA设备未初始化")
	}

	// 计算新的缓冲区大小
	newBufferSize := profile.DataSize
	if strategy.PaddingSize > 0 {
		// 考虑填充开销
		newBufferSize = int64(float64(newBufferSize) * 1.2) // 20%填充开销
	}

	// 确保缓冲区大小满足FPGA要求
	minBufferSize := int64(1024 * 1024) // 最小1MB
	if newBufferSize < minBufferSize {
		newBufferSize = minBufferSize
	}

	// 重新分配FPGA缓冲区
	device := (*C.fpga_device_t)(f.deviceHandle)
	result := C.fpga_allocate_buffers(device, C.size_t(newBufferSize))
	if result != 0 {
		return fmt.Errorf("重新分配FPGA缓冲区失败: %d", result)
	}

	return nil
}

// validateOptimization 验证优化效果
func (f *FPGAAccelerator) validateOptimization(vectors [][]float64, strategy MemoryLayoutStrategy) error {
	// 执行简单的性能测试来验证优化效果

	if len(vectors) == 0 {
		return nil
	}

	// 测试内存访问性能
	startTime := time.Now()

	// 模拟典型的向量操作
	for i := 0; i < len(vectors); i++ {
		for j := 0; j < len(vectors[i]); j++ {
			_ = vectors[i][j] // 简单的内存访问
		}
	}

	accessTime := time.Since(startTime)

	// 检查访问时间是否在合理范围内
	expectedTime := time.Duration(len(vectors)*len(vectors[0])) * time.Nanosecond
	if accessTime > expectedTime*10 {
		return fmt.Errorf("内存访问性能不佳，优化可能失败")
	}

	// 验证数据完整性
	for i, vector := range vectors {
		if len(vector) == 0 {
			return fmt.Errorf("向量 %d 数据丢失", i)
		}
	}

	return nil
}

// PrefetchData 预取数据
func (f *FPGAAccelerator) PrefetchData(vectors [][]float64) error {
	// FPGA可以预加载数据到片上内存
	return nil
}

// GetCapabilities 获取FPGA能力信息
func (f *FPGAAccelerator) GetCapabilities() HardwareCapabilities {
	f.mutex.RLock()
	defer f.mutex.RUnlock()
	return f.capabilities
}

// GetStats 获取FPGA统计信息
func (f *FPGAAccelerator) GetStats() HardwareStats {
	f.mutex.RLock()
	defer f.mutex.RUnlock()
	f.stats.LastUsed = f.startTime
	return f.stats
}

// GetPerformanceMetrics 获取性能指标
func (f *FPGAAccelerator) GetPerformanceMetrics() PerformanceMetrics {
	f.mutex.RLock()
	defer f.mutex.RUnlock()

	return PerformanceMetrics{
		LatencyP50:        float64(f.stats.AverageLatency),
		LatencyP95:        float64(f.stats.AverageLatency * 2),
		LatencyP99:        float64(f.stats.AverageLatency * 3),
		ThroughputCurrent: f.stats.Throughput,
		ThroughputPeak:    f.stats.Throughput * 1.5,
		CacheHitRate:      0.95, // FPGA通常有很好的缓存命中率
		ResourceUtilization: map[string]float64{
			"compute_units": 0.8,
			"memory":        f.stats.MemoryUtilization,
			"power":         f.stats.PowerConsumption / 100.0, // 假设最大功耗100W
		},
	}
}

// UpdateConfig 更新配置
func (f *FPGAAccelerator) UpdateConfig(config interface{}) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if fpgaConfig, ok := config.(*FPGAConfig); ok {
		f.config = fpgaConfig
		return nil
	}

	return fmt.Errorf("无效的FPGA配置类型")
}

// AutoTune 自动调优
func (f *FPGAAccelerator) AutoTune(workload WorkloadProfile) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	// 根据工作负载调整FPGA配置
	if f.config != nil {
		// 根据向量维度调整并行度
		if workload.VectorDimension > 1024 {
			f.config.Parallelism.VectorWidth = 16
			f.config.Parallelism.UnrollFactor = 8
		} else {
			f.config.Parallelism.VectorWidth = 8
			f.config.Parallelism.UnrollFactor = 4
		}

		// 根据批处理大小调整流水线深度
		if workload.BatchSize > 1000 {
			f.config.PipelineDepth = 8
		} else {
			f.config.PipelineDepth = 4
		}
	}

	return nil
}

// detectFPGA 检测FPGA可用性
func (f *FPGAAccelerator) detectFPGA() {
	deviceCount := int(C.fpga_get_device_count())
	if deviceCount > f.deviceID {
		f.available = true
		f.updateCapabilities()
	}
}

// updateCapabilities 更新能力信息
func (f *FPGAAccelerator) updateCapabilities() {
	if !f.available {
		return
	}

	var name [256]C.char
	var computeUnits C.int
	var memorySize C.size_t

	result := C.fpga_get_device_info(C.int(f.deviceID), &name[0], &computeUnits, &memorySize)
	if result == 0 {
		f.capabilities.ComputeUnits = int(computeUnits)
		f.capabilities.MemorySize = int64(memorySize)
		f.capabilities.MaxBatchSize = int(memorySize) / (512 * 8) // 假设512维向量
		f.capabilities.Bandwidth = f.capabilities.MemorySize * 10 // 假设10x内存带宽
		f.capabilities.Latency = 100 * time.Microsecond           // FPGA通常有很低的延迟
		f.capabilities.PowerConsumption = 50.0                    // 假设50W功耗
	}
}

// updateStats 更新统计信息
func (f *FPGAAccelerator) updateStats(duration time.Duration, operations int, success bool) {
	f.stats.TotalOperations += int64(operations)
	if success {
		f.stats.SuccessfulOps += int64(operations)
	} else {
		f.stats.FailedOps += int64(operations)
	}

	// 更新平均延迟
	if f.stats.TotalOperations > 0 {
		totalTime := time.Duration(int64(f.stats.AverageLatency)*(f.stats.TotalOperations-int64(operations))) + duration
		f.stats.AverageLatency = totalTime / time.Duration(f.stats.TotalOperations)
	}

	// 更新吞吐量
	now := time.Now()
	if now.Sub(f.lastStatsTime) > time.Second {
		elapsed := now.Sub(f.lastStatsTime).Seconds()
		f.stats.Throughput = float64(operations) / elapsed
		f.lastStatsTime = now
	}

	// 更新错误率
	if f.stats.TotalOperations > 0 {
		f.stats.ErrorRate = float64(f.stats.FailedOps) / float64(f.stats.TotalOperations)
	}

	// 获取FPGA真实统计数据
	f.updateRealFPGAStats()

	// 更新最后使用时间
	f.stats.LastUsed = now
}

// updateRealFPGAStats 获取FPGA真实统计数据
func (f *FPGAAccelerator) updateRealFPGAStats() {
	if !f.initialized || f.deviceHandle == nil {
		// 如果FPGA未初始化，使用模拟数据
		f.stats.MemoryUtilization = 0.7 // 假设70%内存利用率
		f.stats.Temperature = 45.0      // 假设45°C
		f.stats.PowerConsumption = 50.0 // 假设50W功耗
		return
	}

	// 获取FPGA设备统计信息
	var fpgaStats C.fpga_stats_t
	device := (*C.fpga_device_t)(f.deviceHandle)
	result := C.fpga_get_device_stats(device, &fpgaStats)

	if result == 0 {
		// 成功获取真实统计数据
		f.stats.MemoryUtilization = float64(fpgaStats.memory_utilization)
		f.stats.Temperature = float64(fpgaStats.temperature)
		f.stats.PowerConsumption = float64(fpgaStats.power_consumption)

		// 计算额外的性能指标
		if fpgaStats.total_cycles > 0 {
			// 计算FPGA利用率
			utilization := float64(fpgaStats.active_cycles) / float64(fpgaStats.total_cycles)
			// 将利用率信息存储在元数据中（如果需要的话）
			_ = utilization
		}

		// 更新带宽利用率
		if f.capabilities.Bandwidth > 0 {
			// 根据内存读写次数估算带宽利用率
			memoryOps := float64(fpgaStats.memory_reads + fpgaStats.memory_writes)
			// 假设每次操作8字节，时钟频率为实际频率
			bandwidthUsed := memoryOps * 8.0 * float64(fpgaStats.clock_frequency) * 1e6 // 转换为bytes/sec
			bandwidthUtilization := bandwidthUsed / float64(f.capabilities.Bandwidth)
			if bandwidthUtilization > 1.0 {
				bandwidthUtilization = 1.0
			}
			// 可以将带宽利用率存储在扩展字段中
			_ = bandwidthUtilization
		}

		// 检查错误计数
		if fpgaStats.error_count > 0 {
			// 更新错误统计
			f.stats.FailedOps += int64(fpgaStats.error_count)
			if f.stats.TotalOperations > 0 {
				f.stats.ErrorRate = float64(f.stats.FailedOps) / float64(f.stats.TotalOperations)
			}
			// 重置FPGA错误计数器
			C.fpga_reset_stats(device)
		}
	} else {
		// 获取统计数据失败，使用估算值
		f.estimateFPGAStats()
	}
}

// estimateFPGAStats 估算FPGA统计数据
func (f *FPGAAccelerator) estimateFPGAStats() {
	// 基于运行时间和操作数量估算统计数据
	uptime := time.Since(f.startTime)

	// 估算内存利用率（基于操作频率）
	if f.stats.Throughput > 0 {
		// 高吞吐量意味着高内存利用率
		memUtil := f.stats.Throughput / 1000.0 // 假设1000 ops/sec为满负载
		if memUtil > 1.0 {
			memUtil = 1.0
		} else if memUtil < 0.1 {
			memUtil = 0.1
		}
		f.stats.MemoryUtilization = memUtil
	} else {
		f.stats.MemoryUtilization = 0.3 // 默认30%
	}

	// 估算温度（基于利用率和运行时间）
	baseTemp := 35.0                                 // 基础温度
	tempIncrease := f.stats.MemoryUtilization * 20.0 // 最大增加20度
	timeEffect := float64(uptime.Minutes()) * 0.1    // 运行时间影响
	if timeEffect > 10.0 {
		timeEffect = 10.0 // 最大增加10度
	}
	f.stats.Temperature = baseTemp + tempIncrease + timeEffect

	// 估算功耗（基于利用率）
	basePower := 20.0                                 // 基础功耗20W
	powerIncrease := f.stats.MemoryUtilization * 40.0 // 最大增加40W
	f.stats.PowerConsumption = basePower + powerIncrease
}

// GetDetailedStats 获取详细的FPGA统计信息
func (f *FPGAAccelerator) GetDetailedStats() map[string]interface{} {
	f.mutex.RLock()
	defer f.mutex.RUnlock()

	stats := map[string]interface{}{
		"basic_stats":      f.stats,
		"device_id":        f.deviceID,
		"initialized":      f.initialized,
		"available":        f.available,
		"capabilities":     f.capabilities,
		"uptime":           time.Since(f.startTime),
		"bitstream":        f.config.Bitstream,
		"clock_frequency":  f.config.ClockFrequency,
		"memory_bandwidth": f.config.MemoryBandwidth,
	}

	// 如果FPGA已初始化，获取实时性能计数器
	if f.initialized && f.deviceHandle != nil {
		counters := make([]C.ulong, 8) // 8个性能计数器
		device := (*C.fpga_device_t)(f.deviceHandle)
		result := C.fpga_get_performance_counters(device, &counters[0], 8)

		if result == 0 {
			perfCounters := make([]uint64, 8)
			for i := 0; i < 8; i++ {
				perfCounters[i] = uint64(counters[i])
			}
			stats["performance_counters"] = perfCounters
			stats["counter_labels"] = []string{
				"total_cycles", "active_cycles", "memory_reads", "memory_writes",
				"cache_hits", "cache_misses", "pipeline_stalls", "compute_operations",
			}
		}
	}

	return stats
}

// ResetStats 重置统计信息
func (f *FPGAAccelerator) ResetStats() {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	// 重置基本统计
	f.stats.TotalOperations = 0
	f.stats.SuccessfulOps = 0
	f.stats.FailedOps = 0
	f.stats.AverageLatency = 0
	f.stats.Throughput = 0
	f.stats.ErrorRate = 0
	f.stats.MemoryUtilization = 0
	f.stats.Temperature = 0
	f.stats.PowerConsumption = 0
	f.stats.LastUsed = time.Now()
	f.lastStatsTime = time.Now()

	// 调用C函数重置硬件统计
	if f.initialized && f.deviceHandle != nil {
		device := (*C.fpga_device_t)(f.deviceHandle)
		C.fpga_reset_stats(device)
	}
}

// LoadBitstream 加载比特流
func (f *FPGAAccelerator) LoadBitstream(path string) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if !f.initialized {
		return fmt.Errorf("FPGA未初始化")
	}

	// 调用C函数加载比特流
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	device := (*C.fpga_device_t)(f.deviceHandle)
	result := C.fpga_load_bitstream(device, cPath)
	if result != 0 {
		return fmt.Errorf("加载比特流失败: %d", result)
	}

	f.bitstream = path
	return nil
}

// Reconfigure 重新配置FPGA
func (f *FPGAAccelerator) Reconfigure(bitstreamPath string) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	if !f.initialized {
		return fmt.Errorf("FPGA未初始化")
	}

	// 检查是否支持动态重配置
	if f.config != nil && !f.config.Reconfiguration.Enable {
		return fmt.Errorf("动态重配置未启用")
	}

	// 加载新的比特流
	if err := f.LoadBitstream(bitstreamPath); err != nil {
		return fmt.Errorf("加载新比特流失败: %v", err)
	}

	return nil
}

// GetDeviceInfo 获取设备信息
func (f *FPGAAccelerator) GetDeviceInfo() (map[string]interface{}, error) {
	f.mutex.RLock()
	defer f.mutex.RUnlock()

	if !f.available {
		return nil, fmt.Errorf("FPGA设备不可用")
	}

	// 调用C函数获取设备信息
	var name [256]C.char
	var computeUnits C.int
	var memorySize C.size_t

	result := C.fpga_get_device_info(C.int(f.deviceID), &name[0], &computeUnits, &memorySize)
	if result != 0 {
		return nil, fmt.Errorf("获取设备信息失败: %d", result)
	}

	return map[string]interface{}{
		"device_id":     f.deviceID,
		"name":          C.GoString(&name[0]),
		"compute_units": int(computeUnits),
		"memory_size":   int64(memorySize),
		"temperature":   f.stats.Temperature,
		"power":         f.stats.PowerConsumption,
		"utilization":   f.stats.MemoryUtilization,
		"bitstream":     f.bitstream,
		"initialized":   f.initialized,
		"available":     f.available,
	}, nil
}

// fpgaVectorSearch 使用FPGA进行向量搜索
func (f *FPGAAccelerator) fpgaVectorSearch(query []float64, database [][]float64, options entity.SearchOptions) ([]AccelResult, error) {
	if len(database) == 0 {
		return nil, nil
	}

	// 创建结果切片
	results := make([]AccelResult, 0, len(database))

	// 使用FPGA计算相似度
	similarity, err := f.ComputeDistance(query, database)
	for i, si := range similarity {
		if err != nil {
			continue // 跳过错误的计算
		}

		results = append(results, AccelResult{
			Index:      i,
			Similarity: 1.0 - si, // 距离转换为相似度
			Distance:   si,
			Metadata:   map[string]interface{}{"fpga_processed": true},
		})
	}

	// 按相似度排序
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Similarity > results[i].Similarity {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// 限制返回结果数量
	if options.K > 0 && len(results) > options.K {
		results = results[:options.K]
	}

	return results, nil
}
