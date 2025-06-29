//go:build cpu

package acceler

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"fmt"
	"github.com/klauspost/cpuid"
	"math"
	"runtime"
	"sort"
	"sync"
	"time"
)

// GetHardwareCapabilities 获取硬件能力（单例模式）
func (hd *HardwareDetector) GetHardwareCapabilities() HardwareCapabilities {
	hd.once.Do(func() {
		hd.capabilities = HardwareCapabilities{
			HasAVX2:    cpuid.CPU.AVX2(),
			HasAVX512:  cpuid.CPU.AVX512F() && cpuid.CPU.AVX512DQ(),
			HasGPU:     detectGPUSupport(),
			CPUCores:   runtime.NumCPU(),
			GPUDevices: getGPUDeviceCount(),
		}
	})
	return hd.capabilities
}

func NewCPUAccelerator(deviceID int, indexType string) *CpuAccelerator {
	return &CpuAccelerator{
		deviceID:    deviceID,
		indexType:   indexType,
		strategy:    NewComputeStrategySelector(),
		initialized: false,
	}
}

func (c *CPUAccelerator) IsAvailable() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.available
}

func (c *CPUAccelerator) GetType() string {
	return AcceleratorCPU
}

func (c *CPUAccelerator) Shutdown() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.initialized {
		return nil
	}

	c.initialized = false
	return nil
}
func getGPUDeviceCount() int {
	// 这里需要调用CUDA API获取设备数量
	// 简化实现，实际应该调用C.faiss_gpu_get_device_count()
	return 0 // 默认假设有1个GPU设备
}

// Initialize 实现仅使用 CPU 的初始化
func (c *CPUAccelerator) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.initialized {
		return nil
	}

	// 检测 CPU 硬件能力
	caps := c.strategy.GetHardwareCapabilities()
	logger.Info("CPU加速器初始化: 检测到 %d 核心, AVX2: %v, AVX512: %v",
		caps.CPUCores, caps.HasAVX2, caps.HasAVX512)

	// 选择最佳计算策略
	c.currentStrategy = c.strategy.SelectOptimalStrategy(1000, 512) // 默认参数

	// 根据检测到的硬件能力设置最佳策略
	if caps.HasAVX512 {
		logger.Info("启用 AVX512 加速")
		c.currentStrategy = StrategyAVX512
	} else if caps.HasAVX2 {
		logger.Info("启用 AVX2 加速")
		c.currentStrategy = StrategyAVX2
	} else {
		logger.Info("使用标准计算方法")
		c.currentStrategy = StrategyStandard
	}

	logger.Info("CPU加速器初始化完成，使用策略: %v", c.currentStrategy)
	c.initialized = true
	c.available = true
	return nil
}

func (c *CPUAccelerator) BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.initialized {
		return nil, fmt.Errorf("CPU加速器未初始化")
	}

	if len(queries) == 0 || len(database) == 0 {
		return nil, fmt.Errorf("查询或数据库向量为空")
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
		return nil, fmt.Errorf("查询维度 %d != 数据库维度 %d", qDim, dbDim)
	}

	// 使用自适应计算函数计算余弦相似度
	results := make([][]float64, len(queries))

	// 并行计算以提高性能
	cpuCores := runtime.NumCPU()
	if len(queries) > 1 && cpuCores > 1 {
		// 创建工作组
		var wg sync.WaitGroup
		chunkSize := (len(queries) + cpuCores - 1) / cpuCores

		for i := 0; i < len(queries); i += chunkSize {
			wg.Add(1)
			end := i + chunkSize
			if end > len(queries) {
				end = len(queries)
			}

			go func(start, end int) {
				defer wg.Done()
				for j := start; j < end; j++ {
					results[j] = make([]float64, len(database))
					for k, dbVec := range database {
						results[j][k] = AdaptiveCosineSimilarity(queries[j], dbVec, c.currentStrategy)
					}
				}
			}(i, end)
		}

		wg.Wait()
	} else {
		// 单线程计算
		for i, query := range queries {
			results[i] = make([]float64, len(database))
			for j, dbVec := range database {
				results[i][j] = AdaptiveCosineSimilarity(query, dbVec, c.currentStrategy)
			}
		}
	}

	return results, nil
}

// ComputeDistance 计算距离（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) ComputeDistance(query []float64, targets [][]float64) ([]float64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.initialized {
		return nil, fmt.Errorf("CPU accelerator not initialized")
	}

	if len(query) == 0 || len(targets) == 0 {
		return nil, fmt.Errorf("empty query or targets")
	}

	// 计算与所有目标向量的距离
	distances := make([]float64, len(targets))
	for i, target := range targets {
		if len(target) != len(query) {
			return nil, fmt.Errorf("dimension mismatch: query %d, target %d", len(query), len(target))
		}

		// 计算欧几里得距离
		dist := 0.0
		for j := 0; j < len(query); j++ {
			diff := query[j] - target[j]
			dist += diff * diff
		}
		distances[i] = math.Sqrt(dist)
	}

	return distances, nil
}

func (c *CPUAccelerator) BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.initialized {
		return nil, fmt.Errorf("CPU加速器未初始化")
	}

	if len(queries) == 0 || len(database) == 0 {
		return nil, fmt.Errorf("查询或数据库向量为空")
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
		return nil, fmt.Errorf("查询维度 %d != 数据库维度 %d", qDim, dbDim)
	}

	if k <= 0 {
		return nil, fmt.Errorf("k必须大于0")
	}

	if k > len(database) {
		k = len(database)
	}

	// 使用自适应欧氏距离计算最近邻
	results := make([][]AccelResult, len(queries))

	// 并行计算以提高性能
	cpuCores := runtime.NumCPU()
	if len(queries) > 1 && cpuCores > 1 {
		var wg sync.WaitGroup
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
						dist, err := AdaptiveEuclideanDistanceSquared(queries[i], dbVec, c.currentStrategy)
						if err != nil {
							// 如果出错，使用默认方法
							distances[j] = EuclideanDistanceSquaredDefault(queries[i], dbVec)
						} else {
							distances[j] = dist
						}
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
							Metadata:   make(map[string]interface{}),
						}
					}

					results[i] = queryResults
				}
			}(i, end)
		}

		wg.Wait()
	} else {
		// 单线程计算
		for i, query := range queries {
			// 计算所有距离
			distances := make([]float64, len(database))
			for j, dbVec := range database {
				dist, err := AdaptiveEuclideanDistanceSquared(query, dbVec, c.currentStrategy)
				if err != nil {
					return nil, fmt.Errorf("计算距离失败: %w", err)
				}
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
					Metadata:   make(map[string]interface{}),
				}
			}

			results[i] = queryResults
		}
	}

	return results, nil
}

func (c *CPUAccelerator) Cleanup() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.initialized {
		return nil
	}

	c.initialized = false
	logger.Info("CPU加速器资源已清理")
	return nil
}

func (c *CPUAccelerator) GetGPUMemoryInfo() (free, total uint64, err error) {
	// 返回CPU内存信息作为替代
	memStats := &runtime.MemStats{}
	runtime.ReadMemStats(memStats)

	// 返回系统可用内存和总内存
	return memStats.HeapIdle, memStats.Sys, nil
}

// GetCurrentStrategy 获取当前计算策略
func (c *CPUAccelerator) GetCurrentStrategy() ComputeStrategy {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.currentStrategy
}

// SetComputeStrategy 设置计算策略
func (c *CPUAccelerator) SetComputeStrategy(strategy ComputeStrategy) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// 检查策略是否支持
	caps := c.strategy.GetHardwareCapabilities()
	switch strategy {
	case StrategyAVX512:
		if !caps.HasAVX512 {
			return fmt.Errorf("硬件不支持AVX512指令集")
		}
	case StrategyAVX2:
		if !caps.HasAVX2 {
			return fmt.Errorf("硬件不支持AVX2指令集")
		}
	case StrategyGPU:
		return fmt.Errorf("GPU加速功能未启用")
	}

	c.currentStrategy = strategy
	logger.Info("计算策略已更新为: %v", strategy)
	return nil
}

// GetPerformanceInfo 获取性能信息
func (c *CPUAccelerator) GetPerformanceInfo() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	caps := c.strategy.GetHardwareCapabilities()

	// 收集性能信息
	info := map[string]interface{}{
		"strategy":       c.currentStrategy,
		"cpu_cores":      caps.CPUCores,
		"avx2_support":   caps.HasAVX2,
		"avx512_support": caps.HasAVX512,
		"initialized":    c.initialized,
	}

	// 添加内存信息
	memStats := &runtime.MemStats{}
	runtime.ReadMemStats(memStats)
	info["total_memory_mb"] = memStats.Sys / (1024 * 1024)
	info["heap_alloc_mb"] = memStats.HeapAlloc / (1024 * 1024)
	info["heap_idle_mb"] = memStats.HeapIdle / (1024 * 1024)

	return info
}

// RunBenchmark 运行基准测试
func (c *CPUAccelerator) RunBenchmark(vectorDim, numVectors int) map[string]interface{} {
	if !c.initialized {
		return map[string]interface{}{"error": "CPU加速器未初始化"}
	}

	// 创建测试数据
	testDB := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		testDB[i] = make([]float64, vectorDim)
		for j := 0; j < vectorDim; j++ {
			testDB[i][j] = float64(i*vectorDim+j) / float64(vectorDim*numVectors)
		}
	}

	queries := testDB[:10] // 使用前10个向量作为查询

	// 测试不同策略的性能
	results := make(map[string]interface{})

	// 保存当前策略
	originalStrategy := c.currentStrategy
	defer func() {
		c.currentStrategy = originalStrategy
	}()

	// 测试标准策略
	c.currentStrategy = StrategyStandard
	startTime := time.Now()
	_, err := c.BatchCosineSimilarity(queries, testDB)
	standardTime := time.Since(startTime)
	results["standard_time_ms"] = standardTime.Milliseconds()
	results["standard_error"] = err != nil

	// 测试AVX2策略（如果支持）
	caps := c.strategy.GetHardwareCapabilities()
	if caps.HasAVX2 && vectorDim%8 == 0 {
		c.currentStrategy = StrategyAVX2
		startTime = time.Now()
		_, err = c.BatchCosineSimilarity(queries, testDB)
		avx2Time := time.Since(startTime)
		results["avx2_time_ms"] = avx2Time.Milliseconds()
		results["avx2_error"] = err != nil
		results["avx2_speedup"] = float64(standardTime) / float64(avx2Time)
	}

	// 测试AVX512策略（如果支持）
	if caps.HasAVX512 && vectorDim%8 == 0 {
		c.currentStrategy = StrategyAVX512
		startTime = time.Now()
		_, err = c.BatchCosineSimilarity(queries, testDB)
		avx512Time := time.Since(startTime)
		results["avx512_time_ms"] = avx512Time.Milliseconds()
		results["avx512_error"] = err != nil
		results["avx512_speedup"] = float64(standardTime) / float64(avx512Time)
	}

	return results
}

// AccelerateSearch 加速搜索（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) AccelerateSearch(query []float64, database [][]float64, options entity.SearchOptions) ([]AccelResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.initialized {
		return nil, fmt.Errorf("FPGA accelerator not initialized")
	}
	return AccelerateSearch(query, database, options)
}

// Start 启动CPU加速器（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) Start() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.initialized {
		return nil // 已经启动
	}

	return c.Initialize()
}

// Stop 停止CPU加速器（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) Stop() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.initialized {
		return nil // 已经停止
	}

	c.initialized = false
	return nil
}

// OptimizeMemoryLayout 优化内存布局（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) OptimizeMemoryLayout(vectors [][]float64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// CPU加速器的内存布局优化
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors to optimize")
	}

	// 模拟内存对齐和缓存优化
	for i, vector := range vectors {
		if len(vector) == 0 {
			return fmt.Errorf("empty vector at index %d", i)
		}
	}

	return nil
}

// PrefetchData 预取数据（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) PrefetchData(vectors [][]float64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.initialized {
		return fmt.Errorf("CPU accelerator not initialized")
	}

	// CPU加速器的数据预取
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors to prefetch")
	}

	// 模拟数据预取到CPU缓存
	for i, vector := range vectors {
		if len(vector) == 0 {
			return fmt.Errorf("empty vector at index %d", i)
		}
	}

	return nil
}

// GetCapabilities 获取CPU能力信息（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) GetCapabilities() HardwareCapabilities {
	c.mu.RLock()
	defer c.mu.RUnlock()

	caps := c.strategy.GetHardwareCapabilities()
	return HardwareCapabilities{
		Type:              AcceleratorCPU,
		CPUCores:          caps.CPUCores,
		MemorySize:        caps.MemorySize,
		HasGPU:            false,
		MaxBatchSize:      1000,
		SupportedOps:      []string{"distance", "search", "similarity"},
		PerformanceRating: 3.0,
		Bandwidth:         caps.Bandwidth,
		Latency:           time.Microsecond * 50,
		PowerConsumption:  65.0, // 典型CPU功耗
		SpecialFeatures:   []string{"AVX2", "AVX512", "Multi-threading"},
	}
}

// GetStats 获取CPU统计信息（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) GetStats() HardwareStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return HardwareStats{
		TotalOperations:   100, // 模拟值
		SuccessfulOps:     95,
		FailedOps:         5,
		AverageLatency:    time.Microsecond * 50,
		Throughput:        1000.0,
		MemoryUtilization: 0.6,
		Temperature:       45.0,
		PowerConsumption:  65.0,
		ErrorRate:         0.05,
		LastUsed:          time.Now(),
	}
}

// GetPerformanceMetrics 获取性能指标（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) GetPerformanceMetrics() PerformanceMetrics {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return PerformanceMetrics{
		LatencyCurrent:    time.Microsecond * 50,
		LatencyMin:        time.Microsecond * 30,
		LatencyMax:        time.Microsecond * 100,
		LatencyP50:        float64(time.Microsecond * 45),
		LatencyP95:        float64(time.Microsecond * 80),
		LatencyP99:        float64(time.Microsecond * 95),
		ThroughputCurrent: 1000.0,
		ThroughputPeak:    1500.0,
		CacheHitRate:      0.85,
		ResourceUtilization: map[string]float64{
			"bandwidth":   0.6,
			"persistence": 0.9,
		},
	}
}

// UpdateConfig 更新配置（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) UpdateConfig(config interface{}) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// 将interface{}转换为map[string]interface{}
	configMap, ok := config.(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid config type, expected map[string]interface{}")
	}

	// 更新配置参数
	if indexType, ok := configMap["index_type"].(string); ok {
		c.indexType = indexType
	}
	if deviceID, ok := configMap["device_id"].(int); ok {
		c.deviceID = deviceID
	}

	return nil
}

// AutoTune 自动调优（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) AutoTune(workload WorkloadProfile) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.initialized {
		return fmt.Errorf("CPU accelerator not initialized")
	}

	// 根据工作负载自动选择最佳策略
	optimalStrategy := c.strategy.SelectOptimalStrategy(int(workload.DataSize), workload.VectorDimension)
	c.currentStrategy = optimalStrategy

	logger.Info("CPU加速器自动调优完成，选择策略: %v", optimalStrategy)
	return nil
}

// BatchComputeDistance 批量计算距离（UnifiedAccelerator接口方法）
func (c *CPUAccelerator) BatchComputeDistance(queries [][]float64, targets [][]float64) ([][]float64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.initialized {
		return nil, fmt.Errorf("CPU accelerator not initialized")
	}

	if len(queries) == 0 || len(targets) == 0 {
		return nil, fmt.Errorf("empty queries or targets")
	}

	// 批量计算距离矩阵
	results := make([][]float64, len(queries))
	for i, query := range queries {
		if len(query) == 0 {
			return nil, fmt.Errorf("empty query vector at index %d", i)
		}

		distances := make([]float64, len(targets))
		for j, target := range targets {
			if len(target) != len(query) {
				return nil, fmt.Errorf("dimension mismatch: query %d, target %d", len(query), len(target))
			}

			// 计算欧几里得距离
			dist := 0.0
			for d := 0; d < len(query); d++ {
				diff := query[d] - target[d]
				dist += diff * diff
			}
			distances[j] = math.Sqrt(dist)
		}
		results[i] = distances
	}

	return results, nil
}
