//go:build !gpu

package acceler

import "C"
import (
	"VectorSphere/src/library/log"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"
)

func NewFAISSAccelerator(deviceID int, indexType string) *FAISSAccelerator {
	return &FAISSAccelerator{
		deviceID:    deviceID,
		indexType:   indexType,
		strategy:    NewComputeStrategySelector(),
		initialized: false,
	}
}

func (c *FAISSAccelerator) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.initialized {
		return nil
	}

	// 检测CPU硬件能力
	caps := c.strategy.GetHardwareCapabilities()
	log.Info("CPU加速器初始化: 检测到 %d 核心, AVX2: %v, AVX512: %v",
		caps.CPUCores, caps.HasAVX2, caps.HasAVX512)

	// 选择最佳计算策略
	c.currentStrategy = c.strategy.SelectOptimalStrategy(1000, 512) // 默认参数

	// 根据检测到的硬件能力设置最佳策略
	if caps.HasAVX512 {
		log.Info("启用 AVX512 加速")
		c.currentStrategy = StrategyAVX512
	} else if caps.HasAVX2 {
		log.Info("启用 AVX2 加速")
		c.currentStrategy = StrategyAVX2
	} else {
		log.Info("使用标准计算方法")
		c.currentStrategy = StrategyStandard
	}

	log.Info("CPU加速器初始化完成，使用策略: %v", c.currentStrategy)
	c.initialized = true
	return nil
}

func (c *FAISSAccelerator) BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error) {
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

func (c *FAISSAccelerator) BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
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

func (c *FAISSAccelerator) Cleanup() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.initialized {
		return nil
	}

	c.initialized = false
	log.Info("CPU加速器资源已清理")
	return nil
}

// CheckGPUAvailability 公共方法，供外部调用检查GPU可用性
func (c *FAISSAccelerator) CheckGPUAvailability() error {
	return fmt.Errorf("GPU加速功能未启用，使用CPU加速替代")
}

func (c *FAISSAccelerator) GetGPUMemoryInfo() (free, total uint64, err error) {
	// 返回CPU内存信息作为替代
	memStats := &runtime.MemStats{}
	runtime.ReadMemStats(memStats)

	// 返回系统可用内存和总内存
	return memStats.HeapIdle, memStats.Sys, nil
}

// GetCurrentStrategy 获取当前计算策略
func (c *FAISSAccelerator) GetCurrentStrategy() ComputeStrategy {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.currentStrategy
}

// SetComputeStrategy 设置计算策略
func (c *FAISSAccelerator) SetComputeStrategy(strategy ComputeStrategy) error {
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
	log.Info("计算策略已更新为: %v", strategy)
	return nil
}

// GetPerformanceInfo 获取性能信息
func (c *FAISSAccelerator) GetPerformanceInfo() map[string]interface{} {
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
func (c *FAISSAccelerator) RunBenchmark(vectorDim, numVectors int) map[string]interface{} {
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
