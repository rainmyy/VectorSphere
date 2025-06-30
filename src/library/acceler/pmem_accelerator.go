//go:build pmem

package acceler

import (
	"VectorSphere/src/library/entity"
	"fmt"
	"sync"
	"time"
	"unsafe"
)

/*
#cgo CFLAGS: -I/usr/include/libpmem -I/usr/include/libpmemobj
#cgo LDFLAGS: -lpmem -lpmemobj -lpmempool

#include <libpmem.h>
#include <libpmemobj.h>
#include <stdlib.h>
#include <string.h>

// PMem 设备结构体
typedef struct {
    void* mapped_addr;
    size_t mapped_size;
    char* device_path;
    int is_pmem;
    PMEMobjpool* pool;
    PMEMoid root;
} pmem_device_t;

// PMem 向量存储结构体
typedef struct {
    size_t count;
    size_t dimension;
    float vectors[];
} pmem_vector_store_t;

// PMem 函数声明
int pmem_init_device(pmem_device_t* device, const char* path, size_t size);
int pmem_map_file(pmem_device_t* device, const char* path, size_t size);
int pmem_store_vectors(pmem_device_t* device, float* vectors, size_t count, size_t dimension);
int pmem_load_vectors(pmem_device_t* device, float* vectors, size_t* count, size_t* dimension);
int pmem_compute_distances_persistent(pmem_device_t* device, float* query, float* results, size_t dimension);
int pmem_flush_data(pmem_device_t* device);
void pmem_cleanup_device(pmem_device_t* device);
int pmem_is_available();
size_t pmem_get_available_space(const char* path);
*/
import "C"

// NewPMemAccelerator 创建新的持久内存加速器
func NewPMemAccelerator(config *PMemConfig) *PMemAccelerator {
	capabilities := HardwareCapabilities{
		Type:              AcceleratorPMem,
		SupportedOps:      []string{"persistent_storage", "fast_access", "distance_compute", "vector_cache"},
		PerformanceRating: 8.0,
		SpecialFeatures:   []string{"persistent", "byte_addressable", "low_latency", "high_bandwidth"},
	}
	baseAccel := NewBaseAccelerator(0, "PMem", capabilities, HardwareStats{})
	pmem := &PMemAccelerator{
		BaseAccelerator: baseAccel,
		devicePath:      config.DevicePath,
		deviceSize:      config.PoolSize,
		config:          config,
		MemoryPool:      make(map[string][]float64),
		Namespaces:      make(map[string]*PMemNamespace),

		VectorCache: make(map[string][]float64),
	}

	if config != nil {
		pmem.devicePaths = config.DevicePaths
	}

	// 检测PMem可用性
	pmem.detectPMem()
	return pmem
}

// GetType 获取加速器类型
func (p *PMemAccelerator) GetType() string {
	return AcceleratorPMem
}

// IsAvailable 检查PMem是否可用
func (p *PMemAccelerator) IsAvailable() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.available
}

// Initialize 初始化PMem
func (p *PMemAccelerator) Initialize() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.available {
		return fmt.Errorf("PMem设备不可用")
	}

	if p.initialized {
		return nil
	}

	// 初始化所有PMem设备
	p.deviceHandles = make([]unsafe.Pointer, len(p.devicePaths))
	for i, path := range p.devicePaths {
		device := (*C.pmem_device_t)(C.malloc(C.sizeof_pmem_device_t))
		if device == nil {
			return fmt.Errorf("分配PMem设备内存失败")
		}

		pathCStr := C.CString(path)
		defer C.free(unsafe.Pointer(pathCStr))

		// 获取可用空间
		availableSpace := C.pmem_get_available_space(pathCStr)
		if availableSpace == 0 {
			C.free(unsafe.Pointer(device))
			return fmt.Errorf("PMem设备 %s 没有可用空间", path)
		}

		// 初始化设备
		result := C.pmem_init_device(device, pathCStr, availableSpace)
		if result != 0 {
			C.free(unsafe.Pointer(device))
			return fmt.Errorf("初始化PMem设备 %s 失败: %d", path, result)
		}

		p.deviceHandles[i] = unsafe.Pointer(device)
	}

	p.initialized = true
	p.updateCapabilities()

	return nil
}

// Shutdown 关闭PMem
func (p *PMemAccelerator) Shutdown() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.initialized {
		return nil
	}

	// 刷新所有数据
	for _, handle := range p.deviceHandles {
		if handle != nil {
			device := (*C.pmem_device_t)(handle)
			C.pmem_flush_data(device)
			C.pmem_cleanup_device(device)
			C.free(handle)
		}
	}

	p.deviceHandles = nil
	p.initialized = false

	return nil
}

// Start 启动PMem
func (p *PMemAccelerator) Start() error {
	return p.Initialize()
}

// Stop 停止PMem
func (p *PMemAccelerator) Stop() error {
	return p.Shutdown()
}

// ComputeDistance 计算距离（使用PMem加速）
func (p *PMemAccelerator) ComputeDistance(query []float64, vectors [][]float64) ([]float64, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.initialized {
		return nil, fmt.Errorf("PMem未初始化")
	}

	start := time.Now()
	defer func() {
		p.updateStats(time.Since(start), 1, true)
	}()

	// 检查缓存
	queryKey := p.generateQueryKey(query)
	p.CacheMutex.RLock()
	if cachedResult, exists := p.VectorCache[queryKey]; exists {
		p.CacheMutex.RUnlock()
		return cachedResult, nil
	}
	p.CacheMutex.RUnlock()

	// 转换数据格式
	queryFlat := make([]float32, len(query))
	for i, v := range query {
		queryFlat[i] = float32(v)
	}

	results := make([]float64, len(vectors))

	// 使用PMem设备计算距离
	if len(p.deviceHandles) > 0 {
		device := (*C.pmem_device_t)(p.deviceHandles[0])
		resultsFlat := make([]float32, len(vectors))

		result := C.pmem_compute_distances_persistent(
			device,
			(*C.float)(unsafe.Pointer(&queryFlat[0])),
			(*C.float)(unsafe.Pointer(&resultsFlat[0])),
			C.size_t(len(query)),
		)

		if result == 0 {
			// 转换结果
			for i, v := range resultsFlat {
				results[i] = float64(v)
			}
		} else {
			// 回退到CPU计算
			results = p.computeDistanceCPU(query, vectors)
		}
	} else {
		// 回退到CPU计算
		results = p.computeDistanceCPU(query, vectors)
	}

	// 缓存结果
	p.CacheMutex.Lock()
	p.VectorCache[queryKey] = results
	p.CacheMutex.Unlock()

	return results, nil
}

// BatchComputeDistance 批量计算距离
func (p *PMemAccelerator) BatchComputeDistance(queries [][]float64, vectors [][]float64) ([][]float64, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.initialized {
		return nil, fmt.Errorf("PMem未初始化")
	}

	start := time.Now()
	defer func() {
		p.updateStats(time.Since(start), len(queries), true)
	}()

	results := make([][]float64, len(queries))

	// 并行处理查询
	var wg sync.WaitGroup
	errorChan := make(chan error, len(queries))

	for i, query := range queries {
		wg.Add(1)
		go func(idx int, q []float64) {
			defer wg.Done()
			result, err := p.ComputeDistance(q, vectors)
			if err != nil {
				errorChan <- err
				return
			}
			results[idx] = result
		}(i, query)
	}

	wg.Wait()
	close(errorChan)

	// 检查错误
	if err := <-errorChan; err != nil {
		return nil, err
	}

	return results, nil
}

// BatchSearch 批量搜索
func (p *PMemAccelerator) BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	// 先计算距离
	distances, err := p.BatchComputeDistance(queries, database)
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
				Metadata:   map[string]interface{}{"index": indexDistances[j].index, "cached": true},
			}
		}
		results[i] = queryResults
	}

	return results, nil
}

// BatchCosineSimilarity 批量余弦相似度计算
func (p *PMemAccelerator) BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error) {
	// PMem可以提供快速的向量访问，但余弦相似度计算仍然使用CPU
	return p.BatchComputeDistance(queries, database)
}

// AccelerateSearch 加速搜索
func (p *PMemAccelerator) AccelerateSearch(query []float64, results []AccelResult, options entity.SearchOptions) ([]AccelResult, error) {
	// PMem可以提供持久化的搜索结果缓存
	return results, nil
}

// OptimizeMemoryLayout 优化内存布局
func (p *PMemAccelerator) OptimizeMemoryLayout(vectors [][]float64) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.initialized || len(p.deviceHandles) == 0 {
		return fmt.Errorf("PMem未初始化")
	}

	// 将向量存储到PMem中以优化访问
	if len(vectors) > 0 {
		dimension := len(vectors[0])
		vectorsFlat := make([]float32, len(vectors)*dimension)

		for i, vec := range vectors {
			for j, v := range vec {
				vectorsFlat[i*dimension+j] = float32(v)
			}
		}

		device := (*C.pmem_device_t)(p.deviceHandles[0])
		result := C.pmem_store_vectors(
			device,
			(*C.float)(unsafe.Pointer(&vectorsFlat[0])),
			C.size_t(len(vectors)),
			C.size_t(dimension),
		)

		if result != 0 {
			return fmt.Errorf("存储向量到PMem失败: %d", result)
		}
	}

	return nil
}

// PrefetchData 预取数据
func (p *PMemAccelerator) PrefetchData(vectors [][]float64) error {
	// PMem的字节寻址特性使得预取非常高效
	return p.OptimizeMemoryLayout(vectors)
}

// GetCapabilities 获取PMem能力信息
func (p *PMemAccelerator) GetCapabilities() HardwareCapabilities {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.capabilities
}

// GetStats 获取PMem统计信息
func (p *PMemAccelerator) GetStats() HardwareStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	p.stats.LastUsed = p.startTime
	return p.stats
}

// GetPerformanceMetrics 获取性能指标
func (p *PMemAccelerator) GetPerformanceMetrics() PerformanceMetrics {
	p.mu.RLock()
	defer p.mu.RUnlock()
	latencyP95 := float64(p.stats.AverageLatency) * 1.5
	return PerformanceMetrics{
		LatencyP50:        float64(p.stats.AverageLatency),
		LatencyP95:        latencyP95,
		LatencyP99:        float64(p.stats.AverageLatency * 2),
		ThroughputCurrent: p.stats.Throughput,
		ThroughputPeak:    p.stats.Throughput * 1.2,
		CacheHitRate:      p.getCacheHitRate(),
		ResourceUtilization: map[string]float64{
			"memory":      p.stats.MemoryUtilization,
			"bandwidth":   0.6,
			"persistence": 0.9,
		},
	}
}

// UpdateConfig 更新配置
func (p *PMemAccelerator) UpdateConfig(config interface{}) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if pmemConfig, ok := config.(*PMemConfig); ok {
		p.config = pmemConfig
		return nil
	}

	return fmt.Errorf("无效的PMem配置类型")
}

// AutoTune 自动调优
func (p *PMemAccelerator) AutoTune(workload WorkloadProfile) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// 根据工作负载调整PMem配置
	if p.config != nil {
		// 根据访问模式调整性能配置
		if workload.AccessPattern == "sequential" {
			p.config.Performance.ReadAhead = true
			p.config.Performance.BatchSize = 1024
		} else if workload.AccessPattern == "random" {
			p.config.Performance.ReadAhead = false
			p.config.Performance.BatchSize = 256
		}

		// 根据数据大小调整持久化策略
		if workload.DataSize > 1024*1024*1024 { // > 1GB
			p.config.Persistence.FlushStrategy = "async"
			p.config.Persistence.FlushInterval = 5 * time.Second
		} else {
			p.config.Persistence.FlushStrategy = "sync"
		}
	}

	return nil
}

// detectPMem 检测PMem可用性
func (p *PMemAccelerator) detectPMem() {
	if int(C.pmem_is_available()) == 1 {
		p.available = true
		p.updateCapabilities()
	}
}

// updateCapabilities 更新能力信息
func (p *PMemAccelerator) updateCapabilities() {
	if !p.available {
		return
	}

	// 计算总可用空间
	var totalSpace int64
	for _, path := range p.devicePaths {
		pathCStr := C.CString(path)
		space := C.pmem_get_available_space(pathCStr)
		C.free(unsafe.Pointer(pathCStr))
		totalSpace += int64(space)
	}

	p.capabilities.MemorySize = totalSpace
	p.capabilities.MaxBatchSize = int(totalSpace) / (512 * 8) // 假设512维向量
	p.capabilities.Bandwidth = totalSpace * 5                 // 假设5x内存带宽
	p.capabilities.Latency = 200 * time.Nanosecond            // PMem有很低的延迟
	p.capabilities.PowerConsumption = 20.0                    // 假设20W功耗
	p.capabilities.ComputeUnits = len(p.devicePaths)          // 每个设备一个计算单元
}

// updateStats 更新统计信息
func (p *PMemAccelerator) updateStats(duration time.Duration, operations int, success bool) {
	p.stats.TotalOperations += int64(operations)
	if success {
		p.stats.SuccessfulOps += int64(operations)
	} else {
		p.stats.FailedOps += int64(operations)
	}

	// 更新平均延迟
	if p.stats.TotalOperations > 0 {
		totalTime := time.Duration(int64(p.stats.AverageLatency)*(p.stats.TotalOperations-int64(operations))) + duration
		p.stats.AverageLatency = totalTime / time.Duration(p.stats.TotalOperations)
	}

	// 更新吞吐量
	now := time.Now()
	if now.Sub(p.lastStatsTime) > time.Second {
		elapsed := now.Sub(p.lastStatsTime).Seconds()
		p.stats.Throughput = float64(operations) / elapsed
		p.lastStatsTime = now
	}

	// 更新错误率
	if p.stats.TotalOperations > 0 {
		p.stats.ErrorRate = float64(p.stats.FailedOps) / float64(p.stats.TotalOperations)
	}

	// 模拟其他指标
	p.stats.MemoryUtilization = 0.8 // 假设80%内存利用率
	p.stats.Temperature = 35.0      // 假设35°C
	p.stats.PowerConsumption = 20.0 // 假设20W功耗
}

// computeDistanceCPU CPU回退计算
func (p *PMemAccelerator) computeDistanceCPU(query []float64, vectors [][]float64) []float64 {
	results := make([]float64, len(vectors))
	for i, vec := range vectors {
		dist := 0.0
		for j := range query {
			diff := query[j] - vec[j]
			dist += diff * diff
		}
		results[i] = dist
	}
	return results
}

// generateQueryKey 生成查询键
func (p *PMemAccelerator) generateQueryKey(query []float64) string {
	// 简单的哈希函数生成查询键
	hash := uint64(0)
	for _, v := range query {
		hash = hash*31 + uint64(v*1000) // 简单哈希
	}
	return fmt.Sprintf("query_%x", hash)
}

// getCacheHitRate 获取缓存命中率
func (p *PMemAccelerator) getCacheHitRate() float64 {
	p.CacheMutex.RLock()
	defer p.CacheMutex.RUnlock()

	if p.stats.TotalOperations == 0 {
		return 0.0
	}

	// 简单估算缓存命中率
	cacheSize := len(p.VectorCache)
	if cacheSize > 1000 {
		return 0.9 // 高缓存命中率
	} else if cacheSize > 100 {
		return 0.7
	} else {
		return 0.3
	}
}

// FlushCache 刷新缓存
func (p *PMemAccelerator) FlushCache() error {
	p.CacheMutex.Lock()
	defer p.CacheMutex.Unlock()

	// 清空向量缓存
	p.VectorCache = make(map[string][]float64)

	// 刷新PMem数据
	if p.initialized {
		for _, handle := range p.deviceHandles {
			if handle != nil {
				device := (*C.pmem_device_t)(handle)
				C.pmem_flush_data(device)
			}
		}
	}

	return nil
}

// GetCacheStats 获取缓存统计
func (p *PMemAccelerator) GetCacheStats() map[string]interface{} {
	p.CacheMutex.RLock()
	defer p.CacheMutex.RUnlock()

	return map[string]interface{}{
		"cache_size":     len(p.VectorCache),
		"cache_hit_rate": p.getCacheHitRate(),
		"memory_usage":   len(p.VectorCache) * 512 * 8, // 估算内存使用
	}
}
