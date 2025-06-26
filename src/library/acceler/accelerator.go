package acceler

import (
	"fmt"
	"sync"
)

// Accelerator GPU 加速器接口
type Accelerator interface {
	Initialize() error
	BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error)
	BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error)
	Cleanup() error
}

// FAISSAccelerator FAISS 加速器实现
type FAISSAccelerator struct {
	deviceID        int
	initialized     bool
	available       bool
	mu              sync.RWMutex
	indexType       string
	dimension       int
	strategy        *ComputeStrategySelector
	currentStrategy ComputeStrategy
	dataSize        int
}

// AccelResult 搜索结果结构体
type AccelResult struct {
	ID         string
	Similarity float64
	Distance   float64
	Metadata   map[string]interface{}
	DocIds     []string
	Vector     []float64
	Index      int
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

// CheckGPUAvailability 检查GPU是否可用
func (c *FAISSAccelerator) CheckGPUAvailability() error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// 检查是否已初始化
	if !c.initialized {
		return fmt.Errorf("GPU加速器未初始化")
	}

	// 检查GPU内存状态
	free, total, err := c.GetGPUMemoryInfo()
	if err != nil {
		return fmt.Errorf("GPU内存检查失败: %v", err)
	}

	// 检查可用内存是否足够（至少100MB）
	minRequiredMem := uint64(100 * 1024 * 1024) // 100MB
	if free < minRequiredMem {
		return fmt.Errorf("GPU内存不足: 可用 %d MB, 总计 %d MB", free/(1024*1024), total/(1024*1024))
	}

	// 检查可用内存比例（至少5%）
	if float64(free)/float64(total) < 0.05 {
		return fmt.Errorf("GPU内存使用率过高: 可用 %.2f%%", float64(free)/float64(total)*100)
	}

	return nil
}

func (c *FAISSAccelerator) SelectOptimalBatchSize(vectorDim, numQueries int) int {
	// 获取 GPU 内存信息
	free, _, err := c.GetGPUMemoryInfo()
	if err != nil {
		// 默认保守值
		return 1000
	}

	// 估算每个向量需要的内存
	bytesPerVector := vectorDim * 4 // float32 = 4 bytes

	// 考虑 FAISS 索引和临时内存开销，只使用可用内存的 70%
	safeMemory := float64(free) * 0.7

	// 计算最大批处理大小
	maxBatchSize := int(safeMemory) / bytesPerVector

	// 限制批处理大小
	if maxBatchSize > numQueries {
		return numQueries
	}
	if maxBatchSize < 10 {
		return 10 // 最小批处理大小
	}

	return maxBatchSize
}

//// 新增 EnableHybridMode 方法
//func (c *FAISSAccelerator) EnableHybridMode() error {
//	c.mu.Lock()
//	defer c.mu.Unlock()
//
//	if !c.initialized {
//		return fmt.Errorf("GPU 加速器未初始化")
//	}
//
//	// 检测 CPU 硬件能力
//	cpuDetector := &HardwareDetector{}
//	cpuCaps := cpuDetector.GetHardwareCapabilities()
//
//	// 设置混合模式
//	c.hybridMode = true
//	c.cpuCapabilities = cpuCaps
//
//	log.Info("已启用混合计算模式 - CPU: %d 核心, AVX2: %v, AVX512: %v",
//		cpuCaps.CPUCores, cpuCaps.HasAVX2, cpuCaps.HasAVX512)
//
//	return nil
//}
//
//// 新增 hybridBatchSearch 方法
//func (c *FAISSAccelerator) hybridBatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
//	// 根据数据特性决定处理方式
//	if len(queries) < 10 || len(database) < 1000 {
//		// 小数据量使用 CPU
//		return c.batchSearchCPUFallback(queries, database, k)
//	}
//
//	// 大数据量使用 GPU
//	return c.batchSearchGPU(queries, database, k)
//}
