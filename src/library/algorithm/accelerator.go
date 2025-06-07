package algorithm

import (
	"fmt"
	"sync"
	"unsafe"
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
	deviceID    int
	initialized bool
	mu          sync.RWMutex
	indexType   string
	gpuWrapper  unsafe.Pointer // C.FaissGpuWrapper*
	dimension   int
}

// AccelResult 搜索结果结构体
type AccelResult struct {
	ID         string
	Similarity float64
	Metadata   map[string]interface{}
	DocIds     []string
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
