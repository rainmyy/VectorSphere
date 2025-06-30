package test

import (
	"VectorSphere/src/library/acceler"
	"testing"
	"unsafe"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

//go:generate mockgen -source=gpu_accelerator.go -destination=../mocks/mock_gpu_accelerator.go

func TestGPUAcceleratorInitializeSuccess(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	// Create GPU accelerator
	gpu := acceler.NewGPUAccelerator(0)
	
	// Mock successful GPU device count check
	mockDeviceCount := 1
	
	// Mock successful device properties
	mockProp := struct {
		name  [256]byte
		major int
		minor int
		computeMode int
	}{
		major: 7,
		minor: 5,
		computeMode: 0,
	}
	copy(mockProp.name[:], "Tesla V100")
	
	// Mock successful memory allocation test

	// Mock FAISS resources creation
	mockGpuResources := unsafe.Pointer(uintptr(0x87654321))
	mockGpuWrapper := unsafe.Pointer(uintptr(0x11223344))
	
	// Mock memory info
	mockFree := uint64(4 * 1024 * 1024 * 1024) // 4GB
	mockTotal := uint64(8 * 1024 * 1024 * 1024) // 8GB
	
	// Set up the GPU accelerator state to simulate successful initialization
	gpu.DeviceCount = mockDeviceCount
	gpu.GpuResources = mockGpuResources
	gpu.GpuWrapper = mockGpuWrapper
	gpu.MemoryTotal = int64(mockTotal)
	gpu.MemoryUsed = int64(mockTotal - mockFree)
	gpu.Dimension = 512
	
	// Test initialization
	err := gpu.Initialize()
	
	// Assertions
	assert.NoError(t, err)
	assert.True(t, gpu.Initialized)
	assert.True(t, gpu.Available)
	assert.Equal(t, mockDeviceCount, gpu.DeviceCount)
	assert.NotNil(t, gpu.GpuResources)
	assert.NotNil(t, gpu.GpuWrapper)
}

func TestBatchSearchGPUSuccess(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	// Create and initialize GPU accelerator
	gpu := acceler.NewGPUAccelerator(0)
	gpu.Initialized = true
	gpu.Available = true
	gpu.GpuWrapper = unsafe.Pointer(uintptr(0x11223344))
	gpu.BatchSize = 1000
	
	// Create test data
	queries := [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
	}
	database := [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
		{9.0, 10.0, 11.0, 12.0},
	}
	k := 2
	
	// Mock GPU memory info for batch size calculation
	gpu.MemoryTotal = 8 * 1024 * 1024 * 1024 // 8GB
	gpu.MemoryUsed = 2 * 1024 * 1024 * 1024  // 2GB used
	
	// Execute batch search
	results, err := gpu.BatchSearch(queries, database, k)
	
	// Assertions
	assert.NoError(t, err)
	assert.Len(t, results, len(queries))
	for i, queryResults := range results {
		assert.Len(t, queryResults, k, "Query %d should return %d results", i, k)
		for j, result := range queryResults {
			assert.NotEmpty(t, result.ID, "Result %d for query %d should have ID", j, i)
			assert.GreaterOrEqual(t, result.Similarity, 0.0, "Similarity should be non-negative")
			assert.GreaterOrEqual(t, result.Distance, 0.0, "Distance should be non-negative")
			assert.NotNil(t, result.Metadata, "Metadata should not be nil")
		}
	}
}

func TestGPUMemoryInfoAndBatchSizeOptimization(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	// Create GPU accelerator
	gpu := acceler.NewGPUAccelerator(0)
	gpu.Initialized = true
	gpu.Available = true
	gpu.BatchSize = 1000
	
	// Mock memory info
	mockFree := uint64(4 * 1024 * 1024 * 1024) // 4GB free
	mockTotal := uint64(8 * 1024 * 1024 * 1024) // 8GB total
	
	gpu.MemoryTotal = int64(mockTotal)
	gpu.MemoryUsed = int64(mockTotal - mockFree)
	
	// Test memory info retrieval
	free, total, err := gpu.GetGPUMemoryInfo()
	
	// Assertions for memory info
	assert.NoError(t, err)
	assert.Equal(t, mockFree, free)
	assert.Equal(t, mockTotal, total)
	
	// Test optimal batch size calculation
	vectorDim := 128
	numQueries := 1000
	
	optimalBatch := gpu.SelectOptimalBatchSize(vectorDim, numQueries)
	
	// Assertions for batch size optimization
	assert.Greater(t, optimalBatch, 0, "Optimal batch size should be positive")
	assert.LessOrEqual(t, optimalBatch, numQueries, "Optimal batch size should not exceed number of queries")
	assert.GreaterOrEqual(t, optimalBatch, 100, "Optimal batch size should be at least 100")
	assert.LessOrEqual(t, optimalBatch, 2000, "Optimal batch size should not exceed 2000")
}

func TestGPUAcceleratorInitializeNoDevicesAvailable(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	// Create GPU accelerator
	gpu := acceler.NewGPUAccelerator(0)
	
	// Simulate no GPU devices available
	gpu.DeviceCount = 0
	
	// Test initialization
	err := gpu.Initialize()
	
	// Assertions
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "未检测到GPU设备")
	assert.False(t, gpu.Initialized)
}

func TestBatchSearchGPUFailureFallbackToCPU(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	// Create and initialize GPU accelerator
	gpu := acceler.NewGPUAccelerator(0)
	gpu.Initialized = true
	gpu.Available = true
	gpu.GpuWrapper = nil // Simulate GPU wrapper failure
	
	// Create test data
	queries := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}
	database := [][]float64{
		{1.0, 2.0, 3.0},
		{7.0, 8.0, 9.0},
		{10.0, 11.0, 12.0},
	}
	k := 2
	
	// Execute batch search (should fallback to CPU)
	results, err := gpu.BatchSearch(queries, database, k)
	
	// Assertions
	assert.NoError(t, err, "CPU fallback should succeed")
	assert.Len(t, results, len(queries))
	for i, queryResults := range results {
		assert.Len(t, queryResults, k, "Query %d should return %d results", i, k)
		for j, result := range queryResults {
			assert.NotEmpty(t, result.ID, "Result %d for query %d should have ID", j, i)
			assert.GreaterOrEqual(t, result.Similarity, 0.0, "Similarity should be non-negative")
			assert.GreaterOrEqual(t, result.Distance, 0.0, "Distance should be non-negative")
		}
	}
	
	// Verify that we got reasonable results from CPU computation
	assert.Greater(t, len(results), 0, "Should have results from CPU fallback")
}

func TestBatchSearchMismatchedVectorDimensions(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	// Create and initialize GPU accelerator
	gpu := acceler.NewGPUAccelerator(0)
	gpu.Initialized = true
	gpu.Available = true
	
	// Create test data with mismatched dimensions
	queries := [][]float64{
		{1.0, 2.0, 3.0}, // 3 dimensions
	}
	database := [][]float64{
		{1.0, 2.0, 3.0, 4.0}, // 4 dimensions - mismatch!
		{5.0, 6.0, 7.0, 8.0},
	}
	k := 1
	
	// Execute batch search
	results, err := gpu.BatchSearch(queries, database, k)
	
	// Assertions
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "查询向量维度")
	assert.Contains(t, err.Error(), "数据库向量维度")
	assert.Nil(t, results)
}