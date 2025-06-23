package test

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"VectorSphere/src/vector"
)

// generateTestVectors 生成测试向量
func generateTestVectors(count, dim int) [][]float64 {
	rand.Seed(time.Now().UnixNano())
	vectors := make([][]float64, count)
	for i := 0; i < count; i++ {
		vec := make([]float64, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float64()*2 - 1 // [-1, 1]
		}
		vectors[i] = vec
	}
	return vectors
}

// TestIVFHNSWIndexBasic 测试IVF-HNSW混合索引基本功能
func TestIVFHNSWIndexBasic(t *testing.T) {
	// 创建测试数据库
	db := vector.NewVectorDB("test_db", 10)

	// 添加测试向量
	testVectors := generateTestVectors(100, 128)
	for i, vec := range testVectors {
		id := fmt.Sprintf("vec_%d", i)
		db.Add(id, vec)
	}

	// 初始化IVF-HNSW索引
	err := db.InitializeIVFHNSWIndex()
	if err != nil {
		t.Fatalf("Failed to initialize IVF-HNSW index: %v", err)
	}

	// 执行搜索测试
	queryVector := generateTestVectors(1, 128)[0]
	options := vector.SearchOptions{}
	results, err := db.OptimizedSearch(queryVector, 10, options)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// 验证结果
	if len(results) == 0 {
		t.Error("No search results returned")
	}

	// 验证结果按距离排序
	for i := 1; i < len(results); i++ {
		if results[i-1].Distance > results[i].Distance {
			t.Error("Results not sorted by distance")
		}
	}

	fmt.Printf("Basic test passed: found %d results\n", len(results))
}

// TestIVFHNSWIndexPerformance 测试IVF-HNSW混合索引性能
func TestIVFHNSWIndexPerformance(t *testing.T) {
	// 创建大规模测试数据库
	db := vector.NewVectorDB("perf_test_db", 20)

	// 添加大量测试向量
	testVectors := generateTestVectors(10000, 256)
	start := time.Now()
	for i, vec := range testVectors {
		id := fmt.Sprintf("vec_%d", i)
		db.Add(id, vec)
	}
	addTime := time.Since(start)

	// 初始化IVF-HNSW索引
	start = time.Now()
	err := db.InitializeIVFHNSWIndex()
	if err != nil {
		t.Fatalf("Failed to initialize IVF-HNSW index: %v", err)
	}
	indexTime := time.Since(start)

	// 性能测试
	queryVectors := generateTestVectors(100, 256)
	options := vector.SearchOptions{}
	start = time.Now()
	for _, queryVec := range queryVectors {
		_, err := db.OptimizedSearch(queryVec, 10, options)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}
	searchTime := time.Since(start)

	// 输出性能指标
	fmt.Printf("Performance test results:\n")
	fmt.Printf("Add time: %v\n", addTime)
	fmt.Printf("Index time: %v\n", indexTime)
	fmt.Printf("Search time for 100 queries: %v\n", searchTime)
	fmt.Printf("Average search time: %v\n", searchTime/100)

	// 验证性能要求
	avgSearchTime := searchTime / 100
	if avgSearchTime > 10*time.Millisecond {
		t.Errorf("Average search time too slow: %v", avgSearchTime)
	}
}

// TestIVFHNSWIndexConfiguration 测试IVF-HNSW混合索引配置
func TestIVFHNSWIndexConfiguration(t *testing.T) {
	// 创建测试数据库
	db := vector.NewVectorDB("config_test_db", 15)

	// 添加测试向量
	testVectors := generateTestVectors(500, 64)
	for i, vec := range testVectors {
		id := fmt.Sprintf("vec_%d", i)
		db.Add(id, vec)
	}

	// 测试自定义配置
	// 这里需要根据实际的配置接口进行调整
	err := db.InitializeIVFHNSWIndex()
	if err != nil {
		t.Fatalf("Failed to initialize IVF-HNSW index: %v", err)
	}

	// 验证配置生效
	queryVector := generateTestVectors(1, 64)[0]
	options := vector.SearchOptions{}
	results, err := db.OptimizedSearch(queryVector, 5, options)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("No search results returned")
	}

	fmt.Printf("Configuration test passed: found %d results\n", len(results))
}

// TestIVFHNSWIndexErrorHandling 测试IVF-HNSW混合索引错误处理
func TestIVFHNSWIndexErrorHandling(t *testing.T) {
	// 创建测试数据库
	db := vector.NewVectorDB("error_test_db", 5)

	// 测试空数据库索引初始化
	err := db.InitializeIVFHNSWIndex()
	if err == nil {
		t.Error("Expected error when initializing index on empty database")
	}

	// 添加少量向量
	testVectors := generateTestVectors(5, 32)
	for i, vec := range testVectors {
		id := fmt.Sprintf("vec_%d", i)
		db.Add(id, vec)
	}

	// 测试数据量不足的情况
	err = db.InitializeIVFHNSWIndex()
	if err == nil {
		t.Error("Expected error when initializing index with insufficient data")
	}

	// 添加足够的向量
	moreVectors := generateTestVectors(100, 32)
	for i, vec := range moreVectors {
		id := fmt.Sprintf("more_vec_%d", i)
		db.Add(id, vec)
	}

	// 正常初始化
	err = db.InitializeIVFHNSWIndex()
	if err != nil {
		t.Fatalf("Failed to initialize IVF-HNSW index: %v", err)
	}

	// 测试无效查询向量
	invalidQuery := make([]float64, 16) // 错误的维度
	options := vector.SearchOptions{}
	_, err = db.OptimizedSearch(invalidQuery, 5, options)
	if err == nil {
		t.Error("Expected error when searching with invalid query vector")
	}

	fmt.Println("Error handling test passed")
}