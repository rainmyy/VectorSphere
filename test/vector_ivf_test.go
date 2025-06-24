package test

import (
	"VectorSphere/src/vector"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestIVFConfig 测试IVF配置结构
func TestIVFConfig(t *testing.T) {
	config := &vector.IVFConfig{
		NumClusters:        10,
		TrainingRatio:      0.2,
		RebalanceThreshold: 500,
		UsePQCompression:   true,
		PQSubVectors:       4,
		PQCentroids:        128,
		EnableDynamic:      true,
		MaxClusterSize:     1000,
		MinClusterSize:     5,
	}

	if config.NumClusters != 10 {
		t.Errorf("Expected NumClusters to be 10, got %d", config.NumClusters)
	}
	if config.TrainingRatio != 0.2 {
		t.Errorf("Expected TrainingRatio to be 0.2, got %f", config.TrainingRatio)
	}
	if !config.UsePQCompression {
		t.Error("Expected UsePQCompression to be true")
	}
}

// TestSampleTrainingData 测试训练数据采样功能
func TestSampleTrainingData(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_sample.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 3)
	defer db.Close()

	// 添加测试向量
	testVectors := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
		{10.0, 11.0, 12.0},
		{13.0, 14.0, 15.0},
	}

	for i, vec := range testVectors {
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 注意：sampleTrainingData是未导出方法，无法直接测试
	// 通过构建索引来间接验证采样功能
	config := &vector.IVFConfig{
		NumClusters:      2,   // 减少聚类数量以加快测试
		TrainingRatio:    0.8, // 增加训练比例确保有足够数据
		UsePQCompression: false,
		EnableDynamic:    false, // 禁用动态更新以加快测试
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF index: %v", err)
	}

	// 验证索引构建成功（间接验证采样功能）
	if !db.IsIndexed() {
		t.Error("Expected index to be built")
	}

	// 测试空数据库
	emptyDB := vector.NewVectorDB(filepath.Join(tempDir, "empty.db"), 3)
	defer emptyDB.Close()

	err = emptyDB.BuildEnhancedIVFIndex(nil)
	if err == nil {
		t.Error("Expected error when building index on empty database")
	}
}

// TestBuildEnhancedIVFIndex 测试增强IVF索引构建
func TestBuildEnhancedIVFIndex(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_ivf.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 4)
	defer db.Close()

	// 添加足够的测试向量
	rand.Seed(42)
	for i := 0; i < 50; i++ {
		vec := make([]float64, 4)
		for j := range vec {
			vec[j] = rand.Float64()*10 - 5 // -5到5的随机数
		}
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 测试使用默认配置构建索引
	err := db.BuildEnhancedIVFIndex(nil)
	if err != nil {
		t.Fatalf("Failed to build IVF index with default config: %v", err)
	}

	// 验证索引是否正确构建
	if !db.IsIndexed() {
		t.Error("Database should be marked as indexed")
	}

	// 测试使用自定义配置
	config := &vector.IVFConfig{
		NumClusters:        5,
		TrainingRatio:      0.3,
		RebalanceThreshold: 100,
		UsePQCompression:   true,
		PQSubVectors:       2,
		PQCentroids:        16,
		EnableDynamic:      false,
		MaxClusterSize:     20,
		MinClusterSize:     2,
	}

	err = db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF index with custom config: %v", err)
	}

	// 测试空数据库构建索引
	emptyDB := vector.NewVectorDB(filepath.Join(tempDir, "empty_ivf.db"), 4)
	defer emptyDB.Close()

	err = emptyDB.BuildEnhancedIVFIndex(config)
	if err == nil {
		t.Error("Expected error when building IVF index on empty database")
	}
}

// TestCalculateClusterMetrics 测试聚类指标计算
func TestCalculateClusterMetrics(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_metrics.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 3)
	defer db.Close()

	// 添加测试向量
	testVectors := [][]float64{
		{1.0, 1.0, 1.0},
		{1.1, 1.1, 1.1},
		{1.2, 1.2, 1.2},
		{5.0, 5.0, 5.0},
		{5.1, 5.1, 5.1},
	}

	for i, vec := range testVectors {
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 构建索引
	config := &vector.IVFConfig{
		NumClusters:      2,
		TrainingRatio:    1.0,
		UsePQCompression: false,
		EnableDynamic:    false,
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF index: %v", err)
	}

	// 测试空聚类的指标计算
	emptyCluster := &vector.EnhancedCluster{
		Centroid:  []float64{0, 0, 0},
		VectorIDs: []string{},
	}
	db.CalculateClusterMetrics(emptyCluster)
	if emptyCluster.Metrics.Variance != 0 || emptyCluster.Metrics.Density != 0 || emptyCluster.Metrics.Radius != 0 {
		t.Error("Empty cluster metrics should be zero")
	}

	// 测试nil聚类
	db.CalculateClusterMetrics(nil) // 应该不会panic
}

// TestBuildIVFPQIndex 测试IVF-PQ索引构建
func TestBuildIVFPQIndex(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_pq.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 8) // 8维向量便于PQ分割
	defer db.Close()

	// 添加测试向量
	rand.Seed(42)
	for i := 0; i < 30; i++ {
		vec := make([]float64, 8)
		for j := range vec {
			vec[j] = rand.Float64()*10 - 5
		}
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 构建带PQ压缩的IVF索引
	config := &vector.IVFConfig{
		NumClusters:      3,
		TrainingRatio:    0.8,
		UsePQCompression: true,
		PQSubVectors:     4, // 8维向量分成4个子向量，每个2维
		PQCentroids:      8,
		EnableDynamic:    false,
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF-PQ index: %v", err)
	}

	// 测试PQ压缩禁用的情况
	configNoPQ := &vector.IVFConfig{
		NumClusters:      2,
		UsePQCompression: false,
	}

	// 创建空的增强聚类用于测试
	emptyClusters := []vector.EnhancedCluster{}
	err = db.BuildIVFPQIndex(emptyClusters, configNoPQ)
	if err != nil {
		t.Errorf("BuildIVFPQIndex should not fail when PQ is disabled: %v", err)
	}

	// 测试无效配置
	invalidConfig := &vector.IVFConfig{
		UsePQCompression: true,
		PQSubVectors:     0, // 无效值
		PQCentroids:      0, // 无效值
	}
	err = db.BuildIVFPQIndex(emptyClusters, invalidConfig)
	if err != nil {
		t.Errorf("BuildIVFPQIndex should handle invalid config gracefully: %v", err)
	}
}

// TestEnhancedIVFSearch 测试增强IVF搜索功能
func TestEnhancedIVFSearch(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_search.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 4)
	defer db.Close()

	// 添加测试向量
	testVectors := [][]float64{
		{1.0, 1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0, 2.0},
		{3.0, 3.0, 3.0, 3.0},
		{4.0, 4.0, 4.0, 4.0},
		{5.0, 5.0, 5.0, 5.0},
		{10.0, 10.0, 10.0, 10.0},
		{11.0, 11.0, 11.0, 11.0},
		{12.0, 12.0, 12.0, 12.0},
	}

	for i, vec := range testVectors {
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 构建IVF索引
	config := &vector.IVFConfig{
		NumClusters:      3,
		TrainingRatio:    1.0,
		UsePQCompression: false,
		EnableDynamic:    false,
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF index: %v", err)
	}

	// 测试搜索
	query := []float64{1.5, 1.5, 1.5, 1.5}
	results, err := db.EnhancedIVFSearch(query, 3, 2) // 搜索top-3，探测2个聚类
	if err != nil {
		t.Fatalf("Enhanced IVF search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected at least one search result")
	}

	// 验证结果按相似度排序
	for i := 1; i < len(results); i++ {
		if results[i-1].Similarity < results[i].Similarity {
			t.Error("Results should be sorted by similarity in descending order")
		}
	}

	// 测试没有索引的情况
	noIndexDB := vector.NewVectorDB(filepath.Join(tempDir, "no_index.db"), 4)
	defer noIndexDB.Close()

	// 添加一些向量但不构建索引
	for i, vec := range testVectors[:3] {
		noIndexDB.Add(fmt.Sprintf("%d", i), vec)
	}

	// 应该回退到传统搜索
	results, err = noIndexDB.EnhancedIVFSearch(query, 2, 1)
	if err != nil {
		t.Fatalf("Enhanced IVF search without index should fallback gracefully: %v", err)
	}
}

// TestSearchInCluster 测试单聚类搜索功能
func TestSearchInCluster(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_cluster_search.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 3)
	defer db.Close()

	// 添加测试向量
	testVectors := [][]float64{
		{1.0, 1.0, 1.0},
		{1.1, 1.1, 1.1},
		{5.0, 5.0, 5.0},
		{5.1, 5.1, 5.1},
	}

	for i, vec := range testVectors {
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 构建IVF索引
	config := &vector.IVFConfig{
		NumClusters:      2,
		TrainingRatio:    1.0,
		UsePQCompression: false,
		EnableDynamic:    false,
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF index: %v", err)
	}

	// 测试在有效聚类中搜索
	query := []float64{1.0, 1.0, 1.0}
	results, err := db.SearchInCluster(query, 2, 0) // 在聚类0中搜索
	if err != nil {
		t.Fatalf("SearchInCluster failed: %v", err)
	}

	// 验证结果
	if len(results) == 0 {
		t.Error("Expected at least one result from cluster search")
	}

	// 测试无效聚类ID
	_, err = db.SearchInCluster(query, 2, 999)
	if err == nil {
		t.Error("Expected error for invalid cluster ID")
	}

	// 测试负数聚类ID
	_, err = db.SearchInCluster(query, 2, -1)
	if err == nil {
		t.Error("Expected error for negative cluster ID")
	}
}

// TestSelectCandidateClusters 测试候选聚类选择
func TestSelectCandidateClusters(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_candidates.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 3)
	defer db.Close()

	// 添加测试向量
	for i := 0; i < 20; i++ {
		vec := []float64{float64(i), float64(i), float64(i)}
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 构建IVF索引
	config := &vector.IVFConfig{
		NumClusters:   4,
		TrainingRatio: 1.0,
		EnableDynamic: false,
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF index: %v", err)
	}

	// 测试候选聚类选择
	query := []float64{5.0, 5.0, 5.0}
	candidates := db.SelectCandidateClusters(query, 2)

	if len(candidates) != 2 {
		t.Errorf("Expected 2 candidate clusters, got %d", len(candidates))
	}

	// 测试边界情况
	candidates = db.SelectCandidateClusters(query, 0) // nprobe = 0
	if len(candidates) != 1 {
		t.Errorf("Expected at least 1 candidate cluster when nprobe=0, got %d", len(candidates))
	}

	candidates = db.SelectCandidateClusters(query, 100) // nprobe > 聚类数
	if len(candidates) != 4 {
		t.Errorf("Expected 4 candidate clusters when nprobe > cluster count, got %d", len(candidates))
	}
}

// TestCalculateAdaptiveNprobe 测试自适应nprobe计算
func TestCalculateAdaptiveNprobe(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_adaptive.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 3)
	defer db.Close()

	// 添加测试向量
	for i := 0; i < 15; i++ {
		vec := []float64{float64(i), float64(i), float64(i)}
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 构建IVF索引
	config := &vector.IVFConfig{
		NumClusters:    3,
		TrainingRatio:  1.0,
		MaxClusterSize: 10,
		EnableDynamic:  false,
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF index: %v", err)
	}

	// 测试自适应nprobe计算
	query := []float64{7.0, 7.0, 7.0}
	adaptiveNprobe := db.CalculateAdaptiveNprobe(query, 2)

	if adaptiveNprobe < 1 {
		t.Errorf("Adaptive nprobe should be at least 1, got %d", adaptiveNprobe)
	}
	if adaptiveNprobe > 3 {
		t.Errorf("Adaptive nprobe should not exceed cluster count (3), got %d", adaptiveNprobe)
	}

	// 测试边界情况
	adaptiveNprobe = db.CalculateAdaptiveNprobe(query, 0)
	if adaptiveNprobe < 1 {
		t.Errorf("Adaptive nprobe should be at least 1 when base is 0, got %d", adaptiveNprobe)
	}

	adaptiveNprobe = db.CalculateAdaptiveNprobe(query, 100)
	if adaptiveNprobe > 3 {
		t.Errorf("Adaptive nprobe should not exceed cluster count when base is large, got %d", adaptiveNprobe)
	}
}

// TestCalculateADCDistance 测试ADC距离计算
func TestCalculateADCDistance(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_adc.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 4)
	defer db.Close()

	// 添加测试向量
	for i := 0; i < 20; i++ {
		vec := []float64{float64(i), float64(i), float64(i), float64(i)}
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 构建带PQ的IVF索引
	config := &vector.IVFConfig{
		NumClusters:      2,
		TrainingRatio:    1.0,
		UsePQCompression: true,
		PQSubVectors:     2,
		PQCentroids:      4,
		EnableDynamic:    false,
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF-PQ index: %v", err)
	}

	// 测试ADC距离计算功能（通过搜索间接测试）
	query := []float64{5.0, 5.0, 5.0, 5.0}

	// 执行搜索来间接验证ADC距离计算
	results, err := db.OptimizedSearch(query, 5, vector.SearchOptions{})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected search results, got none")
	}

	// 验证搜索结果的距离是非负数
	for _, result := range results {
		if result.Distance < 0 {
			t.Errorf("Distance should be non-negative, got %f", result.Distance)
		}
	}
}

// TestDynamicClusterUpdates 测试动态聚类更新
func TestDynamicClusterUpdates(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_dynamic.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 3)
	defer db.Close()

	// 添加测试向量
	for i := 0; i < 10; i++ {
		vec := []float64{float64(i), float64(i), float64(i)}
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 测试启用动态更新的配置
	config := &vector.IVFConfig{
		NumClusters:    2,
		TrainingRatio:  1.0,
		EnableDynamic:  true,
		MaxClusterSize: 3, // 设置较小的最大聚类大小以触发重平衡
		MinClusterSize: 1,
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF index with dynamic updates: %v", err)
	}

	// 测试禁用动态更新
	db.StartDynamicClusterUpdates() // 应该检测到动态更新已禁用或索引不存在

	// 测试聚类维护
	// 注意：performClusterMaintenance是未导出方法，无法直接测试
}

// TestFindNearestPQCentroid 测试PQ质心查找功能（通过IVF-PQ索引间接测试）
func TestFindNearestPQCentroid(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_pq_centroid.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 4)
	defer db.Close()

	// 添加测试向量
	for i := 0; i < 20; i++ {
		vec := []float64{float64(i), float64(i), float64(i), float64(i)}
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 构建带PQ的IVF索引来间接测试PQ质心查找
	config := &vector.IVFConfig{
		NumClusters:      2,
		TrainingRatio:    1.0,
		UsePQCompression: true,
		PQSubVectors:     2,
		PQCentroids:      4,
		EnableDynamic:    false,
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF-PQ index: %v", err)
	}

	// 通过搜索来验证PQ质心查找功能正常工作
	query := []float64{5.0, 5.0, 5.0, 5.0}
	results, err := db.EnhancedIVFSearch(query, 3, 1)
	if err != nil {
		t.Fatalf("Enhanced IVF search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected at least one search result")
	}
}

// TestIVFIndexConcurrency 测试IVF索引的并发安全性
func TestIVFIndexConcurrency(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_concurrency.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 3)
	defer db.Close()

	// 添加测试向量
	for i := 0; i < 30; i++ {
		vec := []float64{float64(i), float64(i), float64(i)}
		db.Add(fmt.Sprintf("%d", i), vec)
	}

	// 构建IVF索引
	config := &vector.IVFConfig{
		NumClusters:   3,
		TrainingRatio: 1.0,
		EnableDynamic: false,
	}

	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF index: %v", err)
	}

	// 并发搜索测试
	const numGoroutines = 10
	const numSearches = 5

	done := make(chan bool, numGoroutines)
	errors := make(chan error, numGoroutines*numSearches)

	for i := 0; i < numGoroutines; i++ {
		go func(goroutineID int) {
			defer func() { done <- true }()
			for j := 0; j < numSearches; j++ {
				query := []float64{float64(goroutineID + j), float64(goroutineID + j), float64(goroutineID + j)}
				_, err := db.EnhancedIVFSearch(query, 3, 2)
				if err != nil {
					errors <- err
				}
			}
		}(i)
	}

	// 等待所有goroutine完成
	for i := 0; i < numGoroutines; i++ {
		<-done
	}
	close(errors)

	// 检查是否有错误
	for err := range errors {
		t.Errorf("Concurrent search error: %v", err)
	}
}

// TestIVFIndexPerformance 测试IVF索引性能
func TestIVFIndexPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test_performance.db")
	defer os.RemoveAll(tempDir)

	// 创建测试数据库
	db := vector.NewVectorDB(dbPath, 128) // 高维向量
	defer db.Close()

	// 添加大量测试向量
	const numVectors = 1000
	rand.Seed(42)

	start := time.Now()
	for i := 0; i < numVectors; i++ {
		vec := make([]float64, 128)
		for j := range vec {
			vec[j] = rand.Float64()*10 - 5
		}
		db.Add(fmt.Sprintf("%d", i), vec)
	}
	addTime := time.Since(start)
	t.Logf("Added %d vectors in %v", numVectors, addTime)

	// 构建IVF索引
	config := &vector.IVFConfig{
		NumClusters:      int(math.Sqrt(float64(numVectors))),
		TrainingRatio:    0.1,
		UsePQCompression: true,
		PQSubVectors:     16,
		PQCentroids:      256,
		EnableDynamic:    false,
	}

	start = time.Now()
	err := db.BuildEnhancedIVFIndex(config)
	if err != nil {
		t.Fatalf("Failed to build IVF index: %v", err)
	}
	indexTime := time.Since(start)
	t.Logf("Built IVF index in %v", indexTime)

	// 性能测试搜索
	const numQueries = 100
	queries := make([][]float64, numQueries)
	for i := range queries {
		queries[i] = make([]float64, 128)
		for j := range queries[i] {
			queries[i][j] = rand.Float64()*10 - 5
		}
	}

	start = time.Now()
	for _, query := range queries {
		_, err := db.EnhancedIVFSearch(query, 10, 5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}
	searchTime := time.Since(start)
	avgSearchTime := searchTime / time.Duration(numQueries)
	t.Logf("Performed %d searches in %v (avg: %v per search)", numQueries, searchTime, avgSearchTime)

	// 验证性能指标
	if avgSearchTime > 10*time.Millisecond {
		t.Logf("Warning: Average search time (%v) is higher than expected", avgSearchTime)
	}
}
