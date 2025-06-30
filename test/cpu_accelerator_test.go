package test

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"math"
	"testing"
	"time"
)

// TestCPUAccelerator 测试CPU加速器功能
func TestCPUAccelerator(t *testing.T) {
	t.Run("创建CPU加速器", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		if cpu == nil {
			t.Fatal("创建CPU加速器失败")
		}
		if cpu.GetType() != "CPU" {
			t.Errorf("期望类型为CPU，实际为%s", cpu.GetType())
		}
	})

	t.Run("初始化CPU加速器", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		err := cpu.Initialize()
		if err != nil {
			t.Fatalf("初始化CPU加速器失败: %v", err)
		}
		if !cpu.IsInitialized() {
			t.Error("CPU加速器应该已初始化")
		}
	})

	t.Run("测试可用性", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		if !cpu.IsAvailable() {
			t.Error("CPU加速器应该是可用的")
		}
	})

	t.Run("计算距离", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{
			{1.0, 0.0, 0.0}, // 相同向量，距离应该为0
			{0.0, 1.0, 0.0}, // 垂直向量，余弦距离为1
			{-1.0, 0.0, 0.0}, // 相反向量，余弦距离为2
		}
		
		distances, err := cpu.ComputeDistance(query, vectors)
		if err != nil {
			t.Fatalf("计算距离失败: %v", err)
		}
		
		if len(distances) != 3 {
			t.Errorf("期望3个距离值，实际为%d", len(distances))
		}
		
		// 检查第一个距离（相同向量）
		if math.Abs(distances[0]) > 1e-6 {
			t.Errorf("相同向量的距离应该接近0，实际为%f", distances[0])
		}
	})

	t.Run("批量计算距离", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		
		queries := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
		}
		vectors := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
			{0.0, 0.0, 1.0},
		}
		
		results, err := cpu.BatchComputeDistance(queries, vectors)
		if err != nil {
			t.Fatalf("批量计算距离失败: %v", err)
		}
		
		if len(results) != 2 {
			t.Errorf("期望2个查询结果，实际为%d", len(results))
		}
		
		for i, result := range results {
			if len(result) != 3 {
				t.Errorf("查询%d期望3个距离值，实际为%d", i, len(result))
			}
		}
	})

	t.Run("余弦相似度计算", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		
		queries := [][]float64{
			{1.0, 0.0, 0.0},
		}
		database := [][]float64{
			{1.0, 0.0, 0.0}, // 相同向量，相似度为1
			{0.0, 1.0, 0.0}, // 垂直向量，相似度为0
		}
		
		similarities, err := cpu.BatchCosineSimilarity(queries, database)
		if err != nil {
			t.Fatalf("计算余弦相似度失败: %v", err)
		}
		
		if len(similarities) != 1 {
			t.Errorf("期望1个查询结果，实际为%d", len(similarities))
		}
		
		if len(similarities[0]) != 2 {
			t.Errorf("期望2个相似度值，实际为%d", len(similarities[0]))
		}
		
		// 检查相同向量的相似度
		if math.Abs(similarities[0][0]-1.0) > 1e-6 {
			t.Errorf("相同向量的相似度应该接近1，实际为%f", similarities[0][0])
		}
		
		// 检查垂直向量的相似度
		if math.Abs(similarities[0][1]) > 1e-6 {
			t.Errorf("垂直向量的相似度应该接近0，实际为%f", similarities[0][1])
		}
	})

	t.Run("批量搜索", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		
		queries := [][]float64{
			{1.0, 0.0, 0.0},
		}
		database := [][]float64{
			{1.0, 0.0, 0.0},
			{0.9, 0.1, 0.0},
			{0.0, 1.0, 0.0},
			{0.0, 0.0, 1.0},
		}
		
		results, err := cpu.BatchSearch(queries, database, 2)
		if err != nil {
			t.Fatalf("批量搜索失败: %v", err)
		}
		
		if len(results) != 1 {
			t.Errorf("期望1个查询结果，实际为%d", len(results))
		}
		
		if len(results[0]) != 2 {
			t.Errorf("期望返回2个最近邻，实际为%d", len(results[0]))
		}
		
		// 检查结果是否按相似度排序
		if results[0][0].Similarity < results[0][1].Similarity {
			t.Error("搜索结果应该按相似度降序排列")
		}
	})

	t.Run("加速搜索", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		
		query := []float64{1.0, 0.0, 0.0}
		database := [][]float64{
			{1.0, 0.0, 0.0},
			{0.9, 0.1, 0.0},
			{0.0, 1.0, 0.0},
		}
		
		options := entity.SearchOptions{
			K:         2,
			Threshold: 0.5,
		}
		
		results, err := cpu.AccelerateSearch(query, database, options)
		if err != nil {
			t.Fatalf("加速搜索失败: %v", err)
		}
		
		if len(results) == 0 {
			t.Error("应该有搜索结果")
		}
		
		// 检查阈值过滤
		for _, result := range results {
			if result.Similarity < 0.5 {
				t.Errorf("结果相似度%f低于阈值0.5", result.Similarity)
			}
		}
	})

	t.Run("获取统计信息", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		
		// 执行一些操作
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		cpu.ComputeDistance(query, vectors)
		
		stats := cpu.GetStats()
		if stats.TotalOperations == 0 {
			t.Error("应该有操作统计")
		}
	})

	t.Run("自动调优", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		
		workload := acceler.WorkloadProfile{
			VectorDimension: 128,
			QueriesPerSecond: 100,
			BatchSize: 32,
			Type: "search",
			DataSize: 10000,
		}
		
		err := cpu.AutoTune(workload)
		if err != nil {
			t.Fatalf("自动调优失败: %v", err)
		}
	})

	t.Run("关闭加速器", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		
		err := cpu.Shutdown()
		if err != nil {
			t.Fatalf("关闭CPU加速器失败: %v", err)
		}
	})
}

// TestCPUAcceleratorPerformance 测试CPU加速器性能
func TestCPUAcceleratorPerformance(t *testing.T) {
	cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
	cpu.Initialize()
	defer cpu.Shutdown()
	
	// 生成测试数据
	dimension := 128
	numQueries := 100
	numVectors := 1000
	
	queries := make([][]float64, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			queries[i][j] = float64(i*dimension + j)
		}
	}
	
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = float64(i*dimension + j)
		}
	}
	
	// 测试批量距离计算性能
	start := time.Now()
	_, err := cpu.BatchComputeDistance(queries, vectors)
	if err != nil {
		t.Fatalf("批量距离计算失败: %v", err)
	}
	duration := time.Since(start)
	
	t.Logf("批量距离计算耗时: %v", duration)
	
	// 测试批量搜索性能
	start = time.Now()
	_, err = cpu.BatchSearch(queries, vectors, 10)
	if err != nil {
		t.Fatalf("批量搜索失败: %v", err)
	}
	duration = time.Since(start)
	
	t.Logf("批量搜索耗时: %v", duration)
	
	// 检查性能指标
	metrics := cpu.GetPerformanceMetrics()
	if metrics.LatencyCurrent == 0 {
		t.Error("应该有延迟统计")
	}
}

// TestCPUAcceleratorErrorHandling 测试CPU加速器错误处理
func TestCPUAcceleratorErrorHandling(t *testing.T) {
	t.Run("空查询向量", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		defer cpu.Shutdown()
		
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		_, err := cpu.ComputeDistance(nil, vectors)
		if err == nil {
			t.Error("空查询向量应该返回错误")
		}
	})

	t.Run("空数据库向量", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		defer cpu.Shutdown()
		
		query := []float64{1.0, 0.0, 0.0}
		_, err := cpu.ComputeDistance(query, nil)
		if err == nil {
			t.Error("空数据库向量应该返回错误")
		}
	})

	t.Run("维度不匹配", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		defer cpu.Shutdown()
		
		query := []float64{1.0, 0.0}
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		_, err := cpu.ComputeDistance(query, vectors)
		if err == nil {
			t.Error("维度不匹配应该返回错误")
		}
	})

	t.Run("未初始化操作", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		_, err := cpu.ComputeDistance(query, vectors)
		if err == nil {
			t.Error("未初始化的加速器操作应该返回错误")
		}
	})

	t.Run("无效K值", func(t *testing.T) {
		cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
		cpu.Initialize()
		defer cpu.Shutdown()
		
		queries := [][]float64{{1.0, 0.0, 0.0}}
		database := [][]float64{{1.0, 0.0, 0.0}}
		_, err := cpu.BatchSearch(queries, database, 0)
		if err == nil {
			t.Error("无效K值应该返回错误")
		}
	})
}

// TestCPUAcceleratorConcurrency 测试CPU加速器并发安全性
func TestCPUAcceleratorConcurrency(t *testing.T) {
	cpu := acceler.NewCPUAccelerator(0, "IDMap,Flat")
	cpu.Initialize()
	defer cpu.Shutdown()
	
	query := []float64{1.0, 0.0, 0.0}
	vectors := [][]float64{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	}
	
	// 并发执行距离计算
	for i := 0; i < 100; i++ {
		go func() {
			cpu.ComputeDistance(query, vectors)
		}()
	}
	
	// 等待所有goroutine完成
	time.Sleep(time.Millisecond * 100)
	
	stats := cpu.GetStats()
	if stats.TotalOperations != 100 {
		t.Errorf("期望100次操作，实际为%d", stats.TotalOperations)
	}
}