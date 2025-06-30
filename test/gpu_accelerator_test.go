package test

import (
	"VectorSphere/src/library/acceler"
	"testing"
	"time"
)

// TestGPUAcceleratorCreation 测试GPU加速器创建
func TestGPUAcceleratorCreation(t *testing.T) {
	t.Run("创建GPU加速器", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		if gpu == nil {
			t.Fatal("GPU加速器创建失败")
		}
		
		if gpu.GetType() != "GPU" {
			t.Errorf("期望类型为GPU，实际为%s", gpu.GetType())
		}
	})
	
	t.Run("GPU加速器可用性检查", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		// 注意：在没有GPU的环境中，IsAvailable可能返回false
		// 这是正常的，不应该导致测试失败
		available := gpu.IsAvailable()
		t.Logf("GPU可用性: %v", available)
	})
}

// TestGPUAcceleratorInitialization 测试GPU加速器初始化
func TestGPUAcceleratorInitialization(t *testing.T) {
	t.Run("GPU初始化", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		
		err := gpu.Initialize()
		if err != nil {
			// 在没有GPU的环境中，初始化失败是正常的
			t.Logf("GPU初始化失败（可能是因为没有GPU设备）: %v", err)
			return
		}
		
		defer gpu.Shutdown()
		
		if !gpu.IsAvailable() {
			t.Error("初始化后GPU应该可用")
		}
	})
	
	t.Run("重复初始化", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		
		err1 := gpu.Initialize()
		if err1 != nil {
			t.Logf("GPU初始化失败: %v", err1)
			return
		}
		
		defer gpu.Shutdown()
		
		// 重复初始化应该成功
		err2 := gpu.Initialize()
		if err2 != nil {
			t.Errorf("重复初始化失败: %v", err2)
		}
	})
}

// TestGPUAcceleratorComputation 测试GPU计算功能
func TestGPUAcceleratorComputation(t *testing.T) {
	t.Run("距离计算", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		
		err := gpu.Initialize()
		if err != nil {
			t.Logf("GPU初始化失败，跳过计算测试: %v", err)
			return
		}
		defer gpu.Shutdown()
		
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
			{0.0, 0.0, 1.0},
		}
		
		distances, err := gpu.ComputeDistance(query, vectors)
		if err != nil {
			t.Fatalf("GPU距离计算失败: %v", err)
		}
		
		if len(distances) != len(vectors) {
			t.Errorf("期望距离数量为%d，实际为%d", len(vectors), len(distances))
		}
		
		// 验证第一个向量的距离应该最小（相同向量）
		if len(distances) > 0 && distances[0] > 0.1 {
			t.Errorf("相同向量的距离应该接近0，实际为%f", distances[0])
		}
	})
	
	t.Run("批量搜索", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		
		err := gpu.Initialize()
		if err != nil {
			t.Logf("GPU初始化失败，跳过批量搜索测试: %v", err)
			return
		}
		defer gpu.Shutdown()
		
		queries := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
		}
		database := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
			{0.0, 0.0, 1.0},
			{1.0, 1.0, 0.0},
		}
		
		results, err := gpu.BatchSearch(queries, database, 2)
		if err != nil {
			t.Fatalf("GPU批量搜索失败: %v", err)
		}
		
		if len(results) != len(queries) {
			t.Errorf("期望结果数量为%d，实际为%d", len(queries), len(results))
		}
		
		for i, result := range results {
			if len(result) != 2 {
				t.Errorf("查询%d期望返回2个结果，实际为%d", i, len(result))
			}
		}
	})
}

// TestGPUAcceleratorPerformance 测试GPU性能
func TestGPUAcceleratorPerformance(t *testing.T) {
	t.Run("性能指标", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		
		err := gpu.Initialize()
		if err != nil {
			t.Logf("GPU初始化失败，跳过性能测试: %v", err)
			return
		}
		defer gpu.Shutdown()
		
		// 执行一些操作来生成性能数据
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
		}
		
		_, err = gpu.ComputeDistance(query, vectors)
		if err != nil {
			t.Fatalf("计算失败: %v", err)
		}
		
		// 检查性能指标
		metrics := gpu.GetPerformanceMetrics()
		if metrics.LatencyCurrent < 0 {
			t.Error("性能指标应该有效")
		}
		
		// 检查统计信息
		stats := gpu.GetStats()
		if stats.TotalOperations < 0 {
			t.Error("统计信息应该有效")
		}
	})
	
	t.Run("大规模数据处理", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		
		err := gpu.Initialize()
		if err != nil {
			t.Logf("GPU初始化失败，跳过大规模测试: %v", err)
			return
		}
		defer gpu.Shutdown()
		
		// 创建大规模测试数据
		dimension := 128
		numVectors := 1000
		
		query := make([]float64, dimension)
		for i := range query {
			query[i] = float64(i) / float64(dimension)
		}
		
		vectors := make([][]float64, numVectors)
		for i := range vectors {
			vectors[i] = make([]float64, dimension)
			for j := range vectors[i] {
				vectors[i][j] = float64((i+j)%100) / 100.0
			}
		}
		
		start := time.Now()
		distances, err := gpu.ComputeDistance(query, vectors)
		duration := time.Since(start)
		
		if err != nil {
			t.Fatalf("大规模计算失败: %v", err)
		}
		
		if len(distances) != numVectors {
			t.Errorf("期望距离数量为%d，实际为%d", numVectors, len(distances))
		}
		
		t.Logf("GPU处理%d个%d维向量耗时: %v", numVectors, dimension, duration)
	})
}

// TestGPUAcceleratorErrorHandling 测试GPU错误处理
func TestGPUAcceleratorErrorHandling(t *testing.T) {
	t.Run("无效设备ID", func(t *testing.T) {
		// 使用一个很大的设备ID，应该会失败
		gpu := acceler.NewGPUAccelerator(999)
		
		err := gpu.Initialize()
		if err != nil {
			t.Logf("无效设备ID导致初始化失败（符合预期）: %v", err)
		} else {
			t.Logf("无效设备ID未导致初始化失败，可能使用了默认设备")
		}
	})
	
	t.Run("未初始化操作", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		// 不调用Initialize
		
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		
		_, err := gpu.ComputeDistance(query, vectors)
		if err == nil {
			t.Error("未初始化的GPU应该返回错误")
		}
	})
	
	t.Run("空数据处理", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		
		err := gpu.Initialize()
		if err != nil {
			t.Logf("GPU初始化失败，跳过空数据测试: %v", err)
			return
		}
		defer gpu.Shutdown()
		
		// 测试空查询向量
		_, err = gpu.ComputeDistance(nil, [][]float64{{1.0, 0.0}})
		if err != nil {
			t.Logf("空查询向量返回错误（符合预期）: %v", err)
		} else {
			t.Logf("空查询向量未返回错误，可能有默认处理")
		}
		
		// 测试空数据库向量
		_, err = gpu.ComputeDistance([]float64{1.0, 0.0}, nil)
		if err != nil {
			t.Logf("空数据库向量返回错误（符合预期）: %v", err)
		} else {
			t.Logf("空数据库向量未返回错误，可能有默认处理")
		}
	})
}

// TestGPUAcceleratorConcurrency 测试GPU并发安全性
func TestGPUAcceleratorConcurrency(t *testing.T) {
	gpu := acceler.NewGPUAccelerator(0)
	
	err := gpu.Initialize()
	if err != nil {
		t.Logf("GPU初始化失败，跳过并发测试: %v", err)
		return
	}
	defer gpu.Shutdown()
	
	// 并发执行多个计算任务
	const numGoroutines = 10
	done := make(chan bool, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			query := []float64{float64(id), 0.0, 0.0}
			vectors := [][]float64{
				{1.0, 0.0, 0.0},
				{0.0, 1.0, 0.0},
			}
			
			_, err := gpu.ComputeDistance(query, vectors)
			if err != nil {
				t.Errorf("Goroutine %d 计算失败: %v", id, err)
			}
			
			done <- true
		}(i)
	}
	
	// 等待所有goroutine完成
	for i := 0; i < numGoroutines; i++ {
		<-done
	}
	
	stats := gpu.GetStats()
	if stats.TotalOperations < numGoroutines {
		t.Errorf("期望至少%d次操作，实际为%d", numGoroutines, stats.TotalOperations)
	}
}

// TestGPUAcceleratorMemoryManagement 测试GPU内存管理
func TestGPUAcceleratorMemoryManagement(t *testing.T) {
	t.Run("内存使用监控", func(t *testing.T) {
		gpu := acceler.NewGPUAccelerator(0)
		
		err := gpu.Initialize()
		if err != nil {
			t.Logf("GPU初始化失败，跳过内存测试: %v", err)
			return
		}
		defer gpu.Shutdown()
		
		// 获取初始内存状态
		capabilities := gpu.GetCapabilities()
		if capabilities.Type == "" {
			t.Error("GPU能力信息应该有效")
		}
		
		// 执行一些内存密集型操作
		for i := 0; i < 5; i++ {
			query := make([]float64, 512)
			vectors := make([][]float64, 100)
			for j := range vectors {
				vectors[j] = make([]float64, 512)
			}
			
			_, err := gpu.ComputeDistance(query, vectors)
			if err != nil {
				t.Fatalf("内存密集型操作失败: %v", err)
			}
		}
	})
}