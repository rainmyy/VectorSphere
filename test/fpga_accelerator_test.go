package test

import (
	"VectorSphere/src/library/acceler"
	"testing"
	"time"
)

// TestFPGAAcceleratorCreation 测试FPGA加速器创建
func TestFPGAAcceleratorCreation(t *testing.T) {
	t.Run("创建FPGA加速器", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable:    true,
			Bitstream: "test_bitstream.bit",
			Parallelism: acceler.FPGAParallelismConfig{
				ComputeUnits: 4,
			},
		}
		
		fpga := acceler.NewFPGAAccelerator(0, config)
		if fpga == nil {
			t.Fatal("FPGA加速器创建失败")
		}
		
		if fpga.GetType() != "FPGA" {
			t.Errorf("期望类型为FPGA，实际为%s", fpga.GetType())
		}
	})
	
	t.Run("FPGA加速器可用性检查", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		// 注意：在没有FPGA的环境中，IsAvailable可能返回false
		available := fpga.IsAvailable()
		t.Logf("FPGA可用性: %v", available)
	})
	
	t.Run("无效配置", func(t *testing.T) {
		// 测试nil配置
		fpga := acceler.NewFPGAAccelerator(0, nil)
		if fpga == nil {
			t.Error("即使配置为nil，FPGA加速器也应该能创建")
		}
	})
}

// TestFPGAAcceleratorInitialization 测试FPGA加速器初始化
func TestFPGAAcceleratorInitialization(t *testing.T) {
	t.Run("FPGA初始化", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable:    true,
			Bitstream: "test_bitstream.bit",
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Initialize()
		if err != nil {
			// 在没有FPGA的环境中，初始化失败是正常的
			t.Logf("FPGA初始化失败（可能是因为没有FPGA设备）: %v", err)
			return
		}
		
		defer fpga.Shutdown()
		
		if !fpga.IsAvailable() {
			t.Error("初始化后FPGA应该可用")
		}
	})
	
	t.Run("重复初始化", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err1 := fpga.Initialize()
		if err1 != nil {
			t.Logf("FPGA初始化失败: %v", err1)
			return
		}
		
		defer fpga.Shutdown()
		
		// 重复初始化应该成功
		err2 := fpga.Initialize()
		if err2 != nil {
			t.Errorf("重复初始化失败: %v", err2)
		}
	})
	
	t.Run("启动和停止", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Start()
		if err != nil {
			t.Logf("FPGA启动失败: %v", err)
			return
		}
		
		err = fpga.Stop()
		if err != nil {
			t.Errorf("FPGA停止失败: %v", err)
		}
	})
}

// TestFPGAAcceleratorComputation 测试FPGA计算功能
func TestFPGAAcceleratorComputation(t *testing.T) {
	t.Run("距离计算", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Initialize()
		if err != nil {
			t.Logf("FPGA初始化失败，跳过计算测试: %v", err)
			return
		}
		defer fpga.Shutdown()
		
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
			{0.0, 0.0, 1.0},
		}
		
		distances, err := fpga.ComputeDistance(query, vectors)
		if err != nil {
			t.Fatalf("FPGA距离计算失败: %v", err)
		}
		
		if len(distances) != len(vectors) {
			t.Errorf("期望距离数量为%d，实际为%d", len(vectors), len(distances))
		}
		
		// 验证第一个向量的距离应该最小（相同向量）
		if len(distances) > 0 && distances[0] > 0.1 {
			t.Errorf("相同向量的距离应该接近0，实际为%f", distances[0])
		}
	})
	
	t.Run("批量计算", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Initialize()
		if err != nil {
			t.Logf("FPGA初始化失败，跳过批量计算测试: %v", err)
			return
		}
		defer fpga.Shutdown()
		
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
		
		results, err := fpga.BatchComputeDistance(queries, database)
		if err != nil {
			t.Fatalf("FPGA批量计算失败: %v", err)
		}
		
		if len(results) != len(queries) {
			t.Errorf("期望结果数量为%d，实际为%d", len(queries), len(results))
		}
	})
	
	t.Run("余弦相似度计算", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Initialize()
		if err != nil {
			t.Logf("FPGA初始化失败，跳过余弦相似度测试: %v", err)
			return
		}
		defer fpga.Shutdown()
		
		vec1 := []float64{1.0, 0.0, 0.0}
		vec2 := []float64{1.0, 0.0, 0.0}
		
		queries := [][]float64{vec1}
		database := [][]float64{vec2}
		
		similarities, err := fpga.BatchCosineSimilarity(queries, database)
		if err != nil {
			t.Fatalf("FPGA余弦相似度计算失败: %v", err)
		}
		
		// 相同向量的余弦相似度应该接近1
		if len(similarities) > 0 && len(similarities[0]) > 0 && similarities[0][0] < 0.99 {
			t.Errorf("相同向量的余弦相似度应该接近1，实际为%f", similarities[0][0])
		}
	})
}

// TestFPGAAcceleratorPerformance 测试FPGA性能
func TestFPGAAcceleratorPerformance(t *testing.T) {
	t.Run("性能指标", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Initialize()
		if err != nil {
			t.Logf("FPGA初始化失败，跳过性能测试: %v", err)
			return
		}
		defer fpga.Shutdown()
		
		// 执行一些操作来生成性能数据
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
		}
		
		_, err = fpga.ComputeDistance(query, vectors)
		if err != nil {
			t.Fatalf("计算失败: %v", err)
		}
		
		// 检查性能指标
		metrics := fpga.GetPerformanceMetrics()
		if metrics.LatencyCurrent < 0 {
			t.Error("性能指标应该有效")
		}
		
		// 检查统计信息
		stats := fpga.GetStats()
		if stats.TotalOperations < 0 {
			t.Error("统计信息应该有效")
		}
	})
	
	t.Run("硬件统计", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Initialize()
		if err != nil {
			t.Logf("FPGA初始化失败，跳过硬件统计测试: %v", err)
			return
		}
		defer fpga.Shutdown()
		
		// 检查基本统计信息
		stats := fpga.GetStats()
		if stats.TotalOperations < 0 {
			t.Error("统计信息应该有效")
		}
		
		// 检查性能指标
		metrics := fpga.GetPerformanceMetrics()
		if metrics.LatencyCurrent < 0 {
			t.Error("性能指标应该有效")
		}
	})
	
	t.Run("大规模并行处理", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
			Parallelism: acceler.FPGAParallelismConfig{
				ComputeUnits: 8,
			},
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Initialize()
		if err != nil {
			t.Logf("FPGA初始化失败，跳过大规模测试: %v", err)
			return
		}
		defer fpga.Shutdown()
		
		// 创建大规模测试数据
		dimension := 256
		numVectors := 2000
		
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
		distances, err := fpga.ComputeDistance(query, vectors)
		duration := time.Since(start)
		
		if err != nil {
			t.Fatalf("大规模计算失败: %v", err)
		}
		
		if len(distances) != numVectors {
			t.Errorf("期望距离数量为%d，实际为%d", numVectors, len(distances))
		}
		
		t.Logf("FPGA处理%d个%d维向量耗时: %v", numVectors, dimension, duration)
	})
}

// TestFPGAAcceleratorErrorHandling 测试FPGA错误处理
func TestFPGAAcceleratorErrorHandling(t *testing.T) {
	t.Run("无效设备ID", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		// 使用一个很大的设备ID，应该会失败
		fpga := acceler.NewFPGAAccelerator(999, config)
		
		err := fpga.Initialize()
		if err == nil {
			t.Error("无效设备ID应该导致初始化失败")
		}
	})
	
	t.Run("未初始化操作", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		// 不调用Initialize
		
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		
		_, err := fpga.ComputeDistance(query, vectors)
		if err == nil {
			t.Error("未初始化的FPGA应该返回错误")
		}
	})
	
	t.Run("空数据处理", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Initialize()
		if err != nil {
			t.Logf("FPGA初始化失败，跳过空数据测试: %v", err)
			return
		}
		defer fpga.Shutdown()
		
		// 测试空查询向量
		_, err = fpga.ComputeDistance(nil, [][]float64{{1.0, 0.0}})
		if err == nil {
			t.Error("空查询向量应该返回错误")
		}
		
		// 测试空数据库向量
		_, err = fpga.ComputeDistance([]float64{1.0, 0.0}, nil)
		if err == nil {
			t.Error("空数据库向量应该返回错误")
		}
	})
	
	t.Run("维度不匹配", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable: true,
		}
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Initialize()
		if err != nil {
			t.Logf("FPGA初始化失败，跳过维度测试: %v", err)
			return
		}
		defer fpga.Shutdown()
		
		// 查询向量和数据库向量维度不匹配
		query := []float64{1.0, 0.0}
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		
		_, err = fpga.ComputeDistance(query, vectors)
		if err == nil {
			t.Error("维度不匹配应该返回错误")
		}
	})
}

// TestFPGAAcceleratorConfiguration 测试FPGA配置
func TestFPGAAcceleratorConfiguration(t *testing.T) {
	t.Run("配置验证", func(t *testing.T) {
		// 有效配置
		validConfig := &acceler.FPGAConfig{
			Enable:    true,
			Bitstream: "valid_bitstream.bit",
			Parallelism: acceler.FPGAParallelismConfig{
				ComputeUnits: 4,
			},
		}
		
		fpga := acceler.NewFPGAAccelerator(0, validConfig)
		if fpga == nil {
			t.Error("有效配置应该能创建FPGA加速器")
		}
		
		// 禁用配置
		disabledConfig := &acceler.FPGAConfig{
			Enable: false,
		}
		
		fpga2 := acceler.NewFPGAAccelerator(0, disabledConfig)
		if fpga2 == nil {
			t.Error("即使禁用，也应该能创建FPGA加速器")
		}
	})
	
	t.Run("比特流加载", func(t *testing.T) {
		config := &acceler.FPGAConfig{
			Enable:    true,
			Bitstream: "nonexistent_bitstream.bit",
		}
		
		fpga := acceler.NewFPGAAccelerator(0, config)
		
		err := fpga.Initialize()
		if err != nil {
			// 不存在的比特流文件应该导致初始化失败
			t.Logf("比特流加载失败（预期）: %v", err)
		}
	})
}

// TestFPGAAcceleratorConcurrency 测试FPGA并发安全性
func TestFPGAAcceleratorConcurrency(t *testing.T) {
	config := &acceler.FPGAConfig{
		Enable: true,
	}
	fpga := acceler.NewFPGAAccelerator(0, config)
	
	err := fpga.Initialize()
	if err != nil {
		t.Logf("FPGA初始化失败，跳过并发测试: %v", err)
		return
	}
	defer fpga.Shutdown()
	
	// 并发执行多个计算任务
	const numGoroutines = 8
	done := make(chan bool, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			query := []float64{float64(id), 0.0, 0.0}
			vectors := [][]float64{
				{1.0, 0.0, 0.0},
				{0.0, 1.0, 0.0},
			}
			
			_, err := fpga.ComputeDistance(query, vectors)
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
	
	stats := fpga.GetStats()
	if stats.TotalOperations < numGoroutines {
		t.Errorf("期望至少%d次操作，实际为%d", numGoroutines, stats.TotalOperations)
	}
}