package test

import (
	"VectorSphere/src/library/acceler"
	"testing"
	"time"
)

// TestBaseAccelerator 测试基础加速器功能
func TestBaseAccelerator(t *testing.T) {
	t.Run("创建基础加速器", func(t *testing.T) {
		// 创建基础加速器
		capabilities := acceler.HardwareCapabilities{
			HasAVX2:           true,
			HasAVX512:         false,
			HasGPU:            false,
			CPUCores:          8,
			Type:              "CPU",
			MemorySize:        8 * 1024 * 1024 * 1024,
			MaxBatchSize:      100,
			PerformanceRating: 8.5,
		}
		stats := acceler.HardwareStats{
			TotalOperations: 0,
			SuccessfulOps:   0,
			FailedOps:       0,
			Throughput:      0.0,
		}
		base := acceler.NewBaseAccelerator(0, "ivf", capabilities, stats)
		if base == nil {
			t.Fatal("创建基础加速器失败")
		}
	})

	t.Run("测试设备ID", func(t *testing.T) {
		// 创建基础加速器
		capabilities := acceler.HardwareCapabilities{
			HasAVX2:           true,
			HasAVX512:         true,
			HasGPU:            true,
			CPUCores:          16,
			GPUDevices:        1,
			Type:              "GPU",
			MemorySize:        16 * 1024 * 1024 * 1024,
			MaxBatchSize:      500,
			PerformanceRating: 9.2,
		}
		stats := acceler.HardwareStats{
			TotalOperations: 0,
			SuccessfulOps:   0,
			FailedOps:       0,
			Throughput:      0.0,
		}
		base := acceler.NewBaseAccelerator(1, "hnsw", capabilities, stats)
		if base.GetDeviceID() != 1 {
			t.Errorf("期望设备ID为1，实际为%d", base.GetDeviceID())
		}
	})

	t.Run("测试索引类型", func(t *testing.T) {
		capabilities := acceler.HardwareCapabilities{
			HasAVX2:           true,
			HasAVX512:         false,
			HasGPU:            false,
			CPUCores:          8,
			Type:              "CPU",
			MemorySize:        8 * 1024 * 1024 * 1024,
			MaxBatchSize:      100,
			PerformanceRating: 8.5,
		}
		stats := acceler.HardwareStats{
			TotalOperations: 0,
			SuccessfulOps:   0,
			FailedOps:       0,
			Throughput:      0.0,
		}
		base := acceler.NewBaseAccelerator(0, "IVF,Flat", capabilities, stats)
		if base.GetIndexType() != "IVF,Flat" {
			t.Errorf("期望索引类型为IVF,Flat，实际为%s", base.GetIndexType())
		}
	})

	t.Run("测试可用性", func(t *testing.T) {
		capabilities := acceler.HardwareCapabilities{
			HasAVX2:           true,
			HasAVX512:         false,
			HasGPU:            false,
			CPUCores:          8,
			Type:              "CPU",
			MemorySize:        8 * 1024 * 1024 * 1024,
			MaxBatchSize:      100,
			PerformanceRating: 8.5,
		}
		stats := acceler.HardwareStats{
			TotalOperations: 0,
			SuccessfulOps:   0,
			FailedOps:       0,
			Throughput:      0.0,
		}
		base := acceler.NewBaseAccelerator(0, "test", capabilities, stats)
		if !base.IsAvailable() {
			t.Error("基础加速器应该是可用的")
		}
	})

	t.Run("测试初始化状态", func(t *testing.T) {
		capabilities := acceler.HardwareCapabilities{
			HasAVX2:           true,
			HasAVX512:         false,
			HasGPU:            false,
			CPUCores:          8,
			Type:              "CPU",
			MemorySize:        8 * 1024 * 1024 * 1024,
			MaxBatchSize:      100,
			PerformanceRating: 8.5,
		}
		stats := acceler.HardwareStats{
			TotalOperations: 0,
			SuccessfulOps:   0,
			FailedOps:       0,
			Throughput:      0.0,
		}
		base := acceler.NewBaseAccelerator(0, "test", capabilities, stats)
		if base.IsInitialized() {
			t.Error("新创建的加速器不应该已初始化")
		}
	})

	t.Run("测试统计信息", func(t *testing.T) {
		capabilities := acceler.HardwareCapabilities{
			HasAVX2:           true,
			HasAVX512:         false,
			HasGPU:            false,
			CPUCores:          8,
			Type:              "CPU",
			MemorySize:        8 * 1024 * 1024 * 1024,
			MaxBatchSize:      100,
			PerformanceRating: 8.5,
		}
		stats := acceler.HardwareStats{
			TotalOperations: 0,
			SuccessfulOps:   0,
			FailedOps:       0,
			Throughput:      0.0,
		}
		base := acceler.NewBaseAccelerator(0, "test", capabilities, stats)
		stats2 := base.GetStats()
		if stats2.TotalOperations != 0 {
			t.Errorf("期望总操作数为0，实际为%d", stats2.TotalOperations)
		}
		if stats2.SuccessfulOps != 0 {
			t.Errorf("期望成功操作数为0，实际为%d", stats2.SuccessfulOps)
		}
	})

	t.Run("测试性能指标", func(t *testing.T) {
		capabilities := acceler.HardwareCapabilities{
			HasAVX2:           true,
			HasAVX512:         false,
			HasGPU:            false,
			CPUCores:          8,
			Type:              "CPU",
			MemorySize:        8 * 1024 * 1024 * 1024,
			MaxBatchSize:      100,
			PerformanceRating: 8.5,
		}
		stats := acceler.HardwareStats{
			TotalOperations: 0,
			SuccessfulOps:   0,
			FailedOps:       0,
			Throughput:      0.0,
		}
		base := acceler.NewBaseAccelerator(0, "test", capabilities, stats)
		metrics := base.GetPerformanceMetrics()
		if metrics.LatencyCurrent != 0 {
			t.Errorf("期望当前延迟为0，实际为%v", metrics.LatencyCurrent)
		}
	})

	t.Run("测试能力信息", func(t *testing.T) {
		capabilities := acceler.HardwareCapabilities{
			HasAVX2:           true,
			HasAVX512:         false,
			HasGPU:            false,
			CPUCores:          8,
			Type:              "CPU",
			MemorySize:        8 * 1024 * 1024 * 1024,
			MaxBatchSize:      100,
			PerformanceRating: 8.5,
		}
		stats := acceler.HardwareStats{
			TotalOperations: 0,
			SuccessfulOps:   0,
			FailedOps:       0,
			Throughput:      0.0,
		}
		base := acceler.NewBaseAccelerator(0, "test", capabilities, stats)
		caps := base.GetCapabilities()
		if !caps.HasAVX2 {
			t.Error("期望支持AVX2")
		}
		if caps.Type != "CPU" {
			t.Errorf("期望类型为CPU，实际为%s", caps.Type)
		}
	})

	t.Run("测试更新统计", func(t *testing.T) {
		capabilities := acceler.HardwareCapabilities{
			HasAVX2:           true,
			HasAVX512:         false,
			HasGPU:            false,
			CPUCores:          8,
			Type:              "CPU",
			MemorySize:        8 * 1024 * 1024 * 1024,
			MaxBatchSize:      100,
			PerformanceRating: 8.5,
		}
		stats := acceler.HardwareStats{
			TotalOperations: 0,
			SuccessfulOps:   0,
			FailedOps:       0,
			Throughput:      0.0,
		}
		base := acceler.NewBaseAccelerator(0, "test", capabilities, stats)
		
		// 模拟操作
		base.UpdateStats(time.Millisecond*10, 1, true)
		stats2 := base.GetStats()
		
		if stats2.TotalOperations != 1 {
			t.Errorf("期望总操作数为1，实际为%d", stats2.TotalOperations)
		}
		if stats2.SuccessfulOps != 1 {
			t.Errorf("期望成功操作数为1，实际为%d", stats2.SuccessfulOps)
		}
		if stats2.FailedOps != 0 {
			t.Errorf("期望失败操作数为0，实际为%d", stats2.FailedOps)
		}
	})

	t.Run("测试失败操作统计", func(t *testing.T) {
		capabilities := acceler.HardwareCapabilities{
			HasAVX2:           true,
			HasAVX512:         false,
			HasGPU:            false,
			CPUCores:          8,
			Type:              "CPU",
			MemorySize:        8 * 1024 * 1024 * 1024,
			MaxBatchSize:      100,
			PerformanceRating: 8.5,
		}
		stats := acceler.HardwareStats{
			TotalOperations: 0,
			SuccessfulOps:   0,
			FailedOps:       0,
			Throughput:      0.0,
		}
		base := acceler.NewBaseAccelerator(0, "test", capabilities, stats)
		
		// 模拟失败操作
		base.UpdateStats(time.Millisecond*10, 1, false)
		stats2 := base.GetStats()
		
		if stats2.TotalOperations != 1 {
			t.Errorf("期望总操作数为1，实际为%d", stats2.TotalOperations)
		}
		if stats2.SuccessfulOps != 0 {
			t.Errorf("期望成功操作数为0，实际为%d", stats2.SuccessfulOps)
		}
		if stats2.FailedOps != 1 {
			t.Errorf("期望失败操作数为1，实际为%d", stats2.FailedOps)
		}
	})
}

// TestBaseAcceleratorConcurrency 测试基础加速器并发安全性
func TestBaseAcceleratorConcurrency(t *testing.T) {
	capabilities := acceler.HardwareCapabilities{
		HasAVX2:           true,
		HasAVX512:         false,
		HasGPU:            false,
		CPUCores:          8,
		Type:              "CPU",
		MemorySize:        8 * 1024 * 1024 * 1024,
		MaxBatchSize:      100,
		PerformanceRating: 8.5,
	}
	stats := acceler.HardwareStats{
		TotalOperations: 0,
		SuccessfulOps:   0,
		FailedOps:       0,
		Throughput:      0.0,
	}
	base := acceler.NewBaseAccelerator(0, "test", capabilities, stats)
	
	// 并发更新统计
	for i := 0; i < 100; i++ {
		go func() {
			base.UpdateStats(time.Millisecond, 1, true)
		}()
	}
	
	// 等待所有goroutine完成
	time.Sleep(time.Millisecond * 100)
	
	stats2 := base.GetStats()
	if stats2.TotalOperations != 100 {
		t.Errorf("期望总操作数为100，实际为%d", stats2.TotalOperations)
	}
}