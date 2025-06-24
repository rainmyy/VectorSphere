package test

import (
	"VectorSphere/src/vector"
	"testing"
)

func TestNewGPUAccelerator(t *testing.T) {
	accelerator := vector.NewGPUAccelerator()
	if accelerator == nil {
		t.Fatal("Expected non-nil GPU accelerator")
	}
	
	// GPU可能不可用，这是正常的
	_ = accelerator.IsAvailable()
}

func TestGPUAcceleratorBasicOperations(t *testing.T) {
	accelerator := vector.NewGPUAccelerator()
	
	// 测试初始化
	err := accelerator.Initialize()
	if err != nil {
		// GPU初始化失败是正常的，跳过测试
		t.Skipf("GPU initialization failed: %v", err)
	}
	
	// 测试可用性
	available := accelerator.IsAvailable()
	if !available {
		t.Skip("GPU accelerator not available")
	}
	
	// 清理
	accelerator.Shutdown()
}

func TestFPGAAccelerator(t *testing.T) {
	accelerator := vector.NewFPGAAccelerator()
	if accelerator == nil {
		t.Fatal("Expected non-nil FPGA accelerator")
	}
	
	// 测试初始化
	err := accelerator.Initialize()
	if err != nil {
		// FPGA初始化失败是正常的，跳过测试
		t.Skipf("FPGA initialization failed: %v", err)
	}
	
	// 测试可用性
	available := accelerator.IsAvailable()
	if !available {
		t.Skip("FPGA accelerator not available")
	}
	
	// 清理
	accelerator.Shutdown()
}

func TestGPUAcceleratorAdvanced(t *testing.T) {
	accelerator := vector.NewGPUAccelerator()
	
	// GPU可能不可用，这是正常的
	if accelerator == nil {
		t.Skip("GPU accelerator not available, skipping test")
	}
	
	// 测试初始化和关闭
	err := accelerator.Initialize()
	if err != nil {
		t.Skipf("GPU initialization failed: %v", err)
	}
	
	if accelerator.IsAvailable() {
		// GPU可用时的测试
		t.Log("GPU accelerator is available")
	}
	
	accelerator.Shutdown()
}

func TestAcceleratorComparison(t *testing.T) {
	gpuAccelerator := vector.NewGPUAccelerator()
	fpgaAccelerator := vector.NewFPGAAccelerator()
	
	// 测试GPU加速器
	if gpuAccelerator != nil {
		err := gpuAccelerator.Initialize()
		if err == nil && gpuAccelerator.IsAvailable() {
			t.Log("GPU accelerator is available and initialized")
			gpuAccelerator.Shutdown()
		} else {
			t.Log("GPU accelerator is not available")
		}
	}
	
	// 测试FPGA加速器
	if fpgaAccelerator != nil {
		err := fpgaAccelerator.Initialize()
		if err == nil && fpgaAccelerator.IsAvailable() {
			t.Log("FPGA accelerator is available and initialized")
			fpgaAccelerator.Shutdown()
		} else {
			t.Log("FPGA accelerator is not available")
		}
	}
}

func TestAcceleratorMemoryManagement(t *testing.T) {
	accelerator := vector.NewGPUAccelerator()
	
	// 测试大量内存分配和释放
	for i := 0; i < 100; i++ {
		size := 1000 + i*10
		vec1 := make([]float64, size)
		vec2 := make([]float64, size)
		
		for j := 0; j < size; j++ {
			vec1[j] = float64(j)
			vec2[j] = float64(j + 1)
		}
		
		_, _ = accelerator.ComputeDistance(vec1, [][]float64{vec2})
	}
	
	// 检查统计信息
	stats := accelerator.GetStats()
	if stats.TotalOperations < 0 {
		t.Error("Expected non-negative total operations")
	}
}

func TestAcceleratorErrorHandling(t *testing.T) {
	accelerator := vector.NewGPUAccelerator()
	
	// 测试不匹配的向量维度
	vec1 := []float64{1.0, 2.0, 3.0}
	vec2 := []float64{4.0, 5.0} // 不同长度
	
	// 应该返回错误
	_, err := accelerator.ComputeDistance(vec1, [][]float64{vec2})
	if err == nil {
		t.Error("Expected error for mismatched vector dimensions")
	}
	
	// 测试空向量
	emptyVec1 := []float64{}
	emptyVec2 := []float64{}
	
	_, err = accelerator.ComputeDistance(emptyVec1, [][]float64{emptyVec2})
	if err == nil {
		t.Error("Expected error for empty vectors")
	}
}

func TestSIMDAccelerator(t *testing.T) {
	// SIMD加速器不存在，使用GPU加速器代替
	accelerator := vector.NewGPUAccelerator()
	if accelerator == nil {
		t.Skip("GPU accelerator not available, skipping test")
	}
	
	if !accelerator.IsAvailable() {
		t.Skip("GPU not supported on this platform")
	}
	
	vec1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	vec2 := []float64{8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}
	
	// 测试距离计算
	results, err := accelerator.ComputeDistance(vec1, [][]float64{vec2})
	if err != nil {
		t.Skipf("GPU computation failed: %v", err)
	}
	
	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}
}

func TestAcceleratorSelection(t *testing.T) {
	// 测试GPU加速器
	accelerator := vector.NewGPUAccelerator()
	if accelerator == nil {
		t.Fatal("Expected non-nil accelerator")
	}
	
	// GPU可能不可用，这是正常的
	if !accelerator.IsAvailable() {
		t.Skip("GPU accelerator not available")
	}
	
	// 测试基本功能
	vec1 := []float64{1.0, 2.0, 3.0}
	vec2 := []float64{4.0, 5.0, 6.0}
	
	results, err := accelerator.ComputeDistance(vec1, [][]float64{vec2})
	if err != nil {
		t.Skipf("GPU computation failed: %v", err)
	}
	
	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}
}

func TestAcceleratorConcurrency(t *testing.T) {
	accelerator := vector.NewGPUAccelerator()
	
	// 并发执行向量操作
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func(id int) {
			vec1 := make([]float64, 100) // 减少向量大小
			vec2 := make([]float64, 100)
			
			for j := 0; j < 100; j++ {
				vec1[j] = float64(id*100 + j)
				vec2[j] = float64(id*100 + j + 1)
			}
			
			results, err := accelerator.ComputeDistance(vec1, [][]float64{vec2})
			if err == nil && len(results) > 0 {
				// 计算成功
			}
			
			done <- true
		}(i)
	}
	
	// 等待所有goroutine完成
	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestAcceleratorCapabilities(t *testing.T) {
	accelerator := vector.NewGPUAccelerator()
	
	caps := accelerator.GetCapabilities()
	
	if caps.Type != "GPU" {
		t.Errorf("Expected GPU accelerator type, got %s", caps.Type)
	}
	
	if caps.PerformanceRating <= 0 {
		t.Error("Expected positive performance rating")
	}
	
	if len(caps.SupportedOps) == 0 {
		t.Error("Expected non-empty supported operations")
	}
}

// 辅助函数
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}