package test

import (
	"VectorSphere/library/algorithm"
	"testing"
)

func TestFAISSGPUAccelerator(t *testing.T) {
	// 创建 GPU 加速器
	gpu := algorithm.NewFAISSGPUAccelerator(0, "Flat")

	// 初始化
	err := gpu.Initialize()
	if err != nil {
		t.Skipf("GPU 不可用，跳过测试: %v", err)
	}
	defer gpu.Cleanup()

	// 测试数据
	queries := [][]float64{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
	}

	database := [][]float64{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0},
		{0.0, 0.0, 0.0, 1.0},
	}

	// 执行批量相似度计算
	results, err := gpu.BatchCosineSimilarity(queries, database)
	if err != nil {
		t.Fatalf("批量相似度计算失败: %v", err)
	}

	// 验证结果
	if len(results) != len(queries) {
		t.Errorf("结果数量不匹配，期望 %d，实际 %d", len(queries), len(results))
	}

	t.Logf("GPU 加速器测试通过，结果: %v", results)
}
