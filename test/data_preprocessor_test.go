package test

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/vector"
	"math"
	"testing"
)

func TestNewStandardPreprocessor(t *testing.T) {
	preprocessor := vector.NewStandardPreprocessor()
	if preprocessor == nil {
		t.Fatal("Expected non-nil StandardPreprocessor")
	}

	stats := preprocessor.GetStats()
	if stats.ProcessedVectors != 0 {
		t.Errorf("Expected 0 processed vectors, got %d", stats.ProcessedVectors)
	}
}

func TestNormalizeVectors(t *testing.T) {
	preprocessor := vector.NewStandardPreprocessor()

	vectors := [][]float64{
		{3.0, 4.0},
		{1.0, 1.0},
		{0.0, 5.0},
	}

	normalized := preprocessor.Normalize(vectors)

	if len(normalized) != len(vectors) {
		t.Errorf("Expected %d normalized vectors, got %d", len(vectors), len(normalized))
	}

	// 检查第一个向量的归一化结果
	expectedNorm := math.Sqrt(3.0*3.0 + 4.0*4.0)
	expected := []float64{3.0 / expectedNorm, 4.0 / expectedNorm}

	for i, val := range normalized[0] {
		if math.Abs(val-expected[i]) > 1e-6 {
			t.Errorf("Expected normalized[0][%d] = %f, got %f", i, expected[i], val)
		}
	}

	// 检查归一化后的向量长度是否为1
	norm := math.Sqrt(normalized[0][0]*normalized[0][0] + normalized[0][1]*normalized[0][1])
	if math.Abs(norm-1.0) > 1e-6 {
		t.Errorf("Expected normalized vector length 1.0, got %f", norm)
	}
}

func TestReduceDimension(t *testing.T) {
	preprocessor := vector.NewStandardPreprocessor()

	vectors := [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
		{3.0, 4.0, 5.0, 6.0},
		{4.0, 5.0, 6.0, 7.0},
	}

	reduced, err := preprocessor.ReduceDimension(vectors, 2)
	if err != nil {
		t.Fatalf("Failed to reduce dimension: %v", err)
	}

	if len(reduced) != len(vectors) {
		t.Errorf("Expected %d reduced vectors, got %d", len(vectors), len(reduced))
	}

	for i, vec := range reduced {
		if len(vec) != 2 {
			t.Errorf("Expected reduced vector %d to have 2 dimensions, got %d", i, len(vec))
		}
	}
}

func TestReduceDimensionInvalidTarget(t *testing.T) {
	preprocessor := vector.NewStandardPreprocessor()

	vectors := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	// 目标维度大于原始维度
	_, err := preprocessor.ReduceDimension(vectors, 5)
	if err == nil {
		t.Error("Expected error when target dimension > original dimension")
	}

	// 目标维度为0
	_, err = preprocessor.ReduceDimension(vectors, 0)
	if err == nil {
		t.Error("Expected error when target dimension = 0")
	}
}

func TestQuantizeVectors(t *testing.T) {
	preprocessor := vector.NewStandardPreprocessor()

	vectors := [][]float64{
		{0.1, 0.9, 0.5},
		{0.3, 0.7, 0.2},
		{0.8, 0.4, 0.6},
	}

	quantized, err := preprocessor.Quantize(vectors, 8)
	if err != nil {
		t.Fatalf("Failed to quantize vectors: %v", err)
	}

	if len(quantized) != len(vectors) {
		t.Errorf("Expected %d quantized vectors, got %d", len(vectors), len(quantized))
	}

	// 检查量化值是否在合理范围内
	for i, vec := range quantized {
		for j, val := range vec {
			if val < 0 || val > 255 {
				t.Errorf("Quantized value [%d][%d] = %f out of range [0, 255]", i, j, val)
			}
		}
	}
}

func TestFilterVectors(t *testing.T) {
	preprocessor := vector.NewStandardPreprocessor()

	vectors := [][]float64{
		{0.0, 0.0},   // 零向量
		{1.0, 1.0},   // 正常向量
		{10.0, 10.0}, // 大向量
		{0.1, 0.1},   // 小向量
	}

	criteria := vector.FilterCriteria{
		MinNorm:     0.5,
		MaxNorm:     5.0,
		RemoveZeros: true,
	}

	filtered := preprocessor.Filter(vectors, criteria)

	// 应该只保留第二个向量（norm ≈ 1.414）
	if len(filtered) != 1 {
		t.Errorf("Expected 1 filtered vector, got %d", len(filtered))
	}

	if len(filtered) > 0 {
		expected := []float64{1.0, 1.0}
		for i, val := range filtered[0] {
			if math.Abs(val-expected[i]) > 1e-6 {
				t.Errorf("Expected filtered[0][%d] = %f, got %f", i, expected[i], val)
			}
		}
	}
}

func TestFilterResults(t *testing.T) {
	preprocessor := vector.NewStandardPreprocessor()

	results := []entity.Result{
		{Id: "1", Distance: 0.1, Similarity: 0.9},
		{Id: "2", Distance: 0.5, Similarity: 0.5},
		{Id: "3", Distance: 0.8, Similarity: 0.2},
		{Id: "4", Distance: 1.2, Similarity: 0.1},
	}

	options := vector.SearchOptions{
		QualityLevel: 0.8, // 设置质量等级来过滤结果
	}

	filtered := preprocessor.FilterResults(results, options)

	// 应该返回Similarity >= 0.4的结果（QualityLevel * 0.5 = 0.8 * 0.5 = 0.4）
	// 只有Id为"1"和"2"的结果满足条件
	if len(filtered) != 2 {
		t.Errorf("Expected 2 filtered results, got %d", len(filtered))
	}

	if len(filtered) >= 2 {
		if filtered[0].Id != "1" || filtered[1].Id != "2" {
			t.Errorf("Expected results with ID 1 and 2, got %s and %s", filtered[0].Id, filtered[1].Id)
		}
	}
}

func TestPreprocessorStats(t *testing.T) {
	preprocessor := vector.NewStandardPreprocessor()

	vectors := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	// 执行一些操作
	preprocessor.Normalize(vectors)
	preprocessor.Quantize(vectors, 8)

	stats := preprocessor.GetStats()

	if stats.NormalizedVectors == 0 {
		t.Error("Expected normalized vectors count > 0")
	}

	if stats.QuantizedVectors == 0 {
		t.Error("Expected quantized vectors count > 0")
	}
}

func TestPCATransform(t *testing.T) {
	preprocessor := vector.NewStandardPreprocessor()

	// 创建一些相关的数据
	vectors := [][]float64{
		{1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0},
		{3.0, 3.0, 3.0},
		{4.0, 4.0, 4.0},
		{5.0, 5.0, 5.0},
	}

	// 应用PCA降维到2维
	reduced, err := preprocessor.ReduceDimension(vectors, 2)
	if err != nil {
		t.Fatalf("Failed to apply PCA: %v", err)
	}

	if len(reduced) != len(vectors) {
		t.Errorf("Expected %d reduced vectors, got %d", len(vectors), len(reduced))
	}

	for i, vec := range reduced {
		if len(vec) != 2 {
			t.Errorf("Expected reduced vector %d to have 2 dimensions, got %d", i, len(vec))
		}
	}
}

func TestPreprocessorConcurrency(t *testing.T) {
	preprocessor := vector.NewStandardPreprocessor()

	vectors := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	// 并发执行多个操作
	done := make(chan bool, 3)

	go func() {
		preprocessor.Normalize(vectors)
		done <- true
	}()

	go func() {
		preprocessor.Quantize(vectors, 8)
		done <- true
	}()

	go func() {
		preprocessor.Filter(vectors, vector.FilterCriteria{MinNorm: 0.1, MaxNorm: 100.0})
		done <- true
	}()

	// 等待所有操作完成
	for i := 0; i < 3; i++ {
		<-done
	}

	stats := preprocessor.GetStats()
	if stats.ProcessedVectors == 0 {
		t.Error("Expected some processed vectors after concurrent operations")
	}
}
