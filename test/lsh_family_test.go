package test

import (
	"VectorSphere/src/library/enum"
	"VectorSphere/src/vector"
	"testing"
	"time"
)

// MockLSHHashFunction 模拟LSH哈希函数
type MockLSHHashFunction struct {
	familyType enum.LSHFamilyType
	parameters map[string]interface{}
}

func (m *MockLSHHashFunction) Hash(vector []float64) (uint64, error) {
	// 简单的模拟哈希函数
	hash := uint64(0)
	for i, v := range vector {
		hash += uint64(v * float64(i+1))
	}
	return hash, nil
}

func (m *MockLSHHashFunction) GetType() enum.LSHFamilyType {
	return m.familyType
}

func (m *MockLSHHashFunction) GetParameters() map[string]interface{} {
	return m.parameters
}

// TestNewLSHFamily 测试LSH族创建
func TestNewLSHFamily(t *testing.T) {
	parameters := map[string]interface{}{
		"w": 4.5,
		"r": 2.0,
	}

	lshFamily := vector.NewLSHFamily(
		"test-family-1",
		enum.LSHFamilyRandomProjection,
		128,
		10,
		parameters,
	)

	if lshFamily == nil {
		t.Fatal("LSH族创建失败")
	}

	if lshFamily.ID != "test-family-1" {
		t.Errorf("期望ID为test-family-1，实际为%s", lshFamily.ID)
	}

	if lshFamily.FamilyType != enum.LSHFamilyRandomProjection {
		t.Errorf("期望族类型为RandomProjection，实际为%v", lshFamily.FamilyType)
	}

	if lshFamily.Dimension != 128 {
		t.Errorf("期望维度为128，实际为%d", lshFamily.Dimension)
	}

	if lshFamily.NumHashFunctions != 10 {
		t.Errorf("期望哈希函数数量为10，实际为%d", lshFamily.NumHashFunctions)
	}

	if lshFamily.W != 4.5 {
		t.Errorf("期望W参数为4.5，实际为%f", lshFamily.W)
	}

	if lshFamily.R != 2.0 {
		t.Errorf("期望R参数为2.0，实际为%f", lshFamily.R)
	}

	if lshFamily.UsageCount != 0 {
		t.Errorf("期望初始使用次数为0，实际为%d", lshFamily.UsageCount)
	}

	if lshFamily.Performance == nil {
		t.Error("性能指标不应为nil")
	}

	if lshFamily.Performance.TotalQueries != 0 {
		t.Errorf("期望初始查询次数为0，实际为%d", lshFamily.Performance.TotalQueries)
	}
}

// TestNewLSHFamilyWithNilParameters 测试使用nil参数创建LSH族
func TestNewLSHFamilyWithNilParameters(t *testing.T) {
	lshFamily := vector.NewLSHFamily(
		"test-family-nil",
		enum.LSHFamilyMinHash,
		64,
		5,
		nil,
	)

	if lshFamily == nil {
		t.Fatal("LSH族创建失败")
	}

	// 使用默认值
	if lshFamily.W != 4.0 {
		t.Errorf("期望默认W参数为4.0，实际为%f", lshFamily.W)
	}

	if lshFamily.R != 1.0 {
		t.Errorf("期望默认R参数为1.0，实际为%f", lshFamily.R)
	}
}

// TestNewLSHFamilyWithEmptyParameters 测试使用空参数创建LSH族
func TestNewLSHFamilyWithEmptyParameters(t *testing.T) {
	parameters := map[string]interface{}{}

	lshFamily := vector.NewLSHFamily(
		"test-family-empty",
		enum.LSHFamilyP2LSH,
		256,
		15,
		parameters,
	)

	if lshFamily == nil {
		t.Fatal("LSH族创建失败")
	}

	// 使用默认值
	if lshFamily.W != 4.0 {
		t.Errorf("期望默认W参数为4.0，实际为%f", lshFamily.W)
	}

	if lshFamily.R != 1.0 {
		t.Errorf("期望默认R参数为1.0，实际为%f", lshFamily.R)
	}
}

// TestUpdatePerformance 测试性能指标更新
func TestUpdatePerformance(t *testing.T) {
	lshFamily := vector.NewLSHFamily(
		"test-performance",
		enum.LSHFamilyAngular,
		128,
		8,
		nil,
	)

	// 第一次更新
	recall1 := 0.8
	precision1 := 0.9
	queryTime1 := 50 * time.Millisecond
	collisionRate1 := 0.1

	lshFamily.UpdatePerformance(recall1, precision1, queryTime1, collisionRate1)

	if lshFamily.UsageCount != 1 {
		t.Errorf("期望使用次数为1，实际为%d", lshFamily.UsageCount)
	}

	if lshFamily.Performance.TotalQueries != 1 {
		t.Errorf("期望总查询次数为1，实际为%d", lshFamily.Performance.TotalQueries)
	}

	// 检查指数移动平均计算
	alpha := 0.1
	expectedRecall := recall1 * alpha
	if abs(lshFamily.Performance.AvgRecall-expectedRecall) > 0.001 {
		t.Errorf("期望平均召回率为%f，实际为%f", expectedRecall, lshFamily.Performance.AvgRecall)
	}

	expectedPrecision := precision1 * alpha
	if abs(lshFamily.Performance.AvgPrecision-expectedPrecision) > 0.001 {
		t.Errorf("期望平均精确率为%f，实际为%f", expectedPrecision, lshFamily.Performance.AvgPrecision)
	}

	if lshFamily.Performance.AvgQueryTime != queryTime1 {
		t.Errorf("期望平均查询时间为%v，实际为%v", queryTime1, lshFamily.Performance.AvgQueryTime)
	}

	// 第二次更新
	recall2 := 0.7
	precision2 := 0.85
	queryTime2 := 60 * time.Millisecond
	collisionRate2 := 0.15

	lshFamily.UpdatePerformance(recall2, precision2, queryTime2, collisionRate2)

	if lshFamily.UsageCount != 2 {
		t.Errorf("期望使用次数为2，实际为%d", lshFamily.UsageCount)
	}

	if lshFamily.Performance.TotalQueries != 2 {
		t.Errorf("期望总查询次数为2，实际为%d", lshFamily.Performance.TotalQueries)
	}

	// 检查指数移动平均更新
	expectedRecall2 := expectedRecall*(1-alpha) + recall2*alpha
	if abs(lshFamily.Performance.AvgRecall-expectedRecall2) > 0.001 {
		t.Errorf("期望平均召回率为%f，实际为%f", expectedRecall2, lshFamily.Performance.AvgRecall)
	}
}

// TestUpdatePerformanceWithNilPerformance 测试性能为nil时的更新
func TestUpdatePerformanceWithNilPerformance(t *testing.T) {
	lshFamily := vector.NewLSHFamily(
		"test-nil-performance",
		enum.LSHFamilyEuclidean,
		128,
		8,
		nil,
	)

	// 手动设置Performance为nil
	lshFamily.Performance = nil

	recall := 0.8
	precision := 0.9
	queryTime := 50 * time.Millisecond
	collisionRate := 0.1

	lshFamily.UpdatePerformance(recall, precision, queryTime, collisionRate)

	if lshFamily.Performance == nil {
		t.Error("性能指标应该被重新创建")
	}

	if lshFamily.Performance.TotalQueries != 1 {
		t.Errorf("期望总查询次数为1，实际为%d", lshFamily.Performance.TotalQueries)
	}
}

// TestGetEffectiveness 测试有效性评分计算
func TestGetEffectiveness(t *testing.T) {
	lshFamily := vector.NewLSHFamily(
		"test-effectiveness",
		enum.LSHFamilyRandomProjection,
		128,
		8,
		nil,
	)

	// 初始状态应该返回0
	effectiveness := lshFamily.GetEffectiveness()
	if effectiveness != 0.0 {
		t.Errorf("期望初始有效性为0.0，实际为%f", effectiveness)
	}

	// 更新性能指标
	recall := 0.8
	precision := 0.9
	queryTime := 50 * time.Millisecond // 小于100ms，应该有较高的时间评分
	collisionRate := 0.1

	lshFamily.UpdatePerformance(recall, precision, queryTime, collisionRate)

	effectiveness = lshFamily.GetEffectiveness()
	if effectiveness <= 0.0 {
		t.Errorf("期望有效性大于0，实际为%f", effectiveness)
	}

	// 验证计算公式
	recallWeight := 0.4
	precisionWeight := 0.4
	timeWeight := 0.2
	maxAcceptableTime := 100 * time.Millisecond
	timeScore := 1.0 - float64(queryTime)/float64(maxAcceptableTime)

	// 使用指数移动平均后的实际值计算期望
	alpha := 0.1
	expectedRecall := recall * alpha       // 因为初始值为0
	expectedPrecision := precision * alpha // 因为初始值为0
	expected := expectedRecall*recallWeight + expectedPrecision*precisionWeight + timeScore*timeWeight
	if abs(effectiveness-expected) > 0.001 {
		t.Errorf("期望有效性为%f，实际为%f", expected, effectiveness)
	}
}

// TestGetEffectivenessWithSlowQuery 测试慢查询的有效性评分
func TestGetEffectivenessWithSlowQuery(t *testing.T) {
	lshFamily := vector.NewLSHFamily(
		"test-slow-query",
		enum.LSHFamilyRandomProjection,
		128,
		8,
		nil,
	)

	// 使用很慢的查询时间
	recall := 0.8
	precision := 0.9
	queryTime := 200 * time.Millisecond // 大于100ms
	collisionRate := 0.1

	lshFamily.UpdatePerformance(recall, precision, queryTime, collisionRate)

	effectiveness := lshFamily.GetEffectiveness()

	// 时间评分应该为0（因为查询时间超过了最大可接受时间）
	recallWeight := 0.4
	precisionWeight := 0.4
	timeWeight := 0.2
	timeScore := 0.0 // 查询时间太慢

	// 使用指数移动平均后的实际值计算期望
	alpha := 0.1
	expectedRecall := recall * alpha       // 因为初始值为0
	expectedPrecision := precision * alpha // 因为初始值为0
	expected := expectedRecall*recallWeight + expectedPrecision*precisionWeight + timeScore*timeWeight
	if abs(effectiveness-expected) > 0.001 {
		t.Errorf("期望有效性为%f，实际为%f", expected, effectiveness)
	}
}

// TestIsExpired 测试过期检查
func TestIsExpired(t *testing.T) {
	lshFamily := vector.NewLSHFamily(
		"test-expired",
		enum.LSHFamilyMinHash,
		128,
		8,
		nil,
	)

	// 刚创建的LSH族不应该过期
	if lshFamily.IsExpired(1 * time.Hour) {
		t.Error("刚创建的LSH族不应该过期")
	}

	// 模拟过期情况
	if !lshFamily.IsExpired(0) {
		t.Error("过期时间为0时应该过期")
	}

	// 更新使用时间
	lshFamily.UpdatePerformance(0.8, 0.9, 50*time.Millisecond, 0.1)

	// 更新后不应该过期
	if lshFamily.IsExpired(1 * time.Hour) {
		t.Error("更新后的LSH族不应该过期")
	}
}

// TestConcurrentAccess 测试并发访问
func TestLSHFamilyConcurrentAccess(t *testing.T) {
	lshFamily := vector.NewLSHFamily(
		"test-concurrent",
		enum.LSHFamilyP2LSH,
		128,
		8,
		nil,
	)

	// 并发更新性能指标
	const numGoroutines = 10
	const updatesPerGoroutine = 100

	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			for j := 0; j < updatesPerGoroutine; j++ {
				lshFamily.UpdatePerformance(0.8, 0.9, 50*time.Millisecond, 0.1)
			}
			done <- true
		}()
	}

	// 并发读取有效性
	for i := 0; i < numGoroutines; i++ {
		go func() {
			for j := 0; j < updatesPerGoroutine; j++ {
				lshFamily.GetEffectiveness()
				lshFamily.IsExpired(1 * time.Hour)
			}
			done <- true
		}()
	}

	// 等待所有goroutine完成
	for i := 0; i < numGoroutines*2; i++ {
		<-done
	}

	// 验证最终状态
	expectedUsageCount := int64(numGoroutines * updatesPerGoroutine)
	if lshFamily.UsageCount != expectedUsageCount {
		t.Errorf("期望使用次数为%d，实际为%d", expectedUsageCount, lshFamily.UsageCount)
	}

	if lshFamily.Performance.TotalQueries != expectedUsageCount {
		t.Errorf("期望总查询次数为%d，实际为%d", expectedUsageCount, lshFamily.Performance.TotalQueries)
	}
}

// TestGetFloatParameter 测试参数获取辅助函数
func TestGetFloatParameter(t *testing.T) {
	// 测试nil参数
	lshFamily1 := vector.NewLSHFamily("test1", enum.LSHFamilyRandomProjection, 128, 8, nil)
	if lshFamily1.W != 4.0 {
		t.Errorf("nil参数时期望W为4.0，实际为%f", lshFamily1.W)
	}

	// 测试存在的参数
	params := map[string]interface{}{
		"w": 5.5,
		"r": 3.0,
	}
	lshFamily2 := vector.NewLSHFamily("test2", enum.LSHFamilyRandomProjection, 128, 8, params)
	if lshFamily2.W != 5.5 {
		t.Errorf("期望W为5.5，实际为%f", lshFamily2.W)
	}
	if lshFamily2.R != 3.0 {
		t.Errorf("期望R为3.0，实际为%f", lshFamily2.R)
	}

	// 测试错误类型的参数
	params2 := map[string]interface{}{
		"w": "invalid",
		"r": 123, // int而不是float64
	}
	lshFamily3 := vector.NewLSHFamily("test3", enum.LSHFamilyRandomProjection, 128, 8, params2)
	if lshFamily3.W != 4.0 {
		t.Errorf("无效参数时期望W为默认值4.0，实际为%f", lshFamily3.W)
	}
	if lshFamily3.R != 1.0 {
		t.Errorf("无效参数时期望R为默认值1.0，实际为%f", lshFamily3.R)
	}
}

// TestAllLSHFamilyTypes 测试所有LSH族类型
func TestAllLSHFamilyTypes(t *testing.T) {
	types := []enum.LSHFamilyType{
		enum.LSHFamilyRandomProjection,
		enum.LSHFamilyMinHash,
		enum.LSHFamilyP2LSH,
		enum.LSHFamilyAngular,
		enum.LSHFamilyEuclidean,
	}

	for i, familyType := range types {
		lshFamily := vector.NewLSHFamily(
			string(rune('A'+i)),
			familyType,
			128,
			8,
			nil,
		)

		if lshFamily.FamilyType != familyType {
			t.Errorf("期望族类型为%v，实际为%v", familyType, lshFamily.FamilyType)
		}
	}
}

// TestPerformanceMetricsEdgeCases 测试性能指标边界情况
func TestPerformanceMetricsEdgeCases(t *testing.T) {
	lshFamily := vector.NewLSHFamily(
		"test-edge-cases",
		enum.LSHFamilyRandomProjection,
		128,
		8,
		nil,
	)

	// 测试极值
	lshFamily.UpdatePerformance(0.0, 0.0, 0, 0.0)
	effectiveness := lshFamily.GetEffectiveness()
	if effectiveness < 0.0 || effectiveness > 1.0 {
		t.Errorf("有效性应该在0-1之间，实际为%f", effectiveness)
	}

	lshFamily.UpdatePerformance(1.0, 1.0, 0, 1.0)
	effectiveness = lshFamily.GetEffectiveness()
	if effectiveness < 0.0 || effectiveness > 1.0 {
		t.Errorf("有效性应该在0-1之间，实际为%f", effectiveness)
	}
}

// BenchmarkLSHFamilyOperations 基准测试
func BenchmarkLSHFamilyOperations(b *testing.B) {
	lshFamily := vector.NewLSHFamily(
		"benchmark-test",
		enum.LSHFamilyRandomProjection,
		128,
		8,
		nil,
	)

	b.ResetTimer()

	b.Run("UpdatePerformance", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			lshFamily.UpdatePerformance(0.8, 0.9, 50*time.Millisecond, 0.1)
		}
	})

	b.Run("GetEffectiveness", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			lshFamily.GetEffectiveness()
		}
	})

	b.Run("IsExpired", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			lshFamily.IsExpired(1 * time.Hour)
		}
	})
}
