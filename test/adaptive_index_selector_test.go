package test

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/vector"
	"testing"
	"time"
)

// MockVectorDB 模拟 VectorDB 用于测试
type MockVectorDB struct {
	vectors []entity.Point
}

func (m *MockVectorDB) GetVectors() []entity.Point {
	return m.vectors
}

// TestNewAdaptiveIndexSelector 测试选择器创建和初始状态
func TestNewAdaptiveIndexSelector(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	if selector == nil {
		t.Fatal("NewAdaptiveIndexSelector 返回 nil")
	}

	// 验证初始状态（通过GetInsights间接验证）
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证初始窗口为空
	if insights["current_records"] != 0 {
		t.Errorf("初始记录数应该为 0，实际为 %v", insights["current_records"])
	}

	// 验证默认窗口大小
	if insights["window_size"] != 100 {
		t.Errorf("期望窗口大小为 100，实际为 %v", insights["window_size"])
	}

	// 验证优化间隔
	if insights["optimization_interval"] != "5m0s" {
		t.Errorf("期望优化间隔为 5m0s，实际为 %v", insights["optimization_interval"])
	}
}

// TestRecordPerformance 测试性能记录功能
func TestRecordPerformance(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 记录第一个性能数据
	selector.RecordPerformance(
		vector.StrategyBruteForce,
		100*time.Millisecond,
		0.95,
		1000,
		128,
	)

	// 通过GetInsights验证性能记录
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证记录数量
	if insights["current_records"] != 1 {
		t.Errorf("期望窗口中有 1 条记录，实际有 %v 条", insights["current_records"])
	}

	// 验证平均延迟（应该等于单条记录的延迟）
	if insights["avg_latency"] != "100ms" {
		t.Errorf("期望平均延迟为 100ms，实际为 %v", insights["avg_latency"])
	}

	// 验证平均质量
	if insights["avg_quality"] != 0.95 {
		t.Errorf("期望平均质量为 0.95，实际为 %v", insights["avg_quality"])
	}
}

// TestWindowSizeLimit 测试窗口大小限制
func TestWindowSizeLimit(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 记录超过窗口大小的数据
	for i := 0; i < 150; i++ {
		selector.RecordPerformance(
			vector.StrategyBruteForce,
			time.Duration(i)*time.Millisecond,
			0.9,
			1000,
			128,
		)
	}

	// 通过GetInsights验证窗口大小限制
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证窗口大小限制为100
	if insights["current_records"] != 100 {
		t.Errorf("期望窗口大小为 100，实际为 %v", insights["current_records"])
	}

	// 验证窗口大小设置
	if insights["window_size"] != 100 {
		t.Errorf("期望窗口大小为 100，实际为 %v", insights["window_size"])
	}
}

// TestGetInsights 测试洞察信息获取
func TestGetInsights(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 测试空数据情况
	insights := selector.GetInsights()
	if insights["status"] != "no_data" {
		t.Error("空数据时应该返回 no_data 状态")
	}

	// 添加一些测试数据
	strategies := []vector.IndexStrategy{
		vector.StrategyBruteForce,
		vector.StrategyIVF,
		vector.StrategyHNSW,
	}

	for i := 0; i < 20; i++ {
		strategy := strategies[i%len(strategies)]
		latency := time.Duration(50+i*10) * time.Millisecond
		quality := 0.8 + float64(i%3)*0.1

		selector.RecordPerformance(strategy, latency, quality, 1000+i*100, 128)
	}

	// 获取洞察信息
	insights = selector.GetInsights()

	// 验证基础统计信息
	if insights["window_size"] != 100 {
		t.Error("窗口大小不正确")
	}

	if insights["current_records"] != 20 {
		t.Error("当前记录数不正确")
	}

	// 验证策略统计信息
	strategyStats, ok := insights["strategy_statistics"].(map[string]interface{})
	if !ok {
		t.Error("策略统计信息格式不正确")
	}

	if len(strategyStats) != 3 {
		t.Errorf("期望有 3 种策略的统计信息，实际有 %d 种", len(strategyStats))
	}

	// 验证趋势分析
	trendAnalysis, ok := insights["trend_analysis"].(map[string]interface{})
	if !ok {
		t.Error("趋势分析信息格式不正确")
	}

	if trendAnalysis["status"] != "stable" && trendAnalysis["status"] != "improving" && trendAnalysis["status"] != "degrading" {
		t.Error("趋势分析状态不正确")
	}
}

// TestGetOptimalStrategy 测试最优策略选择
func TestGetOptimalStrategy(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 测试数据不足的情况
	ctx := vector.SearchContext{
		QueryVector:  make([]float64, 128),
		K:            10,
		QualityLevel: 0.8,
	}

	strategy := selector.GetOptimalStrategy(ctx)
	if strategy != -1 {
		t.Error("数据不足时应该返回 -1")
	}

	// 添加足够的测试数据
	for i := 0; i < 10; i++ {
		// 让 HNSW 策略表现更好
		if i%2 == 0 {
			selector.RecordPerformance(
				vector.StrategyHNSW,
				50*time.Millisecond,
				0.95,
				1000,
				128,
			)
		} else {
			selector.RecordPerformance(
				vector.StrategyBruteForce,
				200*time.Millisecond,
				0.85,
				1000,
				128,
			)
		}
	}

	strategy = selector.GetOptimalStrategy(ctx)
	if strategy == -1 {
		t.Error("有足够数据时应该返回有效策略")
	}
}

// TestAnalyzeTrends 测试趋势分析（通过GetInsights间接测试）
func TestAnalyzeTrends(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 测试数据不足的情况
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}
	if insights["status"] != "no_data" {
		t.Error("数据不足时应该返回no_data状态")
	}

	// 添加足够的数据进行趋势分析
	for i := 0; i < 20; i++ {
		// 模拟性能逐渐改善的趋势
		latency := time.Duration(200-i*5) * time.Millisecond
		quality := 0.7 + float64(i)*0.01

		selector.RecordPerformance(
			vector.StrategyHNSW,
			latency,
			quality,
			1000,
			128,
		)
	}

	insights = selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证趋势分析结果
	if insights["trend_analysis"] == nil {
		t.Error("应该包含趋势分析信息")
	}

	if insights["strategy_statistics"] == nil {
		t.Error("应该包含策略统计信息")
	}
}

// TestDetectAnomalies 测试异常检测（通过GetInsights间接测试）
func TestDetectAnomalies(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 测试数据不足的情况
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}
	if insights["status"] != "no_data" {
		t.Error("数据不足时应该返回no_data状态")
	}

	// 添加正常数据
	for i := 0; i < 20; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			100*time.Millisecond, // 稳定的延迟
			0.9,                  // 稳定的质量
			1000,
			128,
		)
	}

	// 添加异常数据
	for i := 0; i < 3; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			500*time.Millisecond, // 异常高的延迟
			0.5,                  // 异常低的质量
			1000,
			128,
		)
	}

	// 测试异常检测
	insights = selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证包含异常检测信息
	if insights["anomalies"] == nil {
		t.Error("应该包含异常检测信息")
	}
	if insights["current_records"] != 23 {
		t.Errorf("期望记录数为 23，实际为 %v", insights["current_records"])
	}
}

// TestAssessDataQuality 测试数据质量评估（通过GetInsights间接测试）
func TestAssessDataQuality(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 测试空数据情况
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}
	if insights["status"] != "no_data" {
		t.Error("数据不足时应该返回no_data状态")
	}

	// 添加高质量数据
	strategies := []vector.IndexStrategy{
		vector.StrategyBruteForce,
		vector.StrategyIVF,
		vector.StrategyHNSW,
		vector.StrategyPQ,
	}

	for i := 0; i < 20; i++ {
		strategy := strategies[i%len(strategies)]
		selector.RecordPerformance(
			strategy,
			100*time.Millisecond,
			0.9,
			1000,
			128,
		)
	}

	insights = selector.GetInsights()

	// 验证数据质量评估结果
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	if insights["data_quality"] == nil {
		t.Error("应该包含数据质量评估")
	}

	if insights["strategy_statistics"] == nil {
		t.Error("应该包含策略统计信息")
	}

	if insights["current_records"] != 20 {
		t.Errorf("期望记录数为 20，实际为 %v", insights["current_records"])
	}
}

// TestGetStrategyRecommendation 测试策略推荐（通过GetInsights间接测试）
func TestGetStrategyRecommendation(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 测试数据不足的情况
	insights := selector.GetInsights()
	if insights["status"] != "no_data" {
		t.Error("数据不足时应该返回 no_data")
	}

	// 添加不同策略的性能数据
	// HNSW 策略表现最好
	for i := 0; i < 5; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			50*time.Millisecond,
			0.95,
			1000,
			128,
		)
	}

	// IVF 策略表现一般
	for i := 0; i < 5; i++ {
		selector.RecordPerformance(
			vector.StrategyIVF,
			100*time.Millisecond,
			0.85,
			1000,
			128,
		)
	}

	insights = selector.GetInsights()
	if insights["status"] == "no_data" {
		t.Error("有足够数据时不应该返回 no_data")
	}

	// 验证推荐结果
	if insights["recommendation"] == nil {
		t.Error("应该包含推荐信息")
	}

	if insights["strategy_statistics"] == nil {
		t.Error("应该包含策略统计信息")
	}

	if insights["current_records"] != 10 {
		t.Errorf("期望记录数为 10，实际为 %v", insights["current_records"])
	}
}

// TestCalculateContextSimilarity 测试上下文相似度计算（通过GetOptimalStrategy间接测试）
func TestCalculateContextSimilarity(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	ctx := vector.SearchContext{
		QueryVector:  make([]float64, 128),
		K:            10,
		QualityLevel: 0.8,
	}

	// 记录一些性能数据
	for i := 0; i < 15; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			time.Duration(50+i*5)*time.Millisecond,
			0.9+float64(i)*0.005,
			1000,
			128,
		)
	}

	// 测试GetOptimalStrategy方法（内部会使用CalculateContextSimilarity）
	strategy := selector.GetOptimalStrategy(ctx)
	if strategy == -1 {
		t.Error("应该返回有效的策略")
	}

	// 验证策略选择的合理性
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}
	if insights["current_records"] != 15 {
		t.Errorf("期望记录数为 15，实际为 %v", insights["current_records"])
	}
}

// TestConcurrentAccess 测试并发访问
func TestConcurrentAccess(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 并发记录性能数据
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 10; j++ {
				selector.RecordPerformance(
					vector.StrategyHNSW,
					time.Duration(id*10+j)*time.Millisecond,
					0.9,
					1000,
					128,
				)
			}
			done <- true
		}(i)
	}

	// 等待所有 goroutine 完成
	for i := 0; i < 10; i++ {
		<-done
	}

	// 通过GetInsights验证数据完整性
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证窗口大小限制为100
	if insights["current_records"] != 100 {
		t.Errorf("期望窗口大小为 100，实际为 %v", insights["current_records"])
	}
}

// TestEstimateSearchQuality 测试搜索质量估算
func TestEstimateSearchQuality(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 记录一些性能数据
	for i := 0; i < 20; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			time.Duration(50+i*5)*time.Millisecond,
			0.9+float64(i%5)*0.02,
			1000,
			128,
		)
	}

	// 测试质量估算
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证质量评估的合理性
	if insights["data_quality"] == nil {
		t.Error("应该包含数据质量评估")
	}
	if insights["strategy_statistics"] == nil {
		t.Error("应该包含策略统计信息")
	}
	if insights["current_records"] != 20 {
		t.Errorf("期望记录数为 20，实际为 %v", insights["current_records"])
	}
}

// TestPerformanceWindowManagement 测试性能窗口管理
func TestPerformanceWindowManagement(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 添加数据超过默认窗口大小（100）
	for i := 0; i < 120; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			time.Duration(i)*time.Millisecond,
			0.9,
			1000,
			128,
		)
	}

	// 通过GetInsights验证窗口大小限制
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证窗口大小限制为100
	if insights["current_records"] != 100 {
		t.Errorf("期望记录数为 100，实际为 %v", insights["current_records"])
	}

	// 验证窗口大小设置
	if insights["window_size"] != 100 {
		t.Errorf("期望窗口大小为 100，实际为 %v", insights["window_size"])
	}
}

// BenchmarkRecordPerformance 性能记录的基准测试
func BenchmarkRecordPerformance(b *testing.B) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			100*time.Millisecond,
			0.9,
			1000,
			128,
		)
	}
}

// BenchmarkGetOptimalStrategy 最优策略选择的基准测试
func BenchmarkGetOptimalStrategy(b *testing.B) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 预填充一些数据
	for i := 0; i < 50; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			time.Duration(50+i)*time.Millisecond,
			0.9,
			1000,
			128,
		)
	}

	ctx := vector.SearchContext{
		QueryVector:  make([]float64, 128),
		K:            10,
		QualityLevel: 0.8,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selector.GetOptimalStrategy(ctx)
	}
}

// BenchmarkGetInsights 洞察信息获取的基准测试
func BenchmarkGetInsights(b *testing.B) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 预填充数据
	strategies := []vector.IndexStrategy{
		vector.StrategyBruteForce,
		vector.StrategyIVF,
		vector.StrategyHNSW,
	}

	for i := 0; i < 100; i++ {
		strategy := strategies[i%len(strategies)]
		selector.RecordPerformance(
			strategy,
			time.Duration(50+i)*time.Millisecond,
			0.8+float64(i%3)*0.1,
			1000,
			128,
		)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		selector.GetInsights()
	}
}

// TestPerformanceWindowAccess 测试性能窗口访问（通过反射或间接方式）
func TestPerformanceWindowAccess(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 记录一些性能数据
	selector.RecordPerformance(vector.StrategyIVF, 10*time.Millisecond, 0.95, 1000, 128)
	selector.RecordPerformance(vector.StrategyHNSW, 5*time.Millisecond, 0.90, 1000, 128)

	// 通过GetInsights间接验证性能窗口
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}
	if insights["current_records"] != 2 {
		t.Errorf("期望记录数为 2，实际为 %v", insights["current_records"])
	}
}

// TestWindowSizeManagement 测试窗口大小管理
func TestWindowSizeManagement(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 记录超过窗口大小的数据
	for i := 0; i < 105; i++ {
		selector.RecordPerformance(vector.StrategyIVF, time.Duration(i)*time.Millisecond, 0.95, 1000, 128)
	}

	// 验证窗口大小限制
	insights := selector.GetInsights()
	if insights["current_records"] != 100 {
		t.Errorf("期望记录数为 100，实际为 %v", insights["current_records"])
	}
}

// TestOptimizationTrigger 测试优化触发机制
func TestOptimizationTrigger(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 记录足够的性能数据
	for i := 0; i < 15; i++ {
		selector.RecordPerformance(vector.StrategyIVF, time.Duration(10+i)*time.Millisecond, 0.95-float64(i)*0.01, 1000, 128)
		selector.RecordPerformance(vector.StrategyHNSW, time.Duration(5+i)*time.Millisecond, 0.90-float64(i)*0.01, 1000, 128)
	}

	// 验证数据记录成功
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}
	if insights["current_records"] != 30 {
		t.Errorf("期望记录数为 30，实际为 %v", insights["current_records"])
	}
}

// TestStrategyPerformanceComparison 测试策略性能比较
func TestStrategyPerformanceComparison(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 记录不同策略的性能数据
	for i := 0; i < 15; i++ {
		// HNSW策略 - 更快但质量稍低
		selector.RecordPerformance(
			vector.StrategyHNSW,
			time.Duration(i*5)*time.Millisecond,
			0.9-float64(i)*0.01,
			1000,
			128,
		)
		// IVF策略 - 较慢但质量更高
		selector.RecordPerformance(
			vector.StrategyIVF,
			time.Duration(i*10)*time.Millisecond,
			0.95-float64(i)*0.005,
			1000,
			128,
		)
	}

	// 验证策略统计信息
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	if insights["strategy_statistics"] == nil {
		t.Error("策略统计信息不应该为空")
	}

	// 验证记录总数
	if insights["current_records"] != 30 {
		t.Errorf("期望记录数为 30，实际为 %v", insights["current_records"])
	}
}

// TestCalculateMean 测试均值计算（通过GetInsights间接测试）
func TestCalculateMean(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 添加一些性能数据来测试均值计算
	for i := 0; i < 5; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			time.Duration(100+i*20)*time.Millisecond, // 100, 120, 140, 160, 180
			0.9,
			1000,
			128,
		)
	}

	// 通过GetInsights验证均值计算功能
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证包含平均延迟信息（间接验证均值计算）
	if insights["avg_latency"] == nil {
		t.Error("应该包含平均延迟信息")
	}

	// 验证平均质量
	if insights["avg_quality"] != 0.9 {
		t.Errorf("期望平均质量为 0.9，实际为 %v", insights["avg_quality"])
	}

	// 验证记录数量
	if insights["current_records"] != 5 {
		t.Errorf("期望记录数为 5，实际为 %v", insights["current_records"])
	}
}

// TestCalculateStdDev 测试标准差计算（通过GetInsights间接测试）
func TestCalculateStdDev(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 添加一些性能数据来测试标准差计算
	for i := 0; i < 10; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			time.Duration(100+i*10)*time.Millisecond,
			0.9,
			1000,
			128,
		)
	}

	// 通过GetInsights验证标准差计算功能
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证包含统计信息（间接验证标准差计算）
	if insights["avg_latency"] == nil {
		t.Error("应该包含平均延迟信息")
	}

	// 验证记录数量
	if insights["current_records"] != 10 {
		t.Errorf("期望记录数为 10，实际为 %v", insights["current_records"])
	}
}

// TestOptimizationTiming 测试优化时机
func TestOptimizationTiming(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 记录性能数据，但数量不足以触发优化
	for i := 0; i < 5; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			100*time.Millisecond,
			0.9,
			1000,
			128,
		)
	}

	// 验证数据记录但优化可能未触发（数据不足）
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}
	if insights["current_records"] != 5 {
		t.Errorf("期望记录数为 5，实际为 %v", insights["current_records"])
	}
}

// TestPerformanceRecordValidation 测试性能记录验证
func TestPerformanceRecordValidation(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 测试负延迟
	selector.RecordPerformance(
		vector.StrategyHNSW,
		-100*time.Millisecond,
		0.9,
		1000,
		128,
	)

	// 测试无效质量值
	selector.RecordPerformance(
		vector.StrategyHNSW,
		100*time.Millisecond,
		-0.5, // 负质量
		1000,
		128,
	)

	selector.RecordPerformance(
		vector.StrategyHNSW,
		100*time.Millisecond,
		1.5, // 质量大于1
		1000,
		128,
	)

	// 测试负向量数量
	selector.RecordPerformance(
		vector.StrategyHNSW,
		100*time.Millisecond,
		0.9,
		-1000,
		128,
	)

	// 测试负维度
	selector.RecordPerformance(
		vector.StrategyHNSW,
		100*time.Millisecond,
		0.9,
		1000,
		-128,
	)

	// 通过GetInsights验证记录数量
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 所有记录都应该被添加，但在数据质量评估中会被标记为无效
	if insights["current_records"] != 5 {
		t.Errorf("期望窗口中有 5 条记录，实际有 %v 条", insights["current_records"])
	}

	// 验证数据质量评估能检测到无效数据（通过GetInsights间接验证）
	if insights["data_quality"] == nil {
		t.Error("应该包含数据质量信息")
	}
}

// TestContextSimilarityEdgeCases 测试上下文相似度计算的边界情况（通过GetOptimalStrategy间接测试）
func TestContextSimilarityEdgeCases(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	ctx := vector.SearchContext{
		QueryVector:  make([]float64, 128),
		K:            10,
		QualityLevel: 0.8,
	}

	// 记录不同维度和向量数量的性能数据
	for i := 0; i < 10; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			time.Duration(50+i*5)*time.Millisecond,
			0.9,
			1000, // 匹配的向量数量
			128,  // 匹配的维度
		)
		selector.RecordPerformance(
			vector.StrategyIVF,
			time.Duration(100+i*10)*time.Millisecond,
			0.95,
			10000, // 不匹配的向量数量
			256,   // 不匹配的维度
		)
	}

	// 测试策略选择（内部会计算上下文相似度）
	strategy := selector.GetOptimalStrategy(ctx)
	if strategy == -1 {
		t.Error("应该返回有效的策略")
	}

	// 验证数据记录
	insights := selector.GetInsights()
	if insights["current_records"] != 20 {
		t.Errorf("期望记录数为 20，实际为 %v", insights["current_records"])
	}
}

// TestGetOptimalStrategyEdgeCases 测试最优策略选择的边界情况
func TestGetOptimalStrategyEdgeCases(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	ctx := vector.SearchContext{
		QueryVector:  make([]float64, 128),
		K:            10,
		QualityLevel: 0.8,
	}

	// 测试所有记录相似度都很低的情况
	for i := 0; i < 10; i++ {
		selector.RecordPerformance(
			vector.StrategyHNSW,
			100*time.Millisecond,
			0.9,
			100000, // 与当前上下文差异很大
			512,    // 与当前上下文差异很大
		)
	}

	strategy := selector.GetOptimalStrategy(ctx)
	if strategy != -1 {
		t.Error("当所有记录相似度都很低时，应该返回 -1")
	}

	// 测试只有一个样本的策略
	selector.RecordPerformance(
		vector.StrategyBruteForce,
		50*time.Millisecond,
		0.95,
		1000, // 与当前上下文相似
		128,  // 与当前上下文相似
	)

	strategy = selector.GetOptimalStrategy(ctx)
	if strategy != -1 {
		t.Error("只有一个样本的策略不应该被选择")
	}

	// 添加足够的相似样本
	selector.RecordPerformance(
		vector.StrategyBruteForce,
		60*time.Millisecond,
		0.92,
		1000,
		128,
	)

	strategy = selector.GetOptimalStrategy(ctx)
	if strategy != vector.StrategyBruteForce {
		t.Errorf("期望选择 StrategyBruteForce，实际选择 %v", strategy)
	}
}

// TestInsightsWithLargeDataset 测试大数据集的洞察信息
func TestInsightsWithLargeDataset(t *testing.T) {
	mockDB := &vector.VectorDB{}
	selector := vector.NewAdaptiveIndexSelector(mockDB)

	// 添加大量数据
	strategies := []vector.IndexStrategy{
		vector.StrategyBruteForce,
		vector.StrategyIVF,
		vector.StrategyHNSW,
		vector.StrategyPQ,
		vector.StrategyHybrid,
	}

	for i := 0; i < 500; i++ {
		strategy := strategies[i%len(strategies)]
		// 模拟不同的性能特征
		latency := time.Duration(50+i%200) * time.Millisecond
		quality := 0.7 + float64(i%30)/100.0
		vectorCount := 1000 + i*10
		dimension := 128 + i%256

		selector.RecordPerformance(strategy, latency, quality, vectorCount, dimension)
	}

	// 获取洞察信息
	insights := selector.GetInsights()
	if insights == nil {
		t.Error("洞察信息不应该为空")
	}

	// 验证窗口大小限制
	if insights["current_records"] != 100 {
		t.Errorf("期望窗口大小为 100，实际为 %v", insights["current_records"])
	}

	// 验证策略统计信息
	strategyStats, ok := insights["strategy_statistics"].(map[string]interface{})
	if !ok {
		t.Error("策略统计信息格式不正确")
	}

	// 应该包含所有策略的统计信息
	if len(strategyStats) != len(strategies) {
		t.Errorf("期望有 %d 种策略的统计信息，实际有 %d 种", len(strategies), len(strategyStats))
	}

	// 验证推荐系统
	recommendation, ok := insights["recommendation"].(map[string]interface{})
	if !ok {
		t.Error("推荐信息格式不正确")
	}

	recommendedStrategy, ok := recommendation["recommended_strategy"].(string)
	if !ok || recommendedStrategy == "" {
		t.Error("应该有推荐策略")
	}

	// 验证异常检测
	anomalies, ok := insights["anomalies"].(map[string]interface{})
	if !ok {
		t.Error("异常检测信息格式不正确")
	}

	if anomalies["status"] == "insufficient_data" {
		t.Error("大数据集不应该返回数据不足")
	}
}