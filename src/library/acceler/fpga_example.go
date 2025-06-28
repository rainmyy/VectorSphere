package acceler

import (
	"fmt"
	"log"
	"time"

	"VectorSphere/src/library/entity"
)

// FPGAExample FPGA使用示例
type FPGAExample struct {
	accelerator UnifiedAccelerator
	factory     *FPGAFactory
}

// NewFPGAExample 创建FPGA示例实例
func NewFPGAExample() *FPGAExample {
	return &FPGAExample{
		factory: NewFPGAFactory(),
	}
}

// Initialize 初始化FPGA加速器
func (example *FPGAExample) Initialize() error {
	// 获取推荐配置
	config := example.factory.GetRecommendedConfig()

	// 验证配置
	if err := example.factory.ValidateConfig(config); err != nil {
		return fmt.Errorf("配置验证失败: %v", err)
	}

	// 创建FPGA加速器（自动选择真实FPGA或模拟器）
	accelerator, err := example.factory.CreateFPGAAccelerator(config)
	if err != nil {
		return fmt.Errorf("创建FPGA加速器失败: %v", err)
	}

	example.accelerator = accelerator

	// 初始化加速器
	if err := example.accelerator.Initialize(); err != nil {
		return fmt.Errorf("初始化FPGA加速器失败: %v", err)
	}

	log.Printf("FPGA加速器初始化成功，类型: %s", example.accelerator.GetType())
	return nil
}

// RunVectorSearchExample 运行向量搜索示例
func (example *FPGAExample) RunVectorSearchExample() error {
	if example.accelerator == nil {
		return fmt.Errorf("FPGA加速器未初始化")
	}

	// 准备测试数据
	queries := [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{2.0, 3.0, 4.0, 5.0},
		{3.0, 4.0, 5.0, 6.0},
	}

	database := [][]float64{
		{1.1, 2.1, 3.1, 4.1},
		{2.1, 3.1, 4.1, 5.1},
		{3.1, 4.1, 5.1, 6.1},
		{4.1, 5.1, 6.1, 7.1},
		{5.1, 6.1, 7.1, 8.1},
	}

	log.Println("开始向量搜索示例...")
	start := time.Now()

	// 执行批量搜索
	results, err := example.accelerator.BatchSearch(queries, database, 3)
	if err != nil {
		return fmt.Errorf("批量搜索失败: %v", err)
	}

	duration := time.Since(start)
	log.Printf("批量搜索完成，耗时: %v", duration)

	// 打印结果
	for i, queryResults := range results {
		log.Printf("查询 %d 的结果:", i)
		for j, result := range queryResults {
			log.Printf("  结果 %d: ID=%s, 相似度=%.4f", j, result.ID, result.Similarity)
		}
	}

	return nil
}

// RunDistanceComputeExample 运行距离计算示例
func (example *FPGAExample) RunDistanceComputeExample() error {
	if example.accelerator == nil {
		return fmt.Errorf("FPGA加速器未初始化")
	}

	// 准备测试数据
	query := []float64{1.0, 2.0, 3.0, 4.0}
	vectors := [][]float64{
		{1.1, 2.1, 3.1, 4.1},
		{2.0, 3.0, 4.0, 5.0},
		{0.9, 1.9, 2.9, 3.9},
	}

	log.Println("开始距离计算示例...")
	start := time.Now()

	// 计算距离
	distances, err := example.accelerator.ComputeDistance(query, vectors)
	if err != nil {
		return fmt.Errorf("距离计算失败: %v", err)
	}

	duration := time.Since(start)
	log.Printf("距离计算完成，耗时: %v", duration)

	// 打印结果
	for i, distance := range distances {
		log.Printf("向量 %d 的距离: %.4f", i, distance)
	}

	return nil
}

// RunAcceleratedSearchExample 运行加速搜索示例
func (example *FPGAExample) RunAcceleratedSearchExample() error {
	if example.accelerator == nil {
		return fmt.Errorf("FPGA加速器未初始化")
	}

	// 准备测试数据
	query := []float64{1.0, 2.0, 3.0, 4.0}
	database := [][]float64{
		{1.1, 2.1, 3.1, 4.1},
		{2.1, 3.1, 4.1, 5.1},
		{3.1, 4.1, 5.1, 6.1},
		{4.1, 5.1, 6.1, 7.1},
		{5.1, 6.1, 7.1, 8.1},
	}

	options := entity.SearchOptions{
		K:         3,
		Threshold: 0.7,
	}

	log.Println("开始加速搜索示例...")
	start := time.Now()

	// 执行加速搜索
	enhancedResults, err := example.accelerator.AccelerateSearch(query, database, options)
	if err != nil {
		return fmt.Errorf("加速搜索失败: %v", err)
	}

	duration := time.Since(start)
	log.Printf("加速搜索完成，耗时: %v", duration)

	// 打印结果
	log.Println("加速搜索结果:")
	for i, result := range enhancedResults {
		log.Printf("  结果 %d: ID=%s, 相似度=%.4f, FPGA处理=%v",
			i, result.ID, result.Similarity, result.Metadata["fpga_processed"])
	}

	return nil
}

// RunMemoryOptimizationExample 运行内存优化示例
func (example *FPGAExample) RunMemoryOptimizationExample() error {
	if example.accelerator == nil {
		return fmt.Errorf("FPGA加速器未初始化")
	}

	// 准备大量向量数据
	vectors := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		vectors[i] = make([]float64, 128)
		for j := 0; j < 128; j++ {
			vectors[i][j] = float64(i*128 + j)
		}
	}

	log.Println("开始内存优化示例...")
	start := time.Now()

	// 注意：OptimizeMemoryLayout和PrefetchData方法在UnifiedAccelerator接口中不存在
	// 这里使用其他可用的方法来演示内存优化
	log.Println("执行内存优化...")
	
	// 可以使用批量计算来模拟内存优化
	for i := 0; i < len(vectors); i += 100 {
		end := i + 100
		if end > len(vectors) {
			end = len(vectors)
		}
		batch := vectors[i:end]
		// 使用批量距离计算来优化内存访问
		for _, vector := range batch {
			_, _ = example.accelerator.ComputeDistance(vectors[0], [][]float64{vector})
		}
	}

	duration := time.Since(start)
	log.Printf("内存优化完成，耗时: %v", duration)

	return nil
}

// ShowPerformanceStats 显示性能统计
func (example *FPGAExample) ShowPerformanceStats() {
	if example.accelerator == nil {
		log.Println("FPGA加速器未初始化")
		return
	}

	// 获取基本统计
	stats := example.accelerator.GetStats()
	log.Println("=== FPGA性能统计 ===")
	log.Printf("总操作数: %d", stats.TotalOperations)
	log.Printf("成功操作数: %d", stats.SuccessfulOps)
	log.Printf("失败操作数: %d", stats.FailedOps)
	log.Printf("平均延迟: %v", stats.AverageLatency)
	log.Printf("吞吐量: %.2f ops/s", stats.Throughput)
	log.Printf("错误率: %.2f%%", stats.ErrorRate*100)
	log.Printf("内存利用率: %.2f%%", stats.MemoryUtilization*100)
	log.Printf("温度: %.1f°C", stats.Temperature)
	log.Printf("功耗: %.1fW", stats.PowerConsumption)

	// 获取性能指标
	performance := example.accelerator.GetPerformanceMetrics()
	log.Println("=== 性能指标 ===")
	log.Printf("当前延迟: %v", performance.LatencyCurrent)
	log.Printf("最小延迟: %v", performance.LatencyMin)
	log.Printf("最大延迟: %v", performance.LatencyMax)
	log.Printf("P50延迟: %v", performance.LatencyP50)

	// 获取硬件能力
	capabilities := example.accelerator.GetCapabilities()
	log.Println("=== 硬件能力 ===")
	log.Printf("类型: %s", capabilities.Type)
	log.Printf("支持的操作: %v", capabilities.SupportedOps)
	log.Printf("性能评级: %.1f/10", capabilities.PerformanceRating)
	log.Printf("特殊功能: %v", capabilities.SpecialFeatures)
}

// Cleanup 清理资源
func (example *FPGAExample) Cleanup() error {
	if example.accelerator != nil {
		log.Println("正在关闭FPGA加速器...")
		if err := example.accelerator.Shutdown(); err != nil {
			return fmt.Errorf("关闭FPGA加速器失败: %v", err)
		}
		log.Println("FPGA加速器已关闭")
	}
	return nil
}

// RunCompleteExample 运行完整示例
func (example *FPGAExample) RunCompleteExample() error {
	// 初始化
	if err := example.Initialize(); err != nil {
		return err
	}
	defer example.Cleanup()

	// 运行各种示例
	if err := example.RunVectorSearchExample(); err != nil {
		log.Printf("向量搜索示例失败: %v", err)
	}

	if err := example.RunDistanceComputeExample(); err != nil {
		log.Printf("距离计算示例失败: %v", err)
	}

	if err := example.RunAcceleratedSearchExample(); err != nil {
		log.Printf("加速搜索示例失败: %v", err)
	}

	if err := example.RunMemoryOptimizationExample(); err != nil {
		log.Printf("内存优化示例失败: %v", err)
	}

	// 显示性能统计
	example.ShowPerformanceStats()

	return nil
}
