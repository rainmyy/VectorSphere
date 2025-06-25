package main

import (
	"VectorSphere/src/library/acceler"
	"fmt"
	"math/rand"
	"time"
)

func main() {
	fmt.Println("=== VectorSphere 整合硬件加速器演示 ===")

	// 创建向量数据库适配器
	adapter := acceler.NewVectorDBAdapter()
	defer adapter.Shutdown()

	fmt.Printf("硬件加速状态: %v\n", adapter.IsEnabled())
	fmt.Printf("可用加速器: %v\n", adapter.GetHardwareManager().GetAvailableAccelerators())

	// 生成测试数据
	dimension := 128
	numVectors := 10000
	numQueries := 100

	fmt.Printf("\n生成测试数据: %d个向量，维度=%d\n", numVectors, dimension)
	database := generateRandomVectors(numVectors, dimension)
	queries := generateRandomVectors(numQueries, dimension)

	// 测试不同工作负载类型
	workloadTypes := []string{
		acceler.WorkloadLowLatency,
		acceler.WorkloadHighThroughput,
		acceler.WorkloadDistributed,
		acceler.WorkloadPersistent,
		acceler.WorkloadBalanced,
	}

	for _, workloadType := range workloadTypes {
		fmt.Printf("\n=== 测试工作负载类型: %s ===\n", workloadType)

		// 优化适配器配置
		err := adapter.OptimizeForWorkload(workloadType, len(queries), dimension)
		if err != nil {
			fmt.Printf("优化配置失败: %v\n", err)
			continue
		}

		// 获取推荐的加速器
		workload := acceler.CreateWorkloadProfile(workloadType, len(queries), dimension)
		recommended := adapter.GetRecommendedAccelerator(workload)
		fmt.Printf("推荐加速器: %s\n", recommended)

		// 估算性能
		performanceEst := adapter.EstimatePerformance(workload)
		fmt.Printf("性能估算: %+v\n", performanceEst)

		// 测试单查询搜索
		testSingleSearch(adapter, queries[0], database, workloadType)

		// 测试批量搜索
		testBatchSearch(adapter, queries[:10], database, workloadType)

		// 测试批量相似度计算
		testBatchSimilarity(adapter, queries[:5], database[:1000], workloadType)
	}

	// 显示性能报告
	fmt.Printf("\n=== 性能报告 ===\n")
	report := adapter.GetPerformanceReport()
	for key, value := range report {
		fmt.Printf("%s: %+v\n", key, value)
	}

	// 测试硬件管理器功能
	testHardwareManager(adapter.GetHardwareManager())

	fmt.Println("\n=== 演示完成 ===")
}

// testSingleSearch 测试单查询搜索
func testSingleSearch(adapter *acceler.VectorDBAdapter, query []float64, database [][]float64, workloadType string) {
	fmt.Printf("\n--- 单查询搜索测试 ---\n")

	start := time.Now()
	results, err := adapter.Search(query, database, 10, workloadType)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("搜索失败: %v\n", err)
		return
	}

	fmt.Printf("搜索耗时: %v\n", duration)
	fmt.Printf("返回结果数: %d\n", len(results))
	if len(results) > 0 {
		fmt.Printf("最佳匹配: ID=%s, 相似度=%.6f\n", results[0].ID, results[0].Similarity)
	}
}

// testBatchSearch 测试批量搜索
func testBatchSearch(adapter *acceler.VectorDBAdapter, queries [][]float64, database [][]float64, workloadType string) {
	fmt.Printf("\n--- 批量搜索测试 ---\n")

	start := time.Now()
	results, err := adapter.BatchSearch(queries, database, 5, workloadType)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("批量搜索失败: %v\n", err)
		return
	}

	fmt.Printf("批量搜索耗时: %v\n", duration)
	fmt.Printf("查询数量: %d\n", len(queries))
	fmt.Printf("平均每查询耗时: %v\n", duration/time.Duration(len(queries)))
	fmt.Printf("吞吐量: %.2f queries/sec\n", float64(len(queries))/duration.Seconds())

	if len(results) > 0 && len(results[0]) > 0 {
		fmt.Printf("第一个查询最佳匹配: ID=%s, 相似度=%.6f\n", results[0][0].ID, results[0][0].Similarity)
	}
}

// testBatchSimilarity 测试批量相似度计算
func testBatchSimilarity(adapter *acceler.VectorDBAdapter, queries [][]float64, database [][]float64, workloadType string) {
	fmt.Printf("\n--- 批量相似度计算测试 ---\n")

	start := time.Now()
	similarities, err := adapter.BatchComputeSimilarity(queries, database, workloadType)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("批量相似度计算失败: %v\n", err)
		return
	}

	fmt.Printf("批量相似度计算耗时: %v\n", duration)
	fmt.Printf("查询数量: %d, 数据库大小: %d\n", len(queries), len(database))
	fmt.Printf("总计算次数: %d\n", len(queries)*len(database))
	fmt.Printf("计算吞吐量: %.2f ops/sec\n", float64(len(queries)*len(database))/duration.Seconds())

	if len(similarities) > 0 && len(similarities[0]) > 0 {
		fmt.Printf("第一个查询第一个结果距离: %.6f\n", similarities[0][0])
	}
}

// testHardwareManager 测试硬件管理器功能
func testHardwareManager(hm *acceler.HardwareManager) {
	fmt.Printf("\n=== 硬件管理器测试 ===\n")

	// 获取系统信息
	systemInfo := hm.GetSystemInfo()
	fmt.Printf("系统信息: %+v\n", systemInfo)

	// 健康检查
	health := hm.HealthCheck()
	fmt.Printf("健康检查: %+v\n", health)

	// 获取所有能力信息
	capabilities := hm.GetAllCapabilities()
	fmt.Printf("\n硬件能力信息:\n")
	for name, caps := range capabilities {
		fmt.Printf("  %s: 类型=%s, 性能评级=%.1f, 内存=%dMB\n",
			name, caps.Type, caps.PerformanceRating, caps.MemorySize/(1024*1024))
	}

	// 获取统计信息
	stats := hm.GetAllStats()
	fmt.Printf("\n统计信息:\n")
	for name, stat := range stats {
		fmt.Printf("  %s: 总操作=%d, 成功=%d, 平均延迟=%v\n",
			name, stat.TotalOperations, stat.SuccessfulOps, stat.AverageLatency)
	}

	// 获取性能指标
	metrics := hm.GetAllPerformanceMetrics()
	fmt.Printf("\n性能指标:\n")
	for name, metric := range metrics {
		fmt.Printf("  %s: 当前吞吐量=%.2f, 缓存命中率=%.2f%%\n",
			name, metric.ThroughputCurrent, metric.CacheHitRate*100)
	}

	// 测试不同类型的最佳加速器选择
	workloadTypes := []string{"low_latency", "high_throughput", "distributed", "persistent"}
	fmt.Printf("\n最佳加速器选择:\n")
	for _, workloadType := range workloadTypes {
		best := hm.GetBestAccelerator(workloadType)
		if best != nil {
			fmt.Printf("  %s: %s\n", workloadType, best.GetType())
		} else {
			fmt.Printf("  %s: 无可用加速器\n", workloadType)
		}
	}
}

// generateRandomVectors 生成随机向量
func generateRandomVectors(count, dimension int) [][]float64 {
	vectors := make([][]float64, count)
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < count; i++ {
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rand.Float64()*2 - 1 // [-1, 1]范围内的随机数
		}
	}

	return vectors
}

// demonstrateSpecificAccelerators 演示特定加速器功能
func demonstrateSpecificAccelerators(hm *acceler.HardwareManager) {
	fmt.Printf("\n=== 特定加速器功能演示 ===\n")

	// 演示GPU加速器
	gpuAccelerators := hm.GetAcceleratorsByType(acceler.AcceleratorGPU)
	if len(gpuAccelerators) > 0 {
		fmt.Printf("\nGPU加速器功能:\n")
		gpu := gpuAccelerators[0]
		caps := gpu.GetCapabilities()
		fmt.Printf("  支持的操作: %v\n", caps.SupportedOps)
		fmt.Printf("  特殊功能: %v\n", caps.SpecialFeatures)
	}

	// 演示FPGA加速器
	fpgaAccelerators := hm.GetAcceleratorsByType(acceler.AcceleratorFPGA)
	if len(fpgaAccelerators) > 0 {
		fmt.Printf("\nFPGA加速器功能:\n")
		fpga := fpgaAccelerators[0]
		caps := fpga.GetCapabilities()
		fmt.Printf("  延迟: %v\n", caps.Latency)
		fmt.Printf("  功耗: %.1fW\n", caps.PowerConsumption)
	}

	// 演示PMem加速器
	pmemAccelerators := hm.GetAcceleratorsByType(acceler.AcceleratorPMem)
	if len(pmemAccelerators) > 0 {
		fmt.Printf("\nPMem加速器功能:\n")
		pmem := pmemAccelerators[0]
		caps := pmem.GetCapabilities()
		fmt.Printf("  内存大小: %dGB\n", caps.MemorySize/(1024*1024*1024))
		fmt.Printf("  带宽: %dGB/s\n", caps.Bandwidth/(1024*1024*1024))
	}

	// 演示RDMA加速器
	rdmaAccelerators := hm.GetAcceleratorsByType(acceler.AcceleratorRDMA)
	if len(rdmaAccelerators) > 0 {
		fmt.Printf("\nRDMA加速器功能:\n")
		rdma := rdmaAccelerators[0]
		caps := rdma.GetCapabilities()
		fmt.Printf("  网络延迟: %v\n", caps.Latency)
		fmt.Printf("  网络带宽: %dGbps\n", caps.Bandwidth/(1024*1024*1024/8))
		
		// 如果是RDMA加速器，显示集群信息
		if rdmaAcc, ok := rdma.(*acceler.RDMAAccelerator); ok {
			clusterInfo := rdmaAcc.GetClusterInfo()
			fmt.Printf("  集群节点数: %d\n", len(clusterInfo))
			for addr, node := range clusterInfo {
				fmt.Printf("    节点 %s: 连接=%v, 延迟=%v\n", addr, node.Connected, node.Latency)
			}
		}
	}
}