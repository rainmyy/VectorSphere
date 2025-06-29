package main

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"fmt"
	"log"
)

func main() {
	// 测试RDMA加速器接口一致性
	fmt.Println("测试RDMA加速器接口一致性...")

	// 创建RDMA配置
	config := &acceler.RDMAConfig{
		Enable:    true,
		QueueSize: 1024,
		Protocol:  "IB",
	}

	// 测试真实RDMA加速器
	rdmaReal := acceler.NewRDMAAccelerator(0, 1, config)
	testAccelerator(rdmaReal, "真实RDMA加速器")

	// 测试模拟RDMA加速器
	rdmaStub := acceler.NewRDMAAccelerator(0, 1, config)
	testAccelerator(rdmaStub, "模拟RDMA加速器")

	fmt.Println("接口一致性测试完成！")
}

func testAccelerator(acc acceler.UnifiedAccelerator, name string) {
	fmt.Printf("\n=== 测试 %s ===\n", name)

	// 测试基础方法
	fmt.Printf("类型: %s\n", acc.GetType())
	fmt.Printf("可用性: %t\n", acc.IsAvailable())

	// 初始化
	err := acc.Initialize()
	if err != nil {
		log.Printf("初始化失败: %v", err)
		return
	}
	fmt.Println("初始化成功")

	// 测试计算功能
	query := []float64{1.0, 2.0, 3.0, 4.0}
	database := [][]float64{
		{1.1, 2.1, 3.1, 4.1},
		{2.0, 3.0, 4.0, 5.0},
		{0.5, 1.5, 2.5, 3.5},
	}

	// 测试距离计算
	distances, err := acc.ComputeDistance(query, database)
	if err != nil {
		log.Printf("距离计算失败: %v", err)
	} else {
		fmt.Printf("距离计算结果: %v\n", distances)
	}

	// 测试批量计算
	queries := [][]float64{query, {2.0, 3.0, 4.0, 5.0}}
	batchDistances, err := acc.BatchComputeDistance(queries, database)
	if err != nil {
		log.Printf("批量距离计算失败: %v", err)
	} else {
		fmt.Printf("批量距离计算结果: %v\n", len(batchDistances))
	}

	// 测试搜索功能
	options := entity.SearchOptions{
		MaxCandidates: 2,
		QualityLevel:  0.8,
		ForceStrategy: "exact",
	}

	searchResults, err := acc.AccelerateSearch(query, database, options)
	if err != nil {
		log.Printf("搜索失败: %v", err)
	} else {
		fmt.Printf("搜索结果数量: %d\n", len(searchResults))
	}

	// 测试批量搜索
	batchSearchResults, err := acc.BatchSearch(queries, database, 2)
	if err != nil {
		log.Printf("批量搜索失败: %v", err)
	} else {
		fmt.Printf("批量搜索结果: %d个查询\n", len(batchSearchResults))
	}

	// 获取能力信息
	capabilities := acc.GetCapabilities()
	fmt.Printf("硬件类型: %s\n", capabilities.Type)

	// 获取统计信息
	stats := acc.GetStats()
	fmt.Printf("总操作数: %d\n", stats.TotalOperations)

	// 关闭
	err = acc.Shutdown()
	if err != nil {
		log.Printf("关闭失败: %v", err)
	} else {
		fmt.Println("关闭成功")
	}
}