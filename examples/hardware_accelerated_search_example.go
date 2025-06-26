package main

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/vector"
	"fmt"
	"math/rand"
	"time"
)

func init() {
	// 初始化随机数生成器
	rand.Seed(time.Now().UnixNano())
}

func main() {
	// 创建向量数据库
	db, err := vector.NewVectorDBWithDimension(128, "cosine")
	if err != nil {
		fmt.Printf("创建向量数据库失败: %v\n", err)
		return
	}

	// 创建硬件配置
	hardwareConfig := &acceler.HardwareConfig{
		CPU: acceler.CPUConfig{
			Enable:    true,
			NumCores:  0, // 0表示使用所有可用核心
			UseAVX2:   true,
			UseAVX512: true,
		},
		GPU: acceler.GPUConfig{
			Enable:      true,
			DeviceID:    0,
			MemoryLimit: 1024 * 1024 * 1024, // 1GB
			BatchSize:   1000,
		},
		FPGA: acceler.FPGAConfig{
			Enable:   false,
			DeviceID: 0,
		},
		PMem: acceler.PMemConfig{
			Enable:     false,
			MountPoint: "/mnt/pmem",
			Size:       8 * 1024 * 1024 * 1024, // 8GB
		},
		RDMA: acceler.RDMAConfig{
			Enable:    false,
			Interface: "eth0",
			Port:      18515,
		},
	}

	// 创建硬件管理器
	hardwareManager := acceler.NewHardwareManagerWithConfig(hardwareConfig)

	// 应用硬件管理器到向量数据库
	if err := db.ApplyHardwareManager(hardwareManager); err != nil {
		fmt.Printf("应用硬件管理器失败: %v\n", err)
		return
	}

	// 生成测试数据
	fmt.Println("生成测试数据...")
	generateExampleData(db, 100000)

	// 生成查询向量
	query := generateRandomVector(128)

	// 展示不同工作负载类型的搜索
	fmt.Println("\n=== 不同工作负载类型的搜索性能比较 ===")

	workloads := []struct {
		name        string
		description string
		options     entity.SearchOptions
	}{
		{
			name:        "平衡型工作负载",
			description: "默认配置，平衡延迟和吞吐量",
			options: entity.SearchOptions{
				K: 10,
			},
		},
		{
			name:        "低延迟工作负载",
			description: "优化延迟，适合实时查询",
			options: entity.SearchOptions{
				K:             10,
				SearchTimeout: 5 * time.Millisecond,
			},
		},
		{
			name:        "高吞吐量工作负载",
			description: "优化吞吐量，适合批量处理",
			options: entity.SearchOptions{
				K:         10,
				BatchSize: 200,
			},
		},
		{
			name:        "分布式工作负载",
			description: "优化分布式环境，适合集群部署",
			options: entity.SearchOptions{
				K:                10,
				DistributedSearch: true,
				ParallelSearch:    true,
			},
		},
		{
			name:        "内存优化工作负载",
			description: "优化内存使用，适合内存受限环境",
			options: entity.SearchOptions{
				K:               10,
				MemoryOptimized: true,
			},
		},
		{
			name:        "持久化存储工作负载",
			description: "优化持久化存储，适合大规模数据",
			options: entity.SearchOptions{
				K:                 10,
				PersistentStorage: true,
			},
		},
		{
			name:        "显式GPU工作负载",
			description: "强制使用GPU加速",
			options: entity.SearchOptions{
				K:      10,
				UseGPU: true,
			},
		},
		{
			name:        "显式FPGA工作负载",
			description: "强制使用FPGA加速",
			options: entity.SearchOptions{
				K:       10,
				UseFPGA: true,
			},
		},
	}

	// 执行搜索并比较性能
	for _, wl := range workloads {
		fmt.Printf("\n%s (%s):\n", wl.name, wl.description)
		
		// 预热
		db.OptimizedSearch(query, wl.options.K, wl.options)
		
		// 计时
		startTime := time.Now()
		results, err := db.OptimizedSearch(query, wl.options.K, wl.options)
		elapsed := time.Since(startTime)

		if err != nil {
			fmt.Printf("  搜索失败: %v\n", err)
			continue
		}

		fmt.Printf("  结果数量: %d\n", len(results))
		fmt.Printf("  搜索耗时: %v\n", elapsed)
		
		// 显示前3个结果
		fmt.Println("  前3个结果:")
		for i := 0; i < 3 && i < len(results); i++ {
			fmt.Printf("    ID: %s, 距离: %.6f\n", results[i].Id, results[i].Distance)
		}
	}

	// 显示硬件加速器性能指标
	fmt.Println("\n=== 硬件加速器性能指标 ===")
	metrics := hardwareManager.GetAllPerformanceMetrics()
	for accType, metric := range metrics {
		fmt.Printf("\n加速器 %s 性能指标:\n", accType)
		fmt.Printf("  当前延迟: %v\n", metric.LatencyCurrent)
		fmt.Printf("  当前吞吐量: %.2f ops/sec\n", metric.ThroughputCurrent)
		fmt.Printf("  资源利用率:\n")
		for resource, utilization := range metric.ResourceUtilization {
			fmt.Printf("    %s: %.2f%%\n", resource, utilization*100)
		}
	}

	// 显示硬件加速器统计信息
	fmt.Println("\n=== 硬件加速器统计信息 ===")
	stats := hardwareManager.GetAllStats()
	for accType, stat := range stats {
		fmt.Printf("\n加速器 %s 统计信息:\n", accType)
		fmt.Printf("  总操作数: %d\n", stat.TotalOperations)
		fmt.Printf("  成功操作数: %d\n", stat.SuccessfulOps)
		fmt.Printf("  失败操作数: %d\n", stat.FailedOps)
		fmt.Printf("  平均延迟: %v\n", stat.AverageLatency)
		fmt.Printf("  吞吐量: %.2f ops/sec\n", stat.Throughput)
		fmt.Printf("  内存利用率: %.2f%%\n", stat.MemoryUtilization*100)
		fmt.Printf("  错误率: %.2f%%\n", stat.ErrorRate*100)
	}

	// 关闭硬件管理器
	hardwareManager.ShutdownAll()
}

// 生成随机向量
func generateRandomVector(dim int) []float64 {
	vector := make([]float64, dim)
	for i := 0; i < dim; i++ {
		vector[i] = rand.Float64()*2 - 1 // 生成 [-1, 1] 范围内的随机数
	}
	return vector
}

// 生成示例数据
func generateExampleData(db *vector.VectorDB, count int) {
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("vec_%d", i)
		vec := generateRandomVector(128)
		db.Add(id, vec)
		
		// 显示进度
		if i%10000 == 0 && i > 0 {
			fmt.Printf("已生成 %d 条数据...\n", i)
		}
	}
}