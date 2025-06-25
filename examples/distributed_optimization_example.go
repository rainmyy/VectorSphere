package main

import (
	"fmt"
	"log"
	"time"

	"VectorSphere/src/vector"
)

// DistributedOptimizationExample 分布式优化示例
func main_1() {
	fmt.Println("VectorSphere 分布式优化示例")
	fmt.Println("==============================")

	// 1. 创建配置管理器
	configPath := "./config/vectorsphere.yaml"
	configManager := vector.NewConfigManager(configPath)

	// 2. 加载配置
	if err := configManager.LoadConfig(); err != nil {
		log.Printf("加载配置失败，使用默认配置: %v", err)
	}

	// 3. 验证配置
	if err := configManager.ValidateConfig(); err != nil {
		log.Fatalf("配置验证失败: %v", err)
	}

	// 4. 创建VectorDB实例
	db := &vector.VectorDB{}

	// 5. 应用配置到数据库
	if err := configManager.ApplyToVectorDB(db); err != nil {
		log.Fatalf("应用配置失败: %v", err)
	}

	// 6. 创建性能监控器
	performanceMonitor := vector.NewStandardPerformanceMonitor()

	// 7. 创建自适应优化引擎
	adaptiveOptimizer := vector.NewAdaptiveOptimizer(configManager, performanceMonitor)

	// 8. 启动自适应优化
	if err := adaptiveOptimizer.Start(1 * time.Minute); err != nil {
		log.Fatalf("启动自适应优化失败: %v", err)
	}
	defer adaptiveOptimizer.Stop()

	// 9. 展示配置摘要
	fmt.Println("\n当前配置摘要:")
	summary := configManager.GetConfigSummary()
	for category, config := range summary {
		fmt.Printf("  %s: %+v\n", category, config)
	}

	// 10. 模拟不同的工作负载场景
	fmt.Println("\n开始模拟工作负载...")

	// 场景1: 高CPU使用率场景
	fmt.Println("\n场景1: 模拟高CPU使用率")
	simulateHighCPUWorkload(db, adaptiveOptimizer)

	// 场景2: 高内存使用率场景
	fmt.Println("\n场景2: 模拟高内存使用率")
	simulateHighMemoryWorkload(db, adaptiveOptimizer)

	// 场景3: 高延迟场景
	fmt.Println("\n场景3: 模拟高查询延迟")
	simulateHighLatencyWorkload(db, adaptiveOptimizer)

	// 场景4: 大数据集场景
	fmt.Println("\n场景4: 模拟大数据集")
	simulatelargeDatasetWorkload(db, adaptiveOptimizer)

	// 11. 展示优化日志
	fmt.Println("\n优化日志:")
	optimizationLog := adaptiveOptimizer.GetOptimizationLog()
	for _, event := range optimizationLog {
		status := "成功"
		if !event.Success {
			status = fmt.Sprintf("失败: %v", event.Error)
		}
		fmt.Printf("  [%s] %s - %s (%s)\n",
			event.Timestamp.Format("15:04:05"),
			event.RuleName,
			event.Description,
			status)
	}

	// 12. 演示不同的优化策略
	fmt.Println("\n演示优化策略切换:")
	demonstrateDifferentStrategies(adaptiveOptimizer)

	// 13. 展示性能指标
	fmt.Println("\n当前性能指标:")
	showPerformanceMetrics(adaptiveOptimizer)

	fmt.Println("\n示例运行完成!")
}

// simulateHighCPUWorkload 模拟高CPU使用率工作负载
func simulateHighCPUWorkload(db *vector.VectorDB, optimizer *vector.AdaptiveOptimizer) {
	fmt.Println("  模拟CPU密集型查询...")

	// 模拟高CPU使用率的指标
	metrics := &vector.PerformanceMetrics{
		AvgLatency:    150 * time.Millisecond,
		Recall:        0.95,
		ThroughputQPS: 500,
		MemoryUsage:   uint64(60 * 1024 * 1024), // 60MB
		LastUpdated:   time.Now(),
	}

	fmt.Printf("  平均延迟: %v, 召回率: %.2f, 吞吐量: %.1f QPS\n",
		metrics.AvgLatency, metrics.Recall, metrics.ThroughputQPS)

	// 等待优化引擎检测并应用优化
	time.Sleep(2 * time.Second)
}

// simulateHighMemoryWorkload 模拟高内存使用率工作负载
func simulateHighMemoryWorkload(db *vector.VectorDB, optimizer *vector.AdaptiveOptimizer) {
	fmt.Println("  模拟内存密集型查询...")

	metrics := &vector.PerformanceMetrics{
		AvgLatency:    120 * time.Millisecond,
		Recall:        0.92,
		ThroughputQPS: 300,
		MemoryUsage:   uint64(90 * 1024 * 1024), // 90MB 高内存使用
		LastUpdated:   time.Now(),
	}

	fmt.Printf("  平均延迟: %v, 召回率: %.2f, 内存使用: %dMB\n",
		metrics.AvgLatency, metrics.Recall, metrics.MemoryUsage/(1024*1024))

	time.Sleep(2 * time.Second)
}

// simulateHighLatencyWorkload 模拟高延迟工作负载
func simulateHighLatencyWorkload(db *vector.VectorDB, optimizer *vector.AdaptiveOptimizer) {
	fmt.Println("  模拟高延迟查询...")

	metrics := &vector.PerformanceMetrics{
		AvgLatency:    350 * time.Millisecond, // 350ms 高延迟
		Recall:        0.88,
		ThroughputQPS: 200,
		MemoryUsage:   uint64(70 * 1024 * 1024), // 70MB
		LastUpdated:   time.Now(),
	}

	fmt.Printf("  平均延迟: %v, 召回率: %.2f, 吞吐量: %.1f QPS\n",
		metrics.AvgLatency, metrics.Recall, metrics.ThroughputQPS)

	time.Sleep(2 * time.Second)
}

// simulatelargeDatasetWorkload 模拟大数据集工作负载
func simulatelargeDatasetWorkload(db *vector.VectorDB, optimizer *vector.AdaptiveOptimizer) {
	fmt.Println("  模拟大数据集查询...")

	metrics := &vector.PerformanceMetrics{
		AvgLatency:    180 * time.Millisecond,
		Recall:        0.90,
		ThroughputQPS: 150,
		MemoryUsage:   uint64(75 * 1024 * 1024), // 75MB
		LastUpdated:   time.Now(),
	}

	fmt.Printf("  平均延迟: %v, 召回率: %.2f, 内存使用: %dMB\n",
		metrics.AvgLatency, metrics.Recall, metrics.MemoryUsage/(1024*1024))

	time.Sleep(2 * time.Second)
}

// demonstrateDifferentStrategies 演示不同的优化策略
func demonstrateDifferentStrategies(optimizer *vector.AdaptiveOptimizer) {
	strategies := []string{"HighPerformance", "MemoryEfficient", "Balanced"}

	for _, strategy := range strategies {
		fmt.Printf("  切换到策略: %s\n", strategy)
		if err := optimizer.ApplyStrategy(strategy); err != nil {
			fmt.Printf("    切换失败: %v\n", err)
		} else {
			fmt.Printf("    切换成功\n")
		}
		time.Sleep(1 * time.Second)
	}
}

// showPerformanceMetrics 展示性能指标
func showPerformanceMetrics(optimizer *vector.AdaptiveOptimizer) {
	metrics, err := optimizer.GetCurrentMetrics()
	if err != nil {
		fmt.Printf("  获取性能指标失败: %v\n", err)
		return
	}

	fmt.Printf("  平均延迟: %v\n", metrics.AvgLatency)
	fmt.Printf("  召回率: %.2f\n", metrics.Recall)
	fmt.Printf("  吞吐量QPS: %.1f\n", metrics.ThroughputQPS)
	fmt.Printf("  内存使用: %dMB\n", metrics.MemoryUsage/(1024*1024))
	fmt.Printf("  最后更新: %v\n", metrics.LastUpdated.Format("15:04:05"))
}

// demonstrateConfigurationScenarios 演示不同配置场景
func demonstrateConfigurationScenarios() {
	fmt.Println("\n配置场景演示:")

	// 场景1: 高性能配置
	fmt.Println("\n场景1: 高性能配置")
	highPerfConfig := vector.GetDefaultDistributedConfig()
	highPerfConfig.IndexConfig.HNSWConfig.Enable = true
	highPerfConfig.IndexConfig.HNSWConfig.MaxConnections = 32
	highPerfConfig.IndexConfig.HNSWConfig.EfConstruction = 400
	fmt.Println("  - 启用HNSW索引")
	fmt.Println("  - 增加连接数到32")
	fmt.Println("  - 提高构建参数到400")

	// 场景2: 内存优化配置
	fmt.Println("\n场景2: 内存优化配置")
	memoryOptConfig := vector.GetDefaultDistributedConfig()
	memoryOptConfig.IndexConfig.PQConfig.Enable = true
	memoryOptConfig.IndexConfig.PQConfig.NumSubVectors = 16
	memoryOptConfig.IndexConfig.PQConfig.NumCentroids = 256
	fmt.Println("  - 启用PQ压缩")
	fmt.Println("  - 设置16个子向量")
	fmt.Println("  - 每个子向量8位")

	// 场景3: 大规模数据配置
	fmt.Println("\n场景3: 大规模数据配置")
	largeScaleConfig := vector.GetDefaultDistributedConfig()
	largeScaleConfig.IndexConfig.IVFConfig.Enable = true
	largeScaleConfig.IndexConfig.IVFConfig.NumClusters = 4096
	largeScaleConfig.IndexConfig.IVFConfig.Nprobe = 128
	fmt.Println("  - 启用IVF索引")
	fmt.Println("  - 设置4096个聚类中心")
	fmt.Println("  - 搜索128个聚类")
}

// demonstrateMonitoringAndAlerting 演示监控和告警
func demonstrateMonitoringAndAlerting() {
	fmt.Println("\n监控和告警演示:")

	monitoringConfig := vector.GetDefaultMonitoringConfig()

	fmt.Println("\n指标收集:")
	fmt.Printf("  - 收集间隔: %v\n", monitoringConfig.Metrics.CollectionInterval)
	fmt.Printf("  - 保留期间: %v\n", monitoringConfig.Metrics.RetentionPeriod)
	fmt.Printf("  - Prometheus端口: %d\n", monitoringConfig.Metrics.Exporter.Prometheus.Port)

	fmt.Println("\n告警规则:")
	for _, rule := range monitoringConfig.Alerting.Rules {
		fmt.Printf("  - %s: %s\n", rule.Name, rule.Description)
		fmt.Printf("    条件: %s %s %.1f\n",
			rule.Metric, rule.Condition.Operator, rule.Condition.Threshold)
		fmt.Printf("    严重级别: %s\n", rule.Severity)
	}

	fmt.Println("\n自动扩缩容:")
	fmt.Printf("  - CPU扩容阈值: %.1f%%\n",
		monitoringConfig.AutoScaling.Metrics.CPUUtilization.ScaleUpThreshold)
	fmt.Printf("  - 内存扩容阈值: %.1f%%\n",
		monitoringConfig.AutoScaling.Metrics.MemoryUtilization.ScaleUpThreshold)
	fmt.Printf("  - 最小计算节点: %d\n",
		monitoringConfig.AutoScaling.Limits.MinComputeNodes)
	fmt.Printf("  - 最大计算节点: %d\n",
		monitoringConfig.AutoScaling.Limits.MaxComputeNodes)
}

// demonstrateCacheStrategies 演示缓存策略
func demonstrateCacheStrategies() {
	fmt.Println("\n缓存策略演示:")

	cacheConfig := vector.GetDefaultCacheConfig()

	fmt.Println("\n结果缓存:")
	fmt.Printf("  - 最大大小: %dMB\n", cacheConfig.ResultCache.MaxSize/(1024*1024))
	fmt.Printf("  - TTL: %v\n", cacheConfig.ResultCache.TTL)
	fmt.Printf("  - 驱逐策略: %s\n", cacheConfig.ResultCache.EvictionPolicy)

	fmt.Println("\n向量缓存:")
	fmt.Printf("  - 最大大小: %dMB\n", cacheConfig.VectorCache.MaxSize/(1024*1024))
	fmt.Printf("  - 热数据策略: %v\n", cacheConfig.VectorCache.HotDataStrategy.Enable)
	fmt.Printf("  - 热数据阈值: %.2f\n", cacheConfig.VectorCache.HotDataStrategy.HotThreshold)

	fmt.Println("\n索引缓存:")
	fmt.Printf("  - 最大大小: %dMB\n", cacheConfig.IndexCache.MaxSize/(1024*1024))
	fmt.Printf("  - 最大索引数: %d\n", cacheConfig.IndexCache.MaxIndices)
	fmt.Printf("  - 预加载: %v\n", cacheConfig.IndexCache.Preloading.Enable)
}
