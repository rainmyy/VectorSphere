package main

import (
	"fmt"
	"time"

	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
)

// CompleteHardwareManagementExample 完整的硬件管理示例
func main() {
	fmt.Println("=== VectorSphere 完整硬件加速器管理系统示例 ===")

	// 1. 创建硬件管理器
	config := acceler.GetDefaultHardwareConfig()
	hm := acceler.NewHardwareManagerWithConfig(config)

	fmt.Println("\n1. 硬件管理器初始化完成")

	// 2. 启动健康监控和恢复管理
	hm.StartHealthMonitoring()
	hm.StartRecoveryManager()
	fmt.Println("2. 健康监控和恢复管理已启动")

	// 3. 检查可用的加速器
	availableAccels := hm.GetAvailableAccelerators()
	fmt.Printf("3. 可用的加速器: %v\n", availableAccels)

	// 4. 获取GPU加速器并进行安全调用
	gpuAccel := hm.GetGPUAccelerator()
	if gpuAccel != nil {
		fmt.Println("4. GPU加速器可用")

		// 使用安全的GPU批量搜索
		testQuery := [][]float64{
			{1.0, 2.0, 3.0, 4.0},
		}
		testDatabase := [][]float64{
			{1.0, 2.0, 3.0, 4.0},
			{2.0, 3.0, 4.0, 5.0},
			{3.0, 4.0, 5.0, 6.0},
		}

		results, err := hm.SafeGPUBatchSearch(testQuery, testDatabase, 2)
		if err != nil {
			fmt.Printf("   GPU搜索出错: %v\n", err)
		} else {
			fmt.Printf("   GPU搜索成功，找到 %d 个结果\n", len(results))
		}
	} else {
		fmt.Println("4. GPU加速器不可用")
	}

	// 5. 等待一段时间让健康监控收集数据
	fmt.Println("\n5. 等待健康监控收集数据...")
	time.Sleep(5 * time.Second)

	// 6. 检查健康状态
	fmt.Println("\n6. 健康状态检查:")
	overallHealth := hm.GetOverallHealth()
	fmt.Printf("   整体健康状态: %s\n", overallHealth)

	allReports := hm.GetAllHealthReports()
	for accelType, report := range allReports {
		if report != nil {
			fmt.Printf("   %s: %s - %s\n", accelType, report.Status, report.Message)
			fmt.Printf("     可用性: %t, 错误率: %.2f%%, 响应时间: %v\n",
				report.Metrics.IsAvailable,
				report.Metrics.ErrorRate*100,
				report.Metrics.ResponseTime)
		}
	}

	// 7. 检查错误统计
	fmt.Println("\n7. 错误统计:")
	errorStats := hm.GetErrorStats()
	if len(errorStats) > 0 {
		for key, count := range errorStats {
			fmt.Printf("   %s: %d 次错误\n", key, count)
		}
	} else {
		fmt.Println("   暂无错误记录")
	}

	// 8. 检查恢复历史
	fmt.Println("\n8. 恢复历史:")
	recoveryHistory := hm.GetRecoveryHistory()
	if len(recoveryHistory) > 0 {
		for _, action := range recoveryHistory {
			status := "失败"
			if action.Success {
				status = "成功"
			}
			fmt.Printf("   %s: %s策略 - %s (%s)\n",
				action.AcceleratorType,
				action.Strategy,
				status,
				action.Timestamp.Format("15:04:05"))
		}
	} else {
		fmt.Println("   暂无恢复记录")
	}

	// 9. 演示加速器性能测试
	fmt.Println("\n9. 性能测试:")
	performanceTest(hm)

	// 10. 演示配置更新
	fmt.Println("\n10. 配置更新演示:")
	configUpdateDemo(hm)

	// 11. 清理资源
	fmt.Println("\n11. 清理资源...")
	hm.StopHealthMonitoring()
	hm.StopRecoveryManager()

	fmt.Println("\n=== 示例完成 ===")
}

// performanceTest 性能测试
func performanceTest(hm *acceler.HardwareManager) {
	// 创建测试数据
	queryVector := make([]float64, 128)
	database := make([][]float64, 1000)
	for i := range database {
		database[i] = make([]float64, 128)
		for j := range database[i] {
			database[i][j] = float64(i*j) * 0.01
		}
	}
	for i := range queryVector {
		queryVector[i] = float64(i) * 0.01
	}

	// 测试不同工作负载类型
	workloadTypes := []string{"search", "compute", "batch"}
	for _, workloadType := range workloadTypes {
		start := time.Now()

		options := entity.SearchOptions{
			K:         10,
			Threshold: 0.8,
		}

		results, err := hm.AccelerateSearch(queryVector, database, options, workloadType)
		duration := time.Since(start)

		if err != nil {
			fmt.Printf("   %s工作负载测试失败: %v\n", workloadType, err)
		} else {
			fmt.Printf("   %s工作负载: %d个结果, 耗时: %v\n", workloadType, len(results), duration)
		}
	}
}

// configUpdateDemo 配置更新演示
func configUpdateDemo(hm *acceler.HardwareManager) {
	// 更新恢复配置
	newRecoveryConfig := &acceler.RecoveryConfig{
		MaxRetries:          5,
		RetryInterval:       60 * time.Second,
		HealthCheckInterval: 30 * time.Second,
		AutoRecoveryEnabled: true,
		FallbackEnabled:     true,
	}

	hm.UpdateRecoveryConfig(newRecoveryConfig)
	fmt.Println("   恢复配置已更新")

	// 重置错误计数
	availableAccels := hm.GetAvailableAccelerators()
	for _, accelType := range availableAccels {
		hm.ResetErrorCount(accelType, "")
		hm.ResetRetryCount(accelType)
	}
	fmt.Println("   错误计数和重试计数已重置")
}

// demonstrateErrorHandling 演示错误处理
func demonstrateErrorHandling(hm *acceler.HardwareManager) {
	fmt.Println("\n=== 错误处理演示 ===")

	// 模拟一个可能失败的操作
	invalidQuery := [][]float64{} // 空查询向量
	invalidDatabase := [][]float64{}

	results, err := hm.SafeGPUBatchSearch(invalidQuery, invalidDatabase, 5)
	if err != nil {
		fmt.Printf("预期的错误: %v\n", err)

		// 检查错误统计
		errorStats := hm.GetErrorStats()
		fmt.Printf("错误统计更新: %v\n", errorStats)
	} else {
		fmt.Printf("意外成功: %d个结果\n", len(results))
	}
}

// demonstrateHealthMonitoring 演示健康监控
func demonstrateHealthMonitoring(hm *acceler.HardwareManager) {
	fmt.Println("\n=== 健康监控演示 ===")

	// 等待健康检查运行
	time.Sleep(2 * time.Second)

	// 检查特定加速器的健康状态
	availableAccels := hm.GetAvailableAccelerators()
	for _, accelType := range availableAccels {
		isHealthy := hm.IsHealthy(accelType)
		report := hm.GetHealthReport(accelType)

		fmt.Printf("%s加速器:\n", accelType)
		fmt.Printf("  健康状态: %t\n", isHealthy)
		if report != nil {
			fmt.Printf("  详细报告: %s\n", report.Message)
			fmt.Printf("  响应时间: %v\n", report.Metrics.ResponseTime)
			fmt.Printf("  内存使用: %.1f%%\n", report.Metrics.MemoryUsage*100)
		}
	}
}

// demonstrateRecoveryManager 演示恢复管理
func demonstrateRecoveryManager(hm *acceler.HardwareManager) {
	fmt.Println("\n=== 恢复管理演示 ===")

	// 检查重试计数
	availableAccels := hm.GetAvailableAccelerators()
	for _, accelType := range availableAccels {
		retryCount := hm.GetRetryCount(accelType)
		fmt.Printf("%s加速器重试次数: %d\n", accelType, retryCount)
	}

	// 获取恢复历史
	history := hm.GetRecoveryHistory()
	fmt.Printf("恢复历史记录数: %d\n", len(history))
	for i, action := range history {
		if i >= 5 { // 只显示最近5条
			break
		}
		fmt.Printf("  %d. %s: %s (%s)\n",
			i+1,
			action.AcceleratorType,
			action.Strategy,
			action.Timestamp.Format("15:04:05"))
	}
}
