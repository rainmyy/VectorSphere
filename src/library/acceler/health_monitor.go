package acceler

import (
	"fmt"
	"sync"
	"time"
)

// HealthStatus 健康状态枚举
type HealthStatus int

const (
	HealthStatusHealthy HealthStatus = iota
	HealthStatusWarning
	HealthStatusCritical
	HealthStatusUnknown
)

// String 返回健康状态的字符串表示
func (hs HealthStatus) String() string {
	switch hs {
	case HealthStatusHealthy:
		return "健康"
	case HealthStatusWarning:
		return "警告"
	case HealthStatusCritical:
		return "严重"
	default:
		return "未知"
	}
}

// HealthReport 健康报告
type HealthReport struct {
	AcceleratorType string        `json:"accelerator_type"`
	Status          HealthStatus  `json:"status"`
	Message         string        `json:"message"`
	Timestamp       time.Time     `json:"timestamp"`
	Metrics         HealthMetrics `json:"metrics"`
}

// HealthMetrics 健康指标
type HealthMetrics struct {
	IsAvailable                 bool          `json:"is_available"`
	IsInitialized               bool          `json:"is_initialized"`
	ErrorRate                   float64       `json:"error_rate"`
	LastErrorTime               *time.Time    `json:"last_error_time,omitempty"`
	ResponseTime                time.Duration `json:"response_time"`
	MemoryUsage                 float64       `json:"memory_usage"`
	CPUUsage                    float64       `json:"cpu_usage"`
	Throughput                  float64       `json:"throughput"`
	LastSuccessfulOperationTime *time.Time    `json:"last_successful_operation_time,omitempty"`
}

// HealthMonitor 健康监控器
type HealthMonitor struct {
	mutex           sync.RWMutex
	hardwareManager *HardwareManager
	reports         map[string]*HealthReport
	monitorInterval time.Duration
	stopCh          chan struct{}
	running         bool
}

// NewHealthMonitor 创建新的健康监控器
func NewHealthMonitor(hm *HardwareManager) *HealthMonitor {
	return &HealthMonitor{
		hardwareManager: hm,
		reports:         make(map[string]*HealthReport),
		monitorInterval: 30 * time.Second,
		stopCh:          make(chan struct{}),
	}
}

// Start 启动健康监控
func (hm *HealthMonitor) Start() {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()

	if hm.running {
		return
	}

	hm.running = true
	go hm.monitorLoop()
}

// Stop 停止健康监控
func (hm *HealthMonitor) Stop() {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()

	if !hm.running {
		return
	}

	hm.running = false
	close(hm.stopCh)
}

// monitorLoop 监控循环
func (hm *HealthMonitor) monitorLoop() {
	ticker := time.NewTicker(hm.monitorInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			hm.performHealthCheck()
		case <-hm.stopCh:
			return
		}
	}
}

// performHealthCheck 执行健康检查
func (hm *HealthMonitor) performHealthCheck() {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()

	// 检查所有注册的加速器
	for name, acc := range hm.hardwareManager.accelerators {
		report := hm.checkAcceleratorHealth(name, acc)
		hm.reports[name] = report
	}
}

// checkAcceleratorHealth 检查单个加速器的健康状态
func (hm *HealthMonitor) checkAcceleratorHealth(name string, acc UnifiedAccelerator) *HealthReport {
	start := time.Now()

	report := &HealthReport{
		AcceleratorType: name,
		Timestamp:       start,
	}

	// 基本可用性检查
	isAvailable := acc.IsAvailable()
	report.Metrics.IsAvailable = isAvailable

	if !isAvailable {
		report.Status = HealthStatusCritical
		report.Message = "加速器不可用"
		return report
	}

	// 检查错误率
	errorStats := hm.hardwareManager.GetErrorStats()
	totalErrors := 0
	for key, count := range errorStats {
		if len(key) > len(name) && key[:len(name)] == name {
			totalErrors += count
		}
	}

	// 计算错误率（假设总操作数为错误数的10倍，这是一个简化的估算）
	totalOps := float64(totalErrors * 10)
	if totalOps > 0 {
		report.Metrics.ErrorRate = float64(totalErrors) / totalOps
	}

	// 获取最后错误时间
	lastError := hm.hardwareManager.GetLastError(name, "")
	if lastError != nil {
		report.Metrics.LastErrorTime = &lastError.Timestamp
	}

	// 响应时间测试（简单的ping测试）
	responseTime := hm.measureResponseTime(acc)
	report.Metrics.ResponseTime = responseTime

	// 获取性能指标
	if metricsProvider, ok := acc.(interface{ GetPerformanceMetrics() PerformanceMetrics }); ok {
		metrics := metricsProvider.GetPerformanceMetrics()
		report.Metrics.MemoryUsage = metrics.MemoryUsage
		report.Metrics.CPUUsage = metrics.CPUUsage
		report.Metrics.Throughput = metrics.Throughput
		if metrics.LastSuccessfulOperationTime != nil {
			report.Metrics.LastSuccessfulOperationTime = metrics.LastSuccessfulOperationTime
		}
	}

	// 确定健康状态
	report.Status = hm.determineHealthStatus(report.Metrics)
	report.Message = hm.generateHealthMessage(report.Status, report.Metrics)

	return report
}

// measureResponseTime 测量响应时间
func (hm *HealthMonitor) measureResponseTime(acc UnifiedAccelerator) time.Duration {
	start := time.Now()

	// 执行一个简单的操作来测试响应时间
	// 这里使用一个小的向量计算作为ping测试
	testQuery := []float64{1.0, 2.0, 3.0}
	testVectors := [][]float64{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}

	_, err := acc.ComputeDistance(testQuery, testVectors)
	if err != nil {
		// 如果出错，返回一个较大的响应时间
		return time.Second
	}

	return time.Since(start)
}

// determineHealthStatus 确定健康状态
func (hm *HealthMonitor) determineHealthStatus(metrics HealthMetrics) HealthStatus {
	if !metrics.IsAvailable {
		return HealthStatusCritical
	}

	// 错误率检查
	if metrics.ErrorRate > 0.1 { // 错误率超过10%
		return HealthStatusCritical
	} else if metrics.ErrorRate > 0.05 { // 错误率超过5%
		return HealthStatusWarning
	}

	// 响应时间检查
	if metrics.ResponseTime > 5*time.Second {
		return HealthStatusCritical
	} else if metrics.ResponseTime > 2*time.Second {
		return HealthStatusWarning
	}

	// 内存使用率检查
	if metrics.MemoryUsage > 0.9 { // 内存使用率超过90%
		return HealthStatusCritical
	} else if metrics.MemoryUsage > 0.8 { // 内存使用率超过80%
		return HealthStatusWarning
	}

	return HealthStatusHealthy
}

// generateHealthMessage 生成健康消息
func (hm *HealthMonitor) generateHealthMessage(status HealthStatus, metrics HealthMetrics) string {
	switch status {
	case HealthStatusHealthy:
		return "加速器运行正常"
	case HealthStatusWarning:
		if metrics.ErrorRate > 0.05 {
			return fmt.Sprintf("错误率较高: %.2f%%", metrics.ErrorRate*100)
		}
		if metrics.ResponseTime > 2*time.Second {
			return fmt.Sprintf("响应时间较慢: %v", metrics.ResponseTime)
		}
		if metrics.MemoryUsage > 0.8 {
			return fmt.Sprintf("内存使用率较高: %.1f%%", metrics.MemoryUsage*100)
		}
		return "性能指标需要关注"
	case HealthStatusCritical:
		if !metrics.IsAvailable {
			return "加速器不可用"
		}
		if metrics.ErrorRate > 0.1 {
			return fmt.Sprintf("错误率过高: %.2f%%", metrics.ErrorRate*100)
		}
		if metrics.ResponseTime > 5*time.Second {
			return fmt.Sprintf("响应时间过长: %v", metrics.ResponseTime)
		}
		if metrics.MemoryUsage > 0.9 {
			return fmt.Sprintf("内存使用率过高: %.1f%%", metrics.MemoryUsage*100)
		}
		return "加速器状态严重异常"
	default:
		return "未知状态"
	}
}

// IsHealthy 检查指定加速器是否健康
func (hm *HealthMonitor) IsHealthy(acceleratorType string) bool {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	report, exists := hm.reports[acceleratorType]
	if !exists {
		return false // 如果没有报告，则认为不健康
	}

	return report.Status == HealthStatusHealthy
}

// GetHealthReport 获取健康报告
func (hm *HealthMonitor) GetHealthReport(acceleratorType string) *HealthReport {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	return hm.reports[acceleratorType]
}

// GetAllHealthReports 获取所有健康报告
func (hm *HealthMonitor) GetAllHealthReports() map[string]*HealthReport {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	reports := make(map[string]*HealthReport)
	for k, v := range hm.reports {
		reports[k] = v
	}
	return reports
}

// GetOverallHealth 获取整体健康状态
func (hm *HealthMonitor) GetOverallHealth() HealthStatus {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	if len(hm.reports) == 0 {
		return HealthStatusUnknown
	}

	overallStatus := HealthStatusHealthy
	for _, report := range hm.reports {
		if report.Status == HealthStatusCritical {
			return HealthStatusCritical
		}
		if report.Status == HealthStatusWarning && overallStatus == HealthStatusHealthy {
			overallStatus = HealthStatusWarning
		}
	}

	return overallStatus
}

// SetMonitorInterval 设置监控间隔
func (hm *HealthMonitor) SetMonitorInterval(interval time.Duration) {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()

	hm.monitorInterval = interval
}

// SetDetectionInterval 设置检测间隔（与SetMonitorInterval相同）
func (hm *HealthMonitor) SetDetectionInterval(interval time.Duration) {
	hm.SetMonitorInterval(interval)
}
