package healthCheck

import (
	"VectorSphere/src/library/log"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// HealthMetricsCollector 健康指标收集器
type HealthMetricsCollector struct {
	healthManager *HealthCheckManager
	metrics       *HealthMetrics
	mu            sync.RWMutex
}

// NewHealthMetricsCollector 创建一个新的健康指标收集器
func NewHealthMetricsCollector(manager *HealthCheckManager) *HealthMetricsCollector {
	return &HealthMetricsCollector{
		healthManager: manager,
		metrics:       &HealthMetrics{}, // 初始化 HealthMetrics
	}
}

// UpdateMetrics 由 HealthCheckManager 调用，用于更新收集器的指标
// 参数应该是 HealthCheckManager 计算好的聚合数据
func (hmc *HealthMetricsCollector) UpdateMetrics(newMetrics *HealthMetrics) {
	hmc.mu.Lock()
	defer hmc.mu.Unlock()
	hmc.metrics = newMetrics
	log.Debug("Health metrics updated: %+v", hmc.metrics)
}

// CollectHealthMetrics 收集健康指标
func (hmc *HealthMetricsCollector) CollectHealthMetrics() map[string]interface{} {
	hmc.mu.RLock()
	defer hmc.mu.RUnlock()

	if hmc.healthManager == nil {
		log.Error("HealthCheckManager is not initialized in HealthMetricsCollector")
		return map[string]interface{}{
			"error": "HealthCheckManager not initialized",
		}
	}

	// 实际项目中，这些指标的更新应该由 HealthCheckManager 在执行健康检查时触发
	hmc.metrics.TotalChecks = hmc.healthManager.GetTotalChecksCount()
	hmc.metrics.SuccessfulChecks = hmc.healthManager.GetSuccessfulChecksCount()
	hmc.metrics.FailedChecks = hmc.metrics.TotalChecks - hmc.metrics.SuccessfulChecks
	hmc.metrics.AverageLatency = hmc.healthManager.GetAverageLatency()
	hmc.metrics.LastCheckTime = hmc.healthManager.GetLastCheckTime()

	return map[string]interface{}{
		"total_services":     len(hmc.healthManager.Services), // 假设 Services 是可公开访问的map或slice
		"healthy_services":   hmc.GetHealthyServiceCount(),
		"unhealthy_services": hmc.getUnhealthyServiceCount(),
		"average_latency_ms": hmc.metrics.AverageLatency.Milliseconds(),
		"success_rate":       hmc.calculateSuccessRate(),
		"last_check_time":    hmc.metrics.LastCheckTime.Format(time.RFC3339),
		"total_checks":       hmc.metrics.TotalChecks,
		"successful_checks":  hmc.metrics.SuccessfulChecks,
		"failed_checks":      hmc.metrics.FailedChecks,
	}
}

// GetHealthyServiceCount 获取健康服务的数量
func (hmc *HealthMetricsCollector) GetHealthyServiceCount() int {
	if hmc.healthManager == nil {
		return 0
	}
	healthyCount := 0
	// 假设 hmc.healthManager.Services 是一个 map[string]*ServiceHealthInfo
	// 并且 ServiceHealthInfo 有一个 IsHealthy() 方法或 Healthy 字段
	for _, serviceInfo := range hmc.healthManager.Services { // 遍历服务
		// 假设 ServiceHealthInfo 有一个 IsHealthy() 方法或类似的状态字段
		// 例如: if serviceInfo.Status == HealthStatusHealthy
		// 或者: if serviceInfo.IsHealthy()
		// 这里我们用一个假设的 IsHealthy 字段
		if serviceInfo.IsHealthy { // 请根据您的 ServiceHealthInfo 结构调整
			healthyCount++
		}
	}
	return healthyCount
}

// getUnhealthyServiceCount 获取不健康服务的数量
func (hmc *HealthMetricsCollector) getUnhealthyServiceCount() int {
	if hmc.healthManager == nil {
		return 0
	}
	// 总服务数减去健康服务数
	totalServices := len(hmc.healthManager.Services)
	healthyServices := hmc.GetHealthyServiceCount()
	return totalServices - healthyServices
}

// calculateSuccessRate 计算健康检查的成功率
func (hmc *HealthMetricsCollector) calculateSuccessRate() float64 {
	if hmc.metrics.TotalChecks == 0 {
		return 0.0 // 避免除以零
	}
	return float64(hmc.metrics.SuccessfulChecks) / float64(hmc.metrics.TotalChecks)
}

// ExposeHealthMetrics 暴露健康指标HTTP端点
func (hmc *HealthMetricsCollector) ExposeHealthMetrics(port int) {
	http.HandleFunc("/metrics/health", func(w http.ResponseWriter, r *http.Request) {
		metrics := hmc.CollectHealthMetrics()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(metrics)
	})

	http.HandleFunc("/health/services", func(w http.ResponseWriter, r *http.Request) {
		if hmc.healthManager == nil {
			http.Error(w, "HealthCheckManager not initialized", http.StatusInternalServerError)
			return
		}
		services := hmc.healthManager.GetAllServiceHealth() // 假设 HealthCheckManager 有此方法
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(services)
	})

	addr := fmt.Sprintf(":%d", port)
	log.Info("Health metrics server starting on http://localhost%s/metrics/health and http://localhost%s/health/services", addr, addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal("Failed to start health metrics server: %v", err)
	}
}
