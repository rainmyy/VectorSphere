package optimization

import (
	"VectorSphere/src/library/log"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"
)

// PerformanceMonitor 性能监控器
type PerformanceMonitor struct {
	metrics        *PerformanceMetrics
	thresholds     *PerformanceThresholds
	alertManager   *AlertManager
	monitoringChan chan *PerformanceEvent
	mu             sync.RWMutex
	isRunning      bool
}

// PerformanceMetrics 性能指标
type PerformanceMetrics struct {
	// 平均延迟
	AvgLatency time.Duration

	// P95延迟
	P95Latency time.Duration

	// P99延迟
	P99Latency time.Duration

	// 吞吐量（QPS）
	ThroughputQPS float64

	// 错误率
	ErrorRate float64

	// 缓存命中率
	CacheHitRate float64

	// 内存使用量
	MemoryUsage int64

	// CPU使用率
	CPUUsage float64

	// 最后更新时间
	LastUpdated time.Time
}

// PerformanceThresholds 性能阈值
type PerformanceThresholds struct {
	// 最大延迟
	MaxLatency time.Duration

	// 最小吞吐量
	MinThroughput float64

	// 最大错误率
	MaxErrorRate float64

	// 最小缓存命中率
	MinCacheHitRate float64

	// 最大内存使用量
	MaxMemoryUsage int64

	// 最大CPU使用率
	MaxCPUUsage float64
}

// PerformanceEvent 性能事件
type PerformanceEvent struct {
	// 事件类型
	Type string

	// 事件消息
	Message string

	// 事件时间戳
	Timestamp time.Time

	// 事件严重程度
	Severity string

	// 事件相关的性能指标
	Metrics *PerformanceMetrics
}

// Alert 告警
type Alert struct {
	// 告警ID
	ID string

	// 告警类型
	Type string

	// 告警消息
	Message string

	// 告警严重程度
	Severity string

	// 告警时间戳
	Timestamp time.Time

	// 是否已解决
	Resolved bool
}

// AlertHandler 告警处理器接口
type AlertHandler interface {
	// Handle 处理告警
	Handle(alert Alert) error
}

// AlertManager 告警管理器
type AlertManager struct {
	// 告警列表
	alerts []Alert

	// 告警处理器映射
	handlers map[string]AlertHandler

	// 互斥锁
	mu sync.RWMutex
}

// NewPerformanceMonitor 创建性能监控器
func NewPerformanceMonitor(thresholds *PerformanceThresholds) *PerformanceMonitor {
	if thresholds == nil {
		thresholds = getDefaultThresholds()
	}

	return &PerformanceMonitor{
		metrics:        NewPerformanceMetrics(),
		thresholds:     thresholds,
		alertManager:   NewAlertManager(),
		monitoringChan: make(chan *PerformanceEvent, 1000),
	}
}

// NewPerformanceMetrics 创建性能指标
func NewPerformanceMetrics() *PerformanceMetrics {
	return &PerformanceMetrics{
		LastUpdated: time.Now(),
	}
}

// getDefaultThresholds 获取默认阈值
func getDefaultThresholds() *PerformanceThresholds {
	return &PerformanceThresholds{
		MaxLatency:      100 * time.Millisecond,
		MinThroughput:   1000,                   // QPS
		MaxErrorRate:    0.01,                   // 1%
		MinCacheHitRate: 0.5,                    // 50%
		MaxMemoryUsage:  8 * 1024 * 1024 * 1024, // 8GB
		MaxCPUUsage:     0.8,                    // 80%
	}
}

// NewAlertManager 创建告警管理器
func NewAlertManager() *AlertManager {
	return &AlertManager{
		alerts:   make([]Alert, 0),
		handlers: make(map[string]AlertHandler),
	}
}

// Start 启动性能监控
func (pm *PerformanceMonitor) Start() error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.isRunning {
		return nil
	}

	pm.isRunning = true

	// 启动监控
	pm.startMonitoring()

	return nil
}

// Stop 停止性能监控
func (pm *PerformanceMonitor) Stop() {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if !pm.isRunning {
		return
	}

	pm.isRunning = false

	// 关闭监控通道
	close(pm.monitoringChan)

	log.Info("性能监控器已停止")
}

// startMonitoring 开始监控
func (pm *PerformanceMonitor) startMonitoring() {
	// 定期收集性能指标
	go pm.collectMetrics()

	// 处理性能事件
	go pm.processEvents()

	log.Info("性能监控器已启动")
}

// collectMetrics 收集性能指标
func (pm *PerformanceMonitor) collectMetrics() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		// 收集系统资源使用情况
		pm.collectSystemMetrics()

		// 检查性能指标是否超过阈值
		pm.checkThresholds()
	}
}

// collectSystemMetrics 收集系统指标
func (pm *PerformanceMonitor) collectSystemMetrics() {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// 收集内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// 更新指标
	pm.metrics.MemoryUsage = int64(m.Alloc)

	// 更新CPU使用率（简化实现，实际应使用系统API）
	pm.metrics.CPUUsage = getCPUUsage()

	// 更新时间戳
	pm.metrics.LastUpdated = time.Now()

	log.Trace("系统指标: 内存=%dMB, CPU=%.1f%%",
		pm.metrics.MemoryUsage/1024/1024,
		pm.metrics.CPUUsage*100)
}

// getCPUUsage 获取CPU使用率
func getCPUUsage() float64 {
	// 这里是简化实现，实际应使用系统API获取真实CPU使用率
	// 例如在Linux上可以解析/proc/stat，在Windows上可以使用WMI
	// 这里返回一个模拟值
	return 0.5 // 50%
}

// checkThresholds 检查阈值
func (pm *PerformanceMonitor) checkThresholds() {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	// 检查延迟
	if pm.metrics.P95Latency > pm.thresholds.MaxLatency {
		pm.createAlert("high_latency", fmt.Sprintf("P95延迟(%.2fms)超过阈值(%.2fms)",
			float64(pm.metrics.P95Latency)/float64(time.Millisecond),
			float64(pm.thresholds.MaxLatency)/float64(time.Millisecond)),
			"warning")
	}

	// 检查吞吐量
	if pm.metrics.ThroughputQPS < pm.thresholds.MinThroughput {
		pm.createAlert("low_throughput", fmt.Sprintf("吞吐量(%.2f QPS)低于阈值(%.2f QPS)",
			pm.metrics.ThroughputQPS, pm.thresholds.MinThroughput),
			"warning")
	}

	// 检查错误率
	if pm.metrics.ErrorRate > pm.thresholds.MaxErrorRate {
		pm.createAlert("high_error_rate", fmt.Sprintf("错误率(%.2f%%)超过阈值(%.2f%%)",
			pm.metrics.ErrorRate*100, pm.thresholds.MaxErrorRate*100),
			"critical")
	}

	// 检查缓存命中率
	if pm.metrics.CacheHitRate < pm.thresholds.MinCacheHitRate {
		pm.createAlert("low_cache_hit_rate", fmt.Sprintf("缓存命中率(%.2f%%)低于阈值(%.2f%%)",
			pm.metrics.CacheHitRate*100, pm.thresholds.MinCacheHitRate*100),
			"warning")
	}

	// 检查内存使用
	if pm.metrics.MemoryUsage > pm.thresholds.MaxMemoryUsage {
		pm.createAlert("high_memory_usage", fmt.Sprintf("内存使用(%.2fGB)超过阈值(%.2fGB)",
			float64(pm.metrics.MemoryUsage)/1024/1024/1024,
			float64(pm.thresholds.MaxMemoryUsage)/1024/1024/1024),
			"warning")
	}

	// 检查CPU使用率
	if pm.metrics.CPUUsage > pm.thresholds.MaxCPUUsage {
		pm.createAlert("high_cpu_usage", fmt.Sprintf("CPU使用率(%.2f%%)超过阈值(%.2f%%)",
			pm.metrics.CPUUsage*100, pm.thresholds.MaxCPUUsage*100),
			"warning")
	}
}

// createAlert 创建告警
func (pm *PerformanceMonitor) createAlert(alertType, message, severity string) {
	event := &PerformanceEvent{
		Type:      alertType,
		Message:   message,
		Timestamp: time.Now(),
		Severity:  severity,
		Metrics:   pm.metrics,
	}

	// 发送到事件通道
	select {
	case pm.monitoringChan <- event:
		log.Warning("性能告警: %s - %s", severity, message)
	default:
		log.Error("告警通道已满，丢弃告警: %s", message)
	}
}

// processEvents 处理事件
func (pm *PerformanceMonitor) processEvents() {
	for event := range pm.monitoringChan {
		// 创建告警
		alert := Alert{
			ID:        fmt.Sprintf("%s_%d", event.Type, time.Now().UnixNano()),
			Type:      event.Type,
			Message:   event.Message,
			Severity:  event.Severity,
			Timestamp: event.Timestamp,
			Resolved:  false,
		}

		// 添加到告警列表
		pm.alertManager.addAlert(alert)

		// 处理告警
		pm.alertManager.handleAlert(alert)
	}
}

// addAlert 添加告警
func (am *AlertManager) addAlert(alert Alert) {
	am.mu.Lock()
	defer am.mu.Unlock()

	// 检查是否已存在相同类型的未解决告警
	for i, a := range am.alerts {
		if a.Type == alert.Type && !a.Resolved {
			// 更新现有告警
			am.alerts[i] = alert
			return
		}
	}

	// 添加新告警
	am.alerts = append(am.alerts, alert)

	// 如果告警数量超过100，移除最旧的已解决告警
	if len(am.alerts) > 100 {
		am.cleanupResolvedAlerts()
	}
}

// cleanupResolvedAlerts 清理已解决的告警
func (am *AlertManager) cleanupResolvedAlerts() {
	// 找出已解决的告警
	resolvedAlerts := make([]int, 0)
	for i, alert := range am.alerts {
		if alert.Resolved {
			resolvedAlerts = append(resolvedAlerts, i)
		}
	}

	// 如果没有已解决的告警，不做处理
	if len(resolvedAlerts) == 0 {
		return
	}

	// 按时间排序，保留最新的
	sort.Slice(resolvedAlerts, func(i, j int) bool {
		return am.alerts[resolvedAlerts[i]].Timestamp.Before(am.alerts[resolvedAlerts[j]].Timestamp)
	})

	// 移除最旧的已解决告警
	toRemove := len(am.alerts) - 100
	if toRemove > len(resolvedAlerts) {
		toRemove = len(resolvedAlerts)
	}

	for i := 0; i < toRemove; i++ {
		// 从后往前删除，避免索引变化
		index := resolvedAlerts[i]
		am.alerts = append(am.alerts[:index], am.alerts[index+1:]...)
	}
}

// handleAlert 处理告警
func (am *AlertManager) handleAlert(alert Alert) {
	am.mu.RLock()
	defer am.mu.RUnlock()

	// 遍历所有处理器
	for _, handler := range am.handlers {
		if err := handler.Handle(alert); err != nil {
			log.Error("处理告警失败: %v", err)
		}
	}
}

// RegisterHandler 注册告警处理器
func (am *AlertManager) RegisterHandler(name string, handler AlertHandler) {
	am.mu.Lock()
	defer am.mu.Unlock()

	am.handlers[name] = handler
	log.Info("注册告警处理器: %s", name)
}

// GetActiveAlerts 获取活跃告警
func (am *AlertManager) GetActiveAlerts() []Alert {
	am.mu.RLock()
	defer am.mu.RUnlock()

	activeAlerts := make([]Alert, 0)
	for _, alert := range am.alerts {
		if !alert.Resolved {
			activeAlerts = append(activeAlerts, alert)
		}
	}

	return activeAlerts
}

// ResolveAlert 解决告警
func (am *AlertManager) ResolveAlert(alertID string) bool {
	am.mu.Lock()
	defer am.mu.Unlock()

	for i, alert := range am.alerts {
		if alert.ID == alertID {
			am.alerts[i].Resolved = true
			log.Info("告警已解决: %s - %s", alert.Severity, alert.Message)
			return true
		}
	}

	return false
}

// recordPerformance 记录性能
func (hto *HighThroughputOptimizer) recordPerformance(strategy string, latency time.Duration, resultCount int, err error) {
	// 更新性能监控器的指标
	pm := hto.performanceMonitor
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// 更新延迟统计
	pm.updateLatencyStats(latency)

	// 更新吞吐量
	pm.updateThroughput(1) // 1个查询

	// 更新错误率
	if err != nil {
		pm.updateErrorRate(true)
	} else {
		pm.updateErrorRate(false)
	}
}

// recordBatchPerformance 记录批量性能
func (hto *HighThroughputOptimizer) recordBatchPerformance(queryCount int, latency time.Duration) {
	// 更新性能监控器的指标
	pm := hto.performanceMonitor
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// 更新延迟统计
	pm.updateLatencyStats(latency)

	// 更新吞吐量
	pm.updateThroughput(queryCount)
}

// updateLatencyStats 更新延迟统计
func (pm *PerformanceMonitor) updateLatencyStats(latency time.Duration) {
	// 更新平均延迟
	pm.metrics.AvgLatency = updateMovingAverage(pm.metrics.AvgLatency, latency, 100)

	// 更新延迟分位数（简化实现）
	pm.metrics.P95Latency = time.Duration(float64(pm.metrics.AvgLatency) * 1.5) // 简化的P95估计
	pm.metrics.P99Latency = time.Duration(float64(pm.metrics.AvgLatency) * 2.0) // 简化的P99估计
}

// updateMovingAverage 更新移动平均值
func updateMovingAverage(current time.Duration, newValue time.Duration, weight int) time.Duration {
	if current == 0 {
		return newValue
	}

	alpha := 1.0 / float64(weight)
	newAvg := time.Duration(float64(current)*(1-alpha) + float64(newValue)*alpha)
	return newAvg
}

// updateThroughput 更新吞吐量
func (pm *PerformanceMonitor) updateThroughput(queryCount int) {
	// 计算时间窗口
	now := time.Now()
	timeWindow := now.Sub(pm.metrics.LastUpdated)

	// 避免除以零
	if timeWindow < time.Millisecond {
		timeWindow = time.Millisecond
	}

	// 计算当前QPS
	currentQPS := float64(queryCount) / timeWindow.Seconds()

	// 更新移动平均QPS
	if pm.metrics.ThroughputQPS == 0 {
		pm.metrics.ThroughputQPS = currentQPS
	} else {
		pm.metrics.ThroughputQPS = pm.metrics.ThroughputQPS*0.9 + currentQPS*0.1
	}
}

// updateErrorRate 更新错误率
func (pm *PerformanceMonitor) updateErrorRate(isError bool) {
	// 使用指数移动平均更新错误率
	alpha := 0.01 // 权重因子

	if isError {
		pm.metrics.ErrorRate = pm.metrics.ErrorRate*(1-alpha) + 1*alpha
	} else {
		pm.metrics.ErrorRate = pm.metrics.ErrorRate * (1 - alpha)
	}
}

// GetMetrics 获取性能指标
func (pm *PerformanceMonitor) GetMetrics() *PerformanceMetrics {
	return pm.GetPerformanceMetrics()
}

// GetPerformanceMetrics 获取性能指标
func (pm *PerformanceMonitor) GetPerformanceMetrics() *PerformanceMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	// 返回指标的副本
	metricsCopy := *pm.metrics
	return &metricsCopy
}

// LogPerformanceStats 记录性能统计
func (pm *PerformanceMonitor) LogPerformanceStats() {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	log.Info("性能统计: 吞吐量=%.2f QPS, 平均延迟=%.2fms, P95延迟=%.2fms, 错误率=%.2f%%, 缓存命中率=%.2f%%",
		pm.metrics.ThroughputQPS,
		float64(pm.metrics.AvgLatency)/float64(time.Millisecond),
		float64(pm.metrics.P95Latency)/float64(time.Millisecond),
		pm.metrics.ErrorRate*100,
		pm.metrics.CacheHitRate*100)
}

// RecordSearchPerformance 记录搜索性能
func (pm *PerformanceMonitor) RecordSearchPerformance(queryCount int, startTime time.Time, duration time.Duration, success bool) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// 更新延迟统计
	pm.updateLatencyStats(duration)

	// 更新吞吐量
	pm.updateThroughput(queryCount)

	// 更新错误率
	pm.updateErrorRate(!success)

	// 检查阈值
	pm.checkThresholds()
}

// RecordBatchSearchPerformance 记录批量搜索性能
func (pm *PerformanceMonitor) RecordBatchSearchPerformance(queryCount int, startTime time.Time, duration time.Duration, success bool) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// 更新延迟统计
	pm.updateLatencyStats(duration)

	// 更新吞吐量
	pm.updateThroughput(queryCount)

	// 更新错误率
	pm.updateErrorRate(!success)

	// 检查阈值
	pm.checkThresholds()
}
