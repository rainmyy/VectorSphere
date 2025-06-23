package vector

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// PerformanceMonitor 性能监控器接口
type PerformanceMonitor interface {
	StartOperation(operationType string) *OperationTracker
	RecordMetric(name string, value float64, tags map[string]string)
	GetMetrics() map[string]MetricData
	GetSystemStats() SystemStats
	ExportPrometheusMetrics() string
	Reset()
}

// OperationTracker 操作跟踪器
type OperationTracker struct {
	operationType string
	startTime     time.Time
	monitor       *StandardPerformanceMonitor
	tags          map[string]string
}

// MetricData 指标数据
type MetricData struct {
	Name        string             `json:"name"`
	Type        string             `json:"type"` // counter, gauge, histogram
	Value       float64            `json:"value"`
	Count       int64              `json:"count"`
	Sum         float64            `json:"sum"`
	Min         float64            `json:"min"`
	Max         float64            `json:"max"`
	Average     float64            `json:"average"`
	Percentiles map[string]float64 `json:"percentiles,omitempty"`
	Tags        map[string]string  `json:"tags,omitempty"`
	Timestamp   time.Time          `json:"timestamp"`
}

// SystemStats 系统统计信息
type SystemStats struct {
	CPUUsage     float64            `json:"cpu_usage"`
	MemoryUsage  MemoryStats        `json:"memory_usage"`
	Goroutines   int                `json:"goroutines"`
	GCCycles     uint32             `json:"gc_cycles"`
	DiskUsage    map[string]float64 `json:"disk_usage,omitempty"`
	NetworkStats NetworkStats       `json:"network_stats,omitempty"`
	Timestamp    time.Time          `json:"timestamp"`
}

// MemoryStats 内存统计信息
type MemoryStats struct {
	Allocated    uint64  `json:"allocated"`     // 当前分配的内存
	TotalAlloc   uint64  `json:"total_alloc"`   // 累计分配的内存
	Sys          uint64  `json:"sys"`           // 系统内存
	Lookups      uint64  `json:"lookups"`       // 指针查找次数
	Mallocs      uint64  `json:"mallocs"`       // 内存分配次数
	Frees        uint64  `json:"frees"`         // 内存释放次数
	HeapAlloc    uint64  `json:"heap_alloc"`    // 堆内存分配
	HeapSys      uint64  `json:"heap_sys"`      // 堆系统内存
	HeapIdle     uint64  `json:"heap_idle"`     // 堆空闲内存
	HeapInuse    uint64  `json:"heap_inuse"`    // 堆使用内存
	HeapReleased uint64  `json:"heap_released"` // 堆释放内存
	HeapObjects  uint64  `json:"heap_objects"`  // 堆对象数量
	StackInuse   uint64  `json:"stack_inuse"`   // 栈使用内存
	StackSys     uint64  `json:"stack_sys"`     // 栈系统内存
	MSpanInuse   uint64  `json:"mspan_inuse"`   // MSpan使用内存
	MSpanSys     uint64  `json:"mspan_sys"`     // MSpan系统内存
	MCacheInuse  uint64  `json:"mcache_inuse"`  // MCache使用内存
	MCacheSys    uint64  `json:"mcache_sys"`    // MCache系统内存
	BuckHashSys  uint64  `json:"buck_hash_sys"` // 分析桶哈希表内存
	GCSys        uint64  `json:"gc_sys"`        // GC系统内存
	OtherSys     uint64  `json:"other_sys"`     // 其他系统内存
	UsagePercent float64 `json:"usage_percent"` // 内存使用百分比
}

// NetworkStats 网络统计信息
type NetworkStats struct {
	BytesReceived   uint64 `json:"bytes_received"`
	BytesSent       uint64 `json:"bytes_sent"`
	PacketsReceived uint64 `json:"packets_received"`
	PacketsSent     uint64 `json:"packets_sent"`
	Connections     int    `json:"connections"`
}

// StandardPerformanceMonitor 标准性能监控器实现
type StandardPerformanceMonitor struct {
	metrics      map[string]*MetricData
	mutex        sync.RWMutex
	startTime    time.Time
	lastCPUTime  time.Time
	lastCPUUsage float64

	// 计数器
	totalOperations int64
	successfulOps   int64
	failedOps       int64

	// 延迟统计
	latencyBuckets map[string][]float64
	latencyMutex   sync.RWMutex
}

// NewStandardPerformanceMonitor 创建新的标准性能监控器
func NewStandardPerformanceMonitor() *StandardPerformanceMonitor {
	return &StandardPerformanceMonitor{
		metrics:        make(map[string]*MetricData),
		startTime:      time.Now(),
		lastCPUTime:    time.Now(),
		latencyBuckets: make(map[string][]float64),
	}
}

// StartOperation 开始操作跟踪
func (spm *StandardPerformanceMonitor) StartOperation(operationType string) *OperationTracker {
	atomic.AddInt64(&spm.totalOperations, 1)

	return &OperationTracker{
		operationType: operationType,
		startTime:     time.Now(),
		monitor:       spm,
		tags:          make(map[string]string),
	}
}

// End 结束操作跟踪
func (ot *OperationTracker) End(success bool, tags map[string]string) {
	duration := time.Since(ot.startTime)

	if success {
		atomic.AddInt64(&ot.monitor.successfulOps, 1)
	} else {
		atomic.AddInt64(&ot.monitor.failedOps, 1)
	}

	// 合并标签
	allTags := make(map[string]string)
	for k, v := range ot.tags {
		allTags[k] = v
	}
	for k, v := range tags {
		allTags[k] = v
	}
	allTags["operation_type"] = ot.operationType
	allTags["success"] = fmt.Sprintf("%t", success)

	// 记录延迟指标
	latencyMs := float64(duration.Nanoseconds()) / 1e6
	ot.monitor.RecordMetric("operation_latency_ms", latencyMs, allTags)

	// 记录到延迟桶中用于百分位数计算
	ot.monitor.latencyMutex.Lock()
	key := ot.operationType
	if _, exists := ot.monitor.latencyBuckets[key]; !exists {
		ot.monitor.latencyBuckets[key] = make([]float64, 0, 1000)
	}
	ot.monitor.latencyBuckets[key] = append(ot.monitor.latencyBuckets[key], latencyMs)

	// 保持桶大小在合理范围内
	if len(ot.monitor.latencyBuckets[key]) > 10000 {
		// 保留最近的5000个样本
		copy(ot.monitor.latencyBuckets[key], ot.monitor.latencyBuckets[key][5000:])
		ot.monitor.latencyBuckets[key] = ot.monitor.latencyBuckets[key][:5000]
	}
	ot.monitor.latencyMutex.Unlock()
}

// AddTag 添加标签
func (ot *OperationTracker) AddTag(key, value string) {
	ot.tags[key] = value
}

// RecordMetric 记录指标
func (spm *StandardPerformanceMonitor) RecordMetric(name string, value float64, tags map[string]string) {
	spm.mutex.Lock()
	defer spm.mutex.Unlock()

	key := name
	if tags != nil && len(tags) > 0 {
		// 为了简化，这里不区分不同标签的指标
		// 在生产环境中，应该为不同的标签组合创建不同的指标
	}

	metric, exists := spm.metrics[key]
	if !exists {
		metric = &MetricData{
			Name:      name,
			Type:      "histogram",
			Min:       value,
			Max:       value,
			Tags:      tags,
			Timestamp: time.Now(),
		}
		spm.metrics[key] = metric
	}

	metric.Count++
	metric.Sum += value
	metric.Value = value
	metric.Average = metric.Sum / float64(metric.Count)
	metric.Timestamp = time.Now()

	if value < metric.Min {
		metric.Min = value
	}
	if value > metric.Max {
		metric.Max = value
	}
}

// GetMetrics 获取所有指标
func (spm *StandardPerformanceMonitor) GetMetrics() map[string]MetricData {
	spm.mutex.RLock()
	defer spm.mutex.RUnlock()

	result := make(map[string]MetricData)
	for k, v := range spm.metrics {
		metric := *v // 复制值

		// 计算百分位数
		spm.latencyMutex.RLock()
		if latencies, exists := spm.latencyBuckets[k]; exists && len(latencies) > 0 {
			metric.Percentiles = spm.calculatePercentiles(latencies)
		}
		spm.latencyMutex.RUnlock()

		result[k] = metric
	}

	return result
}

// calculatePercentiles 计算百分位数
func (spm *StandardPerformanceMonitor) calculatePercentiles(values []float64) map[string]float64 {
	if len(values) == 0 {
		return nil
	}

	// 复制并排序
	sorted := make([]float64, len(values))
	copy(sorted, values)
	spm.quickSort(sorted, 0, len(sorted)-1)

	percentiles := map[string]float64{
		"p50":  spm.percentile(sorted, 0.50),
		"p90":  spm.percentile(sorted, 0.90),
		"p95":  spm.percentile(sorted, 0.95),
		"p99":  spm.percentile(sorted, 0.99),
		"p999": spm.percentile(sorted, 0.999),
	}

	return percentiles
}

// percentile 计算百分位数
func (spm *StandardPerformanceMonitor) percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}

	index := p * float64(len(sorted)-1)
	lower := int(index)
	upper := lower + 1

	if upper >= len(sorted) {
		return sorted[len(sorted)-1]
	}

	weight := index - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

// quickSort 快速排序
func (spm *StandardPerformanceMonitor) quickSort(arr []float64, low, high int) {
	if low < high {
		pi := spm.partition(arr, low, high)
		spm.quickSort(arr, low, pi-1)
		spm.quickSort(arr, pi+1, high)
	}
}

// partition 分区函数
func (spm *StandardPerformanceMonitor) partition(arr []float64, low, high int) int {
	pivot := arr[high]
	i := low - 1

	for j := low; j < high; j++ {
		if arr[j] < pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}

// GetSystemStats 获取系统统计信息
func (spm *StandardPerformanceMonitor) GetSystemStats() SystemStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	memStats := MemoryStats{
		Allocated:    m.Alloc,
		TotalAlloc:   m.TotalAlloc,
		Sys:          m.Sys,
		Lookups:      m.Lookups,
		Mallocs:      m.Mallocs,
		Frees:        m.Frees,
		HeapAlloc:    m.HeapAlloc,
		HeapSys:      m.HeapSys,
		HeapIdle:     m.HeapIdle,
		HeapInuse:    m.HeapInuse,
		HeapReleased: m.HeapReleased,
		HeapObjects:  m.HeapObjects,
		StackInuse:   m.StackInuse,
		StackSys:     m.StackSys,
		MSpanInuse:   m.MSpanInuse,
		MSpanSys:     m.MSpanSys,
		MCacheInuse:  m.MCacheInuse,
		MCacheSys:    m.MCacheSys,
		BuckHashSys:  m.BuckHashSys,
		GCSys:        m.GCSys,
		OtherSys:     m.OtherSys,
		UsagePercent: float64(m.Alloc) / float64(m.Sys) * 100,
	}

	cpuUsage := spm.calculateCPUUsage()

	return SystemStats{
		CPUUsage:    cpuUsage,
		MemoryUsage: memStats,
		Goroutines:  runtime.NumGoroutine(),
		GCCycles:    m.NumGC,
		Timestamp:   time.Now(),
	}
}

// calculateCPUUsage 计算CPU使用率（简化实现）
func (spm *StandardPerformanceMonitor) calculateCPUUsage() float64 {
	// 这是一个简化的CPU使用率计算
	// 在生产环境中，应该使用更精确的方法
	now := time.Now()
	if now.Sub(spm.lastCPUTime) < time.Second {
		return spm.lastCPUUsage
	}

	// 模拟CPU使用率计算
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// 基于GC和内存分配活动估算CPU使用率
	cpuUsage := float64(runtime.NumGoroutine()) / 100.0
	if cpuUsage > 100 {
		cpuUsage = 100
	}

	spm.lastCPUTime = now
	spm.lastCPUUsage = cpuUsage

	return cpuUsage
}

// ExportPrometheusMetrics 导出Prometheus格式的指标
func (spm *StandardPerformanceMonitor) ExportPrometheusMetrics() string {
	metrics := spm.GetMetrics()
	systemStats := spm.GetSystemStats()

	var result string

	// 导出自定义指标
	for name, metric := range metrics {
		result += fmt.Sprintf("# HELP %s %s\n", name, "Custom metric")
		result += fmt.Sprintf("# TYPE %s %s\n", name, metric.Type)

		tagStr := ""
		if metric.Tags != nil && len(metric.Tags) > 0 {
			tagPairs := make([]string, 0, len(metric.Tags))
			for k, v := range metric.Tags {
				tagPairs = append(tagPairs, fmt.Sprintf(`%s="%s"`, k, v))
			}
			tagStr = "{" + fmt.Sprintf("%v", tagPairs) + "}"
		}

		result += fmt.Sprintf("%s%s %f\n", name, tagStr, metric.Value)

		if metric.Percentiles != nil {
			for p, value := range metric.Percentiles {
				result += fmt.Sprintf("%s_%s%s %f\n", name, p, tagStr, value)
			}
		}
	}

	// 导出系统指标
	result += fmt.Sprintf("# HELP vector_db_cpu_usage CPU usage percentage\n")
	result += fmt.Sprintf("# TYPE vector_db_cpu_usage gauge\n")
	result += fmt.Sprintf("vector_db_cpu_usage %f\n", systemStats.CPUUsage)

	result += fmt.Sprintf("# HELP vector_db_memory_usage_bytes Memory usage in bytes\n")
	result += fmt.Sprintf("# TYPE vector_db_memory_usage_bytes gauge\n")
	result += fmt.Sprintf("vector_db_memory_usage_bytes %d\n", systemStats.MemoryUsage.Allocated)

	result += fmt.Sprintf("# HELP vector_db_goroutines Number of goroutines\n")
	result += fmt.Sprintf("# TYPE vector_db_goroutines gauge\n")
	result += fmt.Sprintf("vector_db_goroutines %d\n", systemStats.Goroutines)

	result += fmt.Sprintf("# HELP vector_db_total_operations Total number of operations\n")
	result += fmt.Sprintf("# TYPE vector_db_total_operations counter\n")
	result += fmt.Sprintf("vector_db_total_operations %d\n", atomic.LoadInt64(&spm.totalOperations))

	result += fmt.Sprintf("# HELP vector_db_successful_operations Number of successful operations\n")
	result += fmt.Sprintf("# TYPE vector_db_successful_operations counter\n")
	result += fmt.Sprintf("vector_db_successful_operations %d\n", atomic.LoadInt64(&spm.successfulOps))

	result += fmt.Sprintf("# HELP vector_db_failed_operations Number of failed operations\n")
	result += fmt.Sprintf("# TYPE vector_db_failed_operations counter\n")
	result += fmt.Sprintf("vector_db_failed_operations %d\n", atomic.LoadInt64(&spm.failedOps))

	return result
}

// Reset 重置所有指标
func (spm *StandardPerformanceMonitor) Reset() {
	spm.mutex.Lock()
	defer spm.mutex.Unlock()

	spm.metrics = make(map[string]*MetricData)
	atomic.StoreInt64(&spm.totalOperations, 0)
	atomic.StoreInt64(&spm.successfulOps, 0)
	atomic.StoreInt64(&spm.failedOps, 0)

	spm.latencyMutex.Lock()
	spm.latencyBuckets = make(map[string][]float64)
	spm.latencyMutex.Unlock()

	spm.startTime = time.Now()
}

// AlertManager 告警管理器
type AlertManager struct {
	rules         []AlertRule
	handlers      []AlertHandler
	mutex         sync.RWMutex
	monitor       PerformanceMonitor
	active        bool
	checkInterval time.Duration
}

// AlertHandler 告警处理器接口
type AlertHandler interface {
	Handle(alert Alert) error
}

// Alert 告警信息
type Alert struct {
	Rule      AlertRule
	Value     float64
	Timestamp time.Time
	Message   string
}

// NewAlertManager 创建新的告警管理器
func NewAlertManager(monitor PerformanceMonitor) *AlertManager {
	return &AlertManager{
		rules:         make([]AlertRule, 0),
		handlers:      make([]AlertHandler, 0),
		monitor:       monitor,
		checkInterval: 30 * time.Second,
	}
}

// AddRule 添加告警规则
func (am *AlertManager) AddRule(rule AlertRule) {
	am.mutex.Lock()
	defer am.mutex.Unlock()
	am.rules = append(am.rules, rule)
}

// AddHandler 添加告警处理器
func (am *AlertManager) AddHandler(handler AlertHandler) {
	am.mutex.Lock()
	defer am.mutex.Unlock()
	am.handlers = append(am.handlers, handler)
}

// Start 启动告警管理器
func (am *AlertManager) Start() {
	am.mutex.Lock()
	am.active = true
	am.mutex.Unlock()

	go am.checkAlerts()
}

// Stop 停止告警管理器
func (am *AlertManager) Stop() {
	am.mutex.Lock()
	am.active = false
	am.mutex.Unlock()
}

// checkAlerts 检查告警
func (am *AlertManager) checkAlerts() {
	ticker := time.NewTicker(am.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			am.mutex.RLock()
			if !am.active {
				am.mutex.RUnlock()
				return
			}
			am.mutex.RUnlock()

			am.evaluateRules()
		}
	}
}

// evaluateRules 评估告警规则
func (am *AlertManager) evaluateRules() {
	metrics := am.monitor.GetMetrics()
	systemStats := am.monitor.GetSystemStats()

	am.mutex.RLock()
	rules := make([]AlertRule, len(am.rules))
	copy(rules, am.rules)
	handlers := make([]AlertHandler, len(am.handlers))
	copy(handlers, am.handlers)
	am.mutex.RUnlock()

	for _, rule := range rules {
		if !rule.Enabled {
			continue
		}

		var value float64
		var found bool

		// 检查自定义指标
		if metric, exists := metrics[rule.Metric]; exists {
			value = metric.Value
			found = true
		} else {
			// 检查系统指标
			switch rule.Metric {
			case "cpu_usage":
				value = systemStats.CPUUsage
				found = true
			case "memory_usage":
				value = systemStats.MemoryUsage.UsagePercent
				found = true
			case "goroutines":
				value = float64(systemStats.Goroutines)
				found = true
			}
		}

		if !found {
			continue
		}

		// 评估条件
		triggered := false
		switch rule.Condition.Operator {
		case ">":
			triggered = value > rule.Condition.Threshold
		case "<":
			triggered = value < rule.Condition.Threshold
		case ">=":
			triggered = value >= rule.Condition.Threshold
		case "<=":
			triggered = value <= rule.Condition.Threshold
		case "==":
			triggered = value == rule.Condition.Threshold
		case "!=":
			triggered = value != rule.Condition.Threshold
		}

		if triggered {
			now := time.Now()
			if now.Sub(rule.lastTriggered) >= rule.Duration {
				alert := Alert{
					Rule:      rule,
					Value:     value,
					Timestamp: now,
					Message:   fmt.Sprintf("Alert: %s - %s %s %f (current: %f)", rule.Name, rule.Metric, rule.Condition.Operator, rule.Condition.Threshold, value),
				}

				// 触发告警处理器
				for _, handler := range handlers {
					go func(h AlertHandler, a Alert) {
						if err := h.Handle(a); err != nil {
							fmt.Printf("告警处理失败: %v\n", err)
						}
					}(handler, alert)
				}

				// 更新最后触发时间
				am.mutex.Lock()
				for i := range am.rules {
					if am.rules[i].Name == rule.Name {
						am.rules[i].lastTriggered = now
						break
					}
				}
				am.mutex.Unlock()
			}
		}
	}
}

// LogAlertHandler 日志告警处理器
type LogAlertHandler struct{}

// Handle 处理告警
func (lah *LogAlertHandler) Handle(alert Alert) error {
	fmt.Printf("[%s] %s: %s\n", alert.Rule.Severity, alert.Timestamp.Format(time.RFC3339), alert.Message)
	return nil
}
