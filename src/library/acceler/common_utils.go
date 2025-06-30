package acceler

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"
)

// Logger 统一日志记录器
type Logger struct {
	mu     sync.RWMutex
	Level  LogLevel
	Prefix string
}

// LogLevel 日志级别
type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarn
	LogLevelError
	LogLevelFatal
)

// GlobalLogger 全局日志记录器
var GlobalLogger = &Logger{
	Level:  LogLevelInfo,
	Prefix: "[VectorSphere]",
}

// SetLogLevel 设置日志级别
func (l *Logger) SetLogLevel(level LogLevel) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.Level = level
}

// Debug 记录调试信息
func (l *Logger) Debug(format string, args ...interface{}) {
	l.log(LogLevelDebug, format, args...)
}

// Info 记录信息
func (l *Logger) Info(format string, args ...interface{}) {
	l.log(LogLevelInfo, format, args...)
}

// Warn 记录警告
func (l *Logger) Warn(format string, args ...interface{}) {
	l.log(LogLevelWarn, format, args...)
}

// Error 记录错误
func (l *Logger) Error(format string, args ...interface{}) {
	l.log(LogLevelError, format, args...)
}

// Fatal 记录致命错误
func (l *Logger) Fatal(format string, args ...interface{}) {
	l.log(LogLevelFatal, format, args...)
}

// log 内部日志记录方法
func (l *Logger) log(level LogLevel, format string, args ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if level < l.Level {
		return
	}

	levelStr := l.getLevelString(level)
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	_, file, line, _ := runtime.Caller(2)

	message := fmt.Sprintf(format, args...)
	logMessage := fmt.Sprintf("%s [%s] %s:%d - %s", timestamp, levelStr, file, line, message)

	log.Println(l.Prefix, logMessage)
}

// getLevelString 获取日志级别字符串
func (l *Logger) getLevelString(level LogLevel) string {
	switch level {
	case LogLevelDebug:
		return "DEBUG"
	case LogLevelInfo:
		return "INFO"
	case LogLevelWarn:
		return "WARN"
	case LogLevelError:
		return "ERROR"
	case LogLevelFatal:
		return "FATAL"
	default:
		return "UNKNOWN"
	}
}

// GlobalErrorHandler 全局错误处理器
var GlobalErrorHandler = &ErrorHandler{
	errorCounts:   make(map[string]int),
	lastErrors:    make(map[string]*AcceleratorError),
	maxRetries:    3,
	retryInterval: time.Second * 5,
}

// ResetErrorCount 重置错误计数
func (eh *ErrorHandler) ResetErrorCount(operation string) {
	eh.mu.Lock()
	defer eh.mu.Unlock()
	delete(eh.errorCounts, operation)
	delete(eh.lastErrors, operation)
}

// PerformanceMonitor 性能监控器
type PerformanceMonitor struct {
	mu        sync.RWMutex
	Metrics   map[string]*OperationMetrics
	StartTime time.Time
}

// OperationMetrics 操作指标
type OperationMetrics struct {
	TotalCalls    int64         `json:"total_calls"`
	SuccessCalls  int64         `json:"success_calls"`
	FailedCalls   int64         `json:"failed_calls"`
	TotalDuration time.Duration `json:"total_duration"`
	MinDuration   time.Duration `json:"min_duration"`
	MaxDuration   time.Duration `json:"max_duration"`
	LastCall      time.Time     `json:"last_call"`
}

// GlobalPerformanceMonitor 全局性能监控器
var GlobalPerformanceMonitor = &PerformanceMonitor{
	Metrics:   make(map[string]*OperationMetrics),
	StartTime: time.Now(),
}

// StartOperation 开始操作监控
func (pm *PerformanceMonitor) StartOperation(operation string) *OperationTracker {
	return &OperationTracker{
		Operation: operation,
		StartTime: time.Now(),
		Monitor:   pm,
	}
}

// OperationTracker 操作跟踪器
type OperationTracker struct {
	Operation string
	StartTime time.Time
	Monitor   *PerformanceMonitor
}

// End 结束操作监控
func (ot *OperationTracker) End(success bool) {
	duration := time.Since(ot.StartTime)
	ot.Monitor.recordOperation(ot.Operation, duration, success)
}

// recordOperation 记录操作
func (pm *PerformanceMonitor) recordOperation(operation string, duration time.Duration, success bool) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	metrics, exists := pm.Metrics[operation]
	if !exists {
		metrics = &OperationMetrics{
			MinDuration: duration,
			MaxDuration: duration,
		}
		pm.Metrics[operation] = metrics
	}

	metrics.TotalCalls++
	if success {
		metrics.SuccessCalls++
	} else {
		metrics.FailedCalls++
	}

	metrics.TotalDuration += duration
	metrics.LastCall = time.Now()

	if duration < metrics.MinDuration {
		metrics.MinDuration = duration
	}
	if duration > metrics.MaxDuration {
		metrics.MaxDuration = duration
	}
}

// GetMetrics 获取操作指标
func (pm *PerformanceMonitor) GetMetrics(operation string) *OperationMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	metrics, exists := pm.Metrics[operation]
	if !exists {
		return nil
	}

	// 返回副本
	return &OperationMetrics{
		TotalCalls:    metrics.TotalCalls,
		SuccessCalls:  metrics.SuccessCalls,
		FailedCalls:   metrics.FailedCalls,
		TotalDuration: metrics.TotalDuration,
		MinDuration:   metrics.MinDuration,
		MaxDuration:   metrics.MaxDuration,
		LastCall:      metrics.LastCall,
	}
}

// GetAverageLatency 获取平均延迟
func (pm *PerformanceMonitor) GetAverageLatency(operation string) time.Duration {
	metrics := pm.GetMetrics(operation)
	if metrics == nil || metrics.TotalCalls == 0 {
		return 0
	}
	return metrics.TotalDuration / time.Duration(metrics.TotalCalls)
}

// GetSuccessRate 获取成功率
func (pm *PerformanceMonitor) GetSuccessRate(operation string) float64 {
	metrics := pm.GetMetrics(operation)
	if metrics == nil || metrics.TotalCalls == 0 {
		return 0.0
	}
	return float64(metrics.SuccessCalls) / float64(metrics.TotalCalls)
}

// GetAllMetrics 获取所有操作指标
func (pm *PerformanceMonitor) GetAllMetrics() map[string]*OperationMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	result := make(map[string]*OperationMetrics)
	for operation, metrics := range pm.Metrics {
		result[operation] = &OperationMetrics{
			TotalCalls:    metrics.TotalCalls,
			SuccessCalls:  metrics.SuccessCalls,
			FailedCalls:   metrics.FailedCalls,
			TotalDuration: metrics.TotalDuration,
			MinDuration:   metrics.MinDuration,
			MaxDuration:   metrics.MaxDuration,
			LastCall:      metrics.LastCall,
		}
	}
	return result
}

// ResetMetrics 重置指标
func (pm *PerformanceMonitor) ResetMetrics() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.Metrics = make(map[string]*OperationMetrics)
	pm.StartTime = time.Now()
}

// ValidateConfig 验证配置
func ValidateConfig(config interface{}) error {
	if config == nil {
		return fmt.Errorf("配置不能为空")
	}

	// 根据配置类型进行具体验证
	switch cfg := config.(type) {
	case *CPUConfig:
		return validateCPUConfig(cfg)
	case *GPUConfig:
		return validateGPUConfig(cfg)
	case *FPGAConfig:
		return validateFPGAConfig(cfg)
	case *RDMAConfig:
		return validateRDMAConfig(cfg)
	case *PMemConfig:
		return validatePMemConfig(cfg)
	default:
		return fmt.Errorf("未知的配置类型: %T", config)
	}
}

// validateCPUConfig 验证CPU配置
func validateCPUConfig(config *CPUConfig) error {
	if config.Threads <= 0 {
		return fmt.Errorf("CPU线程数必须大于0")
	}
	return nil
}

// validateGPUConfig 验证GPU配置
func validateGPUConfig(config *GPUConfig) error {
	if len(config.DeviceIDs) == 0 {
		return fmt.Errorf("GPU设备ID列表不能为空")
	}
	for _, id := range config.DeviceIDs {
		if id < 0 {
			return fmt.Errorf("GPU设备ID不能为负数: %d", id)
		}
	}
	return nil
}

// validateFPGAConfig 验证FPGA配置
func validateFPGAConfig(config *FPGAConfig) error {
	if len(config.DeviceIDs) == 0 {
		return fmt.Errorf("FPGA设备ID列表不能为空")
	}
	if config.Bitstream == "" {
		return fmt.Errorf("FPGA比特流文件路径不能为空")
	}
	return nil
}

// validateRDMAConfig 验证RDMA配置
func validateRDMAConfig(config *RDMAConfig) error {
	if config.PortNum <= 0 || config.PortNum > 65535 {
		return fmt.Errorf("RDMA端口号必须在1-65535范围内")
	}
	return nil
}

// validatePMemConfig 验证PMem配置
func validatePMemConfig(config *PMemConfig) error {
	if len(config.DevicePaths) == 0 {
		return fmt.Errorf("PMem设备路径列表不能为空")
	}
	if config.PoolSize <= 0 {
		return fmt.Errorf("PMem池大小必须大于0")
	}
	return nil
}

// RetryOperation 重试操作
func RetryOperation(acceleratorType, operation string, fn func() error, maxRetries int, interval time.Duration) error {
	var lastErr error

	for i := 0; i <= maxRetries; i++ {
		tracker := GlobalPerformanceMonitor.StartOperation(operation)
		err := fn()
		tracker.End(err == nil)

		if err == nil {
			GlobalErrorHandler.ResetErrorCount(operation)
			return nil
		}

		lastErr = err
		GlobalErrorHandler.HandleError(acceleratorType, operation, err)

		if i < maxRetries {
			GlobalLogger.Info("操作 %s 失败，%v 后重试 (第 %d/%d 次)", operation, interval, i+1, maxRetries)
			time.Sleep(interval)
		}
	}

	return fmt.Errorf("操作 %s 在 %d 次重试后仍然失败: %w", operation, maxRetries, lastErr)
}

// SafeExecute 安全执行操作
func SafeExecute(acceleratorType, operation string, fn func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("操作 %s 发生panic: %v", operation, r)
			GlobalLogger.Error("%v", err)
		}
	}()

	tracker := GlobalPerformanceMonitor.StartOperation(operation)
	defer func() {
		tracker.End(err == nil)
	}()

	err = fn()
	if err != nil {
		GlobalErrorHandler.HandleError(acceleratorType, operation, err)
	}

	return err
}
