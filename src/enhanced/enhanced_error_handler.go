package enhanced

import (
	"VectorSphere/src/library/logger"
	"context"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ErrorCategory 错误类别
type ErrorCategory int

const (
	TemporaryError ErrorCategory = iota
	PermanentError
	RetryableError
	NetworkError
	AuthenticationError
	TimeoutError
	ResourceError
	BusinessError
	SystemError
)

// ErrorSeverity 错误严重程度
type ErrorSeverity int

const (
	LowSeverity ErrorSeverity = iota
	MediumSeverity
	HighSeverity
	CriticalSeverity
)

// RetryStrategy 重试策略
type RetryStrategy int

const (
	FixedInterval RetryStrategy = iota
	ExponentialBackoff
	LinearBackoff
	CustomBackoff
	AdaptiveBackoff
)

// EnhancedError 增强错误信息
type EnhancedError struct {
	OriginalError error             `json:"-"`
	ErrorCode     string            `json:"error_code"`
	ErrorMessage  string            `json:"error_message"`
	Category      ErrorCategory     `json:"category"`
	Severity      ErrorSeverity     `json:"severity"`
	Retryable     bool              `json:"retryable"`
	Timestamp     time.Time         `json:"timestamp"`
	Context       map[string]string `json:"context"`
	StackTrace    string            `json:"stack_trace,omitempty"`
	ServiceName   string            `json:"service_name"`
	Operation     string            `json:"operation"`
	CorrelationID string            `json:"correlation_id"`
}

// RetryConfig 重试配置
type RetryConfig struct {
	MaxAttempts     int                                                  `json:"max_attempts"`
	InitialDelay    time.Duration                                        `json:"initial_delay"`
	MaxDelay        time.Duration                                        `json:"max_delay"`
	Multiplier      float64                                              `json:"multiplier"`
	Jitter          bool                                                 `json:"jitter"`
	Strategy        RetryStrategy                                        `json:"strategy"`
	RetryableErrors []ErrorCategory                                      `json:"retryable_errors"`
	BackoffFunc     func(attempt int, delay time.Duration) time.Duration `json:"-"`
}

// RetryAttempt 重试尝试信息
type RetryAttempt struct {
	Attempt   int            `json:"attempt"`
	Delay     time.Duration  `json:"delay"`
	Error     *EnhancedError `json:"error"`
	Timestamp time.Time      `json:"timestamp"`
	Duration  time.Duration  `json:"duration"`
}

// RetryResult 重试结果
type RetryResult struct {
	Success    bool            `json:"success"`
	Attempts   []*RetryAttempt `json:"attempts"`
	TotalTime  time.Duration   `json:"total_time"`
	FinalError *EnhancedError  `json:"final_error,omitempty"`
	Result     interface{}     `json:"result,omitempty"`
}

// ErrorPattern 错误模式
type ErrorPattern struct {
	Pattern     string        `json:"pattern"`
	Category    ErrorCategory `json:"category"`
	Severity    ErrorSeverity `json:"severity"`
	Retryable   bool          `json:"retryable"`
	Description string        `json:"description"`
}

// EnhancedErrorHandler 增强错误处理器
type EnhancedErrorHandler struct {
	client          *clientv3.Client
	errorPatterns   []*ErrorPattern
	retryConfigs    map[string]*RetryConfig
	circuitBreakers map[string]*CircuitBreaker
	rateLimiters    map[string]*RateLimiter
	errorStats      map[string]*ErrorStats
	mu              sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
	metricsEnabled  bool
	auditEnabled    bool
	alertEnabled    bool
	basePrefix      string
}

// ErrorStats 错误统计
type ErrorStats struct {
	ServiceName       string                  `json:"service_name"`
	Operation         string                  `json:"operation"`
	TotalErrors       int64                   `json:"total_errors"`
	ErrorsByCategory  map[ErrorCategory]int64 `json:"errors_by_category"`
	ErrorsBySeverity  map[ErrorSeverity]int64 `json:"errors_by_severity"`
	RetryAttempts     int64                   `json:"retry_attempts"`
	SuccessfulRetries int64                   `json:"successful_retries"`
	FailedRetries     int64                   `json:"failed_retries"`
	AverageRetryTime  time.Duration           `json:"average_retry_time"`
	LastError         time.Time               `json:"last_error"`
	FirstError        time.Time               `json:"first_error"`
}

// EnhancedErrorHandlerConfig 增强错误处理器配置
type EnhancedErrorHandlerConfig struct {
	MetricsEnabled bool   `json:"metrics_enabled"`
	AuditEnabled   bool   `json:"audit_enabled"`
	AlertEnabled   bool   `json:"alert_enabled"`
	BasePrefix     string `json:"base_prefix"`
}

// NewEnhancedErrorHandler 创建增强错误处理器
func NewEnhancedErrorHandler(client *clientv3.Client, config *EnhancedErrorHandlerConfig) *EnhancedErrorHandler {
	ctx, cancel := context.WithCancel(context.Background())

	if config == nil {
		config = &EnhancedErrorHandlerConfig{
			MetricsEnabled: true,
			AuditEnabled:   true,
			AlertEnabled:   true,
			BasePrefix:     "/vector_sphere/errors",
		}
	}

	eeh := &EnhancedErrorHandler{
		client:          client,
		errorPatterns:   make([]*ErrorPattern, 0),
		retryConfigs:    make(map[string]*RetryConfig),
		circuitBreakers: make(map[string]*CircuitBreaker),
		rateLimiters:    make(map[string]*RateLimiter),
		errorStats:      make(map[string]*ErrorStats),
		ctx:             ctx,
		cancel:          cancel,
		metricsEnabled:  config.MetricsEnabled,
		auditEnabled:    config.AuditEnabled,
		alertEnabled:    config.AlertEnabled,
		basePrefix:      config.BasePrefix,
	}

	// 初始化默认错误模式
	eeh.initDefaultErrorPatterns()

	// 初始化默认重试配置
	eeh.initDefaultRetryConfigs()

	// 启动指标收集
	if config.MetricsEnabled {
		go eeh.startMetricsCollection()
	}

	return eeh
}

// ClassifyError 分类错误
func (eeh *EnhancedErrorHandler) ClassifyError(err error, serviceName, operation string) *EnhancedError {
	if err == nil {
		return nil
	}

	enhancedErr := &EnhancedError{
		OriginalError: err,
		ErrorMessage:  err.Error(),
		Timestamp:     time.Now(),
		ServiceName:   serviceName,
		Operation:     operation,
		Context:       make(map[string]string),
		CorrelationID: eeh.generateCorrelationID(),
	}

	// 根据错误模式分类
	for _, pattern := range eeh.errorPatterns {
		if eeh.matchErrorPattern(err, pattern) {
			enhancedErr.Category = pattern.Category
			enhancedErr.Severity = pattern.Severity
			enhancedErr.Retryable = pattern.Retryable
			enhancedErr.ErrorCode = pattern.Pattern
			break
		}
	}

	// 如果没有匹配的模式，使用默认分类
	if enhancedErr.Category == 0 {
		eeh.classifyByErrorType(enhancedErr)
	}

	// 更新统计信息
	if eeh.metricsEnabled {
		eeh.updateErrorStats(enhancedErr)
	}

	// 审计日志
	if eeh.auditEnabled {
		go eeh.auditError(enhancedErr)
	}

	return enhancedErr
}

// ExecuteWithRetry 执行带重试的操作
func (eeh *EnhancedErrorHandler) ExecuteWithRetry(ctx context.Context, operation func() (interface{}, error), serviceName, operationName string, config *RetryConfig) *RetryResult {
	logger.Info("Executing operation with retry: %s.%s", serviceName, operationName)

	if config == nil {
		config = eeh.getRetryConfig(serviceName, operationName)
	}

	result := &RetryResult{
		Attempts: make([]*RetryAttempt, 0),
	}

	start := time.Now()

	for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
		attemptStart := time.Now()

		// 检查熔断器
		if !eeh.checkCircuitBreaker(serviceName, operationName) {
			result.FinalError = &EnhancedError{
				ErrorMessage:  "Circuit breaker is open",
				Category:      SystemError,
				Severity:      HighSeverity,
				Retryable:     false,
				Timestamp:     time.Now(),
				ServiceName:   serviceName,
				Operation:     operationName,
				CorrelationID: eeh.generateCorrelationID(),
			}
			break
		}

		// 检查限流器
		if !eeh.checkRateLimit(serviceName, operationName) {
			result.FinalError = &EnhancedError{
				ErrorMessage:  "Rate limit exceeded",
				Category:      ResourceError,
				Severity:      MediumSeverity,
				Retryable:     true,
				Timestamp:     time.Now(),
				ServiceName:   serviceName,
				Operation:     operationName,
				CorrelationID: eeh.generateCorrelationID(),
			}
			// 限流时等待一段时间再重试
			time.Sleep(config.InitialDelay)
			continue
		}

		// 执行操作
		value, err := operation()
		attemptDuration := time.Since(attemptStart)

		if err == nil {
			// 成功
			result.Success = true
			result.Result = value
			result.TotalTime = time.Since(start)

			// 记录成功到熔断器
			eeh.recordCircuitBreakerSuccess(serviceName, operationName)

			logger.Info("Operation succeeded after %d attempts: %s.%s", attempt, serviceName, operationName)
			return result
		}

		// 分类错误
		enhancedErr := eeh.ClassifyError(err, serviceName, operationName)

		// 记录重试尝试
		retryAttempt := &RetryAttempt{
			Attempt:   attempt,
			Error:     enhancedErr,
			Timestamp: time.Now(),
			Duration:  attemptDuration,
		}
		result.Attempts = append(result.Attempts, retryAttempt)

		// 记录失败到熔断器
		eeh.recordCircuitBreakerFailure(serviceName, operationName)

		// 检查是否应该重试
		if !eeh.shouldRetry(enhancedErr, config) || attempt >= config.MaxAttempts {
			result.FinalError = enhancedErr
			break
		}

		// 计算延迟
		delay := eeh.calculateDelay(attempt, config)
		retryAttempt.Delay = delay

		// 等待重试
		select {
		case <-ctx.Done():
			result.FinalError = &EnhancedError{
				OriginalError: ctx.Err(),
				ErrorMessage:  "Context cancelled during retry",
				Category:      SystemError,
				Severity:      MediumSeverity,
				Retryable:     false,
				Timestamp:     time.Now(),
				ServiceName:   serviceName,
				Operation:     operationName,
				CorrelationID: eeh.generateCorrelationID(),
			}
			return result
		case <-time.After(delay):
			// 继续重试
		}

		logger.Debug("Retrying operation %s.%s, attempt %d after %v", serviceName, operationName, attempt+1, delay)
	}

	result.TotalTime = time.Since(start)
	logger.Warning("Operation failed after %d attempts: %s.%s, final error: %v", len(result.Attempts), serviceName, operationName, result.FinalError)

	return result
}

// RegisterRetryConfig 注册重试配置
func (eeh *EnhancedErrorHandler) RegisterRetryConfig(serviceName, operation string, config *RetryConfig) {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	eeh.mu.Lock()
	eeh.retryConfigs[key] = config
	eeh.mu.Unlock()
	logger.Info("Registered retry config for %s", key)
}

// RegisterCircuitBreaker 注册熔断器
func (eeh *EnhancedErrorHandler) RegisterCircuitBreaker(serviceName, operation string, config *CircuitBreakerConfig) {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	cb := &CircuitBreaker{
		Name:            key,
		config:          config,
		state:           Closed,
		lastStateChange: time.Now(),
		metrics: &CircuitBreakerMetrics{
			CurrentState: "closed",
		},
	}

	eeh.mu.Lock()
	eeh.circuitBreakers[key] = cb
	eeh.mu.Unlock()

	logger.Info("Registered circuit breaker for %s", key)
}

// RegisterRateLimiter 注册限流器
func (eeh *EnhancedErrorHandler) RegisterRateLimiter(serviceName, operation string, config *RateLimiterConfig) {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	rl := &RateLimiter{
		config:     config,
		tokens:     float64(config.Burst),
		lastRefill: time.Now(),
		requests:   make([]time.Time, 0),
		metrics: &RateLimiterMetrics{
			LastReset: time.Now(),
		},
	}

	eeh.mu.Lock()
	eeh.rateLimiters[key] = rl
	eeh.mu.Unlock()

	logger.Info("Registered rate limiter for %s", key)
}

// AddErrorPattern 添加错误模式
func (eeh *EnhancedErrorHandler) AddErrorPattern(pattern *ErrorPattern) {
	eeh.mu.Lock()
	eeh.errorPatterns = append(eeh.errorPatterns, pattern)
	eeh.mu.Unlock()
	logger.Info("Added error pattern: %s", pattern.Pattern)
}

// GetErrorStats 获取错误统计
func (eeh *EnhancedErrorHandler) GetErrorStats(serviceName, operation string) *ErrorStats {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	eeh.mu.RLock()
	defer eeh.mu.RUnlock()
	return eeh.errorStats[key]
}

// GetCircuitBreakerMetrics 获取熔断器指标
func (eeh *EnhancedErrorHandler) GetCircuitBreakerMetrics(serviceName, operation string) *CircuitBreakerMetrics {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	eeh.mu.RLock()
	defer eeh.mu.RUnlock()

	if cb, exists := eeh.circuitBreakers[key]; exists {
		cb.mu.RLock()
		defer cb.mu.RUnlock()
		return cb.metrics
	}
	return nil
}

// GetRateLimiterMetrics 获取限流器指标
func (eeh *EnhancedErrorHandler) GetRateLimiterMetrics(serviceName, operation string) *RateLimiterMetrics {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	eeh.mu.RLock()
	defer eeh.mu.RUnlock()

	if rl, exists := eeh.rateLimiters[key]; exists {
		rl.mu.Lock()
		defer rl.mu.Unlock()
		return rl.metrics
	}
	return nil
}

// 内部方法实现

// initDefaultErrorPatterns 初始化默认错误模式
func (eeh *EnhancedErrorHandler) initDefaultErrorPatterns() {
	patterns := []*ErrorPattern{
		{
			Pattern:     "connection refused",
			Category:    NetworkError,
			Severity:    HighSeverity,
			Retryable:   true,
			Description: "Network connection refused",
		},
		{
			Pattern:     "timeout",
			Category:    TimeoutError,
			Severity:    MediumSeverity,
			Retryable:   true,
			Description: "Operation timeout",
		},
		{
			Pattern:     "authentication failed",
			Category:    AuthenticationError,
			Severity:    HighSeverity,
			Retryable:   false,
			Description: "Authentication failure",
		},
		{
			Pattern:     "resource not found",
			Category:    ResourceError,
			Severity:    MediumSeverity,
			Retryable:   false,
			Description: "Resource not found",
		},
		{
			Pattern:     "internal server error",
			Category:    SystemError,
			Severity:    CriticalSeverity,
			Retryable:   true,
			Description: "Internal server error",
		},
	}

	for _, pattern := range patterns {
		eeh.errorPatterns = append(eeh.errorPatterns, pattern)
	}
}

// initDefaultRetryConfigs 初始化默认重试配置
func (eeh *EnhancedErrorHandler) initDefaultRetryConfigs() {
	// 默认重试配置
	defaultConfig := &RetryConfig{
		MaxAttempts:  3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
		Jitter:       true,
		Strategy:     ExponentialBackoff,
		RetryableErrors: []ErrorCategory{
			TemporaryError,
			NetworkError,
			TimeoutError,
			SystemError,
		},
	}

	// 网络操作重试配置
	networkConfig := &RetryConfig{
		MaxAttempts:  5,
		InitialDelay: 500 * time.Millisecond,
		MaxDelay:     10 * time.Second,
		Multiplier:   1.5,
		Jitter:       true,
		Strategy:     ExponentialBackoff,
		RetryableErrors: []ErrorCategory{
			NetworkError,
			TimeoutError,
			TemporaryError,
		},
	}

	eeh.retryConfigs["default"] = defaultConfig
	eeh.retryConfigs["network"] = networkConfig
}

// matchErrorPattern 匹配错误模式
func (eeh *EnhancedErrorHandler) matchErrorPattern(err error, pattern *ErrorPattern) bool {
	errorMsg := strings.ToLower(err.Error())
	patternStr := strings.ToLower(pattern.Pattern)
	return strings.Contains(errorMsg, patternStr)
}

// classifyByErrorType 根据错误类型分类
func (eeh *EnhancedErrorHandler) classifyByErrorType(enhancedErr *EnhancedError) {
	err := enhancedErr.OriginalError
	errorMsg := strings.ToLower(err.Error())

	// 检查gRPC状态码
	if st, ok := status.FromError(err); ok {
		switch st.Code() {
		case codes.DeadlineExceeded:
			enhancedErr.Category = TimeoutError
			enhancedErr.Severity = MediumSeverity
			enhancedErr.Retryable = true
		case codes.Unavailable:
			enhancedErr.Category = NetworkError
			enhancedErr.Severity = HighSeverity
			enhancedErr.Retryable = true
		case codes.Unauthenticated:
			enhancedErr.Category = AuthenticationError
			enhancedErr.Severity = HighSeverity
			enhancedErr.Retryable = false
		case codes.NotFound:
			enhancedErr.Category = ResourceError
			enhancedErr.Severity = MediumSeverity
			enhancedErr.Retryable = false
		case codes.Internal:
			enhancedErr.Category = SystemError
			enhancedErr.Severity = CriticalSeverity
			enhancedErr.Retryable = true
		default:
			enhancedErr.Category = SystemError
			enhancedErr.Severity = MediumSeverity
			enhancedErr.Retryable = true
		}
		return
	}

	// 基于错误消息的简单分类
	if strings.Contains(errorMsg, "timeout") || strings.Contains(errorMsg, "deadline") {
		enhancedErr.Category = TimeoutError
		enhancedErr.Severity = MediumSeverity
		enhancedErr.Retryable = true
	} else if strings.Contains(errorMsg, "connection") || strings.Contains(errorMsg, "network") {
		enhancedErr.Category = NetworkError
		enhancedErr.Severity = HighSeverity
		enhancedErr.Retryable = true
	} else if strings.Contains(errorMsg, "auth") || strings.Contains(errorMsg, "permission") {
		enhancedErr.Category = AuthenticationError
		enhancedErr.Severity = HighSeverity
		enhancedErr.Retryable = false
	} else {
		enhancedErr.Category = SystemError
		enhancedErr.Severity = MediumSeverity
		enhancedErr.Retryable = true
	}
}

// getRetryConfig 获取重试配置
func (eeh *EnhancedErrorHandler) getRetryConfig(serviceName, operation string) *RetryConfig {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	eeh.mu.RLock()
	defer eeh.mu.RUnlock()

	if config, exists := eeh.retryConfigs[key]; exists {
		return config
	}

	// 尝试服务级别配置
	if config, exists := eeh.retryConfigs[serviceName]; exists {
		return config
	}

	// 返回默认配置
	return eeh.retryConfigs["default"]
}

// shouldRetry 判断是否应该重试
func (eeh *EnhancedErrorHandler) shouldRetry(enhancedErr *EnhancedError, config *RetryConfig) bool {
	if !enhancedErr.Retryable {
		return false
	}

	// 检查错误类别是否在可重试列表中
	for _, retryableCategory := range config.RetryableErrors {
		if enhancedErr.Category == retryableCategory {
			return true
		}
	}

	return false
}

// calculateDelay 计算延迟时间
func (eeh *EnhancedErrorHandler) calculateDelay(attempt int, config *RetryConfig) time.Duration {
	var delay time.Duration

	switch config.Strategy {
	case FixedInterval:
		delay = config.InitialDelay
	case ExponentialBackoff:
		delay = time.Duration(float64(config.InitialDelay) * math.Pow(config.Multiplier, float64(attempt-1)))
	case LinearBackoff:
		delay = time.Duration(int64(config.InitialDelay) * int64(attempt))
	case CustomBackoff:
		if config.BackoffFunc != nil {
			delay = config.BackoffFunc(attempt, config.InitialDelay)
		} else {
			delay = config.InitialDelay
		}
	case AdaptiveBackoff:
		// 自适应退避，基于历史成功率调整
		delay = eeh.calculateAdaptiveDelay(attempt, config)
	default:
		delay = config.InitialDelay
	}

	// 应用最大延迟限制
	if delay > config.MaxDelay {
		delay = config.MaxDelay
	}

	// 应用抖动
	if config.Jitter {
		jitter := time.Duration(rand.Float64() * float64(delay) * 0.1)
		delay += jitter
	}

	return delay
}

// calculateAdaptiveDelay 计算自适应延迟
func (eeh *EnhancedErrorHandler) calculateAdaptiveDelay(attempt int, config *RetryConfig) time.Duration {
	// 基础指数退避
	baseDelay := time.Duration(float64(config.InitialDelay) * math.Pow(config.Multiplier, float64(attempt-1)))

	// 这里可以根据历史成功率、系统负载等因素调整延迟
	// 简化实现：如果系统负载高，增加延迟
	adaptiveFactor := 1.0
	// TODO: 实现基于系统指标的自适应因子计算

	return time.Duration(float64(baseDelay) * adaptiveFactor)
}

// checkCircuitBreaker 检查熔断器
func (eeh *EnhancedErrorHandler) checkCircuitBreaker(serviceName, operation string) bool {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	eeh.mu.RLock()
	cb, exists := eeh.circuitBreakers[key]
	eeh.mu.RUnlock()

	if !exists {
		return true // 没有熔断器，允许通过
	}

	return cb.AllowRequest()
}

// recordCircuitBreakerSuccess 记录熔断器成功
func (eeh *EnhancedErrorHandler) recordCircuitBreakerSuccess(serviceName, operation string) {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	eeh.mu.RLock()
	cb, exists := eeh.circuitBreakers[key]
	eeh.mu.RUnlock()

	if exists {
		cb.recordSuccess()
	}
}

// recordCircuitBreakerFailure 记录熔断器失败
func (eeh *EnhancedErrorHandler) recordCircuitBreakerFailure(serviceName, operation string) {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	eeh.mu.RLock()
	cb, exists := eeh.circuitBreakers[key]
	eeh.mu.RUnlock()

	if exists {
		cb.RecordFailure()
	}
}

// checkRateLimit 检查限流
func (eeh *EnhancedErrorHandler) checkRateLimit(serviceName, operation string) bool {
	key := fmt.Sprintf("%s.%s", serviceName, operation)
	eeh.mu.RLock()
	rl, exists := eeh.rateLimiters[key]
	eeh.mu.RUnlock()

	if !exists {
		return true // 没有限流器，允许通过
	}

	return rl.allowRequest()
}

// updateErrorStats 更新错误统计
func (eeh *EnhancedErrorHandler) updateErrorStats(enhancedErr *EnhancedError) {
	key := fmt.Sprintf("%s.%s", enhancedErr.ServiceName, enhancedErr.Operation)
	eeh.mu.Lock()
	defer eeh.mu.Unlock()

	stats, exists := eeh.errorStats[key]
	if !exists {
		stats = &ErrorStats{
			ServiceName:      enhancedErr.ServiceName,
			Operation:        enhancedErr.Operation,
			ErrorsByCategory: make(map[ErrorCategory]int64),
			ErrorsBySeverity: make(map[ErrorSeverity]int64),
			FirstError:       enhancedErr.Timestamp,
		}
		eeh.errorStats[key] = stats
	}

	stats.TotalErrors++
	stats.ErrorsByCategory[enhancedErr.Category]++
	stats.ErrorsBySeverity[enhancedErr.Severity]++
	stats.LastError = enhancedErr.Timestamp
}

// generateCorrelationID 生成关联ID
func (eeh *EnhancedErrorHandler) generateCorrelationID() string {
	return fmt.Sprintf("err_%d_%d", time.Now().Unix(), rand.Int63())
}

// auditError 审计错误
func (eeh *EnhancedErrorHandler) auditError(enhancedErr *EnhancedError) {
	logger.Info("Error audit: service=%s, operation=%s, category=%d, severity=%d, retryable=%t, correlation_id=%s, message=%s",
		enhancedErr.ServiceName, enhancedErr.Operation, enhancedErr.Category, enhancedErr.Severity,
		enhancedErr.Retryable, enhancedErr.CorrelationID, enhancedErr.ErrorMessage)

	// 这里可以集成更复杂的审计系统
	// 例如发送到日志收集系统、数据库等
}

// startMetricsCollection 启动指标收集
func (eeh *EnhancedErrorHandler) startMetricsCollection() {
	logger.Info("Starting error metrics collection")

	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			eeh.collectMetrics()
		case <-eeh.ctx.Done():
			logger.Info("Error metrics collection stopped")
			return
		}
	}
}

// collectMetrics 收集指标
func (eeh *EnhancedErrorHandler) collectMetrics() {
	eeh.mu.RLock()
	errorStatsCount := len(eeh.errorStats)
	circuitBreakerCount := len(eeh.circuitBreakers)
	rateLimiterCount := len(eeh.rateLimiters)
	eeh.mu.RUnlock()

	logger.Debug("Error handler metrics: error_stats=%d, circuit_breakers=%d, rate_limiters=%d",
		errorStatsCount, circuitBreakerCount, rateLimiterCount)
}

// Close 关闭增强错误处理器
func (eeh *EnhancedErrorHandler) Close() error {
	logger.Info("Closing enhanced error handler")
	eeh.cancel()
	return nil
}
