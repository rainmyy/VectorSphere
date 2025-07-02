package enhanced

import (
	"VectorSphere/src/library/logger"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// CircuitBreakerState 熔断器状态
type CircuitBreakerState int

const (
	Closed CircuitBreakerState = iota
	Open
	HalfOpen
)

// CircuitBreakerConfig 熔断器配置
type CircuitBreakerConfig struct {
	FailureThreshold  int           `json:"failure_threshold"`
	SuccessThreshold  int           `json:"success_threshold"`
	Timeout           time.Duration `json:"timeout"`
	ResetTimeout      time.Duration `json:"reset_timeout"`
	HalfOpenMaxCalls  int           `json:"half_open_max_calls"`
	FailureRate       float64       `json:"failure_rate"`
	MinRequestAmount  int           `json:"min_request_amount"`
	SlidingWindowSize int           `json:"sliding_window_size"`
}

// CircuitBreakerMetrics 熔断器指标
type CircuitBreakerMetrics struct {
	TotalRequests      int64     `json:"total_requests"`
	SuccessfulRequests int64     `json:"successful_requests"`
	FailedRequests     int64     `json:"failed_requests"`
	FailureRate        float64   `json:"failure_rate"`
	CurrentState       string    `json:"current_state"`
	LastStateChange    time.Time `json:"last_state_change"`
	HalfOpenCalls      int       `json:"half_open_calls"`

	SuccessfulCalls int64     `json:"successful_calls"`
	FailedCalls     int64     `json:"failed_calls"`
	RejectedCalls   int64     `json:"rejected_calls"`
	LastFailureTime time.Time `json:"last_failure_time"`
	LastSuccessTime time.Time `json:"last_success_time"`
	StateChanges    int64     `json:"state_changes"`
}

// CircuitBreaker 熔断器
type CircuitBreaker struct {
	Name   string
	config *CircuitBreakerConfig
	state           CircuitBreakerState
	failureCount    int64
	successCount    int64
	requestCount    int64
	lastFailureTime time.Time
	lastStateChange time.Time
	halfOpenCalls   int64
	mu              sync.RWMutex
	metrics         *CircuitBreakerMetrics
}

// shouldOpen 判断是否应该打开熔断器
func (cb *CircuitBreaker) shouldOpen() bool {
	// 请求数量不足，不打开熔断器
	if cb.metrics.TotalRequests < int64(cb.config.MinRequestAmount) {
		return false
	}

	// 失败率超过阈值，打开熔断器
	return cb.metrics.FailureRate >= cb.config.FailureRate
}

// updateFailureRate 更新失败率
func (cb *CircuitBreaker) updateFailureRate() {
	if cb.metrics.TotalRequests > 0 {
		cb.metrics.FailureRate = float64(cb.metrics.FailedRequests) / float64(cb.metrics.TotalRequests)
	} else {
		cb.metrics.FailureRate = 0
	}
}

// 熔断器方法

// AllowRequest 检查是否允许请求
func (cb *CircuitBreaker) AllowRequest() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()

	switch cb.state {
	case Closed:
		return true
	case Open:
		if now.Sub(cb.lastStateChange) > cb.config.ResetTimeout {
			cb.state = HalfOpen
			cb.halfOpenCalls = 0
			cb.lastStateChange = now
			cb.metrics.StateChanges++
			cb.metrics.CurrentState = "half_open"
			logger.Info("Circuit breaker %s transitioned to half-open", cb.Name)
			return true
		}
		return false
	case HalfOpen:
		if cb.halfOpenCalls < int64(cb.config.HalfOpenMaxCalls) {
			cb.halfOpenCalls++
			return true
		}
		return false
	default:
		return false
	}
}

// recordSuccess 记录成功
func (cb *CircuitBreaker) recordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	atomic.AddInt64(&cb.successCount, 1)
	atomic.AddInt64(&cb.requestCount, 1)
	cb.metrics.SuccessfulCalls++
	cb.metrics.TotalRequests++
	cb.metrics.LastSuccessTime = time.Now()

	if cb.state == HalfOpen {
		if cb.successCount >= int64(cb.config.SuccessThreshold) {
			cb.state = Closed
			cb.failureCount = 0
			cb.lastStateChange = time.Now()
			cb.metrics.StateChanges++
			cb.metrics.CurrentState = "closed"
			logger.Info("Circuit breaker %s transitioned to closed", cb.Name)
		}
	}
}

// RecordFailure 记录失败
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	atomic.AddInt64(&cb.failureCount, 1)
	atomic.AddInt64(&cb.requestCount, 1)
	cb.metrics.FailedCalls++
	cb.metrics.TotalRequests++
	cb.metrics.LastFailureTime = time.Now()
	cb.lastFailureTime = time.Now()

	// 计算失败率
	if cb.requestCount >= int64(cb.config.MinRequestAmount) {
		failureRate := float64(cb.failureCount) / float64(cb.requestCount)
		cb.metrics.FailureRate = failureRate

		if cb.state == Closed && failureRate >= cb.config.FailureRate {
			cb.state = Open
			cb.lastStateChange = time.Now()
			cb.metrics.StateChanges++
			cb.metrics.CurrentState = "open"
			logger.Warning("Circuit breaker %s opened due to high failure rate: %.2f", cb.Name, failureRate)
		} else if cb.state == HalfOpen {
			cb.state = Open
			cb.lastStateChange = time.Now()
			cb.metrics.StateChanges++
			cb.metrics.CurrentState = "open"
			logger.Warning("Circuit breaker %s opened from half-open due to failure", cb.Name)
		}
	}
}

// RecordSuccess 记录成功
func (cb *CircuitBreaker) RecordSuccess() {
	cb.recordSuccess()
}

// GetMetrics 获取指标
func (cb *CircuitBreaker) GetMetrics() *CircuitBreakerMetrics {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.metrics
}

// GetState 获取状态
func (cb *CircuitBreaker) GetState() CircuitBreakerState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// Reset 重置熔断器
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.state = Closed
	cb.failureCount = 0
	cb.successCount = 0
	cb.requestCount = 0
	cb.halfOpenCalls = 0
	cb.lastStateChange = time.Now()
	cb.metrics.StateChanges++
	cb.metrics.CurrentState = "closed"
	logger.Info("Circuit breaker %s reset", cb.Name)
}

// EnhancedCircuitBreaker 增强型熔断器管理器
type EnhancedCircuitBreaker struct {
	circuitBreakers map[string]*CircuitBreaker
	mu              sync.RWMutex
	isRunning       bool
}

// NewEnhancedCircuitBreaker 创建增强型熔断器管理器
func NewEnhancedCircuitBreaker() *EnhancedCircuitBreaker {
	return &EnhancedCircuitBreaker{
		circuitBreakers: make(map[string]*CircuitBreaker),
	}
}

// Start 启动增强型熔断器
func (ecb *EnhancedCircuitBreaker) Start() error {
	ecb.mu.Lock()
	defer ecb.mu.Unlock()

	if ecb.isRunning {
		return fmt.Errorf("enhanced circuit breaker is already running")
	}

	ecb.isRunning = true
	logger.Info("Enhanced circuit breaker started")
	return nil
}

// Stop 停止增强型熔断器
func (ecb *EnhancedCircuitBreaker) Stop() error {
	ecb.mu.Lock()
	defer ecb.mu.Unlock()

	if !ecb.isRunning {
		return fmt.Errorf("enhanced circuit breaker is not running")
	}

	ecb.isRunning = false
	logger.Info("Enhanced circuit breaker stopped")
	return nil
}

// CreateCircuitBreaker 创建熔断器
func (ecb *EnhancedCircuitBreaker) CreateCircuitBreaker(name string, config *CircuitBreakerConfig) *CircuitBreaker {
	ecb.mu.Lock()
	defer ecb.mu.Unlock()

	if config == nil {
		config = &CircuitBreakerConfig{
			FailureThreshold:  5,
			SuccessThreshold:  3,
			Timeout:           30 * time.Second,
			ResetTimeout:      60 * time.Second,
			HalfOpenMaxCalls:  3,
			FailureRate:       0.5,
			MinRequestAmount:  10,
			SlidingWindowSize: 100,
		}
	}

	cb := &CircuitBreaker{
		Name:            name,
		config:          config,
		state:           Closed,
		lastStateChange: time.Now(),
		metrics: &CircuitBreakerMetrics{
			CurrentState:    "closed",
			LastStateChange: time.Now(),
		},
	}

	ecb.circuitBreakers[name] = cb
	logger.Info("Circuit breaker created: %s", name)
	return cb
}

// GetCircuitBreaker 获取熔断器
func (ecb *EnhancedCircuitBreaker) GetCircuitBreaker(name string) *CircuitBreaker {
	ecb.mu.RLock()
	defer ecb.mu.RUnlock()
	return ecb.circuitBreakers[name]
}

// RemoveCircuitBreaker 移除熔断器
func (ecb *EnhancedCircuitBreaker) RemoveCircuitBreaker(name string) {
	ecb.mu.Lock()
	defer ecb.mu.Unlock()
	delete(ecb.circuitBreakers, name)
	logger.Info("Circuit breaker removed: %s", name)
}

// GetAllMetrics 获取所有熔断器指标
func (ecb *EnhancedCircuitBreaker) GetAllMetrics() map[string]*CircuitBreakerMetrics {
	ecb.mu.RLock()
	defer ecb.mu.RUnlock()

	result := make(map[string]*CircuitBreakerMetrics)
	for name, cb := range ecb.circuitBreakers {
		result[name] = cb.GetMetrics()
	}
	return result
}

// AllowRequest 检查是否允许请求（使用默认熔断器）
func (ecb *EnhancedCircuitBreaker) AllowRequest() bool {
	ecb.mu.RLock()
	defaultCB, exists := ecb.circuitBreakers["default"]
	ecb.mu.RUnlock()

	if !exists {
		// 如果没有默认熔断器，创建一个
		defaultConfig := &CircuitBreakerConfig{
			FailureThreshold:    5,
			SuccessThreshold:    3,
			Timeout:             30 * time.Second,
			ResetTimeout:        60 * time.Second,
			HalfOpenMaxCalls:    3,
			FailureRate:         0.5,
			MinRequestAmount:    10,
			SlidingWindowSize:   100,
		}
		ecb.CreateCircuitBreaker("default", defaultConfig)
		ecb.mu.RLock()
		defaultCB = ecb.circuitBreakers["default"]
		ecb.mu.RUnlock()
	}

	return defaultCB.AllowRequest()
}

// RecordSuccess 记录成功（使用默认熔断器）
func (ecb *EnhancedCircuitBreaker) RecordSuccess() {
	ecb.mu.RLock()
	defaultCB, exists := ecb.circuitBreakers["default"]
	ecb.mu.RUnlock()

	if exists {
		defaultCB.RecordSuccess()
	}
}

// RecordFailure 记录失败（使用默认熔断器）
func (ecb *EnhancedCircuitBreaker) RecordFailure() {
	ecb.mu.RLock()
	defaultCB, exists := ecb.circuitBreakers["default"]
	ecb.mu.RUnlock()

	if exists {
		defaultCB.RecordFailure()
	}
}

// GetState 获取状态（使用默认熔断器）
func (ecb *EnhancedCircuitBreaker) GetState() string {
	ecb.mu.RLock()
	defaultCB, exists := ecb.circuitBreakers["default"]
	ecb.mu.RUnlock()

	if !exists {
		return "closed"
	}

	state := defaultCB.GetState()
	switch state {
	case Closed:
		return "closed"
	case Open:
		return "open"
	case HalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// GetMetrics 获取指标（使用默认熔断器）
func (ecb *EnhancedCircuitBreaker) GetMetrics() *CircuitBreakerMetrics {
	ecb.mu.RLock()
	defaultCB, exists := ecb.circuitBreakers["default"]
	ecb.mu.RUnlock()

	if !exists {
		return &CircuitBreakerMetrics{}
	}

	return defaultCB.GetMetrics()
}
