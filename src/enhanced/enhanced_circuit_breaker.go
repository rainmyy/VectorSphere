package enhanced

import (
	"VectorSphere/src/library/log"
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
	name            string
	config          *CircuitBreakerConfig
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

// allowRequest 检查是否允许请求
func (cb *CircuitBreaker) allowRequest() bool {
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
			log.Info("Circuit breaker %s transitioned to half-open", cb.name)
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
			log.Info("Circuit breaker %s transitioned to closed", cb.name)
		}
	}
}

// recordFailure 记录失败
func (cb *CircuitBreaker) recordFailure() {
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
			log.Warning("Circuit breaker %s opened due to high failure rate: %.2f", cb.name, failureRate)
		} else if cb.state == HalfOpen {
			cb.state = Open
			cb.lastStateChange = time.Now()
			cb.metrics.StateChanges++
			cb.metrics.CurrentState = "open"
			log.Warning("Circuit breaker %s opened from half-open due to failure", cb.name)
		}
	}
}
