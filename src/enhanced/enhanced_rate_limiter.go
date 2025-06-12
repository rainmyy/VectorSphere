package enhanced

import (
	"math"
	"sync"
	"time"
)

// RateLimiterConfig 限流器配置
type RateLimiterConfig struct {
	Rate      float64       `json:"rate"`      // 每秒允许的请求数
	Burst     int           `json:"burst"`     // 突发请求数
	Window    time.Duration `json:"window"`    // 时间窗口
	Algorithm string        `json:"algorithm"` // "token_bucket", "sliding_window", "fixed_window"
	Enabled   bool          `json:"enabled"`
}

// RateLimiter 限流器
type RateLimiter struct {
	config     *RateLimiterConfig
	tokens     float64
	lastRefill time.Time
	requests   []time.Time
	mu         sync.Mutex
	metrics    *RateLimiterMetrics
}

// RateLimiterMetrics 限流器指标
type RateLimiterMetrics struct {
	TotalRequests    int64     `json:"total_requests"`
	AllowedRequests  int64     `json:"allowed_requests"`
	RejectedRequests int64     `json:"rejected_requests"`
	CurrentRate      float64   `json:"current_rate"`
	LastReset        time.Time `json:"last_reset"`
}

// 限流器方法

// allowRequest 检查是否允许请求
func (rl *RateLimiter) allowRequest() bool {
	if !rl.config.Enabled {
		return true
	}

	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	rl.metrics.TotalRequests++

	switch rl.config.Algorithm {
	case "token_bucket":
		return rl.tokenBucketAllow(now)
	case "sliding_window":
		return rl.slidingWindowAllow(now)
	case "fixed_window":
		return rl.fixedWindowAllow(now)
	default:
		return rl.tokenBucketAllow(now)
	}
}

// tokenBucketAllow 令牌桶算法
func (rl *RateLimiter) tokenBucketAllow(now time.Time) bool {
	// 计算需要添加的令牌数
	elapsed := now.Sub(rl.lastRefill)
	tokensToAdd := elapsed.Seconds() * rl.config.Rate
	rl.tokens = math.Min(rl.tokens+tokensToAdd, float64(rl.config.Burst))
	rl.lastRefill = now

	if rl.tokens >= 1.0 {
		rl.tokens--
		rl.metrics.AllowedRequests++
		return true
	}

	rl.metrics.RejectedRequests++
	return false
}

// slidingWindowAllow 滑动窗口算法
func (rl *RateLimiter) slidingWindowAllow(now time.Time) bool {
	// 清理过期请求
	cutoff := now.Add(-rl.config.Window)
	validRequests := make([]time.Time, 0)
	for _, reqTime := range rl.requests {
		if reqTime.After(cutoff) {
			validRequests = append(validRequests, reqTime)
		}
	}
	rl.requests = validRequests

	// 检查是否超过限制
	maxRequests := int(rl.config.Rate * rl.config.Window.Seconds())
	if len(rl.requests) < maxRequests {
		rl.requests = append(rl.requests, now)
		rl.metrics.AllowedRequests++
		return true
	}

	rl.metrics.RejectedRequests++
	return false
}

// fixedWindowAllow 固定窗口算法
func (rl *RateLimiter) fixedWindowAllow(now time.Time) bool {
	// 检查是否需要重置窗口
	if now.Sub(rl.metrics.LastReset) > rl.config.Window {
		rl.requests = make([]time.Time, 0)
		rl.metrics.LastReset = now
	}

	// 检查是否超过限制
	maxRequests := int(rl.config.Rate * rl.config.Window.Seconds())
	if len(rl.requests) < maxRequests {
		rl.requests = append(rl.requests, now)
		rl.metrics.AllowedRequests++
		return true
	}

	rl.metrics.RejectedRequests++
	return false
}
