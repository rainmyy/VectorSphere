package common

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/cenkalti/backoff/v4"
	etcdv3 "go.etcd.io/etcd/client/v3"
)

// RetryableError 可重试错误接口
type RetryableError interface {
	error
	IsRetryable() bool
}

// EtcdRetryableError etcd可重试错误
type EtcdRetryableError struct {
	OriginalError error
	retryable     bool
}

func (e *EtcdRetryableError) Error() string {
	return e.OriginalError.Error()
}

func (e *EtcdRetryableError) IsRetryable() bool {
	return e.retryable
}

// RetryPolicyConfig 重试策略配置
type RetryPolicyConfig struct {
	InitialInterval     time.Duration
	MaxInterval         time.Duration
	MaxElapsedTime      time.Duration
	Multiplier          float64
	RandomizationFactor float64
	MaxRetries          int
}

// DefaultRetryPolicyConfig 默认重试策略配置
func DefaultRetryPolicyConfig() *RetryPolicyConfig {
	return &RetryPolicyConfig{
		InitialInterval:     1 * time.Second,
		MaxInterval:         30 * time.Second,
		MaxElapsedTime:      5 * time.Minute,
		Multiplier:          2.0,
		RandomizationFactor: 0.1,
		MaxRetries:          10,
	}
}

// CreateRetryPolicy 创建重试策略
func CreateRetryPolicy(config *RetryPolicyConfig) *backoff.ExponentialBackOff {
	policy := backoff.NewExponentialBackOff()
	policy.InitialInterval = config.InitialInterval
	policy.MaxInterval = config.MaxInterval
	policy.MaxElapsedTime = config.MaxElapsedTime
	policy.Multiplier = config.Multiplier
	policy.RandomizationFactor = config.RandomizationFactor
	return policy
}

// IsRetryableEtcdError 判断etcd错误是否可重试
func IsRetryableEtcdError(err error) bool {
	if err == nil {
		return false
	}

	// 检查是否为RetryableError接口
	if retryableErr, ok := err.(RetryableError); ok {
		return retryableErr.IsRetryable()
	}

	// 检查常见的可重试错误
	errorMsg := strings.ToLower(err.Error())
	retryableErrors := []string{
		"timeout",
		"connection refused",
		"connection reset by peer",
		"etcdserver: request timed out",
		"etcdserver: too many requests",
		"context deadline exceeded",
		"temporary failure",
		"network is unreachable",
	}

	for _, retryableError := range retryableErrors {
		if strings.Contains(errorMsg, retryableError) {
			return true
		}
	}

	// 检查etcd特定错误
	if etcdErr, ok := err.(*etcdv3.Error); ok {
		// 某些etcd错误码是可重试的
		retryableCodes := []int{
			// 添加可重试的etcd错误码
		}

		for _, code := range retryableCodes {
			if int(etcdErr.Code) == code {
				return true
			}
		}
	}

	return false
}

// RetryableOperation 执行可重试操作
func RetryableOperation(ctx context.Context, operation func() error, config *RetryPolicyConfig) error {
	policy := CreateRetryPolicy(config)

	return backoff.RetryNotify(
		func() error {
			err := operation()
			if err != nil && !IsRetryableEtcdError(err) {
				// 不可重试错误，立即返回
				return backoff.Permanent(err)
			}
			return err
		},
		backoff.WithContext(policy, ctx),
		func(err error, duration time.Duration) {
			// 记录重试日志
			fmt.Printf("Operation failed, retrying in %s. Error: %v\n", duration, err)
		},
	)
}

// CircuitBreakerConfig 熔断器配置
type CircuitBreakerConfig struct {
	FailureThreshold int
	RecoveryTimeout  time.Duration
	HalfOpenMaxCalls int
}

// CircuitBreaker 简单熔断器实现
type CircuitBreaker struct {
	config       *CircuitBreakerConfig
	failureCount int
	lastFailTime time.Time
	state        string // "closed", "open", "half-open"
}

// NewCircuitBreaker 创建熔断器
func NewCircuitBreaker(config *CircuitBreakerConfig) *CircuitBreaker {
	return &CircuitBreaker{
		config: config,
		state:  "closed",
	}
}

// Execute 执行操作（带熔断）
func (cb *CircuitBreaker) Execute(operation func() error) error {
	if cb.state == "open" {
		if time.Since(cb.lastFailTime) > cb.config.RecoveryTimeout {
			cb.state = "half-open"
		} else {
			return fmt.Errorf("circuit breaker is open")
		}
	}

	err := operation()
	if err != nil {
		cb.onFailure()
	} else {
		cb.onSuccess()
	}

	return err
}

func (cb *CircuitBreaker) onFailure() {
	cb.failureCount++
	cb.lastFailTime = time.Now()

	if cb.failureCount >= cb.config.FailureThreshold {
		cb.state = "open"
	}
}

func (cb *CircuitBreaker) onSuccess() {
	cb.failureCount = 0
	if cb.state == "half-open" {
		cb.state = "closed"
	}
}
