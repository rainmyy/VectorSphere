package bootstrap

import (
	"context"
	"errors"
	"fmt"
	"github.com/cenkalti/backoff/v4"
	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	"strings"
	"time"
)

// ErrorType 错误类型
type ErrorType int

const (
	ErrorTypeTemporary ErrorType = iota
	ErrorTypePermanent
	ErrorTypeRetryable
	ErrorTypeNetwork
	ErrorTypeAuth
	ErrorTypeTimeout
)

// EtcdError 包装的etcd错误
type EtcdError struct {
	Original  error
	Type      ErrorType
	Message   string
	Retryable bool
}

func (e *EtcdError) Error() string {
	return fmt.Sprintf("etcd error [%v]: %s (original: %v)", e.Type, e.Message, e.Original)
}

func (e *EtcdError) Unwrap() error {
	return e.Original
}

// ErrorHandler 错误处理器
type ErrorHandler struct {
	retryPolicies map[ErrorType]*backoff.ExponentialBackOff
}

// NewErrorHandler 创建错误处理器
func NewErrorHandler() *ErrorHandler {
	eh := &ErrorHandler{
		retryPolicies: make(map[ErrorType]*backoff.ExponentialBackOff),
	}

	// 配置不同类型错误的重试策略
	eh.retryPolicies[ErrorTypeTemporary] = &backoff.ExponentialBackOff{
		InitialInterval:     1 * time.Second,
		RandomizationFactor: 0.1,
		Multiplier:          2.0,
		MaxInterval:         30 * time.Second,
		MaxElapsedTime:      5 * time.Minute,
		Clock:               backoff.SystemClock,
	}

	eh.retryPolicies[ErrorTypeNetwork] = &backoff.ExponentialBackOff{
		InitialInterval:     500 * time.Millisecond,
		RandomizationFactor: 0.2,
		Multiplier:          1.5,
		MaxInterval:         10 * time.Second,
		MaxElapsedTime:      2 * time.Minute,
		Clock:               backoff.SystemClock,
	}

	eh.retryPolicies[ErrorTypeTimeout] = &backoff.ExponentialBackOff{
		InitialInterval:     2 * time.Second,
		RandomizationFactor: 0.1,
		Multiplier:          2.0,
		MaxInterval:         60 * time.Second,
		MaxElapsedTime:      10 * time.Minute,
		Clock:               backoff.SystemClock,
	}

	return eh
}

// ClassifyError 分类错误
func (eh *ErrorHandler) ClassifyError(err error) *EtcdError {
	if err == nil {
		return nil
	}

	// 检查是否是etcd特定错误
	if errors.Is(err, rpctypes.ErrLeaderChanged) {
		return &EtcdError{
			Original:  err,
			Type:      ErrorTypeTemporary,
			Message:   "Leader changed, retrying",
			Retryable: true,
		}
	}

	if errors.Is(err, rpctypes.ErrNoLeader) {
		return &EtcdError{
			Original:  err,
			Type:      ErrorTypeTemporary,
			Message:   "No leader available, retrying",
			Retryable: true,
		}
	}

	if errors.Is(err, context.DeadlineExceeded) {
		return &EtcdError{
			Original:  err,
			Type:      ErrorTypeTimeout,
			Message:   "Operation timeout",
			Retryable: true,
		}
	}

	if errors.Is(err, rpctypes.ErrPermissionDenied) {
		return &EtcdError{
			Original:  err,
			Type:      ErrorTypeAuth,
			Message:   "Permission denied",
			Retryable: false,
		}
	}

	// 检查网络错误
	errMsg := strings.ToLower(err.Error())
	if strings.Contains(errMsg, "connection refused") ||
		strings.Contains(errMsg, "connection reset") ||
		strings.Contains(errMsg, "network unreachable") {
		return &EtcdError{
			Original:  err,
			Type:      ErrorTypeNetwork,
			Message:   "Network error",
			Retryable: true,
		}
	}

	// 默认为临时错误
	return &EtcdError{
		Original:  err,
		Type:      ErrorTypeTemporary,
		Message:   "Temporary error",
		Retryable: true,
	}
}

// RetryOperation 重试操作
func (eh *ErrorHandler) RetryOperation(ctx context.Context, operation func() error) error {
	retryableOperation := func() error {
		err := operation()
		if err == nil {
			return nil
		}

		etcdErr := eh.ClassifyError(err)
		if !etcdErr.Retryable {
			return backoff.Permanent(etcdErr)
		}

		return etcdErr
	}

	// 使用默认的重试策略
	bo := eh.retryPolicies[ErrorTypeTemporary]
	return backoff.Retry(retryableOperation, backoff.WithContext(bo, ctx))
}

// RetryOperationWithType 根据错误类型重试操作
func (eh *ErrorHandler) RetryOperationWithType(ctx context.Context, operation func() error, errorType ErrorType) error {
	retryableOperation := func() error {
		err := operation()
		if err == nil {
			return nil
		}

		etcdErr := eh.ClassifyError(err)
		if !etcdErr.Retryable {
			return backoff.Permanent(etcdErr)
		}

		return etcdErr
	}

	bo := eh.retryPolicies[errorType]
	if bo == nil {
		bo = eh.retryPolicies[ErrorTypeTemporary]
	}

	return backoff.Retry(retryableOperation, backoff.WithContext(bo, ctx))
}
