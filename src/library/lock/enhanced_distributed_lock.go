package lock

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/cenkalti/backoff/v4"
	etcdv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
)

// EnhancedDistributedLock 增强的分布式锁
type EnhancedDistributedLock struct {
	client      *etcdv3.Client
	session     *concurrency.Session
	mutex       *concurrency.Mutex
	lockKey     string
	timeout     time.Duration
	retryPolicy *backoff.ExponentialBackOff
	mu          sync.RWMutex
	locked      bool
	ctx         context.Context
	cancel      context.CancelFunc
}

// LockConfig 锁配置
type LockConfig struct {
	Timeout     time.Duration
	SessionTTL  time.Duration
	RetryPolicy *backoff.ExponentialBackOff
}

// NewEnhancedDistributedLock 创建增强的分布式锁
func NewEnhancedDistributedLock(client *etcdv3.Client, lockKey string, config *LockConfig) (*EnhancedDistributedLock, error) {
	ctx, cancel := context.WithCancel(context.Background())

	session, err := concurrency.NewSession(client,
		concurrency.WithTTL(int(config.SessionTTL.Seconds())),
		concurrency.WithContext(ctx),
	)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	mutex := concurrency.NewMutex(session, lockKey)

	return &EnhancedDistributedLock{
		client:      client,
		session:     session,
		mutex:       mutex,
		lockKey:     lockKey,
		timeout:     config.Timeout,
		retryPolicy: config.RetryPolicy,
		ctx:         ctx,
		cancel:      cancel,
	}, nil
}

// TryLock 尝试获取锁（非阻塞）
func (l *EnhancedDistributedLock) TryLock() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.locked {
		return fmt.Errorf("lock already acquired")
	}

	ctx, cancel := context.WithTimeout(l.ctx, 100*time.Millisecond)
	defer cancel()

	err := l.mutex.TryLock(ctx)
	if err != nil {
		return fmt.Errorf("failed to acquire lock: %w", err)
	}

	l.locked = true
	return nil
}

// Lock 获取锁（阻塞，支持超时和重试）
func (l *EnhancedDistributedLock) Lock() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.locked {
		return fmt.Errorf("lock already acquired")
	}

	var lockCtx context.Context
	var cancel context.CancelFunc

	if l.timeout > 0 {
		lockCtx, cancel = context.WithTimeout(l.ctx, l.timeout)
	} else {
		lockCtx, cancel = context.WithCancel(l.ctx)
	}
	defer cancel()

	// 使用重试策略
	retryPolicy := l.retryPolicy
	if retryPolicy != nil {
		retryPolicy.Reset()
		err := backoff.RetryNotify(
			func() error {
				return l.mutex.Lock(lockCtx)
			},
			backoff.WithContext(retryPolicy, lockCtx),
			func(err error, duration time.Duration) {
				// 记录重试日志
			},
		)
		if err != nil {
			return fmt.Errorf("failed to acquire lock with retry: %w", err)
		}
	} else {
		err := l.mutex.Lock(lockCtx)
		if err != nil {
			return fmt.Errorf("failed to acquire lock: %w", err)
		}
	}

	l.locked = true
	return nil
}

// Unlock 释放锁
func (l *EnhancedDistributedLock) Unlock() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if !l.locked {
		return fmt.Errorf("lock not acquired")
	}

	err := l.mutex.Unlock(l.ctx)
	if err != nil {
		return fmt.Errorf("failed to release lock: %w", err)
	}

	l.locked = false
	return nil
}

// IsLocked 检查是否已锁定
func (l *EnhancedDistributedLock) IsLocked() bool {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.locked
}

// Close 关闭锁
func (l *EnhancedDistributedLock) Close() error {
	l.cancel()

	if l.locked {
		l.Unlock()
	}

	if l.session != nil {
		return l.session.Close()
	}

	return nil
}
