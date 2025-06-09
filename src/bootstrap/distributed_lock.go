package bootstrap

import (
	"context"
	"errors"
	"fmt"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"log"
	"sync"
	"time"
)

// DistributedLockManager 分布式锁管理器
type DistributedLockManager struct {
	client  *clientv3.Client
	session *concurrency.Session
	locks   sync.Map // map[string]*concurrency.Mutex
}

// NewDistributedLockManager 创建分布式锁管理器
func NewDistributedLockManager(client *clientv3.Client, session *concurrency.Session) *DistributedLockManager {
	return &DistributedLockManager{
		client:  client,
		session: session,
	}
}

// AcquireLockWithTimeout 获取分布式锁（带超时）
func (dlm *DistributedLockManager) AcquireLockWithTimeout(ctx context.Context, lockName string, timeout time.Duration) (*concurrency.Mutex, error) {
	fullLockPath := "/locks/" + lockName
	mutex := concurrency.NewMutex(dlm.session, fullLockPath)

	// 创建带超时的上下文
	lockCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// 尝试获取锁
	err := mutex.Lock(lockCtx)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("failed to acquire lock %s within timeout %v", lockName, timeout)
		}
		return nil, fmt.Errorf("failed to acquire lock %s: %w", lockName, err)
	}

	// 存储锁引用
	dlm.locks.Store(lockName, mutex)

	log.Printf("Successfully acquired lock: %s", lockName)
	return mutex, nil
}

// ReleaseLock 释放分布式锁
func (dlm *DistributedLockManager) ReleaseLock(ctx context.Context, lockName string) error {
	mutexInterface, exists := dlm.locks.Load(lockName)
	if !exists {
		return fmt.Errorf("lock %s not found", lockName)
	}

	mutex := mutexInterface.(*concurrency.Mutex)
	err := mutex.Unlock(ctx)
	if err != nil {
		return fmt.Errorf("failed to release lock %s: %w", lockName, err)
	}

	// 从存储中移除
	dlm.locks.Delete(lockName)

	log.Printf("Successfully released lock: %s", lockName)
	return nil
}

// TryLock 尝试获取锁（非阻塞）
func (dlm *DistributedLockManager) TryLock(ctx context.Context, lockName string) (*concurrency.Mutex, bool, error) {
	fullLockPath := "/locks/" + lockName
	mutex := concurrency.NewMutex(dlm.session, fullLockPath)

	// 创建立即超时的上下文
	tryCtx, cancel := context.WithTimeout(ctx, 1*time.Millisecond)
	defer cancel()

	err := mutex.Lock(tryCtx)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, false, nil // 锁被占用
		}
		return nil, false, fmt.Errorf("failed to try lock %s: %w", lockName, err)
	}

	dlm.locks.Store(lockName, mutex)
	return mutex, true, nil
}
