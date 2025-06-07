package lock

import (
	"context"
	"fmt"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

type EtcdLock struct {
	client  *clientv3.Client
	lease   clientv3.Lease
	leaseID clientv3.LeaseID
	cancel  context.CancelFunc
	key     string
	timeout time.Duration
}

// NewEtcdLock 创建一个新的分布式锁
func NewEtcdLock(client *clientv3.Client, key string, timeout time.Duration) *EtcdLock {
	return &EtcdLock{
		client:  client,
		key:     key,
		timeout: timeout,
	}
}

// TryLock 尝试获取锁
func (l *EtcdLock) TryLock() error {
	// 创建租约
	lease := clientv3.NewLease(l.client)
	ctx, cancel := context.WithTimeout(context.Background(), l.timeout)
	defer cancel()

	// 设置租约时间，建议至少5秒以上
	leaseResp, err := lease.Grant(ctx, 5)
	if err != nil {
		return err
	}

	// 自动续约
	ctx, cancel = context.WithCancel(context.Background())
	keepAliveChan, err := lease.KeepAlive(ctx, leaseResp.ID)
	if err != nil {
		cancel()
		return err
	}

	<-keepAliveChan
	// 尝试加锁
	txn := l.client.Txn(context.Background())
	txnResp, err := txn.If(
		clientv3.Compare(clientv3.CreateRevision(l.key), "=", 0),
	).Then(
		clientv3.OpPut(l.key, "", clientv3.WithLease(leaseResp.ID)),
	).Else(
		clientv3.OpGet(l.key),
	).Commit()

	if err != nil {
		cancel()
		return err
	}

	if !txnResp.Succeeded {
		cancel()
		return fmt.Errorf("lock is already held by another client")
	}

	l.lease = lease
	l.leaseID = leaseResp.ID
	l.cancel = cancel

	return nil
}

// Unlock 释放锁
func (l *EtcdLock) Unlock() error {
	if l.cancel != nil {
		l.cancel()
	}

	if l.leaseID != 0 {
		_, err := l.lease.Revoke(context.Background(), l.leaseID)
		return err
	}

	return nil
}

// GlobalLock is a distributed global lock interface
// Only one client can hold the lock at any time in the cluster
//
type GlobalLock interface {
	Lock() error
	Unlock() error
}

var _ GlobalLock = (*EtcdLock)(nil)

// Lock blocks until the lock is acquired
func (l *EtcdLock) Lock() error {
	for {
		err := l.TryLock()
		if err == nil {
			return nil
		}
		// Optionally, you can check for context cancellation here
		time.Sleep(100 * time.Millisecond)
	}
}
