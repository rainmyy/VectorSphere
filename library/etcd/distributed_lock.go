package etcd

import (
	"context"
	"fmt"
	clientv3 "go.etcd.io/etcd/client/v3"
	"time"
)

type DistributedLock struct {
	client     *clientv3.Client
	key        string
	leaseID    clientv3.LeaseID
	cancelFunc context.CancelFunc
	timeout    time.Duration
}

// NewDistributedLock 创建一个新的分布式锁
func NewDistributedLock(client *clientv3.Client, key string, timeout time.Duration) *DistributedLock {
	return &DistributedLock{
		client:  client,
		key:     key,
		timeout: timeout,
	}
}

// Lock 尝试获取锁，带重试机制
func (dl *DistributedLock) Lock() error {
	var err error
	var leaseResp *clientv3.LeaseGrantResponse

	// 创建租约
	lease := clientv3.NewLease(dl.client)

	// 尝试获取锁，最多重试3次
	for i := 0; i < 3; i++ {
		// 创建带超时的context
		ctx, cancel := context.WithTimeout(context.Background(), dl.timeout)
		defer cancel()

		// 创建租约
		leaseResp, err = lease.Grant(ctx, 10) // 10秒TTL
		if err != nil {
			time.Sleep(time.Duration(i+1) * 500 * time.Millisecond) // 指数退避
			continue
		}

		// 尝试加锁
		txn := dl.client.Txn(ctx)
		txnResp, err := txn.If(
			clientv3.Compare(clientv3.CreateRevision(dl.key), "=", 0),
		).Then(
			clientv3.OpPut(dl.key, "", clientv3.WithLease(leaseResp.ID)),
		).Else(
			clientv3.OpGet(dl.key),
		).Commit()

		if err != nil {
			lease.Revoke(context.Background(), leaseResp.ID)
			time.Sleep(time.Duration(i+1) * 500 * time.Millisecond)
			continue
		}

		if !txnResp.Succeeded {
			lease.Revoke(context.Background(), leaseResp.ID)
			return fmt.Errorf("lock is already held by another client")
		}

		// 设置自动续约
		keepAliveCtx, cancelFunc := context.WithCancel(context.Background())
		_, err = lease.KeepAlive(keepAliveCtx, leaseResp.ID)
		if err != nil {
			cancelFunc()
			lease.Revoke(context.Background(), leaseResp.ID)
			return err
		}

		dl.leaseID = leaseResp.ID
		dl.cancelFunc = cancelFunc
		return nil
	}

	return fmt.Errorf("failed to acquire lock after 3 attempts: %v", err)
}

// Unlock 释放锁
func (dl *DistributedLock) Unlock() error {
	if dl.cancelFunc != nil {
		dl.cancelFunc()
	}

	if dl.leaseID != 0 {
		lease := clientv3.NewLease(dl.client)
		_, err := lease.Revoke(context.Background(), dl.leaseID)
		return err
	}

	return nil
}
