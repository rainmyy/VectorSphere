package distributed

import (
	"context"
	"testing"
	"time"

	"go.etcd.io/etcd/clientv3"
	"go.etcd.io/etcd/clientv3/concurrency"
)

// TestLeaseKeepAlive 测试租约自动续约功能
func TestLeaseKeepAlive(t *testing.T) {
	// 创建etcd客户端
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{"127.0.0.1:2379"},
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("Failed to create etcd client: %v", err)
	}
	defer client.Close()

	// 创建session
	session, err := concurrency.NewSession(client, concurrency.WithTTL(5)) // 5秒TTL
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	defer session.Close()

	leaseID := session.Lease()
	t.Logf("Created lease with ID: %x", leaseID)

	// 启动租约续约
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	keepAliveCh, err := client.KeepAlive(ctx, leaseID)
	if err != nil {
		t.Fatalf("Failed to start keep-alive: %v", err)
	}

	// 监听续约响应
	go func() {
		for ka := range keepAliveCh {
			if ka != nil {
				t.Logf("Keep-alive response: TTL=%d", ka.TTL)
			} else {
				t.Log("Keep-alive channel closed")
				return
			}
		}
	}()

	// 等待10秒，观察续约情况
	time.Sleep(10 * time.Second)

	// 检查租约是否仍然存在
	resp, err := client.TimeToLive(context.Background(), leaseID)
	if err != nil {
		t.Fatalf("Failed to check lease TTL: %v", err)
	}

	if resp.TTL <= 0 {
		t.Errorf("Lease expired, TTL: %d", resp.TTL)
	} else {
		t.Logf("Lease still active, TTL: %d", resp.TTL)
	}
}

// TestLeaseExpiration 测试租约过期处理
func TestLeaseExpiration(t *testing.T) {
	// 创建etcd客户端
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{"127.0.0.1:2379"},
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("Failed to create etcd client: %v", err)
	}
	defer client.Close()

	// 创建session
	session, err := concurrency.NewSession(client, concurrency.WithTTL(3)) // 3秒TTL
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	leaseID := session.Lease()
	t.Logf("Created lease with ID: %x", leaseID)

	// 不启动续约，让租约自然过期
	session.Close()

	// 等待租约过期
	time.Sleep(5 * time.Second)

	// 检查租约是否已过期
	resp, err := client.TimeToLive(context.Background(), leaseID)
	if err != nil {
		t.Fatalf("Failed to check lease TTL: %v", err)
	}

	if resp.TTL > 0 {
		t.Errorf("Lease should have expired, but TTL: %d", resp.TTL)
	} else {
		t.Logf("Lease expired as expected, TTL: %d", resp.TTL)
	}
}