package main

import (
	"context"
	"fmt"
	"log"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
)

func main() {
	fmt.Println("Testing etcd lease auto-renewal...")

	// 创建etcd客户端
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{"127.0.0.1:2379"},
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		log.Fatalf("Failed to create etcd client: %v", err)
	}
	defer client.Close()

	// 创建session，TTL为5秒
	session, err := concurrency.NewSession(client, concurrency.WithTTL(5))
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	defer session.Close()

	leaseID := session.Lease()
	fmt.Printf("Created lease with ID: %x\n", leaseID)

	// 启动租约续约
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	keepAliveCh, err := client.KeepAlive(ctx, leaseID)
	if err != nil {
		log.Fatalf("Failed to start keep-alive: %v", err)
	}

	fmt.Println("Starting lease keep-alive...")

	// 监听续约响应
	go func() {
		for ka := range keepAliveCh {
			if ka != nil {
				fmt.Printf("[%s] Keep-alive response: TTL=%d\n", time.Now().Format("15:04:05"), ka.TTL)
			} else {
				fmt.Println("Keep-alive channel closed")
				return
			}
		}
	}()

	// 每3秒检查一次租约状态
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	for i := 0; i < 10; i++ {
		select {
		case <-ticker.C:
			// 检查租约是否仍然存在
			resp, err := client.Lease.TimeToLive(context.Background(), leaseID)
			if err != nil {
				fmt.Printf("[%s] Failed to check lease TTL: %v\n", time.Now().Format("15:04:05"), err)
				continue
			}

			if resp.TTL <= 0 {
				fmt.Printf("[%s] Lease expired, TTL: %d\n", time.Now().Format("15:04:05"), resp.TTL)
				return
			} else {
				fmt.Printf("[%s] Lease still active, TTL: %d\n", time.Now().Format("15:04:05"), resp.TTL)
			}
		}
	}

	fmt.Println("Test completed successfully - lease auto-renewal is working!")
}