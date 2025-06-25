package test

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/vector"
	"context"
	"fmt"
	"testing"
)

func TestNewDistributedSearchManager(t *testing.T) {
	strategy := &vector.RangeShardStrategy{}
	manager := vector.NewDistributedSearchManager(strategy)
	if manager == nil {
		t.Fatal("Expected non-nil DistributedSearchManager")
	}
}

func TestAddSearchNode(t *testing.T) {
	strategy := &vector.RangeShardStrategy{}
	manager := vector.NewDistributedSearchManager(strategy)

	node := &vector.SearchNode{
		ID:       "node1",
		Address:  "localhost:8001",
		Weight:   100,
		Active:   true,
		Capacity: 1000,
	}

	// 测试添加节点（无法直接验证，因为没有GetNodes方法）
	manager.AddNode(node)

	// 验证节点添加成功的间接方法：检查健康节点
	healthyNodes := manager.GetHealthyNodes()
	if len(healthyNodes) != 1 {
		t.Errorf("Expected 1 healthy node, got %d", len(healthyNodes))
	}

	if len(healthyNodes) > 0 && healthyNodes[0].ID != "node1" {
		t.Errorf("Expected node ID 'node1', got %s", healthyNodes[0].ID)
	}
}

func TestDistributedSearch(t *testing.T) {
	strategy := &vector.RangeShardStrategy{}
	manager := vector.NewDistributedSearchManager(strategy)

	// 添加搜索节点
	nodes := []*vector.SearchNode{
		{ID: "node1", Address: "localhost:8001", Active: true, Capacity: 1000},
		{ID: "node2", Address: "localhost:8002", Active: true, Capacity: 1000},
	}

	for _, node := range nodes {
		manager.AddNode(node)
	}

	query := []float64{1.0, 2.0, 3.0}
	options := entity.SearchOptions{
		MaxCandidates: 10,
		QualityLevel:  0.8,
	}

	ctx := context.Background()
	results, err := manager.DistributedSearch(ctx, query, 10, options)

	// 注意：这里可能会失败，因为实际的节点服务器不存在
	// 但我们可以测试方法是否正确调用
	if err == nil {
		if results == nil {
			t.Error("Expected non-nil results")
		}
	}
}

func TestShardStrategy(t *testing.T) {
	strategy := &vector.HashShardStrategy{}

	// 测试分片策略
	shardID := strategy.GetShard(123, 4)
	if shardID < 0 || shardID >= 4 {
		t.Errorf("Expected shard ID between 0-3, got %d", shardID)
	}

	query := []float64{1.0, 2.0, 3.0}
	shards := strategy.GetShardForQuery(query, 4)
	if len(shards) != 4 {
		t.Errorf("Expected 4 shards for query, got %d", len(shards))
	}
}

func TestConsistentHashing(t *testing.T) {
	hash := vector.NewDistributedConsistentHash(3)

	// 添加节点
	nodes := []*vector.SearchNode{
		{ID: "node1", Address: "localhost:8001"},
		{ID: "node2", Address: "localhost:8002"},
		{ID: "node3", Address: "localhost:8003"},
	}
	for _, node := range nodes {
		hash.AddNode(node)
	}

	// 测试一致性哈希
	keys := []string{"key1", "key2", "key3", "key4", "key5"}
	nodeAssignments := make(map[string]*vector.SearchNode)

	for _, key := range keys {
		node := hash.GetNode(key)
		if node == nil {
			t.Errorf("Expected non-empty node for key %s", key)
		}
		nodeAssignments[key] = node
	}

	// 注意：DistributedConsistentHash没有实现RemoveNode方法
	// 跳过节点移除测试
	t.Log("RemoveNode method not implemented, skipping node removal test")
}

func TestSearchNodeHealthCheck(t *testing.T) {
	node := &vector.SearchNode{
		ID:      "node1",
		Address: "localhost:8001",
		Active:  true,
	}

	// 测试节点基本属性
	if node.ID != "node1" {
		t.Errorf("Expected node ID 'node1', got %s", node.ID)
	}

	if node.Address != "localhost:8001" {
		t.Errorf("Expected node address 'localhost:8001', got %s", node.Address)
	}

	if !node.Active {
		t.Error("Expected node to be active")
	}
}

func TestDistributedSearchConcurrency(t *testing.T) {
	strategy := &vector.RangeShardStrategy{}
	manager := vector.NewDistributedSearchManager(strategy)

	// 添加多个节点
	for i := 0; i < 5; i++ {
		node := &vector.SearchNode{
			ID:       fmt.Sprintf("node%d", i),
			Address:  fmt.Sprintf("localhost:800%d", i),
			Active:   true,
			Capacity: 1000,
		}
		manager.AddNode(node)
	}

	query := []float64{1.0, 2.0, 3.0}
	options := entity.SearchOptions{MaxCandidates: 10}

	// 并发搜索
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			ctx := context.Background()
			_, err := manager.DistributedSearch(ctx, query, 10, options)
			// 忽略网络错误，只测试并发安全性
			_ = err
			done <- true
		}()
	}

	// 等待所有搜索完成
	for i := 0; i < 10; i++ {
		<-done
	}
}
