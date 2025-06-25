package vector

import (
	"context"
	"fmt"
	"sync"
	"time"

	"VectorSphere/src/library/entity"
)

// DistributedSearchManager 分布式搜索管理器
type DistributedSearchManager struct {
	nodes          []*SearchNode
	loadBalancer   *LoadBalancer
	shardStrategy  ShardStrategy
	consistentHash *DistributedConsistentHash
	mu             sync.RWMutex
}

// SearchNode 搜索节点
type SearchNode struct {
	ID       string
	Address  string
	Weight   int
	Active   bool
	Load     float64
	Latency  time.Duration
	Capacity int
	mu       sync.RWMutex
}

// LoadBalancer 负载均衡器
type LoadBalancer struct {
	strategy LoadBalanceStrategy
	nodes    []*SearchNode
	mu       sync.RWMutex
}

// LoadBalanceStrategy 负载均衡策略
type LoadBalanceStrategy string

const (
	RoundRobin     LoadBalanceStrategy = "round_robin"
	WeightedRandom LoadBalanceStrategy = "weighted_random"
	LeastLoad      LoadBalanceStrategy = "least_load"
	LatencyBased   LoadBalanceStrategy = "latency_based"
	ConsistentHash LoadBalanceStrategy = "consistent_hash"
)

// ShardStrategy 分片策略
type ShardStrategy interface {
	GetShard(vectorID int, totalShards int) int
	GetShardForQuery(query []float64, totalShards int) []int
}

// RangeShardStrategy 范围分片策略
type RangeShardStrategy struct{}

func (r *RangeShardStrategy) GetShard(vectorID int, totalShards int) int {
	return vectorID % totalShards
}

func (r *RangeShardStrategy) GetShardForQuery(query []float64, totalShards int) []int {
	// 查询需要搜索所有分片
	shards := make([]int, totalShards)
	for i := 0; i < totalShards; i++ {
		shards[i] = i
	}
	return shards
}

// HashShardStrategy 哈希分片策略
type HashShardStrategy struct{}

func (h *HashShardStrategy) GetShard(vectorID int, totalShards int) int {
	// 简单哈希分片
	return hash(vectorID) % totalShards
}

func (h *HashShardStrategy) GetShardForQuery(query []float64, totalShards int) []int {
	// 查询需要搜索所有分片
	shards := make([]int, totalShards)
	for i := 0; i < totalShards; i++ {
		shards[i] = i
	}
	return shards
}

// ClusterShardStrategy 聚类分片策略
type ClusterShardStrategy struct {
	clusterCenters [][]float64
}

func (c *ClusterShardStrategy) GetShard(vectorID int, totalShards int) int {
	// 基于向量所属聚类进行分片
	// 这里需要实际的聚类逻辑
	return vectorID % totalShards
}

func (c *ClusterShardStrategy) GetShardForQuery(query []float64, totalShards int) []int {
	// 基于查询向量找到最相关的几个聚类
	if len(c.clusterCenters) == 0 {
		// 如果没有聚类中心，搜索所有分片
		shards := make([]int, totalShards)
		for i := 0; i < totalShards; i++ {
			shards[i] = i
		}
		return shards
	}

	// 找到最近的几个聚类
	nearestClusters := c.findNearestClusters(query, min(3, totalShards))
	return nearestClusters
}

func (c *ClusterShardStrategy) findNearestClusters(query []float64, k int) []int {
	type clusterDistance struct {
		index    int
		distance float64
	}

	distances := make([]clusterDistance, len(c.clusterCenters))
	for i, center := range c.clusterCenters {
		dist := calculateEuclideanDistance(query, center)
		distances[i] = clusterDistance{index: i, distance: dist}
	}

	// 排序并返回前k个
	for i := 0; i < k && i < len(distances); i++ {
		minIdx := i
		for j := i + 1; j < len(distances); j++ {
			if distances[j].distance < distances[minIdx].distance {
				minIdx = j
			}
		}
		distances[i], distances[minIdx] = distances[minIdx], distances[i]
	}

	result := make([]int, min(k, len(distances)))
	for i := 0; i < len(result); i++ {
		result[i] = distances[i].index
	}
	return result
}

// DistributedConsistentHash 分布式一致性哈希
type DistributedConsistentHash struct {
	hashRing     map[uint32]*SearchNode
	sortedHashes []uint32
	replicas     int
	mu           sync.RWMutex
}

// NewDistributedConsistentHash 创建分布式一致性哈希
func NewDistributedConsistentHash(replicas int) *DistributedConsistentHash {
	return &DistributedConsistentHash{
		hashRing: make(map[uint32]*SearchNode),
		replicas: replicas,
	}
}

// AddNode 添加节点
func (ch *DistributedConsistentHash) AddNode(node *SearchNode) {
	ch.mu.Lock()
	defer ch.mu.Unlock()

	for i := 0; i < ch.replicas; i++ {
		hashValue := uint32(hash(fmt.Sprintf("%s:%d", node.ID, i)))
		ch.hashRing[hashValue] = node
		ch.sortedHashes = append(ch.sortedHashes, hashValue)
	}

	// 排序哈希值
	for i := 0; i < len(ch.sortedHashes); i++ {
		for j := i + 1; j < len(ch.sortedHashes); j++ {
			if ch.sortedHashes[i] > ch.sortedHashes[j] {
				ch.sortedHashes[i], ch.sortedHashes[j] = ch.sortedHashes[j], ch.sortedHashes[i]
			}
		}
	}
}

// GetNode 获取节点
func (ch *DistributedConsistentHash) GetNode(key string) *SearchNode {
	ch.mu.RLock()
	defer ch.mu.RUnlock()

	if len(ch.sortedHashes) == 0 {
		return nil
	}

	hashValue := uint32(hash(key))

	// 找到第一个大于等于hashValue的节点
	for _, h := range ch.sortedHashes {
		if h >= hashValue {
			return ch.hashRing[h]
		}
	}

	// 如果没找到，返回第一个节点（环形结构）
	return ch.hashRing[ch.sortedHashes[0]]
}

// NewDistributedSearchManager 创建分布式搜索管理器
func NewDistributedSearchManager(strategy ShardStrategy) *DistributedSearchManager {
	return &DistributedSearchManager{
		nodes:          make([]*SearchNode, 0),
		loadBalancer:   &LoadBalancer{strategy: LeastLoad},
		shardStrategy:  strategy,
		consistentHash: NewDistributedConsistentHash(3),
	}
}

// AddNode 添加搜索节点
func (dsm *DistributedSearchManager) AddNode(node *SearchNode) {
	dsm.mu.Lock()
	defer dsm.mu.Unlock()

	dsm.nodes = append(dsm.nodes, node)
	dsm.loadBalancer.nodes = append(dsm.loadBalancer.nodes, node)
	dsm.consistentHash.AddNode(node)
}

// DistributedSearch 分布式搜索
func (dsm *DistributedSearchManager) DistributedSearch(ctx context.Context, query []float64, k int, options entity.SearchOptions) ([]entity.Result, error) {
	if len(dsm.nodes) == 0 {
		return nil, fmt.Errorf("没有可用的搜索节点")
	}

	// 确定需要搜索的分片
	shards := dsm.shardStrategy.GetShardForQuery(query, len(dsm.nodes))

	// 并行搜索多个分片
	resultChan := make(chan []entity.Result, len(shards))
	errorChan := make(chan error, len(shards))

	var wg sync.WaitGroup
	for _, shardID := range shards {
		wg.Add(1)
		go func(shard int) {
			defer wg.Done()

			node := dsm.selectNodeForShard(shard)
			if node == nil {
				errorChan <- fmt.Errorf("分片 %d 没有可用节点", shard)
				return
			}

			// 执行搜索
			results, err := dsm.searchOnNode(ctx, node, query, k, options)
			if err != nil {
				errorChan <- err
				return
			}

			resultChan <- results
		}(shardID)
	}

	// 等待所有搜索完成
	go func() {
		wg.Wait()
		close(resultChan)
		close(errorChan)
	}()

	// 收集结果
	allResults := make([]entity.Result, 0)
	var errors []error

	for {
		select {
		case results, ok := <-resultChan:
			if !ok {
				resultChan = nil
			} else {
				allResults = append(allResults, results...)
			}
		case err, ok := <-errorChan:
			if !ok {
				errorChan = nil
			} else {
				errors = append(errors, err)
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}

		if resultChan == nil && errorChan == nil {
			break
		}
	}

	// 如果有错误但也有结果，记录错误但继续处理结果
	if len(errors) > 0 && len(allResults) == 0 {
		return nil, fmt.Errorf("所有分片搜索都失败了: %v", errors)
	}

	// 合并和排序结果
	finalResults := dsm.mergeAndRankResults(allResults, k)
	return finalResults, nil
}

// selectNodeForShard 为分片选择节点
func (dsm *DistributedSearchManager) selectNodeForShard(shardID int) *SearchNode {
	dsm.mu.RLock()
	defer dsm.mu.RUnlock()

	if shardID >= len(dsm.nodes) {
		return nil
	}

	return dsm.nodes[shardID]
}

// searchOnNode 在指定节点上执行搜索
func (dsm *DistributedSearchManager) searchOnNode(ctx context.Context, node *SearchNode, query []float64, k int, options entity.SearchOptions) ([]entity.Result, error) {
	// 这里应该实现实际的网络调用逻辑
	// 目前返回空结果作为占位符
	return []entity.Result{}, nil
}

// mergeAndRankResults 合并和排序结果
func (dsm *DistributedSearchManager) mergeAndRankResults(allResults []entity.Result, k int) []entity.Result {
	if len(allResults) == 0 {
		return []entity.Result{}
	}

	// 去重
	seen := make(map[string]bool)
	uniqueResults := make([]entity.Result, 0)
	for _, result := range allResults {
		if !seen[result.Id] {
			seen[result.Id] = true
			uniqueResults = append(uniqueResults, result)
		}
	}

	// 按距离排序
	for i := 0; i < len(uniqueResults); i++ {
		for j := i + 1; j < len(uniqueResults); j++ {
			if uniqueResults[i].Distance > uniqueResults[j].Distance {
				uniqueResults[i], uniqueResults[j] = uniqueResults[j], uniqueResults[i]
			}
		}
	}

	// 返回前k个结果
	if len(uniqueResults) > k {
		return uniqueResults[:k]
	}
	return uniqueResults
}

// UpdateNodeLoad 更新节点负载
func (dsm *DistributedSearchManager) UpdateNodeLoad(nodeID string, load float64, latency time.Duration) {
	dsm.mu.RLock()
	defer dsm.mu.RUnlock()

	for _, node := range dsm.nodes {
		if node.ID == nodeID {
			node.mu.Lock()
			node.Load = load
			node.Latency = latency
			node.mu.Unlock()
			break
		}
	}
}

// GetHealthyNodes 获取健康的节点
func (dsm *DistributedSearchManager) GetHealthyNodes() []*SearchNode {
	dsm.mu.RLock()
	defer dsm.mu.RUnlock()

	healthyNodes := make([]*SearchNode, 0)
	for _, node := range dsm.nodes {
		node.mu.RLock()
		if node.Active && node.Load < 0.8 { // 负载小于80%认为是健康的
			healthyNodes = append(healthyNodes, node)
		}
		node.mu.RUnlock()
	}

	return healthyNodes
}

// hash 简单哈希函数
func hash(key interface{}) int {
	switch v := key.(type) {
	case string:
		h := 0
		for _, c := range v {
			h = 31*h + int(c)
		}
		return h
	case int:
		return v
	default:
		return 0
	}
}

// calculateEuclideanDistance 计算欧几里得距离
func calculateEuclideanDistance(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return float64(^uint(0) >> 1) // 返回最大float64值
	}

	sum := 0.0
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}
	return sum // 返回平方距离，避免开方运算
}
