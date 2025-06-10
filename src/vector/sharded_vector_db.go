package vector

import (
	"hash/fnv"
	"sync"
)

// ShardedVectorDB 分片锁结构
type ShardedVectorDB struct {
	shards    []*VectorShard
	numShards int
}

type VectorShard struct {
	vectors map[string][]float64
	mu      sync.RWMutex
}

// 根据ID确定分片
func (db *ShardedVectorDB) getShardForID(id string) (*VectorShard, error) {
	h := fnv.New32a()
	_, err := h.Write([]byte(id))
	if err != nil {
		return nil, err
	}
	shardIndex := int(h.Sum32()) % db.numShards
	return db.shards[shardIndex], nil
}

// Get 分片查询实现
func (db *ShardedVectorDB) Get(id string) ([]float64, bool) {
	shard, err := db.getShardForID(id)
	if err != nil {
		return nil, false
	}

	shard.mu.RLock()
	defer shard.mu.RUnlock()
	vec, exists := shard.vectors[id]
	return vec, exists
}
