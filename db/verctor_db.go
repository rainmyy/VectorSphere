package db

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

type VectorDB struct {
	vectors map[string][]float64
	mu      sync.RWMutex
}

func NewVectorDB() *VectorDB {
	return &VectorDB{
		vectors: make(map[string][]float64),
	}
}

// Add 添加向量
func (db *VectorDB) Add(id string, vector []float64) {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.vectors[id] = vector
}

// Get 获取向量
func (db *VectorDB) Get(id string) ([]float64, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	vec, exists := db.vectors[id]
	return vec, exists
}

// 计算欧几里得距离
func euclideanDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector dimensions mismatch")
	}
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum), nil
}

// FindNearest 查找最相似的向量
func (db *VectorDB) FindNearest(query []float64, k int) ([]string, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	type result struct {
		id       string
		distance float64
	}

	var results []result

	for id, vec := range db.vectors {
		dist, err := euclideanDistance(query, vec)
		if err != nil {
			return nil, err
		}
		results = append(results, result{id, dist})
	}

	// 按距离排序
	sort.Slice(results, func(i, j int) bool {
		return results[i].distance < results[j].distance
	})

	// 返回前k个结果
	var ids []string
	for i := 0; i < k && i < len(results); i++ {
		ids = append(ids, results[i].id)
	}

	return ids, nil
}
