package hash

import (
	"VectorSphere/src/library/enum"
	"VectorSphere/src/library/log"
	"fmt"
	"math"
)

// EuclideanHash 欧几里得哈希函数（p-stable LSH）
type EuclideanHash struct {
	RandomVector []float64          `json:"random_vector"`
	W            float64            `json:"w"`
	B            float64            `json:"b"`
	FamilyType   enum.LSHFamilyType `json:"family_type"`
}

// Hash 计算欧几里得哈希值
func (eh *EuclideanHash) Hash(vector []float64) (uint64, error) {
	if len(vector) != len(eh.RandomVector) {
		log.Warning("Vector dimension mismatch: expected %d, got %d", len(eh.RandomVector), len(vector))
		return 0, fmt.Errorf("vector dimension mismatch: expected %d, got %d", eh.RandomVector, len(vector))
	}

	// 计算点积
	dotProduct := 0.0
	for i := 0; i < len(vector); i++ {
		dotProduct += vector[i] * eh.RandomVector[i]
	}

	// p-stable LSH 哈希值计算: floor((a·v + b) / w)
	hashValue := math.Floor((dotProduct + eh.B) / eh.W)
	return uint64(math.Abs(hashValue)), nil
}

// GetType 获取哈希函数类型
func (eh *EuclideanHash) GetType() enum.LSHFamilyType {
	return eh.FamilyType
}

// GetParameters 获取哈希函数参数
func (eh *EuclideanHash) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"w":           eh.W,
		"b":           eh.B,
		"random_dim":  len(eh.RandomVector),
		"family_type": eh.FamilyType,
	}
}
