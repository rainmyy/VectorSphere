package hash

import (
	"VectorSphere/src/library/enum"
	"VectorSphere/src/library/logger"

	"fmt"
)

// AngularHash 角度哈希函数（用于余弦相似度）
type AngularHash struct {
	RandomVector []float64          `json:"random_vector"`
	FamilyType   enum.LSHFamilyType `json:"family_type"`
}

// Hash 计算角度哈希值
func (ah *AngularHash) Hash(vector []float64) (uint64, error) {
	if len(vector) != len(ah.RandomVector) {
		logger.Warning("Vector dimension mismatch: expected %d, got %d", len(ah.RandomVector), len(vector))
		return 0, fmt.Errorf("vector dimension mismatch: expected %d, got %d", len(ah.RandomVector), len(vector))
	}

	// 计算点积
	dotProduct := 0.0
	for i := 0; i < len(vector); i++ {
		dotProduct += vector[i] * ah.RandomVector[i]
	}

	// 角度哈希：如果点积 >= 0 返回 1，否则返回 0
	if dotProduct >= 0 {
		return 1, nil
	}

	return 0, fmt.Errorf("invalid hash function")
}

// GetType 获取哈希函数类型
func (ah *AngularHash) GetType() enum.LSHFamilyType {
	return ah.FamilyType
}

// GetParameters 获取哈希函数参数
func (ah *AngularHash) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"random_dim":  len(ah.RandomVector),
		"family_type": ah.FamilyType,
	}
}
