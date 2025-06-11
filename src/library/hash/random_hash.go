package hash

import (
	"VectorSphere/src/library/enum"
	"VectorSphere/src/library/log"
	"fmt"
	"math"
)

// RandomProjectionHash 随机投影哈希函数
type RandomProjectionHash struct {
	ProjectionVector []float64          `json:"projection_vector"`
	W                float64            `json:"w"`
	B                float64            `json:"b"`
	FamilyType       enum.LSHFamilyType `json:"family_type"`
}

// Hash 计算随机投影哈希值
func (rph *RandomProjectionHash) Hash(vector []float64) (uint64, error) {
	if len(vector) != len(rph.ProjectionVector) {
		log.Warning("Vector dimension mismatch: expected %d, got %d", len(rph.ProjectionVector), len(vector))
		return 0, fmt.Errorf("vector dimension mismatch")
	}

	// 计算点积
	dotProduct := 0.0
	for i := 0; i < len(vector); i++ {
		dotProduct += vector[i] * rph.ProjectionVector[i]
	}

	// LSH 哈希值计算: floor((a·v + b) / w)
	hashValue := math.Floor((dotProduct + rph.B) / rph.W)
	return uint64(math.Abs(hashValue)), nil
}

// GetType 获取哈希函数类型
func (rph *RandomProjectionHash) GetType() enum.LSHFamilyType {
	return rph.FamilyType
}

// GetParameters 获取哈希函数参数
func (rph *RandomProjectionHash) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"w":              rph.W,
		"b":              rph.B,
		"projection_dim": len(rph.ProjectionVector),
		"family_type":    rph.FamilyType,
	}
}
