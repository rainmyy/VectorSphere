package algorithm

import (
	"fmt"
	"math"

	"VectorSphere/src/library/enum"
)

// DistanceCalculator 定义了距离计算器接口
type DistanceCalculator interface {
	// Calculate 计算两个向量之间的距离
	Calculate(v1, v2 []float64) float64
	// CalculateSquared 计算两个向量之间的平方距离（避免开方运算）
	CalculateSquared(v1, v2 []float64) float64
	// Type 返回距离计算器的类型
	Type() string
}

// EuclideanDistanceCalculator 欧几里得距离计算器
type EuclideanDistanceCalculator struct {
	// UseSquared 是否使用平方距离（避免开方运算）
	UseSquared bool
}

// Calculate 计算欧几里得距离
func (e *EuclideanDistanceCalculator) Calculate(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return float64(^uint(0) >> 1) // 返回最大float64值
	}

	sum := 0.0
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}

	if e.UseSquared {
		return sum // 返回平方距离，避免开方运算
	}
	return math.Sqrt(sum)
}

// CalculateSquared 计算欧几里得距离的平方
func (e *EuclideanDistanceCalculator) CalculateSquared(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return float64(^uint(0) >> 1) // 返回最大float64值
	}

	sum := 0.0
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}

	return sum // 始终返回平方距离
}

// Type 返回距离计算器的类型
func (e *EuclideanDistanceCalculator) Type() string {
	if e.UseSquared {
		return "EuclideanSquared"
	}
	return "Euclidean"
}

// CosineDistanceCalculator 余弦距离计算器
type CosineDistanceCalculator struct {
	// ReturnSimilarity 是否返回相似度而非距离
	ReturnSimilarity bool
}

// Calculate 计算余弦距离/相似度
func (c *CosineDistanceCalculator) Calculate(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return -1 // 错误情况
	}

	// 计算点积
	dotProduct := 0.0
	for i := 0; i < len(v1); i++ {
		dotProduct += v1[i] * v2[i]
	}

	// 计算向量模长
	magnitudeA := 0.0
	magnitudeB := 0.0
	for i := 0; i < len(v1); i++ {
		magnitudeA += v1[i] * v1[i]
		magnitudeB += v2[i] * v2[i]
	}

	magnitudeA = math.Sqrt(magnitudeA)
	magnitudeB = math.Sqrt(magnitudeB)

	if magnitudeA == 0 || magnitudeB == 0 {
		return -1 // 避免除零错误
	}

	similarity := dotProduct / (magnitudeA * magnitudeB)

	if c.ReturnSimilarity {
		return similarity // 返回相似度（越大越相似）
	}
	return 1.0 - similarity // 返回距离（越小越相似）
}

// CalculateSquared 计算余弦距离/相似度的平方
// 注意：对于余弦距离，平方可能不是有意义的操作，但为了接口一致性实现此方法
func (c *CosineDistanceCalculator) CalculateSquared(v1, v2 []float64) float64 {
	dist := c.Calculate(v1, v2)
	if dist < 0 {
		return dist // 保留错误值
	}
	return dist * dist
}

// Type 返回距离计算器的类型
func (c *CosineDistanceCalculator) Type() string {
	if c.ReturnSimilarity {
		return "CosineSimilarity"
	}
	return "CosineDistance"
}

// ManhattanDistanceCalculator 曼哈顿距离计算器
type ManhattanDistanceCalculator struct{}

// Calculate 计算曼哈顿距离
func (m *ManhattanDistanceCalculator) Calculate(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return float64(^uint(0) >> 1) // 返回最大float64值
	}

	sum := 0.0
	for i := range v1 {
		sum += math.Abs(v1[i] - v2[i])
	}

	return sum
}

// CalculateSquared 计算曼哈顿距离的平方
func (m *ManhattanDistanceCalculator) CalculateSquared(v1, v2 []float64) float64 {
	dist := m.Calculate(v1, v2)
	return dist * dist
}

// Type 返回距离计算器的类型
func (m *ManhattanDistanceCalculator) Type() string {
	return "Manhattan"
}

// NewDistanceCalculator 创建指定类型的距离计算器
func NewDistanceCalculator(distanceType int) (DistanceCalculator, error) {
	switch distanceType {
	case enum.EuclideanDistance:
		return &EuclideanDistanceCalculator{UseSquared: false}, nil
	case enum.EuclideanDistanceSquared:
		return &EuclideanDistanceCalculator{UseSquared: true}, nil
	case enum.CosineDistance:
		return &CosineDistanceCalculator{ReturnSimilarity: false}, nil
	case enum.CosineSimilarity:
		return &CosineDistanceCalculator{ReturnSimilarity: true}, nil
	case enum.ManhattanDistance:
		return &ManhattanDistanceCalculator{}, nil
	default:
		return nil, fmt.Errorf("不支持的距离类型: %d", distanceType)
	}
}

// EuclideanDistance 计算欧几里得距离
func EuclideanDistance(v1, v2 []float64) (float64, error) {
	if len(v1) != len(v2) {
		return 0, fmt.Errorf("向量维度不匹配: %d != %d", len(v1), len(v2))
	}

	sum := 0.0
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum), nil
}
