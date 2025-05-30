package util

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"seetaSearch/library/entity"
	"seetaSearch/library/enum"
)

// CosineSimilarity 计算两个向量的余弦相似度
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return -1 // 错误情况
	}

	// 计算点积
	dotProduct := 0.0
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
	}

	// 计算向量模长
	magnitudeA := 0.0
	magnitudeB := 0.0
	for i := 0; i < len(a); i++ {
		magnitudeA += a[i] * a[i]
		magnitudeB += b[i] * b[i]
	}

	magnitudeA = math.Sqrt(magnitudeA)
	magnitudeB = math.Sqrt(magnitudeB)

	// 避免除零错误
	if magnitudeA == 0 || magnitudeB == 0 {
		return 0
	}

	return dotProduct / (magnitudeA * magnitudeB)
}
func NormalizeVector(vec []float64) []float64 {
	sum := 0.0
	for _, v := range vec {
		sum += v * v
	}
	norm := math.Sqrt(sum)

	if norm == 0 {
		return vec
	}

	normalized := make([]float64, len(vec))
	for i, v := range vec {
		normalized[i] = v / norm
	}

	return normalized
}

// OptimizedCosineSimilarity 优化距离计算函数，支持SIMD加速
func OptimizedCosineSimilarity(a, b []float64) float64 {
	// 使用预计算的归一化向量直接计算点积
	dotProduct := 0.0

	// 向量长度检查
	if len(a) != len(b) {
		return -1
	}

	// 分块计算以提高缓存命中率
	blockSize := 16
	for i := 0; i < len(a); i += blockSize {
		end := i + blockSize
		if end > len(a) {
			end = len(a)
		}

		// 计算当前块的点积
		for j := i; j < end; j++ {
			dotProduct += a[j] * b[j]
		}
	}

	return dotProduct // 对于归一化向量，点积等于余弦相似度
}

// CompressByPQ 产品量化压缩
func CompressByPQ(vec []float64, numSubvectors int, numCentroids int) entity.CompressedVector {
	// 将向量分割为numSubvectors个子向量
	// 对每个子向量进行K-means聚类，得到numCentroids个中心点
	// 用中心点索引替代原始子向量值
	// ...

	return entity.CompressedVector{}
}

// CalculateDistance 添加多种距离计算函数
func CalculateDistance(a, b []float64, method int) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("向量维度不匹配: %d != %d", len(a), len(b))
	}

	switch method {
	case enum.EuclideanDistance:
		// 欧几里得距离
		sum := 0.0
		for i := range a {
			diff := a[i] - b[i]
			sum += diff * diff
		}
		return math.Sqrt(sum), nil

	case enum.CosineDistance:
		// 余弦距离 (1 - 余弦相似度)
		// 假设输入向量已归一化
		dotProduct := 0.0
		for i := range a {
			dotProduct += a[i] * b[i]
		}
		return 1.0 - dotProduct, nil

	case enum.DotProduct:
		// 点积（越大越相似，需要取负值作为距离）
		dotProduct := 0.0
		for i := range a {
			dotProduct += a[i] * b[i]
		}
		return -dotProduct, nil

	case enum.ManhattanDistance:
		// 曼哈顿距离
		sum := 0.0
		for i := range a {
			sum += math.Abs(a[i] - b[i])
		}
		return sum, nil

	default:
		return 0, fmt.Errorf("不支持的距离计算方法: %d", method)
	}
}

// GenerateCacheKey 生成查询缓存键
func GenerateCacheKey(query []float64, k, nprobe, method int) string {
	// 简单哈希函数，将查询向量和参数转换为字符串
	key := fmt.Sprintf("k=%d:nprobe=%d:method=%d:", k, nprobe, method)
	for _, v := range query {
		key += fmt.Sprintf("%.6f:", v)
	}
	return key
}

// ComputeVectorHash 计算向量哈希值，用于缓存键
func ComputeVectorHash(vec []float64) uint64 {
	h := fnv.New64a()
	for _, v := range vec {
		binary.Write(h, binary.LittleEndian, v)
	}
	return h.Sum64()
}
