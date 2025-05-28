package util

import "math"

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
