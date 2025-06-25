package vector

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/algorithm"
	"fmt"
	"math"
	"sync"
)

// DataPreprocessor 数据预处理器接口
type DataPreprocessor interface {
	Normalize(vectors [][]float64) [][]float64
	ReduceDimension(vectors [][]float64, targetDim int) ([][]float64, error)
	Quantize(vectors [][]float64, bits int) ([][]float64, error)
	Filter(vectors [][]float64, criteria FilterCriteria) [][]float64
	FilterResults(results []entity.Result, options entity.SearchOptions) []entity.Result
	GetStats() PreprocessorStats
}

// FilterCriteria 过滤条件
type FilterCriteria struct {
	MinNorm     float64 // 最小向量范数
	MaxNorm     float64 // 最大向量范数
	Dimensions  []int   // 保留的维度索引
	RemoveZeros bool    // 是否移除零向量
}

// PreprocessorStats 预处理器统计信息
type PreprocessorStats struct {
	ProcessedVectors   int64
	NormalizedVectors  int64
	ReducedVectors     int64
	QuantizedVectors   int64
	FilteredVectors    int64
	AverageProcessTime float64 // 毫秒
}

// StandardPreprocessor 标准数据预处理器实现
type StandardPreprocessor struct {
	stats        PreprocessorStats
	mutex        sync.RWMutex
	pcaTransform *PCATransform
}

// PCATransform PCA变换器
type PCATransform struct {
	Components   [][]float64 // 主成分
	Mean         []float64   // 均值
	Variance     []float64   // 方差
	ExplainedVar []float64   // 解释方差比
	TargetDim    int         // 目标维度
	OriginalDim  int         // 原始维度
	Initialized  bool        // 是否已初始化
}

// NewStandardPreprocessor 创建新的标准预处理器
func NewStandardPreprocessor() *StandardPreprocessor {
	return &StandardPreprocessor{
		stats: PreprocessorStats{},
	}
}

// Normalize 向量归一化
func (sp *StandardPreprocessor) Normalize(vectors [][]float64) [][]float64 {
	sp.mutex.Lock()
	defer sp.mutex.Unlock()

	if len(vectors) == 0 {
		return vectors
	}

	normalized := make([][]float64, len(vectors))

	for i, vec := range vectors {
		normalized[i] = sp.normalizeVector(vec)
	}

	sp.stats.ProcessedVectors += int64(len(vectors))
	sp.stats.NormalizedVectors += int64(len(vectors))

	return normalized
}

// normalizeVector 归一化单个向量
func (sp *StandardPreprocessor) normalizeVector(vector []float64) []float64 {
	norm := 0.0
	for _, val := range vector {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return vector // 零向量保持不变
	}

	normalized := make([]float64, len(vector))
	for i, val := range vector {
		normalized[i] = val / norm
	}

	return normalized
}

// ReduceDimension 降维处理
func (sp *StandardPreprocessor) ReduceDimension(vectors [][]float64, targetDim int) ([][]float64, error) {
	sp.mutex.Lock()
	defer sp.mutex.Unlock()

	if len(vectors) == 0 {
		return vectors, nil
	}

	originalDim := len(vectors[0])
	if targetDim <= 0 {
		return nil, fmt.Errorf("目标维度 %d 必须大于0", targetDim)
	}
	if targetDim >= originalDim {
		return vectors, fmt.Errorf("目标维度 %d 必须小于原始维度 %d", targetDim, originalDim)
	}

	// 如果PCA变换器未初始化或维度不匹配，重新训练
	if sp.pcaTransform == nil || !sp.pcaTransform.Initialized ||
		sp.pcaTransform.OriginalDim != originalDim || sp.pcaTransform.TargetDim != targetDim {
		var err error
		sp.pcaTransform, err = sp.trainPCA(vectors, targetDim)
		if err != nil {
			return nil, fmt.Errorf("PCA训练失败: %v", err)
		}
	}

	// 应用PCA变换
	reduced := make([][]float64, len(vectors))
	for i, vec := range vectors {
		reduced[i] = sp.applyPCA(vec)
	}

	sp.stats.ProcessedVectors += int64(len(vectors))
	sp.stats.ReducedVectors += int64(len(vectors))

	return reduced, nil
}

// trainPCA 训练PCA变换器
func (sp *StandardPreprocessor) trainPCA(vectors [][]float64, targetDim int) (*PCATransform, error) {
	n := len(vectors)
	d := len(vectors[0])

	if n < targetDim {
		return nil, fmt.Errorf("样本数量 %d 必须大于等于目标维度 %d", n, targetDim)
	}

	// 计算均值
	mean := make([]float64, d)
	for _, vec := range vectors {
		for j, val := range vec {
			mean[j] += val
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}

	// 中心化数据
	centered := make([][]float64, n)
	for i, vec := range vectors {
		centered[i] = make([]float64, d)
		for j, val := range vec {
			centered[i][j] = val - mean[j]
		}
	}

	// 计算协方差矩阵
	cov := make([][]float64, d)
	for i := range cov {
		cov[i] = make([]float64, d)
		for j := range cov[i] {
			for k := 0; k < n; k++ {
				cov[i][j] += centered[k][i] * centered[k][j]
			}
			cov[i][j] /= float64(n - 1)
		}
	}

	// 简化的特征值分解（实际应用中应使用更高效的算法）
	eigenVectors, eigenValues := sp.simpleEigenDecomposition(cov, targetDim)

	// 计算解释方差比
	totalVar := 0.0
	for _, val := range eigenValues {
		totalVar += val
	}

	explainedVar := make([]float64, len(eigenValues))
	for i, val := range eigenValues {
		explainedVar[i] = val / totalVar
	}

	return &PCATransform{
		Components:   eigenVectors,
		Mean:         mean,
		Variance:     eigenValues,
		ExplainedVar: explainedVar,
		TargetDim:    targetDim,
		OriginalDim:  d,
		Initialized:  true,
	}, nil
}

// simpleEigenDecomposition 简化的特征值分解
func (sp *StandardPreprocessor) simpleEigenDecomposition(matrix [][]float64, k int) ([][]float64, []float64) {
	n := len(matrix)

	// 使用幂迭代法求前k个主成分（简化实现）
	eigenVectors := make([][]float64, k)
	eigenValues := make([]float64, k)

	for i := 0; i < k; i++ {
		// 初始化随机向量
		v := make([]float64, n)
		for j := range v {
			v[j] = math.Sin(float64(i*n + j)) // 伪随机初始化
		}

		// 幂迭代
		for iter := 0; iter < 100; iter++ {
			// Av
			newV := make([]float64, n)
			for j := 0; j < n; j++ {
				for l := 0; l < n; l++ {
					newV[j] += matrix[j][l] * v[l]
				}
			}

			// 正交化（Gram-Schmidt）
			for prevI := 0; prevI < i; prevI++ {
				dot := 0.0
				for j := 0; j < n; j++ {
					dot += newV[j] * eigenVectors[prevI][j]
				}
				for j := 0; j < n; j++ {
					newV[j] -= dot * eigenVectors[prevI][j]
				}
			}

			// 归一化
			norm := 0.0
			for _, val := range newV {
				norm += val * val
			}
			norm = math.Sqrt(norm)

			if norm > 0 {
				for j := range newV {
					newV[j] /= norm
				}
			}

			v = newV
		}

		eigenVectors[i] = v

		// 计算特征值
		lambda := 0.0
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < n; l++ {
				sum += matrix[j][l] * v[l]
			}
			lambda += v[j] * sum
		}
		eigenValues[i] = lambda

		// 从矩阵中减去当前主成分的贡献
		for j := 0; j < n; j++ {
			for l := 0; l < n; l++ {
				matrix[j][l] -= lambda * v[j] * v[l]
			}
		}
	}

	return eigenVectors, eigenValues
}

// applyPCA 应用PCA变换
func (sp *StandardPreprocessor) applyPCA(vector []float64) []float64 {
	if sp.pcaTransform == nil || !sp.pcaTransform.Initialized {
		return vector
	}

	// 中心化
	centered := make([]float64, len(vector))
	for i, val := range vector {
		centered[i] = val - sp.pcaTransform.Mean[i]
	}

	// 投影到主成分
	reduced := make([]float64, sp.pcaTransform.TargetDim)
	for i := 0; i < sp.pcaTransform.TargetDim; i++ {
		for j, val := range centered {
			reduced[i] += val * sp.pcaTransform.Components[i][j]
		}
	}

	return reduced
}

// Quantize 向量量化
func (sp *StandardPreprocessor) Quantize(vectors [][]float64, bits int) ([][]float64, error) {
	sp.mutex.Lock()
	defer sp.mutex.Unlock()

	if len(vectors) == 0 {
		return vectors, nil
	}

	if bits < 1 || bits > 32 {
		return nil, fmt.Errorf("量化位数必须在1-32之间")
	}

	levels := 1 << bits // 2^bits

	// 找到全局最小值和最大值
	minVal, maxVal := sp.findGlobalRange(vectors)

	if minVal == maxVal {
		return vectors, nil // 所有值相同，无需量化
	}

	quantized := make([][]float64, len(vectors))
	for i, vec := range vectors {
		quantized[i] = sp.quantizeVector(vec, minVal, maxVal, levels)
	}

	sp.stats.ProcessedVectors += int64(len(vectors))
	sp.stats.QuantizedVectors += int64(len(vectors))

	return quantized, nil
}

// findGlobalRange 找到全局数值范围
func (sp *StandardPreprocessor) findGlobalRange(vectors [][]float64) (float64, float64) {
	minVal := math.Inf(1)
	maxVal := math.Inf(-1)

	for _, vec := range vectors {
		for _, val := range vec {
			if val < minVal {
				minVal = val
			}
			if val > maxVal {
				maxVal = val
			}
		}
	}

	return minVal, maxVal
}

// quantizeVector 量化单个向量
func (sp *StandardPreprocessor) quantizeVector(vector []float64, minVal, maxVal float64, levels int) []float64 {
	quantized := make([]float64, len(vector))
	range_ := maxVal - minVal
	step := range_ / float64(levels-1)

	for i, val := range vector {
		// 将值映射到[0, levels-1]范围
		normalized := (val - minVal) / range_
		quantLevel := int(normalized*float64(levels-1) + 0.5) // 四舍五入

		// 限制在有效范围内
		if quantLevel < 0 {
			quantLevel = 0
		} else if quantLevel >= levels {
			quantLevel = levels - 1
		}

		// 反量化到原始范围
		quantized[i] = minVal + float64(quantLevel)*step
	}

	return quantized
}

// Filter 向量过滤
func (sp *StandardPreprocessor) Filter(vectors [][]float64, criteria FilterCriteria) [][]float64 {
	sp.mutex.Lock()
	defer sp.mutex.Unlock()

	if len(vectors) == 0 {
		return vectors
	}

	filtered := make([][]float64, 0, len(vectors))

	for _, vec := range vectors {
		if sp.passesFilter(vec, criteria) {
			filtered = append(filtered, vec)
		}
	}

	sp.stats.ProcessedVectors += int64(len(vectors))
	sp.stats.FilteredVectors += int64(len(vectors) - len(filtered))

	return filtered
}

// passesFilter 检查向量是否通过过滤条件
func (sp *StandardPreprocessor) passesFilter(vector []float64, criteria FilterCriteria) bool {
	// 检查零向量
	if criteria.RemoveZeros {
		isZero := true
		for _, val := range vector {
			if val != 0 {
				isZero = false
				break
			}
		}
		if isZero {
			return false
		}
	}

	// 检查向量范数
	if criteria.MinNorm > 0 || criteria.MaxNorm > 0 {
		norm := 0.0
		for _, val := range vector {
			norm += val * val
		}
		norm = math.Sqrt(norm)

		if criteria.MinNorm > 0 && norm < criteria.MinNorm {
			return false
		}
		if criteria.MaxNorm > 0 && norm > criteria.MaxNorm {
			return false
		}
	}

	return true
}

// GetStats 获取预处理器统计信息
func (sp *StandardPreprocessor) GetStats() PreprocessorStats {
	sp.mutex.RLock()
	defer sp.mutex.RUnlock()
	return sp.stats
}

// FilterResults 过滤搜索结果
func (sp *StandardPreprocessor) FilterResults(results []entity.Result, options entity.SearchOptions) []entity.Result {
	sp.mutex.Lock()
	defer sp.mutex.Unlock()

	if len(results) == 0 {
		return results
	}

	filtered := make([]entity.Result, 0, len(results))

	// 应用标量过滤器
	if options.ScalarFilters != nil && len(options.ScalarFilters) > 0 {
		for _, result := range results {
			if sp.passesScalarFilters(result, options.ScalarFilters) {
				filtered = append(filtered, result)
			}
		}
	} else {
		filtered = results
	}

	// 应用相似度阈值过滤
	if options.QualityLevel > 0 {
		minSimilarity := options.QualityLevel * 0.5 // 基于质量等级设置最小相似度
		finalFiltered := make([]entity.Result, 0, len(filtered))
		for _, result := range filtered {
			if result.Similarity >= minSimilarity {
				finalFiltered = append(finalFiltered, result)
			}
		}
		filtered = finalFiltered
	}

	sp.stats.ProcessedVectors += int64(len(results))
	sp.stats.FilteredVectors += int64(len(results) - len(filtered))

	return filtered
}

// passesScalarFilters 检查结果是否通过标量过滤条件
func (sp *StandardPreprocessor) passesScalarFilters(result entity.Result, filters map[string]interface{}) bool {
	// 这里可以根据实际需求实现标量过滤逻辑
	// 例如：根据result.Id查询元数据，然后应用过滤条件
	// 目前返回true，表示所有结果都通过过滤
	return true
}

// ProductQuantizer 乘积量化器
type ProductQuantizer struct {
	subVectors  int           // 子向量数量
	codebooks   [][][]float64 // 码本 [subvector][centroid][dimension]
	subDim      int           // 每个子向量的维度
	centroids   int           // 每个码本的聚类中心数量
	initialized bool
	mutex       sync.RWMutex
}

// NewProductQuantizer 创建新的乘积量化器
func NewProductQuantizer(subVectors, centroids int) *ProductQuantizer {
	return &ProductQuantizer{
		subVectors: subVectors,
		centroids:  centroids,
	}
}

// Train 训练乘积量化器
func (pq *ProductQuantizer) Train(vectors [][]float64) error {
	pq.mutex.Lock()
	defer pq.mutex.Unlock()

	if len(vectors) == 0 {
		return fmt.Errorf("训练数据为空")
	}

	dim := len(vectors[0])
	if dim%pq.subVectors != 0 {
		return fmt.Errorf("向量维度 %d 必须能被子向量数量 %d 整除", dim, pq.subVectors)
	}

	pq.subDim = dim / pq.subVectors
	pq.codebooks = make([][][]float64, pq.subVectors)

	// 为每个子向量训练码本
	for i := 0; i < pq.subVectors; i++ {
		start := i * pq.subDim
		end := start + pq.subDim

		// 提取子向量
		subVecs := make([][]float64, len(vectors))
		for j, vec := range vectors {
			subVecs[j] = vec[start:end]
		}

		// 使用K-means聚类训练码本
		codebook, err := pq.trainCodebook(subVecs, pq.centroids)
		if err != nil {
			return fmt.Errorf("训练第%d个码本失败: %v", i, err)
		}

		pq.codebooks[i] = codebook
	}

	pq.initialized = true
	return nil
}

// trainCodebook 训练单个码本
func (pq *ProductQuantizer) trainCodebook(vectors [][]float64, k int) ([][]float64, error) {
	if len(vectors) < k {
		return nil, fmt.Errorf("样本数量 %d 少于聚类数量 %d", len(vectors), k)
	}

	dim := len(vectors[0])
	centroids := make([][]float64, k)

	// 随机初始化聚类中心
	for i := 0; i < k; i++ {
		centroids[i] = make([]float64, dim)
		copy(centroids[i], vectors[i%len(vectors)])
	}

	// K-means迭代
	for iter := 0; iter < 100; iter++ {
		// 分配样本到最近的聚类中心
		assignments := make([]int, len(vectors))
		for i, vec := range vectors {
			minDist := math.Inf(1)
			for j, centroid := range centroids {
				dist := pq.euclideanDistance(vec, centroid)
				if dist < minDist {
					minDist = dist
					assignments[i] = j
				}
			}
		}

		// 更新聚类中心
		newCentroids := make([][]float64, k)
		counts := make([]int, k)

		for i := 0; i < k; i++ {
			newCentroids[i] = make([]float64, dim)
		}

		for i, vec := range vectors {
			cluster := assignments[i]
			counts[cluster]++
			for j, val := range vec {
				newCentroids[cluster][j] += val
			}
		}

		for i := 0; i < k; i++ {
			if counts[i] > 0 {
				for j := range newCentroids[i] {
					newCentroids[i][j] /= float64(counts[i])
				}
			}
		}

		centroids = newCentroids
	}

	return centroids, nil
}

// euclideanDistance 计算欧几里得距离
func (pq *ProductQuantizer) euclideanDistance(a, b []float64) float64 {
	// 尝试使用全局距离计算器
	if calculator, ok := getGlobalDistanceCalculator(); ok {
		return calculateDistanceWithCalculator(a, b, calculator)
	}
	
	// 回退到算法包中的EuclideanDistance函数
	dist, err := algorithm.EuclideanDistance(a, b)
	if err != nil {
		// 出错时使用备用实现
		dist := 0.0
		for i := range a {
			diff := a[i] - b[i]
			dist += diff * diff
		}
		return math.Sqrt(dist)
	}
	return dist
}

// Encode 编码向量
func (pq *ProductQuantizer) Encode(vector []float64) ([]int, error) {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()

	if !pq.initialized {
		return nil, fmt.Errorf("量化器未初始化")
	}

	if len(vector) != pq.subVectors*pq.subDim {
		return nil, fmt.Errorf("向量维度不匹配")
	}

	codes := make([]int, pq.subVectors)

	for i := 0; i < pq.subVectors; i++ {
		start := i * pq.subDim
		end := start + pq.subDim
		subVec := vector[start:end]

		// 找到最近的码字
		minDist := math.Inf(1)
		for j, codeword := range pq.codebooks[i] {
			dist := pq.euclideanDistance(subVec, codeword)
			if dist < minDist {
				minDist = dist
				codes[i] = j
			}
		}
	}

	return codes, nil
}

// Decode 解码向量
func (pq *ProductQuantizer) Decode(codes []int) ([]float64, error) {
	pq.mutex.RLock()
	defer pq.mutex.RUnlock()

	if !pq.initialized {
		return nil, fmt.Errorf("量化器未初始化")
	}

	if len(codes) != pq.subVectors {
		return nil, fmt.Errorf("编码长度不匹配")
	}

	vector := make([]float64, pq.subVectors*pq.subDim)

	for i, code := range codes {
		if code < 0 || code >= len(pq.codebooks[i]) {
			return nil, fmt.Errorf("无效的编码值: %d", code)
		}

		start := i * pq.subDim
		codeword := pq.codebooks[i][code]
		copy(vector[start:start+pq.subDim], codeword)
	}

	return vector, nil
}
