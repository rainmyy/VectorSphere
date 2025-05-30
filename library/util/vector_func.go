package util

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"seetaSearch/library/algorithm"
	"seetaSearch/library/entity"
	"seetaSearch/library/enum"
	"seetaSearch/library/log"
	"time"
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

// TrainingDataSource 定义了获取训练向量的接口
type TrainingDataSource interface {
	GetTrainingVectors(sampleRate float64, maxVectors int) ([][]float64, error)
	GetVectorDimension() (int, error)
}

// SavePQCodebookToFile 将 PQ 码本保存到文件
func SavePQCodebookToFile(codebook [][]algorithm.Point, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("创建码本文件失败 %s: %w", filePath, err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	// 写入码本中 float64 值的总数
	if err := binary.Write(writer, binary.LittleEndian, int32(len(codebook))); err != nil {
		return fmt.Errorf("写入码本大小失败: %w", err)
	}

	// 写入每个 float64 值
	for _, val := range codebook {
		if err := binary.Write(writer, binary.LittleEndian, val); err != nil {
			return fmt.Errorf("写入码本数据失败: %w", err)
		}
	}

	return writer.Flush()
}

// LoadPQCodebookFromFile 从文件加载 PQ 码本
func LoadPQCodebookFromFile(filePath string) ([][]algorithm.Point, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("打开码本文件失败 %s: %w", filePath, err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)

	// 读取码本中 float64 值的总数
	var size int32
	if err := binary.Read(reader, binary.LittleEndian, &size); err != nil {
		return nil, fmt.Errorf("读取码本大小失败: %w", err)
	}

	if size < 0 {
		return nil, fmt.Errorf("无效的码本大小: %d", size)
	}

	codebook := make([][]algorithm.Point, size)
	// 读取每个 float64 值
	for i := 0; i < int(size); i++ {
		if err := binary.Read(reader, binary.LittleEndian, &codebook[i]); err != nil {
			return nil, fmt.Errorf("读取码本数据失败 (索引 %d): %w", i, err)
		}
	}

	return codebook, nil
}

// TrainPQCodebook 根据训练数据生成产品量化的码本
// dataSource: 提供训练向量的数据源
// numSubvectors: 子向量的数量 (M)
// numCentroidsPerSubvector: 每个子向量空间的质心数量 (K*), 通常为 256
// maxKMeansIterations: K-Means 算法的最大迭代次数
// kMeansTolerance: K-Means 算法的收敛容忍度
// sampleRateForSubspaceTraining: 从 dataSource 获取的向量中，为每个子空间码本训练实际使用的采样率 (0.0 to 1.0)
// maxVectorsForTraining: 从 dataSource 获取用于训练的最大向量数量 (0 表示无限制，但受 sampleRateForSubspaceTraining 影响)
// codebookFilePath: 训练完成后码本的保存路径
func TrainPQCodebook(
	datasource TrainingDataSource,
	numSubvectors int,
	numCentroidsPerSubvector int,
	maxKMeansIterations int,
	kMeansTolerance float64,
	sampleRateForSubspaceTraining float64, // 注意：这个采样率是针对从数据源获取到的向量，再为每个子空间训练采样
	maxVectorsForTraining int,
	codebookFilePath string,
) error {
	if datasource == nil {
		return fmt.Errorf("训练数据源不能为空")
	}
	if numSubvectors <= 0 {
		return fmt.Errorf("子向量数量必须为正")
	}
	firstVectorDim, err := datasource.GetVectorDimension()
	if err != nil {
		return fmt.Errorf("获取向量维度失败: %w", err)
	}
	if firstVectorDim == 0 {
		return fmt.Errorf("训练向量维度不能为0")
	}
	if firstVectorDim%numSubvectors != 0 {
		return fmt.Errorf("向量维度 %d 不能被子向量数量 %d 整除", firstVectorDim, numSubvectors)
	}
	if numCentroidsPerSubvector <= 0 {
		return fmt.Errorf("每个子向量的质心数量必须为正")
	}
	if numCentroidsPerSubvector > 255 {
		return fmt.Errorf("每个子空间的质心数量 %d 超过了byte的最大表示 (255)", numCentroidsPerSubvector)
	}
	if sampleRateForSubspaceTraining <= 0 || sampleRateForSubspaceTraining > 1.0 {
		return fmt.Errorf("子空间训练采样率必须在 (0.0, 1.0] 之间")
	}
	if codebookFilePath == "" {
		return fmt.Errorf("码本保存路径不能为空")
	}

	// 从数据源获取训练向量
	// 注意：这里的 sampleRate 传递给 GetTrainingVectors 是为了初步筛选，
	// 实际用于子空间训练的采样是 sampleRateForSubspaceTraining
	trainingVectors, err := datasource.GetTrainingVectors(1.0, maxVectorsForTraining) // 先获取一批数据
	if err != nil {
		return fmt.Errorf("从数据源获取训练向量失败: %w", err)
	}
	if len(trainingVectors) == 0 {
		return fmt.Errorf("从数据源获取的训练向量集合为空")
	}
	fmt.Printf("从数据源获取了 %d 个向量用于PQ码本训练。\n", len(trainingVectors))

	subvectorDim := firstVectorDim / numSubvectors
	allSubspaceCodebooks := make([][]algorithm.Point, numSubvectors)

	rand.Seed(time.Now().UnixNano())

	fmt.Printf("开始训练PQ码本: %d 个子向量, 每个子空间 %d 个质心。保存路径: %s\n", numSubvectors, numCentroidsPerSubvector, codebookFilePath)

	for i := 0; i < numSubvectors; i++ {
		fmt.Printf("  训练子空间 %d/%d 的码本...\n", i+1, numSubvectors)
		subspaceTrainingPoints := make([]algorithm.Point, 0)

		// 对从数据源获取的 trainingVectors 进行采样，用于当前子空间的训练
		numSamplesForSubspace := int(float64(len(trainingVectors)) * sampleRateForSubspaceTraining)
		if numSamplesForSubspace < numCentroidsPerSubvector && len(trainingVectors) >= numCentroidsPerSubvector {
			fmt.Printf("    警告: 子空间训练采样数量 (%d) 小于质心数量 (%d)，将使用 %d 个向量进行当前子空间训练。\n", numSamplesForSubspace, numCentroidsPerSubvector, numCentroidsPerSubvector)
			numSamplesForSubspace = numCentroidsPerSubvector
		} else if len(trainingVectors) < numCentroidsPerSubvector {
			return fmt.Errorf("用于子空间 %d 训练的向量数 (%d) 少于质心数 (%d)", i, len(trainingVectors), numCentroidsPerSubvector)
		}

		sampledIndices := make([]int, len(trainingVectors))
		for idx := range sampledIndices {
			sampledIndices[idx] = idx
		}
		if sampleRateForSubspaceTraining < 1.0 && numSamplesForSubspace < len(trainingVectors) {
			rand.Shuffle(len(sampledIndices), func(k, l int) { sampledIndices[k], sampledIndices[l] = sampledIndices[l], sampledIndices[k] })
			sampledIndices = sampledIndices[:numSamplesForSubspace]
		} else {
			// 如果采样率是1.0，或者样本数不足，就用所有获取到的向量
			// 或者如果 numSamplesForSubspace >= len(trainingVectors)，也用所有获取到的向量
			sampledIndices = sampledIndices[:len(trainingVectors)]
		}

		for _, originalVecIdx := range sampledIndices {
			originalVec := trainingVectors[originalVecIdx]
			if len(originalVec) != firstVectorDim {
				return fmt.Errorf("训练集中向量维度不一致: 期望 %d, 实际 %d (向量索引 %d)", firstVectorDim, len(originalVec), originalVecIdx)
			}
			subVec := make(algorithm.Point, subvectorDim)
			copy(subVec, originalVec[i*subvectorDim:(i+1)*subvectorDim])
			subspaceTrainingPoints = append(subspaceTrainingPoints, subVec)
		}

		if len(subspaceTrainingPoints) < numCentroidsPerSubvector {
			return fmt.Errorf("子空间 %d 的实际训练点数量 (%d) 少于质心数量 (%d)，无法进行K-Means", i, len(subspaceTrainingPoints), numCentroidsPerSubvector)
		}

		fmt.Printf("    对 %d 个子向量进行K-Means聚类...\n", len(subspaceTrainingPoints))
		centroids, _, err := algorithm.KMeans(subspaceTrainingPoints, numCentroidsPerSubvector, maxKMeansIterations, kMeansTolerance)
		if err != nil {
			return fmt.Errorf("子空间 %d K-Means 聚类失败: %w", i, err)
		}

		allSubspaceCodebooks[i] = centroids // 存储当前子空间的质心列表
		fmt.Printf("    子空间 %d 码本训练完成，获得 %d 个质心。\n", i+1, len(centroids))
	}

	fmt.Printf("PQ码本训练完成。正在保存到 %s ...\n", codebookFilePath)
	err = SavePQCodebookToFile(allSubspaceCodebooks, codebookFilePath)
	if err != nil {
		return fmt.Errorf("保存PQ码本到文件 %s 失败: %w", codebookFilePath, err)
	}
	fmt.Printf("PQ码本已成功保存到 %s。\n", codebookFilePath)
	return nil
}

// CompressByPQ 产品量化压缩
// vec: 原始向量 []float64
// loadedCodebook: 从文件预加载的码本 [][]algorithm.Point
// numSubvectors: 子向量的数量 (M) - 应与加载的码本结构一致
// numCentroidsPerSubvector: 每个子向量空间的质心数量 (K*) - 应与加载的码本结构一致
func CompressByPQ(vec []float64, loadedCodebook [][]algorithm.Point, numSubvectors int, numCentroidsPerSubvector int) (entity.CompressedVector, error) {
	// 1. 参数校验
	if len(vec) == 0 {
		return entity.CompressedVector{}, fmt.Errorf("输入向量不能为空")
	}
	if loadedCodebook == nil || len(loadedCodebook) == 0 {
		return entity.CompressedVector{}, fmt.Errorf("提供的预加载码本不能为空")
	}
	if numSubvectors <= 0 {
		return entity.CompressedVector{}, fmt.Errorf("子向量数量必须为正")
	}
	if len(loadedCodebook) != numSubvectors {
		return entity.CompressedVector{}, fmt.Errorf("加载的码本中的子空间数量 %d 与指定的子向量数量 %d 不匹配", len(loadedCodebook), numSubvectors)
	}
	if len(vec)%numSubvectors != 0 {
		return entity.CompressedVector{}, fmt.Errorf("向量维度 %d 不能被子向量数量 %d 整除", len(vec), numSubvectors)
	}

	subvectorDim := len(vec) / numSubvectors
	compressedData := make([]byte, numSubvectors) // 每个子向量用一个 byte (0-255) 的索引表示质心

	for i := 0; i < numSubvectors; i++ {
		// 获取当前子向量
		subVecData := vec[i*subvectorDim : (i+1)*subvectorDim]
		subVecPoint := algorithm.Point(subVecData)

		// 获取当前子空间的码本
		currentSubspaceCodebook := loadedCodebook[i]
		if len(currentSubspaceCodebook) == 0 {
			return entity.CompressedVector{}, fmt.Errorf("子空间 %d 的码本为空", i)
		}
		// 校验 numCentroidsPerSubvector (如果提供)
		if numCentroidsPerSubvector > 0 && len(currentSubspaceCodebook) != numCentroidsPerSubvector {
			log.Warning("子空间 %d 的码本中的质心数量 %d 与期望的 %d 不符。将使用码本中的实际数量。", i, len(currentSubspaceCodebook), numCentroidsPerSubvector)
			// 不再是致命错误，但需要记录。实际压缩会基于码本的真实大小。
		}
		// 校验子空间码本中质心的维度
		if len(currentSubspaceCodebook[0]) != subvectorDim {
			return entity.CompressedVector{}, fmt.Errorf("子空间 %d 的码本中质心维度 %d 与子向量维度 %d 不符", i, len(currentSubspaceCodebook[0]), subvectorDim)
		}

		minDistSq := math.MaxFloat64
		assignedCentroidIndex := -1

		for centroidIdx, centroid := range currentSubspaceCodebook {
			distSq, err := algorithm.EuclideanDistanceSquared(subVecPoint, centroid)
			if err != nil {
				return entity.CompressedVector{}, fmt.Errorf("计算到质心 %d (子空间 %d) 的距离失败: %w", centroidIdx, i, err)
			}

			if distSq < minDistSq {
				minDistSq = distSq
				assignedCentroidIndex = centroidIdx
			}
		}

		if assignedCentroidIndex == -1 {
			return entity.CompressedVector{}, fmt.Errorf("未能为子向量 %d 分配质心", i)
		}
		if assignedCentroidIndex >= 256 {
			// 这个检查理论上不应该触发，因为 TrainPQCodebook 限制了 numCentroidsPerSubvector <= 255
			// 但如果码本是外部生成的，这个检查仍然有用。
			return entity.CompressedVector{}, fmt.Errorf("质心索引 %d 超出byte表示范围，请确保每个子空间的质心数量不超过255", assignedCentroidIndex)
		}
		compressedData[i] = byte(assignedCentroidIndex)
	}

	return entity.CompressedVector{
		Data:     compressedData,
		Codebook: nil, // 压缩结果不包含码本，码本是全局/外部管理的
	}, nil
}

// flattenPoints 将 Point 列表展平成 float64 切片
func flattenPoints(points []algorithm.Point) []float64 {
	if len(points) == 0 {
		return nil
	}
	dim := len(points[0])
	flat := make([]float64, 0, len(points)*dim)
	for _, p := range points {
		flat = append(flat, p...)
	}
	return flat
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
