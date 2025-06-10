package vector

import (
	"VectorSphere/src/library/algorithm"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/log"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// IVF 配置结构
type IVFConfig struct {
	NumClusters        int     `json:"num_clusters"`        // 聚类数量
	TrainingRatio      float64 `json:"training_ratio"`      // 训练数据比例
	RebalanceThreshold int     `json:"rebalance_threshold"` // 重平衡阈值
	UsePQCompression   bool    `json:"use_pq_compression"`  // 是否使用 PQ 压缩
	PQSubVectors       int     `json:"pq_sub_vectors"`      // PQ 子向量数量
	PQCentroids        int     `json:"pq_centroids"`        // PQ 质心数量
	EnableDynamic      bool    `json:"enable_dynamic"`      // 启用动态更新
	MaxClusterSize     int     `json:"max_cluster_size"`    // 最大聚类大小
	MinClusterSize     int     `json:"min_cluster_size"`    // 最小聚类大小
}

// EnhancedIVFIndex 增强 IVF 索引结构
type EnhancedIVFIndex struct {
	Clusters         []EnhancedCluster `json:"clusters"`
	ClusterCentroids [][]float64       `json:"centroids"`
	ClusterSizes     []int             `json:"sizes"`
	ClusterMetrics   []ClusterMetrics  `json:"metrics"`
	LastUpdateTime   time.Time         `json:"last_update"`
	TotalVectors     int               `json:"total_vectors"`
	IndexVersion     int               `json:"version"`
	mu               sync.RWMutex
	Enable           bool `json:"enable"`
}

// EnhancedCluster 增强聚类结构
type EnhancedCluster struct {
	Centroid     []float64      `json:"centroid"`
	VectorIDs    []string       `json:"vector_ids"`
	SubClusters  []SubCluster   `json:"sub_clusters"` // 二级聚类
	PQCodes      [][]byte       `json:"pq_codes"`     // PQ 编码
	Metrics      ClusterMetrics `json:"metrics"`
	LastAccessed time.Time      `json:"last_accessed"`
	AccessCount  int64          `json:"access_count"`
}

// ClusterMetrics 聚类指标
type ClusterMetrics struct {
	Variance       float64   `json:"variance"`        // 方差
	Density        float64   `json:"density"`         // 密度
	Radius         float64   `json:"radius"`          // 半径
	QueryFrequency float64   `json:"query_frequency"` // 查询频率
	LastRebalance  time.Time `json:"last_rebalance"`
}

// SubCluster 子聚类结构（用于层次化 IVF）
type SubCluster struct {
	Centroid        []float64 `json:"centroid"`
	VectorIDs       []string  `json:"vector_ids"`
	ParentClusterID int       `json:"parent_id"`
}

// IVFPQIndex IVF-PQ 混合索引
type IVFPQIndex struct {
	IVFIndex      *EnhancedIVFIndex `json:"ivf_index"`
	PQCodebooks   [][][]float64     `json:"pq_codebooks"` // 每个聚类的 PQ 码本
	PQCodes       map[string][]byte `json:"pq_codes"`     // 向量的 PQ 编码
	SubVectorDim  int               `json:"sub_vector_dim"`
	NumSubVectors int               `json:"num_sub_vectors"`
	NumCentroids  int               `json:"num_centroids"`
}

// ClusterUpdate 聚类更新事件
type ClusterUpdate struct {
	Type      UpdateType `json:"type"`
	ClusterID int        `json:"cluster_id"`
	VectorID  string     `json:"vector_id"`
	Vector    []float64  `json:"vector"`
	Timestamp time.Time  `json:"timestamp"`
}

type UpdateType int

const (
	UpdateTypeAdd UpdateType = iota
	UpdateTypeRemove
	UpdateTypeRebalance
)

// sampleTrainingData 从数据库中采样训练数据
// trainingRatio: 采样比例 (0.0 to 1.0)
// 返回: 训练向量列表和对应的ID列表
func (db *VectorDB) sampleTrainingData(trainingRatio float64) ([][]float64, []string) {
	if trainingRatio <= 0 || trainingRatio > 1.0 {
		trainingRatio = 0.1 // 默认采样10%的数据进行训练
		log.Warning("Invalid trainingRatio, defaulting to 0.1")
	}

	db.mu.RLock() // Read lock for accessing db.vectors
	defer db.mu.RUnlock()

	if len(db.vectors) == 0 {
		return [][]float64{}, []string{}
	}

	numSamples := int(float64(len(db.vectors)) * trainingRatio)
	if numSamples == 0 && len(db.vectors) > 0 {
		numSamples = 1 // 至少采样一个数据点，如果数据库不为空
	}
	if numSamples > len(db.vectors) {
		numSamples = len(db.vectors)
	}

	trainingVectors := make([][]float64, 0, numSamples)
	trainingIDs := make([]string, 0, numSamples)

	// 为了可复现性和随机性，创建一个临时的ID列表并打乱
	allIDs := make([]string, 0, len(db.vectors))
	for id := range db.vectors {
		allIDs = append(allIDs, id)
	}

	rand.Seed(time.Now().UnixNano()) // 使用当前时间作为随机种子
	rand.Shuffle(len(allIDs), func(i, j int) {
		allIDs[i], allIDs[j] = allIDs[j], allIDs[i]
	})

	for i := 0; i < numSamples; i++ {
		id := allIDs[i]
		if vec, ok := db.vectors[id]; ok {
			trainingVectors = append(trainingVectors, vec)
			trainingIDs = append(trainingIDs, id)
		}
	}

	log.Info("Sampled %d vectors for training (%.2f%% of total %d vectors)", len(trainingVectors), trainingRatio*100, len(db.vectors))
	return trainingVectors, trainingIDs
}

// BuildEnhancedIVFIndex 构建增强 IVF 索引
func (db *VectorDB) BuildEnhancedIVFIndex(config *IVFConfig) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if config == nil {
		// 默认配置，如果外部没有提供
		numClustersDefault := int(math.Sqrt(float64(len(db.vectors))))
		if numClustersDefault == 0 {
			numClustersDefault = 1 // 至少一个簇
		}
		config = &IVFConfig{
			NumClusters:        numClustersDefault,
			TrainingRatio:      0.1, // 默认采样10%的数据
			RebalanceThreshold: 1000,
			UsePQCompression:   true,
			PQSubVectors:       8,
			PQCentroids:        256,
			EnableDynamic:      true,
			MaxClusterSize:     10000,
			MinClusterSize:     10,
		}
		log.Info("No IVFConfig provided, using default configuration: %+v", *config)
	}

	db.ivfConfig = config

	// 0. 检查向量数据是否为空
	if len(db.vectors) == 0 {
		log.Warning("Cannot build Enhanced IVF Index: no vectors in the database.")
		db.indexed = false // 标记为未索引
		return fmt.Errorf("cannot build Enhanced IVF Index: no vectors in the database")
	}

	// 1. 采样训练数据
	trainingVectors, _ := db.sampleTrainingData(config.TrainingRatio) // trainingIDs 暂时未使用
	if len(trainingVectors) == 0 {
		log.Warning("Cannot build Enhanced IVF Index: no training data sampled.")
		db.indexed = false
		return fmt.Errorf("cannot build Enhanced IVF Index: no training data sampled")
	}

	// 确保聚类数量不超过训练样本数量
	if config.NumClusters > len(trainingVectors) {
		log.Warning("Number of clusters (%d) exceeds number of training samples (%d). Adjusting NumClusters to %d.", config.NumClusters, len(trainingVectors), len(trainingVectors))
		config.NumClusters = len(trainingVectors)
	}
	if config.NumClusters <= 0 {
		log.Warning("Number of clusters must be positive. Setting to 1.")
		config.NumClusters = 1
	}

	// 2. 执行聚类
	log.Info("Starting KMeans clustering with %d training vectors and %d clusters.", len(trainingVectors), config.NumClusters)
	centroids, _, err := algorithm.KMeans(algorithm.ConvertToPoints(trainingVectors), config.NumClusters, 100, 0.001)
	if err != nil {
		db.indexed = false
		return fmt.Errorf("KMeans 聚类失败: %w", err)
	}
	log.Info("KMeans clustering completed. Generated %d centroids.", len(centroids))

	// 3. 构建增强聚类
	enhancedClusters := make([]EnhancedCluster, config.NumClusters)
	for i := 0; i < config.NumClusters; i++ {
		enhancedClusters[i] = EnhancedCluster{
			Centroid:     centroids[i],
			VectorIDs:    make([]string, 0),
			SubClusters:  make([]SubCluster, 0),
			PQCodes:      make([][]byte, 0),
			Metrics:      ClusterMetrics{},
			LastAccessed: time.Now(),
			AccessCount:  0,
		}
	}

	// 4. 分配所有向量到聚类
	for id, vector := range db.vectors {
		// 使用 []entity.Point 类型的质心进行匹配
		clusterID := db.findNearestCluster(vector, centroids) // centroids 是 []entity.Point 类型
		enhancedClusters[clusterID].VectorIDs = append(enhancedClusters[clusterID].VectorIDs, id)
	}

	// 5. 构建 PQ 压缩（如果启用）
	if config.UsePQCompression {
		if err := db.BuildIVFPQIndex(enhancedClusters, config); err != nil {
			log.Warning("构建 IVF-PQ 索引失败: %v", err)
			// 根据策略，这里可以选择是否因为PQ构建失败而整体失败
			// return fmt.Errorf("构建 IVF-PQ 索引失败: %w", err)
		}
	}

	// 6. 计算聚类指标
	for i := range enhancedClusters {
		db.calculateClusterMetrics(&enhancedClusters[i])
	}

	// 7. 创建增强索引
	clusterCentroids := algorithm.ConvertToFloat64Slice(centroids)
	db.ivfIndex = &EnhancedIVFIndex{
		Clusters:         enhancedClusters,
		ClusterCentroids: clusterCentroids,
		ClusterSizes:     make([]int, config.NumClusters),
		ClusterMetrics:   make([]ClusterMetrics, config.NumClusters),
		LastUpdateTime:   time.Now(),
		TotalVectors:     len(db.vectors),
		IndexVersion:     1,
	}

	// 8. 启动动态更新（如果启用）
	if config.EnableDynamic {
		db.startDynamicClusterUpdates()
	}

	db.indexed = true
	log.Info("增强 IVF 索引构建完成，共 %d 个聚类", config.NumClusters)
	return nil
}

// BuildIVFPQIndex 为 IVF 索引构建 PQ 压缩
// enhancedClusters: 增强聚类列表
// config: IVF 配置
func (db *VectorDB) BuildIVFPQIndex(enhancedClusters []EnhancedCluster, config *IVFConfig) error {
	if !config.UsePQCompression || config.PQSubVectors <= 0 || config.PQCentroids <= 0 {
		log.Info("PQ Compression is not enabled or configuration is invalid. Skipping PQ index build.")
		return nil
	}

	log.Info("Starting to build PQ index for %d clusters. PQSubVectors: %d, PQCentroids: %d", len(enhancedClusters), config.PQSubVectors, config.PQCentroids)

	db.ivfPQIndex = &IVFPQIndex{
		IVFIndex:      db.ivfIndex, // db.ivfIndex 此时可能还未完全初始化，但其引用是需要的
		PQCodebooks:   make([][][]float64, len(enhancedClusters)),
		PQCodes:       make(map[string][]byte),
		NumSubVectors: config.PQSubVectors,
		NumCentroids:  config.PQCentroids,
	}

	var originalVectorDimension int
	// 获取向量维度，假设所有向量维度相同
	// 优化：可以从 db.vectors 中随机取一个向量来确定维度，或者在 VectorDB 结构中存储维度信息
	if len(db.vectors) > 0 {
		for _, vec := range db.vectors {
			originalVectorDimension = len(vec)
			break
		}
	} else {
		return fmt.Errorf("cannot build PQ index without any vectors to determine dimension")
	}

	if originalVectorDimension == 0 {
		return fmt.Errorf("cannot build PQ index with zero dimension vectors")
	}

	db.ivfPQIndex.SubVectorDim = originalVectorDimension / config.PQSubVectors
	if originalVectorDimension%config.PQSubVectors != 0 {
		log.Warning("Original vector dimension %d is not perfectly divisible by PQSubVectors %d. SubVectorDim will be %d. Some parts of vectors might be ignored or padded.",
			originalVectorDimension, config.PQSubVectors, db.ivfPQIndex.SubVectorDim)
		// 实际应用中可能需要更复杂的处理，例如填充或报错
		if db.ivfPQIndex.SubVectorDim == 0 && originalVectorDimension > 0 { // 确保子维度至少为1
			db.ivfPQIndex.SubVectorDim = 1
		}
	}
	if db.ivfPQIndex.SubVectorDim == 0 && originalVectorDimension > 0 { // 再次检查，如果除法结果为0但原始维度不为0
		log.Error("SubVectorDim is 0 even though original dimension is %d and PQSubVectors is %d. This indicates a problem.", originalVectorDimension, config.PQSubVectors)
		return fmt.Errorf("calculated SubVectorDim is 0, PQSubVectors might be too large for the dimension")
	}

	for i, cluster := range enhancedClusters {
		if len(cluster.VectorIDs) == 0 {
			log.Info("Cluster %d has no vectors, skipping PQ codebook generation.", i)
			db.ivfPQIndex.PQCodebooks[i] = make([][]float64, 0) // 初始化为空码本
			continue
		}

		// 1. 收集当前聚类的所有向量数据
		clusterVectors := make([][]float64, 0, len(cluster.VectorIDs))
		for _, vecID := range cluster.VectorIDs {
			if vec, ok := db.vectors[vecID]; ok {
				clusterVectors = append(clusterVectors, vec)
			} else {
				log.Warning("Vector ID %s not found in db.vectors while building PQ for cluster %d", vecID, i)
			}
		}

		if len(clusterVectors) == 0 {
			log.Info("No valid vectors found for cluster %d after lookup, skipping PQ codebook generation.", i)
			db.ivfPQIndex.PQCodebooks[i] = make([][]float64, 0)
			continue
		}

		// 2. 为每个子向量空间训练PQ码本
		codebookForCluster := make([][][]float64, config.PQSubVectors)
		for subVecIdx := 0; subVecIdx < config.PQSubVectors; subVecIdx++ {
			subVectorTrainingData := make([][]float64, 0, len(clusterVectors))
			startDim := subVecIdx * db.ivfPQIndex.SubVectorDim
			endDim := startDim + db.ivfPQIndex.SubVectorDim

			// 安全检查，确保 endDim 不超过原始向量维度
			if endDim > originalVectorDimension {
				endDim = originalVectorDimension
			}
			// 如果 startDim 已经等于或超过 originalVectorDimension，说明子向量划分有问题
			if startDim >= originalVectorDimension {
				log.Warning("Subvector start dimension %d is out of bounds for original dimension %d in cluster %d, subvector %d. Skipping this subvector.", startDim, originalVectorDimension, i, subVecIdx)
				codebookForCluster[subVecIdx] = make([][]float64, 0) // 空码本
				continue
			}

			for _, vec := range clusterVectors {
				if len(vec) >= endDim {
					subVectorTrainingData = append(subVectorTrainingData, vec[startDim:endDim])
				} else {
					log.Warning("Vector dimension mismatch for PQ. Expected at least %d, got %d for a vector in cluster %d. Skipping this vector for subvector %d.", endDim, len(vec), i, subVecIdx)
				}
			}

			if len(subVectorTrainingData) == 0 {
				log.Warning("No training data for subvector %d in cluster %d. Skipping codebook generation for this subvector.", subVecIdx, i)
				codebookForCluster[subVecIdx] = make([][]float64, 0) // 空码本
				continue
			}

			numPqCentroids := config.PQCentroids
			if numPqCentroids > len(subVectorTrainingData) {
				log.Warning("Number of PQ centroids (%d) for subvector %d in cluster %d exceeds training samples (%d). Adjusting to %d.",
					numPqCentroids, subVecIdx, i, len(subVectorTrainingData), len(subVectorTrainingData))
				numPqCentroids = len(subVectorTrainingData)
			}
			if numPqCentroids <= 0 { // 确保至少有一个质心
				log.Warning("Number of PQ centroids is %d for subvector %d in cluster %d. Setting to 1.", numPqCentroids, subVecIdx, i)
				numPqCentroids = 1
			}

			log.Info("Training PQ codebook for cluster %d, subvector %d with %d samples and %d centroids.", i, subVecIdx, len(subVectorTrainingData), numPqCentroids)
			pqCentroids, _, err := algorithm.KMeans(algorithm.ConvertToPoints(subVectorTrainingData), numPqCentroids, 50, 0.0001) // 使用较少的迭代次数和较小的容差
			if err != nil {
				log.Error("Failed to train PQ codebook for cluster %d, subvector %d: %v", i, subVecIdx, err)
				codebookForCluster[subVecIdx] = make([][]float64, 0) // 错误时设置为空码本
				continue
			}
			codebookForCluster[subVecIdx] = algorithm.ConvertToFloat64Slice(pqCentroids)
		}
		db.ivfPQIndex.PQCodebooks[i] = codebookForCluster

		// 3. 为当前聚类的所有向量生成 PQ 编码
		for _, vecID := range cluster.VectorIDs {
			if vec, ok := db.vectors[vecID]; ok {
				pqCode := make([]byte, config.PQSubVectors)
				for subVecIdx := 0; subVecIdx < config.PQSubVectors; subVecIdx++ {
					startDim := subVecIdx * db.ivfPQIndex.SubVectorDim
					endDim := startDim + db.ivfPQIndex.SubVectorDim

					if endDim > originalVectorDimension {
						endDim = originalVectorDimension
					}
					if startDim >= originalVectorDimension || len(db.ivfPQIndex.PQCodebooks[i]) <= subVecIdx || len(db.ivfPQIndex.PQCodebooks[i][subVecIdx]) == 0 {
						// 如果子向量超出边界，或者该子向量的码本为空，则无法编码
						log.Warning("Cannot generate PQ code for vector %s, subvector %d in cluster %d due to out-of-bounds or empty codebook. Assigning default code (0).", vecID, subVecIdx, i)
						pqCode[subVecIdx] = 0 // 或者其他默认值/错误处理
						continue
					}

					if len(vec) >= endDim {
						subVec := vec[startDim:endDim]
						nearestCentroidIdx := findNearestPQCentroid(subVec, db.ivfPQIndex.PQCodebooks[i][subVecIdx])
						pqCode[subVecIdx] = byte(nearestCentroidIdx)
					} else {
						log.Warning("Vector %s dimension mismatch for PQ encoding. Expected at least %d, got %d for subvector %d in cluster %d. Assigning default code (0).", vecID, endDim, len(vec), subVecIdx, i)
						pqCode[subVecIdx] = 0 // 默认编码
					}
				}
				db.ivfPQIndex.PQCodes[vecID] = pqCode
				enhancedClusters[i].PQCodes = append(enhancedClusters[i].PQCodes, pqCode) // 同时存储在 EnhancedCluster 中
			} else {
				log.Warning("Vector ID %s not found in db.vectors while generating PQ codes for cluster %d", vecID, i)
			}
		}
		log.Info("Finished PQ processing for cluster %d. Generated %d PQ codes.", i, len(enhancedClusters[i].PQCodes))
	}

	log.Info("IVF-PQ index build completed.")
	return nil
}

// findNearestPQCentroid 找到距离给定子向量最近的PQ码本中的质心
// subVector: 要编码的子向量
// pqCodebookForSubVector: 特定子向量的PQ码本 ([][]float64)
// 返回: 最近的PQ质心的索引
func findNearestPQCentroid(subVector []float64, pqCodebookForSubVector [][]float64) int {
	if len(pqCodebookForSubVector) == 0 {
		log.Warning("findNearestPQCentroid called with empty PQ codebook for subvector.")
		return 0 // 或者一个特殊的错误指示值
	}
	minDist := math.MaxFloat64
	nearestCentroidIdx := 0
	for idx, centroid := range pqCodebookForSubVector {
		dist, err := algorithm.EuclideanDistanceSquared(subVector, centroid)
		if err != nil {
			log.Warning("Error calculating distance for PQ centroid %d: %v", idx, err)
			continue
		}
		if dist < minDist {
			minDist = dist
			nearestCentroidIdx = idx
		}
	}
	return nearestCentroidIdx
}

// findNearestCluster 找到距离给定向量最近的聚类中心
// vector: 要分配的向量
// centroids: 聚类中心的列表 (类型为 []entity.Point)
// 返回: 最近的聚类中心的索引
func (db *VectorDB) findNearestCluster(vector []float64, centroids []entity.Point) int {
	if len(centroids) == 0 {
		// 应该在调用此函数之前处理这种情况，但作为安全措施
		log.Error("findNearestCluster called with no centroids")
		return -1 // 或者 panic，取决于错误处理策略
	}

	minDist := math.MaxFloat64
	nearestClusterID := 0

	for i, centroidPoint := range centroids {
		// entity.Point 内部存储的是 []float64
		dist, err := algorithm.EuclideanDistanceSquared(vector, centroidPoint) // entity.Point 可以直接用于距离计算
		if err != nil {
			log.Warning("Error calculating distance between vector and centroid %d: %v", i, err)
			continue // 跳过这个质心或采取其他错误处理
		}

		if dist < minDist {
			minDist = dist
			nearestClusterID = i
		}
	}
	return nearestClusterID
}

// EnhancedIVFSearch 增强 IVF 搜索
func (db *VectorDB) EnhancedIVFSearch(query []float64, k int, nprobe int) ([]entity.Result, error) {
	if db.ivfIndex == nil {
		return db.ivfSearchWithScores(query, k, nprobe, db.GetOptimalStrategy(query))
	}

	db.ivfIndex.mu.RLock()
	defer db.ivfIndex.mu.RUnlock()

	// 1. 自适应 nprobe 调整
	adaptiveNprobe := db.calculateAdaptiveNprobe(query, nprobe)

	// 2. 智能聚类选择
	candidateClusters := db.selectCandidateClusters(query, adaptiveNprobe)

	// 3. 并行搜索候选聚类
	results := make(chan []entity.Result, len(candidateClusters))
	var wg sync.WaitGroup

	for _, clusterID := range candidateClusters {
		wg.Add(1)
		go func(cid int) {
			defer wg.Done()
			clusterResults := db.searchInCluster(query, k, cid)
			results <- clusterResults
		}(clusterID)
	}

	wg.Wait()
	close(results)

	// 4. 合并和排序结果
	allResults := make([]entity.Result, 0)
	for clusterResults := range results {
		allResults = append(allResults, clusterResults...)
	}

	// 5. 返回 top-k 结果
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Similarity > allResults[j].Similarity
	})

	if k > len(allResults) {
		k = len(allResults)
	}

	return allResults[:k], nil
}
