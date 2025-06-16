package vector

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/algorithm"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// IVFConfig IVF 配置结构
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
	PQCodebooks   [][][][]float64   `json:"pq_codebooks"` // 每个聚类的 PQ 码本
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
		logger.Warning("Invalid trainingRatio, defaulting to 0.1")
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

	logger.Info("Sampled %d vectors for training (%.2f%% of total %d vectors)", len(trainingVectors), trainingRatio*100, len(db.vectors))
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
		logger.Info("No IVFConfig provided, using default configuration: %+v", *config)
	}

	db.ivfConfig = config

	// 0. 检查向量数据是否为空
	if len(db.vectors) == 0 {
		logger.Warning("Cannot build Enhanced IVF Index: no vectors in the database.")
		db.indexed = false // 标记为未索引
		return fmt.Errorf("cannot build Enhanced IVF Index: no vectors in the database")
	}

	// 1. 采样训练数据
	trainingVectors, _ := db.sampleTrainingData(config.TrainingRatio) // trainingIDs 暂时未使用
	if len(trainingVectors) == 0 {
		logger.Warning("Cannot build Enhanced IVF Index: no training data sampled.")
		db.indexed = false
		return fmt.Errorf("cannot build Enhanced IVF Index: no training data sampled")
	}

	// 确保聚类数量不超过训练样本数量
	if config.NumClusters > len(trainingVectors) {
		logger.Warning("Number of clusters (%d) exceeds number of training samples (%d). Adjusting NumClusters to %d.", config.NumClusters, len(trainingVectors), len(trainingVectors))
		config.NumClusters = len(trainingVectors)
	}
	if config.NumClusters <= 0 {
		logger.Warning("Number of clusters must be positive. Setting to 1.")
		config.NumClusters = 1
	}

	// 2. 执行聚类
	logger.Info("Starting KMeans clustering with %d training vectors and %d clusters.", len(trainingVectors), config.NumClusters)
	centroids, _, err := algorithm.KMeans(algorithm.ConvertToPoints(trainingVectors), config.NumClusters, 100, 0.001)
	if err != nil {
		db.indexed = false
		return fmt.Errorf("KMeans 聚类失败: %w", err)
	}
	logger.Info("KMeans clustering completed. Generated %d centroids.", len(centroids))

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
			logger.Warning("构建 IVF-PQ 索引失败: %v", err)
			// 根据策略，这里可以选择是否因为PQ构建失败而整体失败
			// return fmt.Errorf("构建 IVF-PQ 索引失败: %w", err)
		}
	}

	// 6. 计算聚类指标
	for i := range enhancedClusters {
		db.CalculateClusterMetrics(&enhancedClusters[i])
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
		db.StartDynamicClusterUpdates()
	}

	db.indexed = true
	logger.Info("增强 IVF 索引构建完成，共 %d 个聚类", config.NumClusters)
	return nil
}

// StartDynamicClusterUpdates 启动后台 goroutine 以进行动态聚类更新
func (db *VectorDB) StartDynamicClusterUpdates() {
	if db.ivfIndex == nil || !db.ivfConfig.EnableDynamic {
		logger.Info("Dynamic cluster updates are not enabled or IVF index is not built.")
		return
	}

	logger.Info("Starting dynamic cluster updates goroutine...")
	go func() {
		// TODO: 从配置中读取更新间隔
		ticker := time.NewTicker(1 * time.Minute) // 例如，每分钟检查一次
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				db.performClusterMaintenance()
				// TODO: 添加一个停止通道，以便在数据库关闭时优雅地停止此 goroutine
				// case <-db.stopDynamicUpdatesCh: // 假设有一个停止通道
				// 	log.Info("Stopping dynamic cluster updates goroutine.")
				// 	return
			}
		}
	}()
}

// performClusterMaintenance 执行聚类维护操作，例如重平衡
func (db *VectorDB) performClusterMaintenance() {
	if db.ivfIndex == nil {
		return
	}
	db.ivfIndex.mu.Lock()
	defer db.ivfIndex.mu.Unlock()

	logger.Info("Performing cluster maintenance...")
	var needsRebuild bool = false
	for i := range db.ivfIndex.Clusters {
		cluster := &db.ivfIndex.Clusters[i]
		// 检查是否需要重平衡
		// 例如：如果聚类大小超出阈值，或者长时间未访问等
		if len(cluster.VectorIDs) > db.ivfConfig.MaxClusterSize || len(cluster.VectorIDs) < db.ivfConfig.MinClusterSize && len(cluster.VectorIDs) > 0 {
			logger.Info("Cluster %d (size: %d) needs rebalancing (max: %d, min: %d). Triggering index rebuild.", i, len(cluster.VectorIDs), db.ivfConfig.MaxClusterSize, db.ivfConfig.MinClusterSize)
			// 标记需要重建索引，实际的重平衡可能需要更复杂的逻辑
			// 例如，分裂过大的簇，合并过小的簇，或者完全重建索引
			needsRebuild = true
			break // 一旦发现需要重建，就可以跳出循环
		}

		// 示例：更新访问统计信息或执行其他维护任务
		// cluster.AccessCount = 0 // 例如，定期重置访问计数
		// cluster.Metrics.LastRebalance = time.Now() // 更新重平衡时间戳
	}

	if needsRebuild {
		logger.Info("Triggering full IVF index rebuild due to cluster maintenance requirements.")
		// 注意：直接在后台 goroutine 中调用 BuildEnhancedIVFIndex 可能需要小心处理并发和锁
		// 最好是通过一个任务队列或者其他机制来触发重建
		// 简单的示例是直接调用，但这在生产环境中可能需要更健壮的处理
		go func() {
			if err := db.BuildEnhancedIVFIndex(db.ivfConfig); err != nil {
				logger.Error("Error during background IVF index rebuild: %v", err)
			}
		}()
	}
	logger.Info("Cluster maintenance check completed.")
}

// CalculateClusterMetrics 计算并更新给定增强聚类的指标
func (db *VectorDB) CalculateClusterMetrics(cluster *EnhancedCluster) {
	if cluster == nil {
		logger.Warning("calculateClusterMetrics called with nil cluster")
		return
	}

	numVectors := len(cluster.VectorIDs)
	if numVectors == 0 {
		cluster.Metrics = ClusterMetrics{ // 重置或设置默认指标
			Variance:       0,
			Density:        0,
			Radius:         0,
			QueryFrequency: cluster.Metrics.QueryFrequency, // 保留查询频率
			LastRebalance:  cluster.Metrics.LastRebalance,  // 保留上次重平衡时间
		}
		return
	}

	var dimension int
	if len(cluster.Centroid) > 0 {
		dimension = len(cluster.Centroid)
	} else if numVectors > 0 {
		if vec, ok := db.vectors[cluster.VectorIDs[0]]; ok {
			dimension = len(vec)
		} else {
			logger.Warning("Cannot determine dimension for cluster metric calculation as centroid is empty and first vector %s not found.", cluster.VectorIDs[0])
			return
		}
	}
	if dimension == 0 {
		logger.Warning("Cannot calculate metrics for zero-dimension vectors.")
		return
	}

	// 计算方差 (Variance) 和 半径 (Radius)
	var sumSquaredDist float64
	maxDistToCentroid := 0.0
	meanVector := make([]float64, dimension)

	for _, vecID := range cluster.VectorIDs {
		if vector, ok := db.vectors[vecID]; ok {
			distSq, err := algorithm.EuclideanDistanceSquared(vector, cluster.Centroid)
			if err != nil {
				logger.Warning("Error calculating distance for variance/radius for vector %s: %v", vecID, err)
				continue
			}
			sumSquaredDist += distSq
			dist := math.Sqrt(distSq)
			if dist > maxDistToCentroid {
				maxDistToCentroid = dist
			}
			// 累加用于计算均值向量
			for i := 0; i < dimension; i++ {
				meanVector[i] += vector[i]
			}
		} else {
			logger.Warning("Vector ID %s not found in db.vectors during metric calculation.", vecID)
		}
	}

	cluster.Metrics.Variance = sumSquaredDist / float64(numVectors)
	cluster.Metrics.Radius = maxDistToCentroid

	// 计算密度 (Density)
	// 密度可以有多种定义，这里使用一种简单的定义：单位体积内的向量数
	// 假设聚类大致呈球形，体积 V = (4/3) * pi * R^3 (高维时更复杂，这里简化)
	if cluster.Metrics.Radius > 0 {
		// 对于高维空间，球体体积公式为 V_n(R) = (pi^(n/2) / Gamma(n/2 + 1)) * R^n
		// 这里使用一个简化的概念，或者直接使用半径的倒数等作为密度指标
		// 或者定义为：单位距离内的平均向量数
		// 另一种简单的密度定义： numVectors / (Radius^dimension) - 但要注意量纲和数值范围
		// 这里采用更稳健的：numVectors / (1 + Radius) 来避免 Radius 为0或过小导致的问题
		cluster.Metrics.Density = float64(numVectors) / (1.0 + cluster.Metrics.Radius)
	} else if numVectors > 0 { // 如果半径为0（例如只有一个点），密度可以认为很高
		cluster.Metrics.Density = float64(numVectors) // 或者一个预设的最大密度值
	} else {
		cluster.Metrics.Density = 0
	}

	// QueryFrequency 和 LastRebalance 通常在其他地方更新
	// cluster.Metrics.QueryFrequency = ... (由查询逻辑更新)
	// cluster.Metrics.LastRebalance = ... (由重平衡逻辑更新)

	logger.Trace("Calculated metrics for cluster with %d vectors: Variance=%.4f, Radius=%.4f, Density=%.4f",
		numVectors, cluster.Metrics.Variance, cluster.Metrics.Radius, cluster.Metrics.Density)
}

// BuildIVFPQIndex 为 IVF 索引构建 PQ 压缩
// enhancedClusters: 增强聚类列表
// config: IVF 配置
func (db *VectorDB) BuildIVFPQIndex(enhancedClusters []EnhancedCluster, config *IVFConfig) error {
	if !config.UsePQCompression || config.PQSubVectors <= 0 || config.PQCentroids <= 0 {
		logger.Info("PQ Compression is not enabled or configuration is invalid. Skipping PQ index build.")
		return nil
	}

	logger.Info("Starting to build PQ index for %d clusters. PQSubVectors: %d, PQCentroids: %d", len(enhancedClusters), config.PQSubVectors, config.PQCentroids)

	db.ivfPQIndex = &IVFPQIndex{
		IVFIndex:      db.ivfIndex, // db.ivfIndex 此时可能还未完全初始化，但其引用是需要的
		PQCodebooks:   make([][][][]float64, len(enhancedClusters)),
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
		logger.Warning("Original vector dimension %d is not perfectly divisible by PQSubVectors %d. SubVectorDim will be %d. Some parts of vectors might be ignored or padded.",
			originalVectorDimension, config.PQSubVectors, db.ivfPQIndex.SubVectorDim)
		// 实际应用中可能需要更复杂的处理，例如填充或报错
		if db.ivfPQIndex.SubVectorDim == 0 && originalVectorDimension > 0 { // 确保子维度至少为1
			db.ivfPQIndex.SubVectorDim = 1
		}
	}
	if db.ivfPQIndex.SubVectorDim == 0 && originalVectorDimension > 0 { // 再次检查，如果除法结果为0但原始维度不为0
		logger.Error("SubVectorDim is 0 even though original dimension is %d and PQSubVectors is %d. This indicates a problem.", originalVectorDimension, config.PQSubVectors)
		return fmt.Errorf("calculated SubVectorDim is 0, PQSubVectors might be too large for the dimension")
	}

	for i, cluster := range enhancedClusters {
		if len(cluster.VectorIDs) == 0 {
			logger.Info("Cluster %d has no vectors, skipping PQ codebook generation.", i)
			db.ivfPQIndex.PQCodebooks[i] = make([][][]float64, 0) // 初始化为空码本
			continue
		}

		// 1. 收集当前聚类的所有向量数据
		clusterVectors := make([][]float64, 0, len(cluster.VectorIDs))
		for _, vecID := range cluster.VectorIDs {
			if vec, ok := db.vectors[vecID]; ok {
				clusterVectors = append(clusterVectors, vec)
			} else {
				logger.Warning("Vector ID %s not found in db.vectors while building PQ for cluster %d", vecID, i)
			}
		}

		if len(clusterVectors) == 0 {
			logger.Info("No valid vectors found for cluster %d after lookup, skipping PQ codebook generation.", i)
			db.ivfPQIndex.PQCodebooks[i] = make([][][]float64, 0)
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
				logger.Warning("Subvector start dimension %d is out of bounds for original dimension %d in cluster %d, subvector %d. Skipping this subvector.", startDim, originalVectorDimension, i, subVecIdx)
				codebookForCluster[subVecIdx] = make([][]float64, 0) // 空码本
				continue
			}

			for _, vec := range clusterVectors {
				if len(vec) >= endDim {
					subVectorTrainingData = append(subVectorTrainingData, vec[startDim:endDim])
				} else {
					logger.Warning("Vector dimension mismatch for PQ. Expected at least %d, got %d for a vector in cluster %d. Skipping this vector for subvector %d.", endDim, len(vec), i, subVecIdx)
				}
			}

			if len(subVectorTrainingData) == 0 {
				logger.Warning("No training data for subvector %d in cluster %d. Skipping codebook generation for this subvector.", subVecIdx, i)
				codebookForCluster[subVecIdx] = make([][]float64, 0) // 空码本
				continue
			}

			numPqCentroids := config.PQCentroids
			if numPqCentroids > len(subVectorTrainingData) {
				logger.Warning("Number of PQ centroids (%d) for subvector %d in cluster %d exceeds training samples (%d). Adjusting to %d.",
					numPqCentroids, subVecIdx, i, len(subVectorTrainingData), len(subVectorTrainingData))
				numPqCentroids = len(subVectorTrainingData)
			}
			if numPqCentroids <= 0 { // 确保至少有一个质心
				logger.Warning("Number of PQ centroids is %d for subvector %d in cluster %d. Setting to 1.", numPqCentroids, subVecIdx, i)
				numPqCentroids = 1
			}

			logger.Info("Training PQ codebook for cluster %d, subvector %d with %d samples and %d centroids.", i, subVecIdx, len(subVectorTrainingData), numPqCentroids)
			pqCentroids, _, err := algorithm.KMeans(algorithm.ConvertToPoints(subVectorTrainingData), numPqCentroids, 50, 0.0001) // 使用较少的迭代次数和较小的容差
			if err != nil {
				logger.Error("Failed to train PQ codebook for cluster %d, subvector %d: %v", i, subVecIdx, err)
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
						logger.Warning("Cannot generate PQ code for vector %s, subvector %d in cluster %d due to out-of-bounds or empty codebook. Assigning default code (0).", vecID, subVecIdx, i)
						pqCode[subVecIdx] = 0 // 或者其他默认值/错误处理
						continue
					}

					if len(vec) >= endDim {
						subVec := vec[startDim:endDim]
						nearestCentroidIdx := findNearestPQCentroid(subVec, db.ivfPQIndex.PQCodebooks[i][subVecIdx])
						pqCode[subVecIdx] = byte(nearestCentroidIdx)
					} else {
						logger.Warning("Vector %s dimension mismatch for PQ encoding. Expected at least %d, got %d for subvector %d in cluster %d. Assigning default code (0).", vecID, endDim, len(vec), subVecIdx, i)
						pqCode[subVecIdx] = 0 // 默认编码
					}
				}
				db.ivfPQIndex.PQCodes[vecID] = pqCode
				enhancedClusters[i].PQCodes = append(enhancedClusters[i].PQCodes, pqCode) // 同时存储在 EnhancedCluster 中
			} else {
				logger.Warning("Vector ID %s not found in db.vectors while generating PQ codes for cluster %d", vecID, i)
			}
		}
		logger.Info("Finished PQ processing for cluster %d. Generated %d PQ codes.", i, len(enhancedClusters[i].PQCodes))
	}

	logger.Info("IVF-PQ index build completed.")
	return nil
}

// findNearestPQCentroid 找到距离给定子向量最近的PQ码本中的质心
// subVector: 要编码的子向量
// pqCodebookForSubVector: 特定子向量的PQ码本 ([][]float64)
// 返回: 最近的PQ质心的索引
func findNearestPQCentroid(subVector []float64, pqCodebookForSubVector [][]float64) int {
	if len(pqCodebookForSubVector) == 0 {
		logger.Warning("findNearestPQCentroid called with empty PQ codebook for subvector.")
		return 0 // 或者一个特殊的错误指示值
	}
	minDist := math.MaxFloat64
	nearestCentroidIdx := 0
	for idx, centroid := range pqCodebookForSubVector {
		dist, err := algorithm.EuclideanDistanceSquared(subVector, centroid)
		if err != nil {
			logger.Warning("Error calculating distance for PQ centroid %d: %v", idx, err)
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
		logger.Error("findNearestCluster called with no centroids")
		return -1 // 或者 panic，取决于错误处理策略
	}

	minDist := math.MaxFloat64
	nearestClusterID := 0

	for i, centroidPoint := range centroids {
		// entity.Point 内部存储的是 []float64
		dist, err := algorithm.EuclideanDistanceSquared(vector, centroidPoint) // entity.Point 可以直接用于距离计算
		if err != nil {
			logger.Warning("Error calculating distance between vector and centroid %d: %v", i, err)
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
	// 修复循环调用问题：当ivfIndex为nil时，直接使用传统IVF搜索，不再调用ivfSearchWithScores
	if db.ivfIndex == nil || !db.ivfIndex.Enable {
		// 使用传统IVF搜索逻辑，避免循环调用
		// 粗排：找到最近的nprobe个簇
		candidateClusters := make([]int, 0, nprobe)
		clusterDistances := make([]float64, len(db.clusters))

		for i, cluster := range db.clusters {
			// 使用自适应距离计算
			selectStrategy := db.GetOptimalStrategy(query)
			switch selectStrategy {
			case acceler.StrategyAVX512, acceler.StrategyAVX2:
				sim := acceler.AdaptiveCosineSimilarity(query, cluster.Centroid, selectStrategy)
				clusterDistances[i] = 1.0 - sim
			default:
				sim := acceler.CosineSimilarity(query, cluster.Centroid)
				clusterDistances[i] = 1.0 - sim
			}
		}

		// 选择距离最近的nprobe个簇
		type clusterDist struct {
			index    int
			distance float64
		}

		clusterList := make([]clusterDist, len(db.clusters))
		for i, dist := range clusterDistances {
			clusterList[i] = clusterDist{index: i, distance: dist}
		}

		sort.Slice(clusterList, func(i, j int) bool {
			return clusterList[i].distance < clusterList[j].distance
		})

		for i := 0; i < nprobe && i < len(clusterList); i++ {
			candidateClusters = append(candidateClusters, clusterList[i].index)
		}

		// 精排：在候选簇中搜索最近邻
		candidateVectors := make([]string, 0)
		for _, clusterIdx := range candidateClusters {
			candidateVectors = append(candidateVectors, db.clusters[clusterIdx].VectorIDs...)
		}

		return db.fineRankingWithScores(query, candidateVectors, k, db.GetOptimalStrategy(query))
	}

	db.ivfIndex.mu.RLock()
	defer db.ivfIndex.mu.RUnlock()

	// 1. 自适应 nprobe 调整
	adaptiveNprobe := db.CalculateAdaptiveNprobe(query, nprobe)

	// 2. 智能聚类选择
	candidateClusters := db.SelectCandidateClusters(query, adaptiveNprobe)

	// 3. 并行搜索候选聚类
	results := make(chan []entity.Result, len(candidateClusters))
	var wg sync.WaitGroup

	for _, clusterID := range candidateClusters {
		wg.Add(1)
		go func(cid int) {
			defer wg.Done()
			clusterResults, _ := db.SearchInCluster(query, k, cid)
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

// SearchInCluster 在指定的单个聚类中搜索 top-k 向量
// query: 查询向量
// k: 要返回的最近邻数量
// clusterID: 要搜索的聚类ID
// 返回: 在该聚类中找到的 top-k 结果列表
func (db *VectorDB) SearchInCluster(query []float64, k int, clusterID int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.ivfIndex == nil || !db.ivfIndex.Enable {
		return nil, fmt.Errorf("IVF index is not built or not enabled")
	}

	if clusterID < 0 || clusterID >= len(db.ivfIndex.Clusters) {
		logger.Warning("searchInCluster called with invalid clusterID (%d) or nil IVF index.", clusterID)
		return nil, fmt.Errorf("invalid clusterID: %d", clusterID)
	}

	cluster := db.ivfIndex.Clusters[clusterID]
	if len(cluster.VectorIDs) == 0 {
		return []entity.Result{}, nil // 空聚类，返回空结果
	}

	results := make([]entity.Result, 0, len(cluster.VectorIDs))

	// 检查是否可以使用 PQ 加速
	usePQ := db.ivfPQIndex != nil && db.ivfConfig != nil && db.ivfConfig.UsePQCompression &&
		clusterID < len(db.ivfPQIndex.PQCodebooks) && len(db.ivfPQIndex.PQCodebooks[clusterID]) == db.ivfPQIndex.NumSubVectors

	for _, vecID := range cluster.VectorIDs {
		var dist float64
		var err error

		if usePQ {
			// 使用 ADC (Asymmetric Distance Computation)
			pqCode, ok := db.ivfPQIndex.PQCodes[vecID]
			if !ok || len(pqCode) != db.ivfPQIndex.NumSubVectors {
				// PQ code 不存在或长度不匹配，回退到精确计算
				originalVec, vecExists := db.vectors[vecID]
				if !vecExists {
					logger.Warning("Vector ID %s not found in db.vectors during PQ fallback in SearchInCluster for cluster %d", vecID, clusterID)
					continue
				}
				dist, err = algorithm.EuclideanDistanceSquared(query, originalVec)
			} else {
				dist, err = db.calculateADCDistance(query, clusterID, pqCode)
			}
		} else {
			// 精确距离计算
			originalVec, vecExists := db.vectors[vecID]
			if !vecExists {
				logger.Warning("Vector ID %s not found in db.vectors during exact search in SearchInCluster for cluster %d", vecID, clusterID)
				continue
			}
			dist, err = algorithm.EuclideanDistanceSquared(query, originalVec)
		}

		if err != nil {
			logger.Warning("Error calculating distance for vector %s in cluster %d: %v", vecID, clusterID, err)
			continue
		}
		results = append(results, entity.Result{Id: vecID, Distance: dist})
	}

	// 按距离升序排序
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	// 返回 top-k
	if k > 0 && k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

// calculateADCDistance 使用PQ码本计算查询向量和PQ编码向量之间的近似距离（非对称距离计算）
func (db *VectorDB) calculateADCDistance(query []float64, clusterID int, pqCode []byte) (float64, error) {
	if db.ivfPQIndex == nil || clusterID >= len(db.ivfPQIndex.PQCodebooks) || len(pqCode) != db.ivfPQIndex.NumSubVectors {
		return 0, fmt.Errorf("invalid parameters for ADC distance calculation")
	}

	var totalDistSq float64
	codebooksForCluster := db.ivfPQIndex.PQCodebooks[clusterID]

	for subVecIdx := 0; subVecIdx < db.ivfPQIndex.NumSubVectors; subVecIdx++ {
		startDim := subVecIdx * db.ivfPQIndex.SubVectorDim
		endDim := startDim + db.ivfPQIndex.SubVectorDim
		if endDim > len(query) {
			endDim = len(query)
		}
		if startDim >= len(query) {
			// log.Warning("Skipping sub-vector %d for ADC due to boundary or codebook issue.", subVecIdx)
			continue // 查询向量的子部分超出了其维度
		}

		querySubVector := query[startDim:endDim]
		centroidIndex := int(pqCode[subVecIdx])

		if subVecIdx >= len(codebooksForCluster) || centroidIndex >= len(codebooksForCluster[subVecIdx]) {
			// log.Warning("Skipping sub-vector %d for ADC due to boundary or codebook issue.", subVecIdx)
			continue // 码本索引超出范围
		}

		centroid := codebooksForCluster[subVecIdx][centroidIndex]

		// 确保子向量和质心维度匹配，以防 SubVectorDim 未能整除原始维度导致末尾子向量维度不一致
		minDim := len(querySubVector)
		if len(centroid) < minDim {
			minDim = len(centroid)
		}

		subDistSq, err := algorithm.EuclideanDistanceSquared(querySubVector[:minDim], centroid[:minDim])
		if err != nil {
			// log.Warning("Error calculating sub-distance for ADC: %v", err)
			return 0, fmt.Errorf("error calculating sub-distance for ADC: %w", err)
		}
		totalDistSq += subDistSq
	}

	return math.Sqrt(totalDistSq), nil
}

// SelectCandidateClusters 根据查询向量和 nprobe 智能选择候选聚类
// query: 查询向量
// nprobe: 要选择的聚类数量
// 返回: 候选聚类的 ID 列表
func (db *VectorDB) SelectCandidateClusters(query []float64, nprobe int) []int {
	if db.ivfIndex == nil || len(db.ivfIndex.ClusterCentroids) == 0 {
		logger.Warning("selectCandidateClusters called with nil or empty IVF index.")
		return []int{}
	}

	db.ivfIndex.mu.RLock() // Ensure read lock for accessing shared index data
	defer db.ivfIndex.mu.RUnlock()

	if nprobe <= 0 {
		nprobe = 1 // 至少选择一个聚类
	}
	if nprobe > len(db.ivfIndex.ClusterCentroids) {
		nprobe = len(db.ivfIndex.ClusterCentroids) // 最多选择所有聚类
	}

	// 计算查询向量到所有聚类中心的距离
	type clusterDistance struct {
		ID       int
		Distance float64
		// 可以添加其他用于排序的指标，例如聚类密度、大小、查询频率等
		Density   float64
		Size      int
		QueryFreq float64
	}

	distances := make([]clusterDistance, 0, len(db.ivfIndex.ClusterCentroids))

	for i, centroid := range db.ivfIndex.ClusterCentroids {
		dist, err := algorithm.EuclideanDistanceSquared(query, centroid)
		if err != nil {
			logger.Warning("Error calculating distance to centroid %d for candidate selection: %v", i, err)
			continue // 跳过计算错误的质心
		}
		metrics := ClusterMetrics{} // 默认空指标
		if i < len(db.ivfIndex.ClusterMetrics) {
			metrics = db.ivfIndex.ClusterMetrics[i]
		}
		clusterSize := 0
		if i < len(db.ivfIndex.Clusters) {
			clusterSize = len(db.ivfIndex.Clusters[i].VectorIDs)
		}

		distances = append(distances, clusterDistance{
			ID:        i,
			Distance:  dist,
			Density:   metrics.Density,
			Size:      clusterSize,
			QueryFreq: metrics.QueryFrequency,
		})
	}

	if len(distances) == 0 {
		return []int{}
	}

	// 根据距离和其他智能因素对聚类进行排序
	sort.Slice(distances, func(i, j int) bool {
		// 主要按距离排序
		if distances[i].Distance != distances[j].Distance {
			return distances[i].Distance < distances[j].Distance
		}
		// 次要排序因素：可以考虑聚类密度（倾向于密度大的），或查询频率（倾向于常被查询的）
		// 示例：如果距离相同，优先选择密度更大的聚类
		// if distances[i].Density != distances[j].Density {
		// 	 return distances[i].Density > distances[j].Density
		// }
		// 示例：如果距离和密度都相同，优先选择规模适中的聚类（避免过小或过大，除非有特定策略）
		// idealSize := db.ivfConfig.MaxClusterSize / 2
		// diffI := math.Abs(float64(distances[i].Size - idealSize))
		// diffJ := math.Abs(float64(distances[j].Size - idealSize))
		// if diffI != diffJ {
		// 	 return diffI < diffJ
		// }
		return distances[i].ID < distances[j].ID // 最终通过ID保证排序稳定性
	})

	selectedClusterIDs := make([]int, 0, nprobe)
	for i := 0; i < len(distances) && i < nprobe; i++ {
		selectedClusterIDs = append(selectedClusterIDs, distances[i].ID)
		// 更新被选中聚类的访问统计（如果需要）
		if distances[i].ID < len(db.ivfIndex.Clusters) {
			// 加锁操作，因为可能会有并发写
			// db.ivfIndex.mu.Lock() // 这可能导致死锁，因为外层已经有RLock
			// 考虑将更新操作移到其他地方，或者使用更细粒度的锁
			// db.ivfIndex.Clusters[distances[i].ID].LastAccessed = time.Now()
			// db.ivfIndex.Clusters[distances[i].ID].AccessCount++
			// db.ivfIndex.mu.Unlock()
		}
		if distances[i].ID < len(db.ivfIndex.ClusterMetrics) {
			// db.ivfIndex.ClusterMetrics[distances[i].ID].QueryFrequency++ // 同上，注意并发安全
		}
	}

	logger.Trace("Selected %d candidate clusters for query: %v", len(selectedClusterIDs), selectedClusterIDs)
	return selectedClusterIDs
}

// CalculateAdaptiveNprobe 根据查询向量和当前聚类状态动态调整 nprobe 值
func (db *VectorDB) CalculateAdaptiveNprobe(query []float64, baseNprobe int) int {
	if db.ivfIndex == nil || len(db.ivfIndex.Clusters) == 0 {
		return baseNprobe
	}

	// 1. 基础检查和限制
	if baseNprobe <= 0 {
		baseNprobe = 1
	}
	if baseNprobe > len(db.ivfIndex.Clusters) {
		baseNprobe = len(db.ivfIndex.Clusters)
	}

	// 2. 计算查询向量到所有聚类中心的距离
	distances := make([]struct {
		clusterID int
		distance  float64
	}, len(db.ivfIndex.ClusterCentroids))

	for i, centroid := range db.ivfIndex.ClusterCentroids {
		dist, err := algorithm.EuclideanDistanceSquared(query, centroid)
		if err != nil {
			logger.Warning("Error calculating distance to centroid %d: %v", i, err)
			continue
		}
		distances[i] = struct {
			clusterID int
			distance  float64
		}{i, dist}
	}

	// 3. 根据距离分布调整 nprobe
	// 对距离进行排序
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	// 计算最近和最远的距离比率
	if len(distances) < 2 {
		return baseNprobe
	}
	minDist := distances[0].distance
	maxDist := distances[len(distances)-1].distance
	distanceRatio := minDist / maxDist

	// 4. 动态调整因子计算
	// 基于距离分布的调整因子
	distanceAdjustment := 1.0
	if distanceRatio < 0.1 { // 距离差异很大，可以减少 nprobe
		distanceAdjustment = 0.8
	} else if distanceRatio > 0.5 { // 距离差异小，需要增加 nprobe
		distanceAdjustment = 1.5
	}

	// 基于聚类大小的调整
	sizeAdjustment := 1.0
	largestClusterSize := 0
	for _, cluster := range db.ivfIndex.Clusters {
		if len(cluster.VectorIDs) > largestClusterSize {
			largestClusterSize = len(cluster.VectorIDs)
		}
	}
	if largestClusterSize > db.ivfConfig.MaxClusterSize/2 {
		// 如果有很大的聚类，增加 nprobe 以提高召回率
		sizeAdjustment = 1.3
	}

	// 基于查询频率的调整
	frequencyAdjustment := 1.0
	totalQueryFrequency := 0.0
	for _, metrics := range db.ivfIndex.ClusterMetrics {
		totalQueryFrequency += metrics.QueryFrequency
	}
	averageFrequency := totalQueryFrequency / float64(len(db.ivfIndex.ClusterMetrics))
	if averageFrequency > 0 {
		// 如果有频繁查询的聚类，适当增加 nprobe
		frequencyAdjustment = 1.2
	}

	// 5. 计算最终的 nprobe 值
	adjustedNprobe := int(float64(baseNprobe) * distanceAdjustment * sizeAdjustment * frequencyAdjustment)

	// 6. 确保 nprobe 在合理范围内
	minNprobe := 1
	maxNprobe := len(db.ivfIndex.Clusters)
	if adjustedNprobe < minNprobe {
		adjustedNprobe = minNprobe
	}
	if adjustedNprobe > maxNprobe {
		adjustedNprobe = maxNprobe
	}

	logger.Trace("Adaptive nprobe calculation: base=%d, distance_adj=%.2f, size_adj=%.2f, freq_adj=%.2f, final=%d",
		baseNprobe, distanceAdjustment, sizeAdjustment, frequencyAdjustment, adjustedNprobe)

	return adjustedNprobe
}
