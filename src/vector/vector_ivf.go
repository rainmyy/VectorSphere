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
	centroids, _, err := algorithm.KMeans(trainingVectors, config.NumClusters, 100, 0.001)
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
		clusterID := db.findNearestCluster(vector, centroids)
		enhancedClusters[clusterID].VectorIDs = append(enhancedClusters[clusterID].VectorIDs, id)
	}

	// 5. 构建 PQ 压缩（如果启用）
	if config.UsePQCompression {
		if err := db.buildIVFPQIndex(enhancedClusters, config); err != nil {
			log.Warning("构建 IVF-PQ 索引失败: %v", err)
		}
	}

	// 6. 计算聚类指标
	for i := range enhancedClusters {
		db.calculateClusterMetrics(&enhancedClusters[i])
	}

	// 7. 创建增强索引
	db.ivfIndex = &EnhancedIVFIndex{
		Clusters:         enhancedClusters,
		ClusterCentroids: centroids,
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
