package db

import (
	"container/heap"
	"encoding/gob"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"runtime"
	"seetaSearch/library/algorithm"
	"seetaSearch/library/entity"
	"seetaSearch/library/log"
	"seetaSearch/library/tree"
	"seetaSearch/library/util"
	"sort"
	"strings"
	"sync"
	"time"
)

// Cluster 代表一个向量簇
type Cluster struct {
	Centroid  algorithm.Point // 簇的中心点
	VectorIDs []string        // 属于该簇的向量ID列表
}

// 添加查询缓存结构
type queryCache struct {
	results   []string
	timestamp int64
}

// PerformanceStats 性能统计结构
type PerformanceStats struct {
	TotalQueries    int64
	CacheHits       int64
	AvgQueryTime    time.Duration
	IndexBuildTime  time.Duration
	LastReindexTime time.Time
	MemoryUsage     uint64
}

type VectorDB struct {
	vectors       map[string][]float64
	mu            sync.RWMutex
	filePath      string              // 数据库文件的存储路径
	clusters      []Cluster           // 存储簇信息，用于IVF索引
	numClusters   int                 // K-Means中的K值，即簇的数量
	indexed       bool                // 标记数据库是否已建立索引
	invertedIndex map[string][]string // 倒排索引，关键词 -> 文件ID列表
	// 优化2: 添加倒排索引锁，细化锁粒度
	invertedMu sync.RWMutex
	// 优化3: 添加查询缓存
	queryCache map[string]queryCache
	cacheMu    sync.RWMutex
	cacheTTL   int64 // 缓存有效期（秒）

	// 新增字段
	vectorDim         int                  // 向量维度，用于验证
	vectorizedType    int                  // 默认向量化类型
	normalizedVectors map[string][]float64 // 预计算的归一化向量

	compressedVectors map[string]entity.CompressedVector // 压缩后的向量
	useCompression    bool                               // 是否使用压缩
	stats             PerformanceStats
	statsMu           sync.RWMutex
	config            AdaptiveConfig
}

const (
	DefaultVectorized = iota
	SimpleVectorized
	TfidfVectorized
	WordEmbeddingVectorized
)

// GetStats 获取性能统计信息
func (db *VectorDB) GetStats() PerformanceStats {
	db.statsMu.RLock()
	defer db.statsMu.RUnlock()
	return db.stats
}

func (db *VectorDB) Close() {

}

func (db *VectorDB) IsIndexed() bool {
	return db.indexed
}

// NewVectorDB 创建一个新的 VectorDB 实例。
// 如果 filePath 非空且文件存在，则尝试从中加载数据。
// numClusters 指定了用于索引的簇数量，如果 <=0，则不启用索引功能。
func NewVectorDB(filePath string, numClusters int) *VectorDB {
	db := &VectorDB{
		vectors:           make(map[string][]float64),
		filePath:          filePath,
		numClusters:       numClusters,
		clusters:          make([]Cluster, 0),
		indexed:           false,
		invertedIndex:     make(map[string][]string),
		queryCache:        make(map[string]queryCache),
		cacheTTL:          300, // 默认缓存5分钟
		vectorDim:         0,
		vectorizedType:    DefaultVectorized,
		normalizedVectors: make(map[string][]float64),
		config:            AdaptiveConfig{},
	}
	if filePath != "" {
		if err := db.LoadFromFile(); err != nil {
			log.Warning("警告: 从 %s 加载向量数据库时出错: %v。将使用空数据库启动。\n", filePath, err)
			db.vectors = make(map[string][]float64)
			db.clusters = make([]Cluster, 0)
			db.indexed = false
			db.normalizedVectors = make(map[string][]float64)
		}
	}
	return db
}

func (db *VectorDB) GetVectors() map[string][]float64 {
	return db.vectors
}

// 优化7: 添加向量预处理函数
func (db *VectorDB) preprocessVector(id string, vector []float64) {
	// 更新向量维度
	if db.vectorDim == 0 && len(vector) > 0 {
		db.vectorDim = len(vector)
	}

	// 预计算并存储归一化向量
	db.normalizedVectors[id] = util.NormalizeVector(vector)
}

// AddDocument 添加文档并将其转换为向量后存入数据库
func (db *VectorDB) AddDocument(id string, doc string, vectorizedType int) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	vector, err := db.GetVectorForText(doc, vectorizedType) // Use GetVectorForText internally
	if err != nil {
		return fmt.Errorf("failed to vectorize document %s for AddDocument: %w", id, err)
	}

	// 设置向量维度（如果尚未设置）
	if db.vectorDim == 0 && len(vector) > 0 {
		db.vectorDim = len(vector)
	} else if db.vectorDim > 0 && len(vector) != db.vectorDim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d for document %s", db.vectorDim, len(vector), id)
	}

	// 将向量添加到数据库
	db.vectors[id] = vector
	// 预计算并存储归一化向量
	db.normalizedVectors[id] = util.NormalizeVector(vector)

	// 更新倒排索引
	db.invertedMu.Lock()
	words := strings.Fields(doc)
	for _, word := range words {
		if _, exists := db.invertedIndex[word]; !exists {
			db.invertedIndex[word] = make([]string, 0, 1)
		}
		// 检查ID是否已存在，避免重复
		found := false
		for _, existingID := range db.invertedIndex[word] {
			if existingID == id {
				found = true
				break
			}
		}
		if !found {
			db.invertedIndex[word] = append(db.invertedIndex[word], id)
		}
	}
	db.invertedMu.Unlock()

	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		log.Info("提示: 添加新文档向量后，索引已失效，请重新调用 BuildIndex()。")
	}
	return nil
}

// GetVectorForText 将文本根据指定的向量化类型转换为向量
func (db *VectorDB) GetVectorForText(text string, vectorizedType int) ([]float64, error) {
	var vectorized DocumentVectorized
	// 注意：WordEmbeddingVectorized 可能需要预加载词向量文件路径，这里暂时硬编码或假设已加载
	// 实际应用中，这个路径应该可配置
	switch vectorizedType {
	case TfidfVectorized:
		vectorized = TFIDFVectorized() // TFIDFVectorized 内部管理其状态，每次调用可能基于已处理文档更新
	case SimpleVectorized:
		vectorized = SimpleBagOfWordsVectorized()
	case WordEmbeddingVectorized:
		// 假设 LoadWordEmbeddings 在 NewVectorDB 或其他初始化阶段被调用并存储了 embeddings
		// 或者在这里按需加载，但这效率较低。更好的方式是 VectorDB 持有 embeddings。
		// For now, let's assume a path or that embeddings are globally available/configured.
		embeddings, err := LoadWordEmbeddings("path/to/pretrained_embeddings.txt") // Placeholder path
		if err != nil {
			return nil, fmt.Errorf("failed to load word embeddings for GetVectorForText: %w", err)
		}
		vectorized = EnhancedWordEmbeddingVectorized(embeddings) // Assuming EnhancedWordEmbeddingVectorized exists
	case DefaultVectorized:
		// Define what DefaultVectorized means, e.g., SimpleBagOfWordsVectorized
		vectorized = SimpleBagOfWordsVectorized()
	default:
		return nil, fmt.Errorf("unhandled vectorizedType: %d", vectorizedType)
	}

	if vectorized == nil {
		return nil, fmt.Errorf("vectorized function is nil for type: %d", vectorizedType)
	}

	vector, err := vectorized(text)
	if err != nil {
		return nil, fmt.Errorf("failed to vectorize text: %w", err)
	}
	return vector, nil
}

// Add 添加向量。如果已建立索引，则将索引标记为失效。
func (db *VectorDB) Add(id string, vector []float64) {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 设置向量维度（如果尚未设置）
	if db.vectorDim == 0 && len(vector) > 0 {
		db.vectorDim = len(vector)
	} else if len(vector) != db.vectorDim && db.vectorDim > 0 {
		log.Fatal("向量维度不匹配: 期望 %d, 实际 %d", db.vectorDim, len(vector))
	}

	db.vectors[id] = vector
	// 预计算并存储归一化向量
	db.normalizedVectors[id] = util.NormalizeVector(vector)

	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		log.Info("提示: 添加新向量后，索引已失效，请重新调用 BuildIndex()。")
	}
}

// DeleteVector 从数据库中删除指定ID的向量
func (db *VectorDB) DeleteVector(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if _, exists := db.vectors[id]; !exists {
		return fmt.Errorf("vector with id %s not found for deletion", id)
	}
	delete(db.vectors, id)
	delete(db.normalizedVectors, id)
	// TODO: 如果使用了压缩，也需要删除压缩向量
	// delete(db.compressedVectors, id)

	// TODO: 如果使用了IVF索引 (db.indexed is true and db.clusters is populated),
	// 需要从相应的簇中移除该 vectorID。
	// 这部分逻辑会比较复杂，需要遍历 clusters 或维护一个反向映射。
	if db.indexed {
		for i := range db.clusters {
			newVectorIDs := make([]string, 0, len(db.clusters[i].VectorIDs))
			for _, vecID := range db.clusters[i].VectorIDs {
				if vecID != id {
					newVectorIDs = append(newVectorIDs, vecID)
				}
			}
			db.clusters[i].VectorIDs = newVectorIDs
		}
	}

	// 清除查询缓存，因为数据已更改
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	log.Info("Vector with id %s deleted successfully.", id)
	return nil
}

// UpdateIndexIncrementally 增量更新索引
func (db *VectorDB) UpdateIndexIncrementally(id string, vector []float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 如果索引未构建，则不需要增量更新
	if !db.indexed || len(db.clusters) == 0 {
		return nil
	}

	// 找到最近的簇
	minDist := math.MaxFloat64
	nearestCluster := -1

	for i, cluster := range db.clusters {
		dist, err := algorithm.EuclideanDistanceSquared(vector, cluster.Centroid)
		if err != nil {
			continue
		}
		if dist < minDist {
			minDist = dist
			nearestCluster = i
		}
	}

	if nearestCluster >= 0 {
		// 将向量添加到最近的簇
		db.clusters[nearestCluster].VectorIDs = append(
			db.clusters[nearestCluster].VectorIDs, id)

		// 可选：更新簇中心（可能需要定期执行而不是每次都更新）
		// ...
	}

	return nil
}

// Get 获取向量
func (db *VectorDB) Get(id string) ([]float64, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	vec, exists := db.vectors[id]
	return vec, exists
}

// Update 更新向量。如果已建立索引，则将索引标记为失效。
func (db *VectorDB) Update(id string, vector []float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if _, exists := db.vectors[id]; !exists {
		return fmt.Errorf("未找到ID为 '%s' 的向量", id)
	}

	// 检查向量维度
	if len(vector) != db.vectorDim && db.vectorDim > 0 {
		return fmt.Errorf("向量维度不匹配: 期望 %d, 实际 %d", db.vectorDim, len(vector))
	}

	db.vectors[id] = vector
	// 更新归一化向量
	db.normalizedVectors[id] = util.NormalizeVector(vector)

	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		log.Info("提示: 更新向量后，索引已失效，请重新调用 BuildIndex()。")
	}
	return nil
}

// Delete 删除向量。如果已建立索引，则将索引标记为失效。
func (db *VectorDB) Delete(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if _, exists := db.vectors[id]; !exists {
		return fmt.Errorf("未找到ID为 '%s' 的向量", id)
	}
	delete(db.vectors, id)
	delete(db.normalizedVectors, id)

	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	// 从倒排索引中删除
	db.invertedMu.Lock()
	for word, ids := range db.invertedIndex {
		newIDs := make([]string, 0, len(ids))
		for _, existingID := range ids {
			if existingID != id {
				newIDs = append(newIDs, existingID)
			}
		}
		if len(newIDs) == 0 {
			delete(db.invertedIndex, word)
		} else {
			db.invertedIndex[word] = newIDs
		}
	}
	db.invertedMu.Unlock()

	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		log.Info("提示: 删除向量后，索引已失效，请重新调用 BuildIndex()。")
	}
	return nil
}

// MultiLevelIndex 多级索引结构
type MultiLevelIndex struct {
	// 一级索引：簇中心
	clusters []Cluster

	// 二级索引：每个簇内部的KD树或其他数据结构
	subIndices []interface{} // 可以是KDTree或其他索引结构

	// 索引元数据
	numClusters int
	indexed     bool
	buildTime   time.Time
}

// BuildMultiLevelIndex 在BuildIndex方法中构建多级索引
func (db *VectorDB) BuildMultiLevelIndex(maxIterations int, tolerance float64) error {
	// 检查参数有效性
	if db.numClusters <= 0 {
		return fmt.Errorf("未配置有效的簇数量 (numClusters: %d)，无法构建索引", db.numClusters)
	}

	if len(db.vectors) < db.numClusters {
		db.indexed = false
		return fmt.Errorf("向量数量 (%d) 少于簇数量 (%d)，无法构建有效索引", len(db.vectors), db.numClusters)
	}

	fmt.Println("开始构建多级索引...")
	startTime := time.Now()

	// 1. 收集所有向量及其ID
	var allVectorsData []algorithm.Point
	var vectorIDs []string // 保持与allVectorsData顺序一致的ID
	for id, vec := range db.vectors {
		allVectorsData = append(allVectorsData, algorithm.Point(vec))
		vectorIDs = append(vectorIDs, id)
	}

	// 2. 第一级：构建IVF索引（调用KMeans算法）
	centroids, assignments, err := algorithm.KMeans(allVectorsData, db.numClusters, maxIterations, tolerance)
	if err != nil {
		return fmt.Errorf("KMeans聚类失败: %w", err)
	}

	// 3. 根据KMeans结果填充db.clusters
	db.clusters = make([]Cluster, db.numClusters)
	for i := 0; i < db.numClusters; i++ {
		db.clusters[i] = Cluster{
			Centroid:  centroids[i],
			VectorIDs: make([]string, 0),
		}
	}

	for i, clusterIndex := range assignments {
		if clusterIndex >= 0 && clusterIndex < db.numClusters { // 确保索引有效
			db.clusters[clusterIndex].VectorIDs = append(db.clusters[clusterIndex].VectorIDs, vectorIDs[i])
		} else {
			fmt.Printf("警告: 向量 %s 被分配到无效的簇索引 %d\n", vectorIDs[i], clusterIndex)
		}
	}

	// 4. 第二级：为每个簇构建KD树子索引
	multiIndex := MultiLevelIndex{
		clusters:    db.clusters,
		subIndices:  make([]interface{}, db.numClusters),
		numClusters: db.numClusters,
		indexed:     true,
		buildTime:   time.Now(),
	}

	// 并行构建每个簇的KD树
	var wg sync.WaitGroup
	for i := range db.clusters {
		wg.Add(1)
		go func(clusterIdx int) {
			defer wg.Done()

			// 跳过空簇
			if len(db.clusters[clusterIdx].VectorIDs) == 0 {
				return
			}

			// 创建KD树
			tree := tree.NewKDTree(db.vectorDim)

			// 将簇内所有向量插入KD树
			for _, id := range db.clusters[clusterIdx].VectorIDs {
				vec, exists := db.vectors[id]
				if exists {
					tree.Insert(vec, id)
				}
			}

			// 保存KD树到多级索引
			multiIndex.subIndices[clusterIdx] = tree
		}(i)
	}

	// 等待所有KD树构建完成
	wg.Wait()

	// 更新索引状态
	db.indexed = true

	// 更新性能统计
	db.statsMu.Lock()
	db.stats.IndexBuildTime = time.Since(startTime)
	db.stats.LastReindexTime = time.Now()

	// 估算内存使用
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	db.stats.MemoryUsage = memStats.Alloc
	db.statsMu.Unlock()

	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	fmt.Printf("多级索引构建完成，共 %d 个簇，耗时 %v\n", db.numClusters, time.Since(startTime))
	return nil
}

// BuildIndex 使用K-Means算法为数据库中的向量构建索引。
// maxIterations: K-Means的最大迭代次数。
// tolerance: K-Means的收敛容忍度。
func (db *VectorDB) BuildIndex(maxIterations int, tolerance float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 检查是否启用了多级索引
	if db.config.UseMultiLevelIndex {
		// 使用多级索引构建
		return db.BuildMultiLevelIndex(maxIterations, tolerance)
	}

	// 以下是原有的单级索引构建逻辑
	if db.numClusters <= 0 {
		return fmt.Errorf("未配置有效的簇数量 (numClusters: %d)，无法构建索引", db.numClusters)
	}

	if len(db.vectors) < db.numClusters {
		db.indexed = false
		return fmt.Errorf("向量数量 (%d) 少于簇数量 (%d)，无法构建有效索引", len(db.vectors), db.numClusters)
	}

	log.Info("开始构建索引...")
	// 1. 收集所有向量及其ID
	var allVectorsData []algorithm.Point
	var vectorIDs []string // 保持与allVectorsData顺序一致的ID
	for id, vec := range db.vectors {
		allVectorsData = append(allVectorsData, algorithm.Point(vec))
		vectorIDs = append(vectorIDs, id)
	}

	// 2. 调用KMeans算法
	centroids, assignments, err := algorithm.KMeans(allVectorsData, db.numClusters, maxIterations, tolerance)
	if err != nil {
		return fmt.Errorf("KMeans聚类失败: %w", err)
	}

	// 3. 根据KMeans结果填充db.clusters
	db.clusters = make([]Cluster, db.numClusters)
	for i := 0; i < db.numClusters; i++ {
		db.clusters[i] = Cluster{
			Centroid:  centroids[i],
			VectorIDs: make([]string, 0),
		}
	}

	for i, clusterIndex := range assignments {
		if clusterIndex >= 0 && clusterIndex < db.numClusters { // 确保索引有效
			db.clusters[clusterIndex].VectorIDs = append(db.clusters[clusterIndex].VectorIDs, vectorIDs[i])
		} else {
			log.Warning("警告: 向量 %s 被分配到无效的簇索引 %d\n", vectorIDs[i], clusterIndex)
		}
	}

	db.indexed = true
	// 清除查询缓存
	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache)
	db.cacheMu.Unlock()

	log.Info("索引构建完成，共 %d 个簇。\n", db.numClusters)
	return nil
}

// dataToSave 结构用于 gob 编码，包含所有需要持久化的字段
type dataToSave struct {
	Vectors           map[string][]float64
	Clusters          []Cluster
	NumClusters       int
	Indexed           bool
	InvertedIndex     map[string][]string
	VectorDim         int
	NormalizedVectors map[string][]float64
}

// SaveToFile 将当前数据库状态（包括索引）保存到其配置的文件中。
func (db *VectorDB) SaveToFile() error {
	if db.filePath == "" {
		return fmt.Errorf("此 VectorDB 实例未配置 filePath")
	}
	db.mu.RLock()
	db.invertedMu.RLock()
	defer db.mu.RUnlock()
	defer db.invertedMu.RUnlock()

	file, err := os.Create(db.filePath)
	if err != nil {
		return fmt.Errorf("创建文件 %s 失败: %w", db.filePath, err)
	}
	defer file.Close()

	data := dataToSave{
		Vectors:           db.vectors,
		Clusters:          db.clusters,
		NumClusters:       db.numClusters,
		Indexed:           db.indexed,
		InvertedIndex:     db.invertedIndex,
		VectorDim:         db.vectorDim,
		NormalizedVectors: db.normalizedVectors,
	}

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("将数据编码到文件 %s 失败: %w", db.filePath, err)
	}
	return nil
}

// LoadFromFile 从其配置的文件中加载数据库状态（包括索引）。
func (db *VectorDB) LoadFromFile() error {
	if db.filePath == "" {
		return fmt.Errorf("此 VectorDB 实例未配置 filePath")
	}
	db.mu.Lock()
	db.invertedMu.Lock()
	defer db.mu.Unlock()
	defer db.invertedMu.Unlock()

	file, err := os.Open(db.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // 文件未找到对于初始加载不是错误
		}
		return fmt.Errorf("打开文件 %s 失败: %w", db.filePath, err)
	}
	defer file.Close()

	var loadedData dataToSave
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&loadedData); err != nil {
		return fmt.Errorf("从文件 %s 解码数据失败: %w", db.filePath, err)
	}

	db.vectors = loadedData.Vectors
	db.clusters = loadedData.Clusters
	db.numClusters = loadedData.NumClusters
	db.indexed = loadedData.Indexed
	db.invertedIndex = loadedData.InvertedIndex
	db.vectorDim = loadedData.VectorDim
	db.normalizedVectors = loadedData.NormalizedVectors

	// 确保 map 和 slice 在加载后不是 nil，即使它们是空的
	if db.vectors == nil {
		db.vectors = make(map[string][]float64)
	}
	if db.clusters == nil {
		db.clusters = make([]Cluster, 0)
	}
	if db.invertedIndex == nil {
		db.invertedIndex = make(map[string][]string)
	}
	if db.normalizedVectors == nil {
		db.normalizedVectors = make(map[string][]float64)
		// 如果没有加载到归一化向量，则重新计算
		for id, vec := range db.vectors {
			db.normalizedVectors[id] = util.NormalizeVector(vec)
		}
	}

	log.Info("从 %s 加载数据库成功。索引状态: %t, 簇数量: %d\n", db.filePath, db.indexed, db.numClusters)
	return nil
}

// FindNearest 优化后的FindNearest方法
func (db *VectorDB) FindNearest(query []float64, k int, nprobe int) ([]string, error) {
	startTime := time.Now()
	// 更新查询计数
	db.statsMu.Lock()
	db.stats.TotalQueries++
	db.statsMu.Unlock()
	// 检查缓存
	cacheKey := util.GenerateCacheKey(query, k, nprobe, 0)
	db.cacheMu.RLock()
	if cache, exists := db.queryCache[cacheKey]; exists {
		db.cacheMu.RUnlock()

		// 更新缓存命中统计
		db.statsMu.Lock()
		db.stats.CacheHits++
		db.statsMu.Unlock()

		return cache.results, nil
	}
	db.cacheMu.RUnlock()

	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		return []string{}, nil
	}

	// 预处理查询向量 - 归一化可以提高相似度计算的准确性
	normalizedQuery := util.NormalizeVector(query)
	var ids []string
	if db.indexed && len(db.clusters) > 0 && db.numClusters > 0 {
		// --- 使用IVF索引进行搜索 ---
		if nprobe <= 0 {
			nprobe = 1 // 默认搜索最近的一个簇
		}
		if nprobe > db.numClusters {
			nprobe = db.numClusters // 不能超过总簇数
		}

		// 使用堆结构来维护最近的nprobe个簇，避免全排序
		centroidHeap := make(entity.CentroidHeap, 0, db.numClusters)

		// 1. 找到查询向量最近的 nprobe 个簇中心
		for i, cluster := range db.clusters {
			distSq, err := algorithm.EuclideanDistanceSquared(normalizedQuery, cluster.Centroid)
			if err != nil {
				continue
			}
			centroidHeap = append(centroidHeap, entity.CentroidDist{Index: i, Distance: distSq})
		}

		// 堆化并提取最近的nprobe个簇
		heap.Init(&centroidHeap)
		var nearestClusters []int
		for i := 0; i < nprobe && len(centroidHeap) > 0; i++ {
			nearestClusters = append(nearestClusters, heap.Pop(&centroidHeap).(entity.CentroidDist).Index)
		}

		// 使用优先队列维护k个最近的向量
		resultHeap := make(entity.ResultHeap, 0, k)
		heap.Init(&resultHeap)

		// 2. 在这 nprobe 个簇中搜索向量
		for _, clusterIndex := range nearestClusters {
			selectedCluster := db.clusters[clusterIndex]

			// 批量处理每个簇中的向量，减少循环开销
			batchSize := 100
			for i := 0; i < len(selectedCluster.VectorIDs); i += batchSize {
				end := i + batchSize
				if end > len(selectedCluster.VectorIDs) {
					end = len(selectedCluster.VectorIDs)
				}

				// 并行处理每批向量
				var wg sync.WaitGroup
				resultChan := make(chan entity.Result, end-i)

				for j := i; j < end; j++ {
					wg.Add(1)
					go func(vecID string) {
						defer wg.Done()
						vec, exists := db.vectors[vecID]
						if !exists {
							return
						}
						// 使用余弦相似度代替欧几里得距离，更适合高维向量
						sim := util.CosineSimilarity(normalizedQuery, vec)
						resultChan <- entity.Result{Id: vecID, Similarity: sim}
					}(selectedCluster.VectorIDs[j])
				}

				// 等待所有goroutine完成
				go func() {
					wg.Wait()
					close(resultChan)
				}()

				// 收集结果并维护最大堆
				for result := range resultChan {
					if len(resultHeap) < k {
						heap.Push(&resultHeap, result)
					} else if result.Similarity > resultHeap[0].Similarity {
						heap.Pop(&resultHeap)
						heap.Push(&resultHeap, result)
					}
				}
			}
		}

		// 3. 从堆中提取结果，按相似度降序排列
		ids = make([]string, 0, len(resultHeap))
		for resultHeap.Len() > 0 {
			ids = append([]string{heap.Pop(&resultHeap).(entity.Result).Id}, ids...)
		}

	} else {
		// --- 回退到暴力搜索，但使用优化的实现 ---
		resultHeap := make(entity.ResultHeap, 0, k)
		heap.Init(&resultHeap)

		// 使用工作池并行处理向量比较
		numWorkers := runtime.NumCPU()
		workChan := make(chan string, len(db.vectors))
		resultChan := make(chan entity.Result, len(db.vectors))

		// 启动工作协程
		var wg sync.WaitGroup
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for id := range workChan {
					vec := db.vectors[id]
					sim := util.CosineSimilarity(normalizedQuery, vec)
					resultChan <- entity.Result{Id: id, Similarity: sim}
				}
			}()
		}

		// 发送工作
		go func() {
			for id := range db.vectors {
				workChan <- id
			}
			close(workChan)
		}()

		// 等待所有工作完成并关闭结果通道
		go func() {
			wg.Wait()
			close(resultChan)
		}()

		// 收集结果
		for result := range resultChan {
			if len(resultHeap) < k {
				heap.Push(&resultHeap, result)
			} else if result.Similarity > resultHeap[0].Similarity {
				heap.Pop(&resultHeap)
				heap.Push(&resultHeap, result)
			}
		}

		// 从堆中提取结果，按相似度降序排列
		ids = make([]string, 0, len(resultHeap))
		for resultHeap.Len() > 0 {
			ids = append([]string{heap.Pop(&resultHeap).(entity.Result).Id}, ids...)
		}
	}
	// 更新平均查询时间统计
	queryTime := time.Since(startTime)
	db.statsMu.Lock()
	// 使用指数移动平均更新平均查询时间
	alpha := 0.1 // 平滑因子
	db.stats.AvgQueryTime = time.Duration(float64(db.stats.AvgQueryTime)*(1-alpha) + float64(queryTime)*alpha)
	db.statsMu.Unlock()
	return ids, nil
}

// FindNearestWithScores 查找最近的k个向量，并返回它们的ID和相似度分数
func (db *VectorDB) FindNearestWithScores(query []float64, k int, nprobe int) ([]entity.Result, error) {
	startTime := time.Now()
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}
	if len(db.vectors) == 0 {
		return []entity.Result{}, nil
	}
	normalizedQuery := util.NormalizeVector(query)
	var results []entity.Result
	if db.indexed && len(db.clusters) > 0 && db.numClusters > 0 {
		if nprobe <= 0 {
			nprobe = 1
		}
		if nprobe > db.numClusters {
			nprobe = db.numClusters
		}
		centroidHeap := make(entity.CentroidHeap, 0, db.numClusters)
		for i, cluster := range db.clusters {
			distSq, err := algorithm.EuclideanDistanceSquared(normalizedQuery, cluster.Centroid)
			if err != nil {
				continue
			}
			centroidHeap = append(centroidHeap, entity.CentroidDist{Index: i, Distance: distSq})
		}
		heap.Init(&centroidHeap)
		var nearestClusters []int
		for i := 0; i < nprobe && len(centroidHeap) > 0; i++ {
			nearestClusters = append(nearestClusters, heap.Pop(&centroidHeap).(entity.CentroidDist).Index)
		}
		resultHeap := make(entity.ResultHeap, 0, k)
		heap.Init(&resultHeap)
		for _, clusterIndex := range nearestClusters {
			selectedCluster := db.clusters[clusterIndex]
			batchSize := 100
			for i := 0; i < len(selectedCluster.VectorIDs); i += batchSize {
				end := i + batchSize
				if end > len(selectedCluster.VectorIDs) {
					end = len(selectedCluster.VectorIDs)
				}
				var wg sync.WaitGroup
				resultChan := make(chan entity.Result, end-i)
				for j := i; j < end; j++ {
					wg.Add(1)
					go func(vecID string) {
						defer wg.Done()
						vec, exists := db.vectors[vecID]
						if !exists {
							return
						}
						sim := util.CosineSimilarity(normalizedQuery, vec)
						resultChan <- entity.Result{Id: vecID, Similarity: sim}
					}(selectedCluster.VectorIDs[j])
				}
				go func() {
					wg.Wait()
					close(resultChan)
				}()
				for res := range resultChan {
					if len(resultHeap) < k {
						heap.Push(&resultHeap, res)
					} else if res.Similarity > resultHeap[0].Similarity {
						heap.Pop(&resultHeap)
						heap.Push(&resultHeap, res)
					}
				}
			}
		}
		results = make([]entity.Result, len(resultHeap))
		for i := len(resultHeap) - 1; i >= 0; i-- {
			results[i] = heap.Pop(&resultHeap).(entity.Result)
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].Similarity > results[j].Similarity
		})
	} else {
		resultHeap := make(entity.ResultHeap, 0, k)
		heap.Init(&resultHeap)
		numWorkers := runtime.NumCPU()
		workChan := make(chan string, len(db.vectors))
		innerResultChan := make(chan entity.Result, len(db.vectors))
		var wg sync.WaitGroup
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for id := range workChan {
					vec := db.vectors[id]
					sim := util.CosineSimilarity(normalizedQuery, vec)
					innerResultChan <- entity.Result{Id: id, Similarity: sim}
				}
			}()
		}
		go func() {
			for id := range db.vectors {
				workChan <- id
			}
			close(workChan)
		}()
		go func() {
			wg.Wait()
			close(innerResultChan)
		}()
		for res := range innerResultChan {
			if len(resultHeap) < k {
				heap.Push(&resultHeap, res)
			} else if res.Similarity > resultHeap[0].Similarity {
				heap.Pop(&resultHeap)
				heap.Push(&resultHeap, res)
			}
		}
		results = make([]entity.Result, len(resultHeap))
		for i := len(resultHeap) - 1; i >= 0; i-- {
			results[i] = heap.Pop(&resultHeap).(entity.Result)
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].Similarity > results[j].Similarity
		})
	}
	queryTime := time.Since(startTime)
	db.statsMu.Lock()
	alpha := 0.1
	db.stats.AvgQueryTime = time.Duration(float64(db.stats.AvgQueryTime)*(1-alpha) + float64(queryTime)*alpha)
	db.statsMu.Unlock()
	return results, nil
}

// AdaptiveConfig 自适应配置结构
type AdaptiveConfig struct {
	// 索引参数
	NumClusters           int     // 簇数量
	IndexRebuildThreshold float64 // 更新比例阈值，超过此值重建索引

	// 查询参数
	DefaultNprobe int           // 默认探测簇数量
	CacheTimeout  time.Duration // 缓存超时时间

	// 系统参数
	MaxWorkers         int  // 最大工作协程数
	VectorCompression  bool // 是否启用向量压缩
	UseMultiLevelIndex bool // 是否使用多级索引
}

// AdjustConfig 自适应配置调整
func (db *VectorDB) AdjustConfig() {
	db.mu.RLock()
	vectorCount := len(db.vectors)
	db.mu.RUnlock()

	config := db.config

	// 根据向量数量调整簇数量
	if vectorCount > 1000000 {
		config.NumClusters = 1000
	} else if vectorCount > 100000 {
		config.NumClusters = 100
	} else if vectorCount > 10000 {
		config.NumClusters = 50
	} else {
		config.NumClusters = 10
	}

	// 根据系统资源调整工作协程数
	config.MaxWorkers = runtime.NumCPU()

	// 根据内存使用情况决定是否启用向量压缩
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	if m.Alloc > 1024*1024*1024 { // 如果内存使用超过1GB
		config.VectorCompression = true
	}

	db.mu.Lock()
	db.config = config
	db.mu.Unlock()
}

// HybridSearch 混合搜索策略
func (db *VectorDB) HybridSearch(query []float64, k int, options SearchOptions) ([]entity.Result, error) {
	// 根据向量维度和数据规模自动选择最佳搜索策略
	if len(db.vectors) < 1000 || !db.indexed {
		// 小数据集使用暴力搜索
		return db.bruteForceSearch(query, k)
	} else if len(query) > 1000 {
		// 高维向量使用LSH (Locality-Sensitive Hashing)
		return db.lshSearch(query, k, options.NumHashTables)
	} else {
		// 默认使用IVF索引
		return db.ivfSearch(query, k, options.Nprobe)
	}
}

// bruteForceSearch 实现暴力搜索策略
// 适用于小数据集或索引未构建的情况
func (db *VectorDB) bruteForceSearch(query []float64, k int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 参数验证
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("vector is nil")
	}

	// 预处理查询向量 - 归一化可以提高相似度计算的准确性
	normalizedQuery := util.NormalizeVector(query)

	// 使用优先队列维护k个最近的向量
	resultHeap := make(entity.ResultHeap, 0, k)
	heap.Init(&resultHeap)

	// 使用工作池并行处理向量比较
	numWorkers := runtime.NumCPU()
	workChan := make(chan string, len(db.vectors))
	resultChan := make(chan entity.Result, len(db.vectors))

	// 启动工作协程
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for id := range workChan {
				vec := db.vectors[id]
				// 使用余弦相似度计算
				sim := util.CosineSimilarity(normalizedQuery, vec)
				resultChan <- entity.Result{Id: id, Similarity: sim}
			}
		}()
	}

	// 发送工作
	go func() {
		for id := range db.vectors {
			workChan <- id
		}
		close(workChan)
	}()

	// 等待所有工作完成并关闭结果通道
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 收集结果
	for result := range resultChan {
		if len(resultHeap) < k {
			heap.Push(&resultHeap, result)
		} else if result.Similarity > resultHeap[0].Similarity {
			heap.Pop(&resultHeap)
			heap.Push(&resultHeap, result)
		}
	}

	// 从堆中提取结果，按相似度降序排列
	ids := make([]entity.Result, 0, len(resultHeap))
	for resultHeap.Len() > 0 {
		ids = append([]entity.Result{heap.Pop(&resultHeap).(entity.Result)}, ids...)
	}

	return ids, nil
}

// LSHTable 局部敏感哈希表结构
type LSHTable struct {
	HashFunctions [][]float64         // 哈希函数参数
	Buckets       map[uint64][]string // 哈希桶，存储向量ID
}

// lshSearch 实现局部敏感哈希搜索
// 适用于高维向量搜索
func (db *VectorDB) lshSearch(query []float64, k int, numTables int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 参数验证
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("vector is nil")
	}

	// 如果LSH索引未构建，则动态构建
	lshTables, err := db.buildLSHIndex(numTables)
	if err != nil {
		// 如果构建LSH索引失败，回退到暴力搜索
		log.Error("构建LSH索引失败: %v，回退到暴力搜索\n", err)
		return db.bruteForceSearch(query, k)
	}

	// 预处理查询向量
	normalizedQuery := util.NormalizeVector(query)

	// 使用LSH索引进行搜索
	candidateSet := make(map[string]struct{})

	// 对每个哈希表进行查询
	for _, table := range lshTables {
		// 计算查询向量的哈希值
		hashValue := db.computeLSHHash(normalizedQuery, table.HashFunctions)

		// 获取对应哈希桶中的向量ID
		if ids, exists := table.Buckets[hashValue]; exists {
			for _, id := range ids {
				candidateSet[id] = struct{}{}
			}
		}
	}

	// 如果候选集为空，回退到暴力搜索
	if len(candidateSet) == 0 {
		return db.bruteForceSearch(query, k)
	}

	// 对候选集中的向量计算精确距离
	resultHeap := make(entity.ResultHeap, 0, k)
	heap.Init(&resultHeap)

	for id := range candidateSet {
		vec, exists := db.vectors[id]
		if !exists {
			continue
		}

		// 计算余弦相似度
		sim := util.CosineSimilarity(normalizedQuery, vec)

		if len(resultHeap) < k {
			heap.Push(&resultHeap, entity.Result{Id: id, Similarity: sim})
		} else if sim > resultHeap[0].Similarity {
			heap.Pop(&resultHeap)
			heap.Push(&resultHeap, entity.Result{Id: id, Similarity: sim})
		}
	}

	// 从堆中提取结果，按相似度降序排列
	ids := make([]entity.Result, 0, len(resultHeap))
	for resultHeap.Len() > 0 {
		ids = append([]entity.Result{heap.Pop(&resultHeap).(entity.Result)}, ids...)
	}

	return ids, nil
}

// buildLSHIndex 构建LSH索引
func (db *VectorDB) buildLSHIndex(numTables int) ([]LSHTable, error) {
	if numTables <= 0 {
		numTables = 5 // 默认使用5个哈希表
	}

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("数据库为空，无法构建LSH索引")
	}

	// 获取向量维度
	var dim int
	for _, vec := range db.vectors {
		dim = len(vec)
		break
	}

	// 创建LSH表
	tables := make([]LSHTable, numTables)

	// 为每个表生成随机哈希函数
	for i := range tables {
		// 每个表使用8个哈希函数
		numHashFunctions := 8
		hashFunctions := make([][]float64, numHashFunctions)

		// 生成随机哈希函数参数
		for j := range hashFunctions {
			hashFunctions[j] = make([]float64, dim)
			for k := range hashFunctions[j] {
				// 使用标准正态分布生成随机向量
				hashFunctions[j][k] = rand.NormFloat64()
			}
		}

		// 初始化哈希表
		tables[i] = LSHTable{
			HashFunctions: hashFunctions,
			Buckets:       make(map[uint64][]string),
		}

		// 将所有向量添加到哈希表中
		for id, vec := range db.vectors {
			hashValue := db.computeLSHHash(vec, hashFunctions)
			tables[i].Buckets[hashValue] = append(tables[i].Buckets[hashValue], id)
		}
	}

	return tables, nil
}

// computeLSHHash 计算向量的LSH哈希值
func (db *VectorDB) computeLSHHash(vec []float64, hashFunctions [][]float64) uint64 {
	// 使用随机超平面哈希
	hashBits := make([]uint64, len(hashFunctions))

	// 计算每个哈希函数的结果
	for i, hashFunc := range hashFunctions {
		// 计算向量与哈希函数的点积
		dotProduct := 0.0
		for j := range vec {
			if j < len(hashFunc) { // 确保索引有效
				dotProduct += vec[j] * hashFunc[j]
			}
		}

		// 如果点积大于0，哈希位为1，否则为0
		if dotProduct > 0 {
			hashBits[i] = 1
		}
	}

	// 将哈希位组合成一个64位整数
	var hashValue uint64
	for i, bit := range hashBits {
		if i < 64 { // 最多使用64位
			hashValue |= bit << i
		}
	}

	return hashValue
}

// ivfSearch 实现倒排文件索引搜索
// 适用于已构建索引的一般情况
func (db *VectorDB) ivfSearch(query []float64, k int, nprobe int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 参数验证
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("vector is nil")
	}

	// 检查索引状态
	if !db.indexed || len(db.clusters) == 0 || db.numClusters <= 0 {
		// 如果索引未构建，回退到暴力搜索
		log.Warning("索引未构建，回退到暴力搜索")
		return db.bruteForceSearch(query, k)
	}

	// 预处理查询向量
	normalizedQuery := util.NormalizeVector(query)

	// 设置默认nprobe值
	if nprobe <= 0 {
		nprobe = 1 // 默认搜索最近的一个簇
	}
	if nprobe > db.numClusters {
		nprobe = db.numClusters // 不能超过总簇数
	}

	// 使用堆结构来维护最近的nprobe个簇
	centroidHeap := make(entity.CentroidHeap, 0, db.numClusters)

	// 找到查询向量最近的nprobe个簇中心
	for i, cluster := range db.clusters {
		distSq, err := algorithm.EuclideanDistanceSquared(normalizedQuery, cluster.Centroid)
		if err != nil {
			continue
		}
		centroidHeap = append(centroidHeap, entity.CentroidDist{Index: i, Distance: distSq})
	}

	// 堆化并提取最近的nprobe个簇
	heap.Init(&centroidHeap)
	var nearestClusters []int
	for i := 0; i < nprobe && len(centroidHeap) > 0; i++ {
		nearestClusters = append(nearestClusters, heap.Pop(&centroidHeap).(entity.CentroidDist).Index)
	}

	// 使用优先队列维护k个最近的向量
	resultHeap := make(entity.ResultHeap, 0, k)
	heap.Init(&resultHeap)

	// 在这nprobe个簇中搜索向量
	for _, clusterIndex := range nearestClusters {
		selectedCluster := db.clusters[clusterIndex]

		// 批量处理每个簇中的向量
		batchSize := 100
		for i := 0; i < len(selectedCluster.VectorIDs); i += batchSize {
			end := i + batchSize
			if end > len(selectedCluster.VectorIDs) {
				end = len(selectedCluster.VectorIDs)
			}

			// 并行处理每批向量
			var wg sync.WaitGroup
			resultChan := make(chan entity.Result, end-i)

			for j := i; j < end; j++ {
				wg.Add(1)
				go func(vecID string) {
					defer wg.Done()
					vec, exists := db.vectors[vecID]
					if !exists {
						return
					}
					// 计算余弦相似度
					sim := util.CosineSimilarity(normalizedQuery, vec)
					resultChan <- entity.Result{Id: vecID, Similarity: sim}
				}(selectedCluster.VectorIDs[j])
			}

			// 等待所有goroutine完成
			go func() {
				wg.Wait()
				close(resultChan)
			}()

			// 收集结果并维护最大堆
			for result := range resultChan {
				if len(resultHeap) < k {
					heap.Push(&resultHeap, result)
				} else if result.Similarity > resultHeap[0].Similarity {
					heap.Pop(&resultHeap)
					heap.Push(&resultHeap, result)
				}
			}
		}
	}

	// 从堆中提取结果，按相似度降序排列
	ids := make([]entity.Result, 0, len(resultHeap))
	for resultHeap.Len() > 0 {
		ids = append([]entity.Result{heap.Pop(&resultHeap).(entity.Result)}, ids...)
	}

	return ids, nil
}

// FileSystemSearch 优化后的FileSystemSearch方法
func (db *VectorDB) FileSystemSearch(query string, vectorizedType int, k int, nprobe int) ([]string, error) {
	// 参数验证
	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	// 将查询转换为向量
	var vectorized DocumentVectorized
	switch vectorizedType {
	case TfidfVectorized:
		vectorized = TFIDFVectorized()
	case SimpleVectorized:
		vectorized = SimpleBagOfWordsVectorized()
	case WordEmbeddingVectorized:
		embeddings, err := LoadWordEmbeddings("path/to/pretrained_embeddings.txt")
		if err != nil {
			return nil, fmt.Errorf("加载词向量文件失败: %w", err)
		}
		vectorized = EnhancedWordEmbeddingVectorized(embeddings)
	default:
		return nil, fmt.Errorf("不支持的向量化类型: %d", vectorizedType)
	}

	queryVector, err := vectorized(query)
	if err != nil {
		return nil, fmt.Errorf("将查询转换为向量时出错: %w", err)
	}

	// 提取查询关键词，用于倒排索引过滤
	words := strings.Fields(query)
	if len(words) == 0 {
		return nil, fmt.Errorf("查询为空")
	}

	// 使用读锁保护并发访问
	db.mu.RLock()
	defer db.mu.RUnlock()

	// 首先使用倒排索引快速筛选候选集
	candidateSet := make(map[string]int)
	for _, word := range words {
		if docIDs, exists := db.invertedIndex[word]; exists {
			for _, id := range docIDs {
				candidateSet[id]++
			}
		}
	}

	// 如果候选集为空，直接使用向量搜索
	if len(candidateSet) == 0 {
		if nprobe > 0 {
			return db.FindNearest(queryVector, k, nprobe)
		} else {
			return db.ParallelFindNearest(queryVector, k)
		}
	}

	// 根据匹配关键词数量排序候选集
	type candidate struct {
		id    string
		count int
	}
	candidates := make([]candidate, 0, len(candidateSet))
	for id, count := range candidateSet {
		candidates = append(candidates, candidate{id, count})
	}

	// 按匹配关键词数量降序排序
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].count > candidates[j].count
	})

	// 限制候选集大小，只保留匹配度最高的前N个
	maxCandidates := 1000
	if len(candidates) > maxCandidates {
		candidates = candidates[:maxCandidates]
	}

	// 对筛选后的候选集进行向量相似度排序
	type result struct {
		id         string
		similarity float64
		wordCount  int
	}

	// 使用工作池并行计算相似度
	numWorkers := runtime.NumCPU()
	workChan := make(chan candidate, len(candidates))
	resultChan := make(chan result, len(candidates))

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range workChan {
				vec, exists := db.vectors[c.id]
				if !exists {
					continue
				}
				sim := util.CosineSimilarity(queryVector, vec)
				resultChan <- result{id: c.id, similarity: sim, wordCount: c.count}
			}
		}()
	}

	// 发送工作
	go func() {
		for _, c := range candidates {
			workChan <- c
		}
		close(workChan)
	}()

	// 等待所有工作完成并关闭结果通道
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 收集结果
	results := make([]result, 0, len(candidates))
	for r := range resultChan {
		results = append(results, r)
	}

	// 使用混合排序：先按关键词匹配数量，再按向量相似度
	sort.Slice(results, func(i, j int) bool {
		// 如果关键词匹配数量相差超过阈值，优先考虑匹配数量
		if math.Abs(float64(results[i].wordCount-results[j].wordCount)) > 2 {
			return results[i].wordCount > results[j].wordCount
		}
		// 否则按相似度排序
		return results[i].similarity > results[j].similarity
	})

	// 提取前k个结果
	count := k
	if len(results) < k {
		count = len(results)
	}

	ids := make([]string, count)
	for i := 0; i < count; i++ {
		ids[i] = results[i].id
	}

	return ids, nil
}

// 改进的查询缓存结构
type enhancedQueryCache struct {
	results    []string  // 结果ID列表
	timestamp  time.Time // 缓存创建时间
	vectorHash uint64    // 查询向量的哈希值
}

// ShardedVectorDB 分片锁结构
type ShardedVectorDB struct {
	shards    []*VectorShard
	numShards int
}

type VectorShard struct {
	vectors map[string][]float64
	mu      sync.RWMutex
}

// 根据ID确定分片
func (db *ShardedVectorDB) getShardForID(id string) (*VectorShard, error) {
	h := fnv.New32a()
	_, err := h.Write([]byte(id))
	if err != nil {
		return nil, err
	}
	shardIndex := int(h.Sum32()) % db.numShards
	return db.shards[shardIndex], nil
}

// Get 分片查询实现
func (db *ShardedVectorDB) Get(id string) ([]float64, bool) {
	shard, err := db.getShardForID(id)
	if err != nil {
		return nil, false
	}

	shard.mu.RLock()
	defer shard.mu.RUnlock()
	vec, exists := shard.vectors[id]
	return vec, exists
}

// ParallelFindNearest 优化的并行查询实现
func (db *VectorDB) ParallelFindNearest(query []float64, k int) ([]string, error) {
	// 创建固定大小的工作池
	numWorkers := runtime.NumCPU()

	// 使用更高效的任务分配策略
	vectorIDs := make([]string, 0, len(db.vectors))
	for id := range db.vectors {
		vectorIDs = append(vectorIDs, id)
	}

	// 计算每个工作协程处理的向量数量
	vectorsPerWorker := (len(vectorIDs) + numWorkers - 1) / numWorkers

	// 创建结果通道
	resultChan := make(chan []entity.Result, numWorkers)

	// 启动工作协程
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			// 计算此工作协程处理的向量范围
			startIdx := workerID * vectorsPerWorker
			endIdx := startIdx + vectorsPerWorker
			if endIdx > len(vectorIDs) {
				endIdx = len(vectorIDs)
			}

			// 跳过空范围
			if startIdx >= len(vectorIDs) {
				resultChan <- nil
				return
			}

			// 处理分配的向量
			localResults := make([]entity.Result, 0, endIdx-startIdx)
			for idx := startIdx; idx < endIdx; idx++ {
				id := vectorIDs[idx]
				vec := db.vectors[id]
				sim := util.OptimizedCosineSimilarity(query, vec)
				localResults = append(localResults, entity.Result{Id: id, Similarity: sim})
			}

			// 本地排序，减少全局合并开销
			sort.Slice(localResults, func(i, j int) bool {
				return localResults[i].Similarity > localResults[j].Similarity
			})

			// 只保留前k个结果
			if len(localResults) > k {
				localResults = localResults[:k]
			}

			resultChan <- localResults
		}(i)
	}

	// 等待所有工作协程完成并关闭结果通道
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 合并结果
	allResults := make([]entity.Result, 0, k*numWorkers)
	for results := range resultChan {
		if results != nil {
			allResults = append(allResults, results...)
		}
	}

	// 全局排序并截取前k个结果
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Similarity > allResults[j].Similarity
	})

	if len(allResults) > k {
		allResults = allResults[:k]
	}

	// 提取ID
	ids := make([]string, len(allResults))
	for i, result := range allResults {
		ids[i] = result.Id
	}

	return ids, nil
}

func (db *VectorDB) GetDataSize() int {
	return len(db.vectors)
}

// SearchOptions 搜索选项结构
type SearchOptions struct {
	Nprobe        int           // IVF搜索探测的簇数量
	NumHashTables int           // LSH哈希表数量
	UseANN        bool          // 是否使用近似最近邻
	SearchTimeout time.Duration // 搜索超时时间
}
