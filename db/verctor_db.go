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
	// 添加倒排索引锁，细化锁粒度
	invertedMu sync.RWMutex
	// 添加查询缓存
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
	multiIndex        *MultiLevelIndex // 存储构建的多级索引
	config            AdaptiveConfig

	// 新增 PQ 相关字段
	pqCodebook               [][]algorithm.Point // 从文件加载的 PQ 码本
	pqCodebookFilePath       string              // PQ 码本文件路径，用于热加载
	numSubvectors            int                 // PQ 的子向量数量
	numCentroidsPerSubvector int                 // 每个子向量空间的质心数量
	usePQCompression         bool                // 标志是否启用 PQ 压缩
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
	db.mu.Lock()
	defer db.mu.Unlock()

	log.Info("Closing VectorDB...")

	// 尝试保存数据到文件
	if db.filePath != "" {
		log.Info("Attempting to save VectorDB data to %s before closing...", db.filePath)
		if err := db.SaveToFile(); err != nil {
			log.Error("Error saving VectorDB data to %s: %v", db.filePath, err)
		}
	}

	// 清理内存中的数据结构
	db.vectors = make(map[string][]float64) // 清空向量
	db.clusters = make([]Cluster, 0)        // 清空簇信息
	db.indexed = false                      // 重置索引状态

	db.invertedMu.Lock()
	db.invertedIndex = make(map[string][]string) // 清空倒排索引
	db.invertedMu.Unlock()

	db.cacheMu.Lock()
	db.queryCache = make(map[string]queryCache) // 清空查询缓存
	db.cacheMu.Unlock()

	db.normalizedVectors = make(map[string][]float64)               // 清空归一化向量
	db.compressedVectors = make(map[string]entity.CompressedVector) // 清空压缩向量

	// 重置其他可能的状态字段
	db.vectorDim = 0

	log.Info("VectorDB closed successfully.")
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

		// 初始化 PQ 相关字段
		pqCodebook:               nil,
		numSubvectors:            0, // 默认为0，表示未配置或不使用
		numCentroidsPerSubvector: 0, // 默认为0
		usePQCompression:         false,
	}
	if filePath != "" {
		if err := db.LoadFromFile(); err != nil {
			log.Warning("警告: 从 %s 加载向量数据库时出错: %v。将使用空数据库启动。\n", filePath, err)
			db.vectors = make(map[string][]float64)
			db.clusters = make([]Cluster, 0)
			db.indexed = false
			db.normalizedVectors = make(map[string][]float64)
			db.compressedVectors = make(map[string]entity.CompressedVector) // 确保加载失败也初始化
		}
	}
	return db
}

// GetTrainingVectors 从数据库中获取用于训练的向量样本
// sampleRate: 采样率 (0.0 to 1.0)
// maxVectors: 最大采样向量数 (0 表示无限制，除非 sampleRate 也为0)
func (db *VectorDB) GetTrainingVectors(sampleRate float64, maxVectors int) ([][]float64, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.vectors) == 0 {
		return nil, fmt.Errorf("数据库中没有向量可供训练")
	}

	var sampledVectors [][]float64
	allVectorIDs := make([]string, 0, len(db.vectors))
	for id := range db.vectors {
		allVectorIDs = append(allVectorIDs, id)
	}

	rand.Shuffle(len(allVectorIDs), func(i, j int) {
		allVectorIDs[i], allVectorIDs[j] = allVectorIDs[j], allVectorIDs[i]
	})

	numToSample := 0
	if sampleRate > 0 {
		numToSample = int(float64(len(db.vectors)) * sampleRate)
	}

	if maxVectors > 0 {
		if numToSample == 0 || numToSample > maxVectors { // 如果采样数超过最大数，或未通过采样率设置
			numToSample = maxVectors
		}
	}

	if numToSample == 0 { // 如果两者都未有效设置，则默认采样一小部分或全部（如果数据量小）
		numToSample = len(db.vectors)
		if numToSample > 10000 { // 避免采样过多数据，设定一个上限
			numToSample = 10000
		}
	}

	if numToSample > len(allVectorIDs) {
		numToSample = len(allVectorIDs)
	}

	sampledVectors = make([][]float64, 0, numToSample)
	for i := 0; i < numToSample; i++ {
		id := allVectorIDs[i]
		// 需要复制一份，避免外部修改影响原始数据
		vecCopy := make([]float64, len(db.vectors[id]))
		copy(vecCopy, db.vectors[id])
		sampledVectors = append(sampledVectors, vecCopy)
	}

	log.Info("从 VectorDB 采样了 %d 个向量用于训练", len(sampledVectors))
	return sampledVectors, nil
}

// LoadPQCodebookFromFile 从文件加载 PQ 码本
func (db *VectorDB) LoadPQCodebookFromFile(filePath string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if filePath == "" {
		log.Warning("PQ 码本文件路径为空，跳过加载。")
		db.pqCodebook = nil
		db.usePQCompression = false // 如果码本路径为空，则禁用PQ
		return nil
	}

	file, err := os.Open(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Warning("PQ 码本文件 %s 不存在，PQ 压缩将不可用。", filePath)
			db.pqCodebook = nil
			db.usePQCompression = false
			return nil // 文件不存在不是致命错误，只是PQ不可用
		}
		return fmt.Errorf("打开 PQ 码本文件 %s 失败: %v", filePath, err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var codebook [][]algorithm.Point
	if err := decoder.Decode(&codebook); err != nil {
		return fmt.Errorf("解码 PQ 码本文件 %s 失败: %v", filePath, err)
	}

	db.pqCodebook = codebook
	db.pqCodebookFilePath = filePath // 存储路径以备将来热更新检查
	// 可以在这里根据码本结构验证 numSubvectors 和 numCentroidsPerSubvector
	if len(codebook) > 0 {
		db.numSubvectors = len(codebook)
		if len(codebook[0]) > 0 {
			db.numCentroidsPerSubvector = len(codebook[0])
		} else {
			log.Warning("加载的 PQ 码本子空间为空，PQ 参数可能不正确。")
			db.numCentroidsPerSubvector = 0
		}
	} else {
		log.Warning("加载的 PQ 码本为空，PQ 参数可能不正确。")
		db.numSubvectors = 0
		db.numCentroidsPerSubvector = 0
	}

	log.Info("成功从 %s 加载 PQ 码本。子空间数: %d, 每子空间质心数: %d", filePath, db.numSubvectors, db.numCentroidsPerSubvector)
	return nil
}

// EnablePQCompression 启用 PQ 压缩并设置相关参数
// codebookPath: 码本文件路径。如果为空，则尝试使用之前配置的路径加载，或禁用PQ。
// numSubvectors, numCentroidsPerSubvector: 这些参数现在主要用于信息展示和潜在的校验，实际值会从加载的码本中推断。
func (db *VectorDB) EnablePQCompression(codebookPath string, numSubvectors int, numCentroidsPerSubvector int) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// 如果传入了新的 codebookPath，则使用它
	// 如果 codebookPath 为空，则尝试使用 db.pqCodebookFilePath (如果之前设置过)
	pathToLoad := codebookPath
	if pathToLoad == "" {
		pathToLoad = db.pqCodebookFilePath
	}

	if pathToLoad == "" {
		db.mu.Lock()
		log.Warning("未提供 PQ 码本文件路径，且之前未配置，PQ 压缩将禁用。")
		db.usePQCompression = false
		db.pqCodebook = nil
		db.numSubvectors = 0
		db.numCentroidsPerSubvector = 0
		db.mu.Unlock()
		return nil // 不是错误，只是禁用
	}
	if numSubvectors <= 0 {
		return fmt.Errorf("子向量数量必须为正")
	}
	if numCentroidsPerSubvector <= 0 {
		return fmt.Errorf("每个子向量的质心数量必须为正")
	}

	if err := db.LoadPQCodebookFromFile(pathToLoad); err != nil {
		// LoadPQCodebookFromFile 内部会处理文件不存在的情况并禁用PQ，这里只处理其他加载错误
		db.mu.Lock()
		db.usePQCompression = false // 加载失败，禁用PQ
		db.pqCodebook = nil
		db.numSubvectors = 0
		db.numCentroidsPerSubvector = 0
		db.mu.Unlock()
		return fmt.Errorf("启用 PQ 压缩失败，加载码本时出错: %v", err)
	}
	db.mu.Lock() // 确保在更新 usePQCompression 之前获取锁
	// 只有当码本成功加载且非空时，才真正启用PQ
	if db.pqCodebook != nil && len(db.pqCodebook) > 0 {
		db.usePQCompression = true
		// 更新 numSubvectors 和 numCentroidsPerSubvector 以匹配加载的码本
		db.numSubvectors = len(db.pqCodebook)
		if len(db.pqCodebook[0]) > 0 {
			db.numCentroidsPerSubvector = len(db.pqCodebook[0])
		} else {
			db.numCentroidsPerSubvector = 0 // 或者报错，取决于策略
		}
		log.Info("PQ 压缩已启用。码本路径: %s, 子向量数: %d, 每子空间质心数: %d", db.pqCodebookFilePath, db.numSubvectors, db.numCentroidsPerSubvector)

		// 提示用户可能需要压缩现有向量
		if len(db.vectors) > 0 && len(db.compressedVectors) < len(db.vectors) {
			log.Info("VectorDB 中存在未压缩的向量。您可能需要调用 CompressExistingVectors 来压缩它们。")
		}
	} else {
		db.usePQCompression = false
		log.Warning("PQ 码本加载后为空或加载失败，PQ 压缩已禁用。")
	}
	db.mu.Unlock()

	return nil
}

// CompressExistingVectors 对数据库中所有尚未压缩的向量进行 PQ 压缩
func (db *VectorDB) CompressExistingVectors() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if !db.usePQCompression || db.pqCodebook == nil {
		return fmt.Errorf("PQ 压缩未启用或码本未设置")
	}

	log.Info("开始压缩现有向量...")
	compressedCount := 0
	for id, vec := range db.vectors {
		if _, exists := db.compressedVectors[id]; !exists {
			compressedVec, err := util.CompressByPQ(vec, db.pqCodebook, db.numSubvectors, db.numCentroidsPerSubvector)
			if err != nil {
				log.Error("压缩向量 %s 失败: %v", id, err)
				continue // 跳过压缩失败的向量
			}
			db.compressedVectors[id] = compressedVec
			compressedCount++
		}
	}
	log.Info("现有向量压缩完成，共压缩了 %d 个向量。", compressedCount)
	return nil
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
	// 如果启用了 PQ 压缩，则压缩向量
	if db.usePQCompression && db.pqCodebook != nil {
		compressedVec, err := util.CompressByPQ(vector, db.pqCodebook, db.numSubvectors, db.numCentroidsPerSubvector)
		if err != nil {
			// 即使压缩失败，原始向量也已添加，这里只记录错误
			log.Error("为文档 %s 添加时压缩向量失败: %v", id, err)
		} else {
			db.compressedVectors[id] = compressedVec
		}
	}
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
	// 如果启用了 PQ 压缩，则压缩向量
	if db.usePQCompression && db.pqCodebook != nil {
		compressedVec, err := util.CompressByPQ(vector, db.pqCodebook, db.numSubvectors, db.numCentroidsPerSubvector)
		if err != nil {
			log.Error("向量 %s 压缩失败: %v。该向量将只以原始形式存储。", id, err)
			// 根据策略，可以选择是否回滚添加操作或仅记录错误
		} else {
			if db.compressedVectors == nil {
				db.compressedVectors = make(map[string]entity.CompressedVector)
			}
			db.compressedVectors[id] = compressedVec
			log.Trace("向量 %s 已压缩并存储。", id)
		}
	}

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
	delete(db.compressedVectors, id)

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
	// 如果启用了 PQ 压缩，则更新压缩向量
	if db.usePQCompression && db.pqCodebook != nil {
		compressedVec, err := util.CompressByPQ(vector, db.pqCodebook, db.numSubvectors, db.numCentroidsPerSubvector)
		if err != nil {
			log.Error("为向量 %s 更新时压缩向量失败: %v", id, err)
			// 即使压缩失败，原始向量也已更新
			delete(db.compressedVectors, id) // 删除旧的压缩向量，因为它不再有效
		} else {
			db.compressedVectors[id] = compressedVec
		}
	}
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
	delete(db.compressedVectors, id) // 删除压缩向量

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
		allVectorsData = append(allVectorsData, vec)
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
	multiIndex := &MultiLevelIndex{
		clusters:    db.clusters,                         // 注意：这里可能需要深拷贝或调整，取决于 MultiLevelIndex 的设计
		subIndices:  make([]interface{}, db.numClusters), // 假设 subIndices 在 goroutine 中填充
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
			kdTree := tree.NewKDTree(db.vectorDim)

			// 将簇内所有向量插入KD树
			for _, id := range db.clusters[clusterIdx].VectorIDs {
				vec, exists := db.vectors[id]
				if exists {
					kdTree.Insert(vec, id)
				}
			}

			// 保存KD树到多级索引
			multiIndex.subIndices[clusterIdx] = kdTree
		}(i)
	}

	// 等待所有KD树构建完成
	wg.Wait()

	db.multiIndex = multiIndex // 保存构建好的多级索引
	db.config.UseMultiLevelIndex = true
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
		allVectorsData = append(allVectorsData, vec)
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
	Vectors                  map[string][]float64
	Clusters                 []Cluster
	NumClusters              int
	Indexed                  bool
	InvertedIndex            map[string][]string
	VectorDim                int
	NormalizedVectors        map[string][]float64
	CompressedVectors        map[string]entity.CompressedVector
	PQCodebook               []float64
	NumSubvectors            int
	NumCentroidsPerSubvector int
	UsePQCompression         bool
}

// SaveToFile 将当前数据库状态（包括索引）保存到其配置的文件中。
func (db *VectorDB) SaveToFile() error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.filePath == "" {
		return fmt.Errorf("文件路径未设置，无法保存数据库")
	}

	file, err := os.Create(db.filePath)
	if err != nil {
		return fmt.Errorf("创建数据库文件 %s 失败: %v", db.filePath, err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	// 序列化 VectorDB 的核心数据
	// 为了向前兼容和模块化，可以考虑为每个主要部分创建独立的结构进行序列化
	data := struct {
		Vectors           map[string][]float64
		Clusters          []Cluster
		NumClusters       int
		Indexed           bool
		InvertedIndex     map[string][]string
		VectorDim         int
		VectorizedType    int
		NormalizedVectors map[string][]float64
		CompressedVectors map[string]entity.CompressedVector
		UseCompression    bool
		// PQ 相关字段也需要保存，以便下次加载时能正确恢复状态
		PqCodebookFilePath       string // 保存码本路径，而不是码本本身，码本由外部文件管理
		NumSubvectors            int
		NumCentroidsPerSubvector int
		UsePQCompression         bool
		// MultiIndex 和 Config 可能也需要保存，取决于其具体实现和是否可序列化
		// MultiIndex *MultiLevelIndex // 如果 MultiLevelIndex 可序列化
		// Config AdaptiveConfig // 如果 AdaptiveConfig 可序列化
	}{
		Vectors:                  db.vectors,
		Clusters:                 db.clusters,
		NumClusters:              db.numClusters,
		Indexed:                  db.indexed,
		InvertedIndex:            db.invertedIndex,
		VectorDim:                db.vectorDim,
		VectorizedType:           db.vectorizedType,
		NormalizedVectors:        db.normalizedVectors,
		CompressedVectors:        db.compressedVectors,
		UseCompression:           db.useCompression,
		PqCodebookFilePath:       db.pqCodebookFilePath, // 保存码本文件路径
		NumSubvectors:            db.numSubvectors,
		NumCentroidsPerSubvector: db.numCentroidsPerSubvector,
		UsePQCompression:         db.usePQCompression,
	}

	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("序列化数据库到 %s 失败: %v", db.filePath, err)
	}

	log.Info("VectorDB 数据成功保存到 %s", db.filePath)
	return nil
}

// LoadFromFile 从其配置的文件中加载数据库状态（包括索引）。
func (db *VectorDB) LoadFromFile() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.filePath == "" {
		return fmt.Errorf("文件路径未设置，无法加载数据库")
	}

	file, err := os.Open(db.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Info("数据库文件 %s 不存在，将创建一个新的空数据库。", db.filePath)
			// 初始化为空数据库状态，确保所有 map 都已创建
			db.vectors = make(map[string][]float64)
			db.clusters = make([]Cluster, 0)
			db.indexed = false
			db.invertedIndex = make(map[string][]string)
			db.queryCache = make(map[string]queryCache)
			db.normalizedVectors = make(map[string][]float64)
			db.compressedVectors = make(map[string]entity.CompressedVector)
			db.pqCodebook = nil
			return nil // 文件不存在不是错误，是正常启动流程
		}
		return fmt.Errorf("打开数据库文件 %s 失败: %v", db.filePath, err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	data := struct {
		Vectors                  map[string][]float64
		Clusters                 []Cluster
		NumClusters              int
		Indexed                  bool
		InvertedIndex            map[string][]string
		VectorDim                int
		VectorizedType           int
		NormalizedVectors        map[string][]float64
		CompressedVectors        map[string]entity.CompressedVector
		UseCompression           bool
		PqCodebookFilePath       string
		NumSubvectors            int
		NumCentroidsPerSubvector int
		UsePQCompression         bool
	}{}

	if err := decoder.Decode(&data); err != nil {
		// 如果解码失败，可能是文件损坏或格式不兼容
		// 记录错误，并以空数据库启动，避免程序崩溃
		log.Error("从 %s 反序列化数据库失败: %v。将使用空数据库启动。", db.filePath, err)
		db.vectors = make(map[string][]float64)
		db.clusters = make([]Cluster, 0)
		db.indexed = false
		db.invertedIndex = make(map[string][]string)
		db.queryCache = make(map[string]queryCache)
		db.normalizedVectors = make(map[string][]float64)
		db.compressedVectors = make(map[string]entity.CompressedVector)
		db.pqCodebook = nil
		return nil // 即使加载失败，也返回nil，让程序继续运行
	}

	// 恢复数据
	db.vectors = data.Vectors
	db.clusters = data.Clusters
	db.numClusters = data.NumClusters
	db.indexed = data.Indexed
	db.invertedIndex = data.InvertedIndex
	db.vectorDim = data.VectorDim
	db.vectorizedType = data.VectorizedType
	db.normalizedVectors = data.NormalizedVectors
	db.compressedVectors = data.CompressedVectors
	db.useCompression = data.UseCompression
	db.pqCodebookFilePath = data.PqCodebookFilePath
	db.numSubvectors = data.NumSubvectors
	db.numCentroidsPerSubvector = data.NumCentroidsPerSubvector
	db.usePQCompression = data.UsePQCompression

	// 确保 map 在 nil 的情况下被初始化
	if db.vectors == nil {
		db.vectors = make(map[string][]float64)
	}
	if db.invertedIndex == nil {
		db.invertedIndex = make(map[string][]string)
	}
	if db.normalizedVectors == nil {
		db.normalizedVectors = make(map[string][]float64)
	}
	if db.compressedVectors == nil {
		db.compressedVectors = make(map[string]entity.CompressedVector)
	}
	if db.queryCache == nil { // queryCache 不在 gob 中，需要单独初始化
		db.queryCache = make(map[string]queryCache)
	}

	// 如果启用了 PQ 压缩且有码本路径，则尝试加载码本
	if db.usePQCompression && db.pqCodebookFilePath != "" {
		// 这里使用临时变量，避免在 LoadPQCodebookFromFile 中发生死锁
		tempPath := db.pqCodebookFilePath
		db.pqCodebookFilePath = "" // 暂时清除，避免 LoadPQCodebookFromFile 内部逻辑冲突
		db.usePQCompression = false

		db.mu.Unlock() // 解锁以便 LoadPQCodebookFromFile 可以获取锁
		errLoadCodebook := db.LoadPQCodebookFromFile(tempPath)
		db.mu.Lock() // 重新获取锁

		if errLoadCodebook != nil {
			log.Error("从 %s 加载数据库后，尝试加载 PQ 码本 %s 失败: %v。PQ 压缩将禁用。", db.filePath, tempPath, errLoadCodebook)
			db.usePQCompression = false
			db.pqCodebook = nil
		} else {
			// LoadPQCodebookFromFile 会更新 db.pqCodebook, db.numSubvectors, db.numCentroidsPerSubvector
			// 它也会在成功加载码本后设置 db.usePQCompression = true (如果码本非空)
			// 所以这里我们只需要确保 db.usePQCompression 反映了加载结果
			if db.pqCodebook == nil || len(db.pqCodebook) == 0 {
				db.usePQCompression = false
			} else {
				db.usePQCompression = true // 确保与加载的码本状态一致
			}
		}
		// 恢复原始配置的 usePQCompression 状态，如果码本加载失败，则它会被设为 false
		// 如果码本加载成功，LoadPQCodebookFromFile 内部会处理
		// 实际上，我们应该信任 LoadPQCodebookFromFile 设置的 usePQCompression
		// 所以，如果 tempUsePQ 为 true 但加载失败，usePQCompression 会是 false，这是正确的
	} else if db.usePQCompression && db.pqCodebookFilePath == "" {
		log.Warning("数据库配置为使用 PQ 压缩，但未指定码本文件路径。PQ 压缩将禁用。")
		db.usePQCompression = false
		db.pqCodebook = nil
	}

	log.Info("VectorDB 数据成功从 %s 加载。向量数: %d, 是否已索引: %t, PQ压缩: %t", db.filePath, len(db.vectors), db.indexed, db.usePQCompression)
	return nil
}

// FindNearest 查找与查询向量最相似的k个向量。
// ... (此函数需要重大修改以支持基于 PQ 的 ADC/SDC 距离计算)
// 如果启用了 PQ 压缩，并且希望利用它进行快速近似搜索，
// 则需要实现相应的距离计算函数，例如 util.ApproximateDistancePQ。
// 当前的 FindNearest 仍然基于原始向量进行精确搜索。
// ... existing code ...

// CalculateApproximateDistancePQ 以下是一个基于 PQ 的近似距离计算的简化示例，需要集成到 FindNearest 或新的搜索函数中
// CalculateApproximateDistancePQ 计算查询向量与数据库中压缩向量的近似距离 (ADC)
func (db *VectorDB) CalculateApproximateDistancePQ(queryVector []float64, compressedDBVector entity.CompressedVector) (float64, error) {
	if !db.usePQCompression || db.pqCodebook == nil {
		return 0, fmt.Errorf("PQ 压缩未启用或码本未设置")
	}
	if len(queryVector) != db.vectorDim {
		return 0, fmt.Errorf("查询向量维度 %d 与数据库向量维度 %d 不匹配", len(queryVector), db.vectorDim)
	}

	// 调用 util 中的 PQ 近似距离计算函数 (假设已实现)
	// dist, err := util.ApproximateDistanceADC(queryVector, compressedDBVector.Data, db.pqCodebook, db.numSubvectors, db.numCentroidsPerSubvector, db.vectorDim/db.numSubvectors)
	// if err != nil {
	// 	return 0, fmt.Errorf("计算 PQ 近似距离失败: %w", err)
	// }
	// return dist

	// 这是一个占位符，您需要实现实际的 PQ ADC/SDC 距离计算
	// 例如，可以先将查询向量也进行量化（如果使用 SDC），或者直接与码本中的子质心计算距离（ADC）
	// 然后根据 compressedDBVector.Data 中的索引组合这些子距离。
	log.Warning("CalculateApproximateDistancePQ 尚未完全实现，返回占位符距离。")
	return math.MaxFloat64, fmt.Errorf("PQ 近似距离计算尚未实现")
}
func (db *VectorDB) CalculateAvgQueryTime(startTime time.Time) time.Duration {
	avgTimeMicroSeconds := db.stats.AvgQueryTime.Microseconds()
	startTimeMicroSeconds := time.Since(startTime).Microseconds()
	return time.Duration(((avgTimeMicroSeconds * (db.stats.TotalQueries - 1)) + startTimeMicroSeconds) / db.stats.TotalQueries)
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
	cached, found := db.queryCache[cacheKey]
	if found && (time.Now().Unix()-cached.timestamp) < db.cacheTTL {
		db.cacheMu.RUnlock()
		db.statsMu.Lock()
		db.stats.CacheHits++
		db.stats.AvgQueryTime = db.CalculateAvgQueryTime(startTime)
		db.statsMu.Unlock()
		return cached.results, nil
	}
	db.cacheMu.RUnlock()

	db.mu.RLock()
	defer db.mu.RUnlock()

	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		return []string{}, nil
	}
	var results []entity.Result
	var err error
	if !db.indexed {
		// 如果未索引，执行暴力搜索
		log.Warning("VectorDB is not indexed. Performing brute-force search.")
		results, err = db.bruteForceSearch(query, k)
		if err != nil {
			return nil, err
		}
	} else if db.config.UseMultiLevelIndex && db.multiIndex != nil && db.multiIndex.indexed {
		results, err = db.multiIndexSearch(query, k, nprobe)
		if err != nil {
			return nil, err
		}
	} else {
		vectorCount := len(db.GetVectors())
		options := SearchOptions{
			Nprobe:        nprobe,
			NumHashTables: 4 + vectorCount/10000, // 根据数据规模调整哈希表数量
			UseANN:        true,
		}

		results, err = db.ivfSearch(query, k, options.Nprobe)
		if err != nil {
			return nil, err
		}
	}
	finalResults := make([]string, len(results))
	for i := len(results) - 1; i >= 0; i-- {
		finalResults[len(results)-1-i] = results[i].Id
	}

	// 更新平均查询时间统计
	db.statsMu.Lock()
	// 使用指数移动平均更新平均查询时间
	// queryTime := time.Since(startTime)
	// alpha := 0.1
	// db.stats.AvgQueryTime = time.Duration(float64(db.stats.AvgQueryTime)*(1-alpha) + float64(queryTime)*alpha)
	db.stats.AvgQueryTime = db.CalculateAvgQueryTime(startTime)
	db.statsMu.Unlock()

	return finalResults, nil
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
func (db *VectorDB) HybridSearch(query []float64, k int, options SearchOptions, nprobe int) ([]entity.Result, error) {
	// 根据向量维度和数据规模自动选择最佳搜索策略
	if len(db.vectors) < 1000 || !db.indexed {
		// 小数据集使用暴力搜索
		return db.bruteForceSearch(query, k)
	} else if len(query) > 1000 {
		// 高维向量使用LSH (Locality-Sensitive Hashing)
		return db.lshSearch(query, k, options.NumHashTables)
	} else if db.config.UseMultiLevelIndex && db.multiIndex != nil && db.multiIndex.indexed {
		return db.multiIndexSearch(query, k, nprobe)
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

// 多级索引搜索
func (db *VectorDB) multiIndexSearch(query []float64, k int, nprobe int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	var results *entity.ResultHeap
	log.Trace("Using Multi-Level Index for FindNearest.")
	// 1. 找到 nprobe 个最近的簇中心 (与之前逻辑类似)
	clusterDist := make([]struct {
		Index int
		Dist  float64
	}, len(db.multiIndex.clusters))

	for i, cluster := range db.multiIndex.clusters {
		dist, err := algorithm.EuclideanDistanceSquared(query, cluster.Centroid)
		if err != nil {
			return nil, fmt.Errorf("error calculating distance to centroid %d: %w", i, err)
		}
		clusterDist[i] = struct {
			Index int
			Dist  float64
		}{i, dist}
	}

	sort.Slice(clusterDist, func(i, j int) bool {
		return clusterDist[i].Dist < clusterDist[j].Dist
	})

	numToProbe := nprobe
	if numToProbe > len(clusterDist) {
		numToProbe = len(clusterDist)
	}

	resultHeap := make(entity.ResultHeap, 0, k)
	heap.Init(&resultHeap)

	// 2. 在选中的簇的二级索引中搜索
	for i := 0; i < numToProbe; i++ {
		clusterIdx := clusterDist[i].Index
		selectedCluster := db.multiIndex.clusters[clusterIdx]

		if clusterIdx >= len(db.multiIndex.subIndices) || db.multiIndex.subIndices[clusterIdx] == nil {
			log.Warning("Sub-index for cluster %d not found or nil. Performing brute-force in this cluster.", clusterIdx)
			// 回退到暴力搜索该簇内的向量
			for _, id := range selectedCluster.VectorIDs {
				if vec, exists := db.vectors[id]; exists {
					dist, _ := algorithm.EuclideanDistanceSquared(query, vec)
					heap.Push(&resultHeap, entity.Result{Id: id, Similarity: dist})
					if results.Len() > k {
						heap.Pop(results)
					}
				}
			}
			continue
		}

		// 假设二级索引是 KDTree，并且有 FindNearest 方法
		kdTree, ok := db.multiIndex.subIndices[clusterIdx].(*tree.KDTree) // 类型断言
		if !ok || kdTree == nil {
			log.Warning("Sub-index for cluster %d is not a KDTree or is nil. Performing brute-force.", clusterIdx)
			for _, id := range selectedCluster.VectorIDs {
				if vec, exists := db.vectors[id]; exists {
					dist, _ := algorithm.EuclideanDistanceSquared(query, vec)
					heap.Push(&resultHeap, entity.Result{Id: id, Similarity: dist})
					if results.Len() > k {
						heap.Pop(results)
					}
				}
			}
			continue
		}

		kdResults := kdTree.FindNearest(query, k) // 在KD树中搜索K个最近的，或者一个合理的数量
		if kdResults == nil {
			log.Error("Error searching in KDTree for cluster %d. Skipping this sub-index.", clusterIdx)
			continue
		}

		for _, item := range kdResults {
			heap.Push(results, &item)
			if results.Len() > k {
				heap.Pop(results)
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
