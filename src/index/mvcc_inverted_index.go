package index

import (
	"VectorSphere/src/db"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/log"
	"VectorSphere/src/library/tree"
	"VectorSphere/src/messages"
	"fmt"
	"hash/fnv"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// InvertedList 倒排列表类型
type InvertedList []MvccSkipListValue

// MvccSkipListValue 结构体定义包含 scoreId 字段
type MvccSkipListValue struct {
	Id          string
	BitsFeature uint64
	ScoreId     int64
	// 新增字段，用于向量相似度计算
	Vector []float64
}

// PerformanceStats 扩展性能统计结构
type PerformanceStats struct {
	TotalQueries     int64
	CacheHits        int64
	AvgQueryTime     time.Duration
	LastOptimizeTime time.Time
	MemoryUsage      uint64

	// 添加更多统计指标
	KeywordSearchCount   int64
	VectorSearchCount    int64
	HybridSearchCount    int64
	AvgKeywordSearchTime time.Duration
	AvgVectorSearchTime  time.Duration
	AvgHybridSearchTime  time.Duration
	IndexSize            int64 // 索引大小（字节）
	DocumentCount        int64 // 文档数量
	KeywordCount         int64 // 关键词数量
}

// 优化的查询缓存结构
type queryCache struct {
	results   []string
	timestamp int64
	// 添加查询热度统计
	hitCount int
}

// AdaptiveConfig 自适应配置结构
type AdaptiveConfig struct {
	// 索引参数
	IndexRebuildThreshold float64 // 更新比例阈值，超过此值重建索引

	// 查询参数
	CacheTimeout time.Duration // 缓存超时时间

	// 系统参数
	MaxWorkers          int  // 最大工作协程数
	UseVectorSimilarity bool // 是否使用向量相似度增强搜索
}

type MVCCBPlusTreeInvertedIndex struct {
	tree     *tree.MVCCBPlusTree
	order    int
	mu       sync.RWMutex
	vectorDB *db.VectorDB // Add VectorDB field
	// 新增字段
	stats   PerformanceStats
	statsMu sync.RWMutex
	config  AdaptiveConfig

	// 查询缓存
	queryCache map[string]queryCache
	cacheMu    sync.RWMutex
	cacheTTL   int64 // 缓存有效期（秒）
}

func NewMVCCBPlusTreeInvertedIndex(order int, txMgr *tree.TransactionManager, lockMgr *tree.LockManager, wal *tree.WALManager, vectorDB *db.VectorDB) *MVCCBPlusTreeInvertedIndex {
	return &MVCCBPlusTreeInvertedIndex{
		tree:  tree.NewMVCCBPlusTree(order, txMgr, lockMgr, wal),
		order: order,
		// 初始化新增字段
		queryCache: make(map[string]queryCache),
		cacheTTL:   300,      // 默认缓存5分钟
		vectorDB:   vectorDB, // Initialize vectorDB
		config: AdaptiveConfig{
			IndexRebuildThreshold: 0.2,
			CacheTimeout:          5 * time.Minute,
			MaxWorkers:            runtime.NumCPU(),
			UseVectorSimilarity:   true,
		},
	}
}

// GetStats 获取性能统计信息
func (idx *MVCCBPlusTreeInvertedIndex) GetStats() PerformanceStats {
	idx.statsMu.RLock()
	defer idx.statsMu.RUnlock()
	return idx.stats
}

// Optimize 优化倒排索引，清理无效数据，重新平衡B+树，更新统计信息
func (idx *MVCCBPlusTreeInvertedIndex) Optimize() error {
	// 获取开始时间，用于计算优化耗时
	startTime := time.Now()

	// 加锁保护索引结构
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// 清理查询缓存
	idx.cacheMu.Lock()
	idx.queryCache = make(map[string]queryCache)
	idx.cacheMu.Unlock()

	// 获取所有关键词
	keywords := idx.getAllKeywords()

	// 记录优化前的统计信息
	beforeKeywordCount := len(keywords)

	// 执行B+树的优化操作

	if optimizer, ok := interface{}(idx.tree).(interface{ Optimize() error }); ok {
		if err := optimizer.Optimize(); err != nil {
			return fmt.Errorf("优化B+树失败: %v", err)
		}
	}

	// 更新统计信息
	idx.statsMu.Lock()
	idx.stats.LastOptimizeTime = time.Now()

	// 更新关键词数量
	idx.stats.KeywordCount = int64(len(idx.getAllKeywords()))

	// 更新内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	idx.stats.MemoryUsage = m.Alloc

	// 记录优化结果
	optimizeTime := time.Since(startTime)
	idx.statsMu.Unlock()

	// 自适应调整配置
	idx.AdjustConfig()

	// 输出优化结果
	log.Info("倒排索引优化完成，耗时 %v，关键词数量: %d -> %d",
		optimizeTime, beforeKeywordCount, idx.stats.KeywordCount)

	return nil
}

// 更新统计信息的方法
func (idx *MVCCBPlusTreeInvertedIndex) updateStats(queryType string, duration time.Duration) {
	idx.statsMu.Lock()
	defer idx.statsMu.Unlock()

	idx.stats.TotalQueries++

	switch queryType {
	case "keyword":
		idx.stats.KeywordSearchCount++
		// 计算移动平均
		if idx.stats.AvgKeywordSearchTime == 0 {
			idx.stats.AvgKeywordSearchTime = duration
		} else {
			idx.stats.AvgKeywordSearchTime = (idx.stats.AvgKeywordSearchTime*9 + duration) / 10
		}
	case "vector":
		idx.stats.VectorSearchCount++
		if idx.stats.AvgVectorSearchTime == 0 {
			idx.stats.AvgVectorSearchTime = duration
		} else {
			idx.stats.AvgVectorSearchTime = (idx.stats.AvgVectorSearchTime*9 + duration) / 10
		}
	case "hybrid":
		idx.stats.HybridSearchCount++
		if idx.stats.AvgHybridSearchTime == 0 {
			idx.stats.AvgHybridSearchTime = duration
		} else {
			idx.stats.AvgHybridSearchTime = (idx.stats.AvgHybridSearchTime*9 + duration) / 10
		}
	}

	// 更新总体平均查询时间
	if idx.stats.AvgQueryTime == 0 {
		idx.stats.AvgQueryTime = duration
	} else {
		idx.stats.AvgQueryTime = (idx.stats.AvgQueryTime*9 + duration) / 10
	}

	// 定期更新索引大小和文档数量统计
	if idx.stats.TotalQueries%100 == 0 {
		// 更新关键词数量
		idx.stats.KeywordCount = int64(len(idx.getAllKeywords()))

		// 更新内存使用情况
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		idx.stats.MemoryUsage = m.Alloc
	}
}

// AdjustConfig 自适应配置调整
func (idx *MVCCBPlusTreeInvertedIndex) AdjustConfig() {
	// 获取所有关键词数量作为数据规模参考
	allKeywords := idx.getAllKeywords()
	keywordCount := len(allKeywords)

	// 获取当前性能统计
	stats := idx.GetStats()

	config := idx.config

	// 根据关键词数量调整工作协程数
	if keywordCount > 100000 {
		config.MaxWorkers = runtime.NumCPU() * 2
	} else if keywordCount > 10000 {
		config.MaxWorkers = runtime.NumCPU()
	} else {
		config.MaxWorkers = runtime.NumCPU() / 2
	}

	// 根据内存使用情况调整缓存策略
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// 内存压力大时减少缓存时间
	if m.Alloc > 1024*1024*1024 { // 如果内存使用超过1GB
		config.CacheTimeout = 2 * time.Minute
	} else {
		config.CacheTimeout = 5 * time.Minute
	}

	// 根据查询类型分布调整索引策略
	totalSearches := stats.KeywordSearchCount + stats.VectorSearchCount + stats.HybridSearchCount
	if totalSearches > 0 {
		vectorRatio := float64(stats.VectorSearchCount+stats.HybridSearchCount) / float64(totalSearches)

		// 如果向量搜索比例高，优化向量索引
		if vectorRatio > 0.7 {
			config.UseVectorSimilarity = true
			// 可以考虑增加向量索引的优化参数
		} else if vectorRatio < 0.3 {
			// 如果关键词搜索为主，优化B+树索引
			// 可以考虑调整B+树的参数
		}
	}

	// 根据查询性能调整索引重建阈值
	if stats.AvgQueryTime > 100*time.Millisecond {
		// 如果查询性能下降，降低重建阈值，更频繁地重建索引
		config.IndexRebuildThreshold = 0.1
	} else {
		config.IndexRebuildThreshold = 0.2
	}

	idx.mu.Lock()
	idx.config = config
	idx.mu.Unlock()
}

// GenerateScoreId 结合 Document 信息和时间戳生成 scoreId
func GenerateScoreId(doc messages.Document) int64 {
	// 获取当前时间戳
	timestamp := time.Now().UnixNano()

	// 使用 FNV-1a 哈希算法对文档 ID 进行哈希处理
	h := fnv.New64a()
	_, err := h.Write([]byte(doc.Id))
	if err != nil {
		return 0
	}
	hashValue := h.Sum64()

	// 结合时间戳和哈希值生成 scoreId
	// 这里简单地将时间戳和哈希值的低 32 位组合
	scoreId := (timestamp << 32) | (int64(hashValue) & 0xFFFFFFFF)

	return scoreId
}

func (idx *MVCCBPlusTreeInvertedIndex) Close() {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.vectorDB != nil {
		idx.vectorDB.Close()
	}

	idx.cacheMu.Lock()
	idx.queryCache = make(map[string]queryCache)
	idx.cacheMu.Unlock()
}

// 添加LRU缓存淘汰策略
func (idx *MVCCBPlusTreeInvertedIndex) cleanupCache() {
	idx.cacheMu.Lock()
	defer idx.cacheMu.Unlock()

	now := time.Now().Unix()
	// 如果缓存项过多，进行LRU淘汰
	if len(idx.queryCache) > 10000 { // 设置合理的缓存上限
		// 按照热度和时间排序
		type cacheItem struct {
			key       string
			timestamp int64
			hitCount  int
		}

		items := make([]cacheItem, 0, len(idx.queryCache))
		for k, v := range idx.queryCache {
			items = append(items, cacheItem{k, v.timestamp, v.hitCount})
		}

		// 按照热度和时间排序，保留热门和较新的缓存
		sort.Slice(items, func(i, j int) bool {
			// 优先考虑热度，其次考虑时间
			if items[i].hitCount != items[j].hitCount {
				return items[i].hitCount > items[j].hitCount
			}
			return items[i].timestamp > items[j].timestamp
		})

		// 保留前50%的缓存
		newCache := make(map[string]queryCache)
		for i := 0; i < len(items)/2; i++ {
			item := items[i]
			newCache[item.key] = idx.queryCache[item.key]
		}
		idx.queryCache = newCache
	}

	// 清理过期缓存
	for k, v := range idx.queryCache {
		if now-v.timestamp > idx.cacheTTL {
			delete(idx.queryCache, k)
		}
	}
}

// Add 插入文档到倒排索引
func (idx *MVCCBPlusTreeInvertedIndex) Add(tx *tree.Transaction, doc messages.Document) error {
	idx.mu.Lock() // Consider lock granularity; this locks the whole Add operation
	defer idx.mu.Unlock()
	scoreId := GenerateScoreId(doc)

	// 获取文档向量
	var docVector []float64
	if idx.vectorDB != nil {
		// 从vectorDB获取向量
		var docTextForVectorDB string
		if len(doc.Bytes) > 0 {
			docTextForVectorDB = string(doc.Bytes)
		} else if len(doc.KeWords) > 0 {
			var words []string
			for _, kw := range doc.KeWords {
				words = append(words, kw.Word)
			}
			docTextForVectorDB = strings.Join(words, " ")
		}

		if docTextForVectorDB != "" {
			// 使用GetVectorForText获取向量
			vector, err := idx.vectorDB.GetVectorForTextWithCache(docTextForVectorDB, db.DefaultVectorized)
			if err == nil {
				docVector = vector
			}
		}
	}

	var keywords []*messages.KeyWord // Track keywords successfully added to B+Tree

	// Add to B+Tree (Inverted List for keywords)
	for _, keyword := range doc.KeWords {
		val, ok := idx.tree.Get(tx, keyword) // Get within transaction
		var list InvertedList
		if ok && val != nil {
			list = val.(InvertedList)
			// 检查是否已存在相同的 doc.Id，避免重复添加
			exist := false
			for _, v := range list {
				if v.Id == doc.Id {
					exist = true
					break
				}
			}
			if !exist {
				list = append(list, MvccSkipListValue{
					Id:          doc.Id,
					BitsFeature: doc.BitsFeature,
					ScoreId:     scoreId,
					Vector:      docVector,
				})
			}
		} else {
			list = InvertedList{
				MvccSkipListValue{
					Id:          doc.Id,
					BitsFeature: doc.BitsFeature,
					ScoreId:     scoreId,
					Vector:      docVector,
				},
			}
		}
		if err := idx.tree.Put(tx, keyword, list); err != nil { // Put within transaction
			// If B+Tree Put fails, we stop and return error. No VectorDB operation was attempted yet.
			return fmt.Errorf("failed to put keyword %v to B+Tree: %w", keyword, err)
		}
		keywords = append(keywords, keyword) // Mark as successfully added
	}

	// 2. Add to VectorDB
	if idx.vectorDB != nil {
		var docTextForVectorDB string
		if len(doc.Bytes) > 0 {
			docTextForVectorDB = string(doc.Bytes)
		} else if len(doc.KeWords) > 0 {
			var words []string
			for _, kw := range doc.KeWords {
				words = append(words, kw.Word) // Assuming KeyWord has a 'Word' field
			}
			docTextForVectorDB = strings.Join(words, " ")
		} else {
			// No content to vectorize, maybe log a warning or skip
			// fmt.Printf("Warning: Document %s has no content (Bytes or KeWords) for vectorization.\n", doc.Id)
			// If no content for vectorization, we are done.
			return nil
		}

		if docTextForVectorDB != "" {
			// Use a default vectorization type or make it configurable
			if err := idx.vectorDB.AddDocument(doc.Id, docTextForVectorDB, db.DefaultVectorized); err != nil {
				// VectorDB Add failed, attempt to roll back B+Tree changes for successfully added keywords
				fmt.Printf("Warning: Failed to add document %s to VectorDB: %v. Attempting B+Tree rollback.\n", doc.Id, err)
				rollbackErr := idx.rollbackBTreeAdd(tx, doc.Id, keywords)
				if rollbackErr != nil {
					fmt.Printf("CRITICAL ERROR: Failed to rollback B+Tree changes for document %s after VectorDB add failure: %v. System might be in an inconsistent state.\n", doc.Id, rollbackErr)
					// A more robust system would require a transaction manager coordinating both DBs
					// or a background process to reconcile inconsistencies.
				}
				return fmt.Errorf("failed to add document %s to VectorDB: %w", doc.Id, err)
			}
		}
	}
	// 清除查询缓存
	idx.cacheMu.Lock()
	idx.queryCache = make(map[string]queryCache)
	idx.cacheMu.Unlock()
	return nil
}

// rollbackBTreeAdd attempts to remove a document from the B+Tree for a list of keywords.
// This is a best-effort rollback mechanism.
func (idx *MVCCBPlusTreeInvertedIndex) rollbackBTreeAdd(tx *tree.Transaction, docId string, keywords []*messages.KeyWord) error {
	var firstErr error
	for _, keyword := range keywords {
		val, ok := idx.tree.Get(tx, keyword)
		if !ok || val == nil {
			continue // Keyword isn't found or no list associated, nothing to rollback for this keyword
		}
		list, ok := val.(InvertedList)
		if !ok {
			// Log error: type assertion failed, cannot roll back for this keyword
			if firstErr == nil {
				firstErr = fmt.Errorf("type assertion failed for keyword %v during rollback", keyword)
			}
			continue
		}

		newList := make(InvertedList, 0, len(list))
		found := false
		for _, v := range list {
			if v.Id != docId {
				newList = append(newList, v)
			} else {
				found = true
			}
		}

		if found {
			if len(newList) == 0 {
				if err := idx.tree.Delete(tx, keyword); err != nil {
					// Log or handle error: failed to delete keyword from B+Tree during rollback
					if firstErr == nil {
						firstErr = fmt.Errorf("failed to delete keyword %v from B+Tree during rollback: %w", keyword, err)
					}
				}
			} else {
				if err := idx.tree.Put(tx, keyword, newList); err != nil {
					// Log or handle error: failed to update keyword list in B+Tree during rollback
					if firstErr == nil {
						firstErr = fmt.Errorf("failed to update keyword %v in B+Tree during rollback: %w", keyword, err)
					}
				}
			}
		}
	}
	return firstErr // Return the first error encountered during rollback, if any
}

// Delete 从倒排索引中删除文档 (B+树和VectorDB)
// Note: The original Delete took scoreId and a specific keyword.
// A more robust delete would take doc.Id and remove it from all relevant keyword lists.
// For now, let's adapt the existing signature and logic, but ideally, it should be doc.Id based for VectorDB.
func (idx *MVCCBPlusTreeInvertedIndex) Delete(tx *tree.Transaction, docId string, keywords []*messages.KeyWord) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// 1. Delete from B+Tree
	for _, keyword := range keywords {
		val, ok := idx.tree.Get(tx, keyword)
		if !ok || val == nil {
			continue // Keyword isn't found or no list associated
		}
		list, ok := val.(InvertedList)
		if !ok {
			// Log error: type assertion failed
			continue
		}

		newList := make(InvertedList, 0, len(list))
		found := false
		for _, v := range list {
			if v.Id != docId {
				newList = append(newList, v)
			} else {
				found = true
			}
		}

		if !found {
			continue
		}

		if len(newList) == 0 {
			if err := idx.tree.Delete(tx, keyword); err != nil {
				// Log or handle error: failed to delete keyword from B+Tree
				// Consider implications for overall consistency
				return fmt.Errorf("failed to delete keyword %v from B+Tree: %w", keyword, err)
			}
		} else {
			if err := idx.tree.Put(tx, keyword, newList); err != nil {
				// Log or handle error: failed to update a keyword list in B+Tree
				return fmt.Errorf("failed to update keyword %v in B+Tree: %w", keyword, err)
			}
		}

	}

	// 2. Delete from VectorDB
	if idx.vectorDB != nil {
		if err := idx.vectorDB.DeleteVector(docId); err != nil {
			// Log or handle error. What if BTree deletion succeeded, but VectorDB failed?
			// This indicates a need for a more robust 2PC or compensation mechanism if strict consistency is required.
			return fmt.Errorf("failed to delete document %s from VectorDB: %w", docId, err)
		}
	}

	return nil
}

// DelDocument Delete 从倒排索引中删除文档
func (idx *MVCCBPlusTreeInvertedIndex) DelDocument(tx *tree.Transaction, scoreId int64, keyword *messages.KeyWord) error {
	doc, err := idx.GetDocumentByScoreId(scoreId)
	if err != nil {
		return err
	}

	val, ok := idx.tree.Get(tx, keyword)
	if !ok || val == nil {
		return nil
	}
	list := val.(InvertedList)
	newList := make(InvertedList, 0, len(list))
	for _, v := range list {
		if v.Id != doc.Id {
			newList = append(newList, v)
		}
	}
	return idx.tree.Put(tx, keyword, newList)
}

// GetDocumentByScoreId 通过 scoreId 获取 Document 信息
func (idx *MVCCBPlusTreeInvertedIndex) GetDocumentByScoreId(scoreId int64) (messages.Document, error) {
	// 假设存在一个获取所有关键词的方法，这里需要根据实际情况实现
	allKeywords := idx.getAllKeywords()
	var doc messages.Document

	for _, keyword := range allKeywords {
		val, ok := idx.tree.Get(nil, &keyword) // 假设使用 nil 事务，实际需要根据情况调整
		if ok && val != nil {
			list := val.(InvertedList)
			for _, v := range list {
				if v.ScoreId == scoreId {
					// 找到匹配的 scoreId，构建 Document 对象
					doc.Id = v.Id
					doc.BitsFeature = v.BitsFeature
					// 假设 KeWords 字段需要根据实际情况填充
					// 这里简单添加当前关键词
					doc.KeWords = append(doc.KeWords, &keyword)
				}
			}
		}
	}

	if doc.Id == "" {
		// 未找到匹配的文档
		return messages.Document{}, fmt.Errorf("document with scoreId %d not found", scoreId)
	}

	return doc, nil
}

// getAllKeywords 获取所有关键词，需要根据实际情况实现
func (idx *MVCCBPlusTreeInvertedIndex) getAllKeywords() []messages.KeyWord {
	var keywords []messages.KeyWord
	seen := make(map[interface{}]struct{})
	current := idx.getFirstLeaf()

	for current != nil {
		for _, key := range current.GetKeys() {
			if kw, ok := key.(messages.KeyWord); ok {
				if _, exists := seen[kw]; !exists {
					keywords = append(keywords, kw)
					seen[kw] = struct{}{}
				}
			}
		}
		current = current.GetNext()
	}

	return keywords
}

// getMinMaxKeywords 获取最小和最大关键词，需要根据实际情况实现
func (idx *MVCCBPlusTreeInvertedIndex) getMinMaxKeywords() (messages.KeyWord, messages.KeyWord) {
	// 这里需要实现获取最小和最大关键词的逻辑
	// 可以遍历 B+ 树的叶子节点链表
	// 以下是一个简单示例，实际需要根据 B+ 树的实现调整
	var minKey, maxKey messages.KeyWord
	firstLeaf := idx.getFirstLeaf()
	lastLeaf := idx.getLastLeaf()
	if firstLeaf != nil && len(firstLeaf.GetKeys()) > 0 {
		keys := firstLeaf.GetKeys()
		if keys == nil {
			return minKey, maxKey
		}
		minKey = keys[0].(messages.KeyWord)
	}
	if lastLeaf != nil && len(lastLeaf.GetKeys()) > 0 {
		keys := lastLeaf.GetKeys()
		if keys == nil {
			return minKey, maxKey
		}
		maxKey = keys[len(keys)-1].(messages.KeyWord)
	}
	return minKey, maxKey
}

// getFirstLeaf 获取第一个叶子节点，需要根据实际情况实现
func (idx *MVCCBPlusTreeInvertedIndex) getFirstLeaf() *tree.MVCCNode {
	// 这里需要实现获取第一个叶子节点的逻辑
	// 可以从根节点开始遍历到最左边的叶子节点
	current := idx.tree.GetRoot()
	for current != nil && !current.IsLeaf() {
		children := current.GetChildren()
		current = children[0].(*tree.MVCCNode)
	}
	return current
}

// getLastLeaf 获取最后一个叶子节点，需要根据实际情况实现
func (idx *MVCCBPlusTreeInvertedIndex) getLastLeaf() *tree.MVCCNode {
	// 这里需要实现获取最后一个叶子节点的逻辑
	// 可以从根节点开始遍历到最右边的叶子节点
	current := idx.tree.GetRoot()
	for current != nil && !current.IsLeaf() {
		children := current.GetChildren()
		current = children[len(children)-1].(*tree.MVCCNode)
	}
	return current
}

// Search 查询倒排索引 (结合B+树和VectorDB)
// Parameters for vector search (vectorQueryText, kNearest) are added.
/*
- tx - 事务
- query - 查询表达式
- onFlag - 开启的特征位
- offFlag - 关闭的特征位
- orFlags - 或操作的特征位
- vectorQueryText - 向量查询文本
- kNearest - 最近邻数量
- useANN - 是否使用近似最近邻搜索
*/
func (idx *MVCCBPlusTreeInvertedIndex) Search(
	tx *tree.Transaction,
	query *messages.TermQuery,
	onFlag uint64,
	offFlag uint64,
	orFlags []uint64,
	vectorQueryText string,
	kNearest int,
	useANN bool,
) ([]string, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var bTreeDocIds map[string]MvccSkipListValue
	var vectorSearchDocs map[string]float64

	var bTreeErr, vectorErr error
	var wg sync.WaitGroup

	performBTreeSearch := query != nil && (query.Keyword != nil || len(query.Must) > 0 || len(query.Should) > 0)
	performVectorSearch := vectorQueryText != "" && idx.vectorDB != nil && kNearest > 0

	// 并行执行B+树搜索和向量搜索
	if performBTreeSearch {
		wg.Add(1)
		go func() {
			defer wg.Done()
			bTreeResults := idx.searchQuery(tx, query, onFlag, offFlag, orFlags)
			if bTreeResults != nil {
				bTreeErr = fmt.Errorf("B+Tree search failed")
				return
			}

			bTreeDocIds = make(map[string]MvccSkipListValue)
			for _, v := range bTreeResults {
				bTreeDocIds[v.Id] = v
			}
		}()
	}

	if performVectorSearch {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// 生成缓存键
			vectorCacheKey := fmt.Sprintf("vec:%s:%d:%t", vectorQueryText, kNearest, useANN)

			// 检查缓存
			idx.cacheMu.RLock()
			if cache, ok := idx.queryCache[vectorCacheKey]; ok && time.Now().Unix()-cache.timestamp < idx.cacheTTL {
				vectorSearchDocs = make(map[string]float64)
				// 从缓存恢复结果
				for _, id := range cache.results {
					parts := strings.Split(id, ":")
					if len(parts) == 2 {
						score, _ := strconv.ParseFloat(parts[1], 64)
						vectorSearchDocs[parts[0]] = score
					}
				}
				// 更新缓存命中统计
				cacheItem := idx.queryCache[vectorCacheKey]
				cacheItem.hitCount++
				idx.queryCache[vectorCacheKey] = cacheItem

				idx.cacheMu.RUnlock()
			} else {
				idx.cacheMu.RUnlock()

				// 缓存未命中，执行向量搜索
				queryVec, errVec := idx.vectorDB.GetVectorForTextWithCache(vectorQueryText, db.DefaultVectorized)
				if errVec != nil {
					vectorErr = fmt.Errorf("error vectorizing query text for vector search: %w", errVec)
					return
				}

				if queryVec != nil {
					vectorSearchDocs = make(map[string]float64)
					var ids []entity.Result
					var err error

					// 根据数据规模自动选择搜索策略
					vectorCount := len(idx.vectorDB.GetVectors())

					// 大规模数据使用ANN
					if useANN || vectorCount > 1000 {
						// 动态调整ANN参数
						nprobe := 8
						if vectorCount > 10000 {
							nprobe = 16
						} else if vectorCount > 100000 {
							nprobe = 32
						}

						options := db.SearchOptions{
							Nprobe:        nprobe,
							NumHashTables: 4 + vectorCount/10000, // 根据数据规模调整哈希表数量
							UseANN:        true,
						}
						ids, err = idx.vectorDB.HybridSearch(queryVec, kNearest, options, nprobe)
					} else {
						// 小规模数据使用精确搜索
						ids, err = idx.vectorDB.FindNearestWithScores(queryVec, kNearest, 0.0)
					}

					if err != nil {
						vectorErr = fmt.Errorf("vector search failed: %w", err)
						return
					}

					// 处理搜索结果
					for i, id := range ids {
						vectorSearchDocs[id.Id] = ids[i].Similarity
					}

					// 缓存结果
					if len(vectorSearchDocs) > 0 {
						cacheResults := make([]string, 0, len(vectorSearchDocs))
						for id, score := range vectorSearchDocs {
							cacheResults = append(cacheResults, fmt.Sprintf("%s:%f", id, score))
						}

						idx.cacheMu.Lock()
						idx.queryCache[vectorCacheKey] = queryCache{
							results:   cacheResults,
							timestamp: time.Now().Unix(),
							hitCount:  1,
						}
						idx.cacheMu.Unlock()
					}
				}
			}
		}()
	}

	// 等待所有搜索完成
	wg.Wait()
	// 检查错误
	if bTreeErr != nil {
		return nil, bTreeErr
	}
	if vectorErr != nil {
		return nil, vectorErr
	}

	// Combine results and apply scoring
	finalResultCandidates := make(map[string]float64) // docId -> combined_score

	if performBTreeSearch && performVectorSearch {
		// 使用更复杂的评分策略
		for vecId, vecScore := range vectorSearchDocs {
			if bTreeVal, exists := bTreeDocIds[vecId]; exists {
				// 归一化向量相似度分数 (通常在0-1之间)
				normalizedVecScore := vecScore

				// 计算关键词匹配分数
				keywordScore := 1.0

				// 可以根据关键词匹配的数量增加分数
				if query != nil && query.Keyword != nil {
					keywordScore = 1.0
				} else if query != nil && len(query.Must) > 0 {
					// 必须匹配的关键词越多，分数越高
					keywordScore = 1.0 + float64(len(query.Must))*0.2
				} else if query != nil && len(query.Should) > 0 {
					// 可选匹配的关键词也增加分数，但权重较低
					keywordScore = 1.0 + float64(len(query.Should))*0.1
				}

				// 考虑文档特征位
				featureScore := 0.0
				if bTreeVal.BitsFeature&onFlag == onFlag {
					featureScore += 0.5 // 匹配所有必需特征位加分
				}

				// 加权组合分数 (可以调整权重)
				keywordWeight := 0.4
				vectorWeight := 0.5
				featureWeight := 0.1

				combinedScore := keywordWeight*keywordScore +
					vectorWeight*normalizedVecScore +
					featureWeight*featureScore

				finalResultCandidates[vecId] = combinedScore
			}
		}
	} else if performBTreeSearch {
		// Only B+Tree search was performed, assign a default score for sorting
		for id := range bTreeDocIds {
			finalResultCandidates[id] = 1.0 // Default score for BTree matches
		}
	} else if performVectorSearch {
		// Only VectorDB search was performed, use vector similarity score
		for id, score := range vectorSearchDocs {
			finalResultCandidates[id] = score
		}
	} else {
		// No search criteria provided, or neither search type was applicable
		return []string{}, nil
	}

	// Convert a map to slice of structs for sorting
	type ScoredDoc struct {
		Id    string
		Score float64
	}
	scoredDocs := make([]ScoredDoc, 0, len(finalResultCandidates))
	for id, score := range finalResultCandidates {
		scoredDocs = append(scoredDocs, ScoredDoc{Id: id, Score: score})
	}

	// Sort by score in descending order
	sort.Slice(scoredDocs, func(i, j int) bool {
		return scoredDocs[i].Score > scoredDocs[j].Score
	})

	// Extract sorted IDs
	finalResultIds := make([]string, 0, len(scoredDocs))
	for _, doc := range scoredDocs {
		finalResultIds = append(finalResultIds, doc.Id)
	}

	return finalResultIds, nil
}

// 内部递归查询
func (idx *MVCCBPlusTreeInvertedIndex) searchQuery(tx *tree.Transaction, query *messages.TermQuery, onFlag uint64, offFlag uint64, orFlags []uint64) InvertedList {
	switch {
	case query.Keyword != nil:
		val, ok := idx.tree.Get(tx, query.Keyword)
		if !ok || val == nil {
			return nil
		}
		list := val.(InvertedList)
		result := make(InvertedList, 0, len(list))
		for _, v := range list {
			flag := v.BitsFeature
			if idx.FilterBits(flag, onFlag, offFlag, orFlags) {
				result = append(result, v)
			}
		}
		return result
	case len(query.Must) > 0:
		results := make([]InvertedList, 0, len(query.Must))
		for _, q := range query.Must {
			subResult := idx.searchQuery(tx, q, onFlag, offFlag, orFlags)
			if subResult != nil {
				results = append(results, subResult)
			}
		}
		return idx.IntersectionList(results...)
	case len(query.Should) > 0:
		results := make([]InvertedList, 0, len(query.Should))
		for _, q := range query.Should {
			subResult := idx.searchQuery(tx, q, onFlag, offFlag, orFlags)
			if subResult != nil {
				results = append(results, subResult)
			}
		}
		return idx.UnionList(results...)
	}
	return nil
}

// FilterBits 位过滤
func (idx *MVCCBPlusTreeInvertedIndex) FilterBits(bits, onFlag, offFlag uint64, orFlags []uint64) bool {
	if bits&onFlag != onFlag {
		return false
	}
	if bits&offFlag != uint64(0) {
		return false
	}
	for _, orFlag := range orFlags {
		if orFlag > 0 && bits&orFlag <= 0 {
			return false
		}
	}
	return true
}

// UnionList 并集
func (idx *MVCCBPlusTreeInvertedIndex) UnionList(lists ...InvertedList) InvertedList {
	idMap := make(map[string]MvccSkipListValue)
	for _, list := range lists {
		for _, v := range list {
			idMap[v.Id] = v
		}
	}
	result := make(InvertedList, 0, len(idMap))
	for _, v := range idMap {
		result = append(result, v)
	}
	return result
}

// IntersectionList 交集
func (idx *MVCCBPlusTreeInvertedIndex) IntersectionList(lists ...InvertedList) InvertedList {
	if len(lists) == 0 {
		return nil
	}
	if len(lists) == 1 {
		return lists[0]
	}
	idCount := make(map[string]int)
	valMap := make(map[string]MvccSkipListValue)
	for _, list := range lists {
		for _, v := range list {
			idCount[v.Id]++
			valMap[v.Id] = v
		}
	}
	result := make(InvertedList, 0)
	for id, cnt := range idCount {
		if cnt == len(lists) {
			result = append(result, valMap[id])
		}
	}
	return result
}
