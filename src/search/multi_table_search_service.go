package search

import (
	"VectorSphere/src/index"
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/log"
	"VectorSphere/src/library/tree"
	"VectorSphere/src/messages"
	"VectorSphere/src/vector"
	"fmt"
	"math/rand"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// TableInstance 包含一个表的倒排索引和向量数据库实例
type TableInstance struct {
	InvertedIndex *index.MVCCBPlusTreeInvertedIndex
	VectorDB      *vector.VectorDB
}

// MultiTableSearchService 结构体中添加缓存字段
type MultiTableSearchService struct {
	tables  map[string]*TableInstance
	mu      sync.RWMutex
	TxMgr   *tree.TransactionManager
	LockMgr *tree.LockManager
	WAL     *tree.WALManager
	// 添加查询缓存
	queryCache   map[string]queryCacheEntry
	queryCacheMu sync.RWMutex
	maxCacheSize int
	cacheTTL     time.Duration
	config       ServiceConfig
}

// 缓存条目结构
type queryCacheEntry struct {
	results    []string
	timestamp  time.Time
	vectorHash uint64 // 用于向量查询的哈希
}

// NewMultiTableSearchService 中初始化配置
func NewMultiTableSearchService(txMgr *tree.TransactionManager, lockMgr *tree.LockManager, wal *tree.WALManager) *MultiTableSearchService {
	mts := &MultiTableSearchService{
		tables:     make(map[string]*TableInstance),
		TxMgr:      txMgr,
		LockMgr:    lockMgr,
		WAL:        wal,
		queryCache: make(map[string]queryCacheEntry),
		config: ServiceConfig{
			MaxCacheSize:          1000,
			CacheTTL:              5 * time.Minute,
			DefaultVectorized:     vector.SimpleVectorized,
			DefaultK:              100,
			DefaultProbe:          10,
			UseAdaptiveConfig:     true,
			MaxWorkers:            runtime.NumCPU(),
			IndexRebuildThreshold: 0.1, // 10%的数据变化触发重建索引
		},
	}

	return mts
}

// ListTables 返回所有表的名称列表
func (mts *MultiTableSearchService) ListTables() ([]string, error) {
	mts.mu.RLock()
	defer mts.mu.RUnlock()

	tableNames := make([]string, 0, len(mts.tables))
	for name := range mts.tables {
		tableNames = append(tableNames, name)
	}

	return tableNames, nil
}

// 添加缓存管理方法
func (mts *MultiTableSearchService) getCachedResults(cacheKey string) ([]string, bool) {
	mts.queryCacheMu.RLock()
	defer mts.queryCacheMu.RUnlock()

	entry, exists := mts.queryCache[cacheKey]
	if !exists {
		return nil, false
	}

	// 检查缓存是否过期
	if time.Since(entry.timestamp) > mts.cacheTTL {
		return nil, false
	}

	return entry.results, true
}

//	func (mts *MultiTableSearchService) GetCacheStats() map[string]interface{} {
//		stats := map[string]interface{}{
//			"size":     len(mts.queryCache),
//			"max_size": mts.maxCacheSize,
//			"hit_rate": float64(mts.stats.CacheHits) / float64(mts.stats.TotalQueries),
//		}
//		return stats
//	}
func (mts *MultiTableSearchService) cacheResults(cacheKey string, results []string, vectorHash uint64) {
	mts.queryCacheMu.Lock()
	defer mts.queryCacheMu.Unlock()

	// 如果缓存已满，清除最旧的条目
	if len(mts.queryCache) >= mts.maxCacheSize {
		mts.evictOldestCacheEntry()
	}

	mts.queryCache[cacheKey] = queryCacheEntry{
		results:    results,
		timestamp:  time.Now(),
		vectorHash: vectorHash,
	}
}

func (mts *MultiTableSearchService) evictOldestCacheEntry() {
	var oldestKey string
	var oldestTime time.Time

	// 找到最旧的缓存条目
	for key, entry := range mts.queryCache {
		if oldestTime.IsZero() || entry.timestamp.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.timestamp
		}
	}

	if oldestKey != "" {
		delete(mts.queryCache, oldestKey)
	}
}

// 生成缓存键
func generateCacheKey(tableName string, query *messages.TermQuery, vectorizedType int, k int, probe int, onFlag uint64, offFlag uint64, orFlags []uint64, useANN bool) string {
	// 将查询参数序列化为缓存键
	key := fmt.Sprintf("%s:%s:%d:%d:%d:%d:%d:%v:%v", tableName,
		query.Keyword.ToString(), vectorizedType, k, probe, onFlag, offFlag, orFlags, useANN)
	return key
}

// CreateTable 创建一个新的表
func (mts *MultiTableSearchService) CreateTable(tableName string, vectorDBBasePath string, numClusters int, invertedIndexOrder int) error {
	mts.mu.Lock()
	defer mts.mu.Unlock()

	if _, exists := mts.tables[tableName]; exists {
		err := fmt.Errorf("table '%s' already exists", tableName)
		mts.logError("CreateTable", err)
		return err
	}
	// 为新表创建独立的 VectorDB 和 InvertedIndex 实例
	vectorDBPath := fmt.Sprintf("%s/%s", vectorDBBasePath, tableName) // 假设每个表有独立的 VectorDB 存储路径
	vectorDB := vector.NewVectorDB(vectorDBPath, numClusters)
	// InvertedIndex 需要 VectorDB 实例，所以先创建 VectorDB
	invertedIndex := index.NewMVCCBPlusTreeInvertedIndex(invertedIndexOrder, mts.TxMgr, mts.LockMgr, mts.WAL, vectorDB)

	mts.tables[tableName] = &TableInstance{
		InvertedIndex: invertedIndex,
		VectorDB:      vectorDB,
	}

	fmt.Printf("Table '%s' created successfully.\n", tableName)

	return nil
}

// GetTable 获取指定名称的表实例
func (mts *MultiTableSearchService) GetTable(tableName string) (*TableInstance, error) {
	mts.mu.RLock()
	defer mts.mu.RUnlock()

	table, ok := mts.tables[tableName]
	if !ok {
		return nil, fmt.Errorf("table '%s' not found", tableName)
	}

	return table, nil
}

// DeleteTable 删除指定名称的表
func (mts *MultiTableSearchService) DeleteTable(tableName string) error {
	mts.mu.Lock()
	defer mts.mu.Unlock()

	table, exists := mts.tables[tableName]
	if !exists {
		return fmt.Errorf("table '%s' not found", tableName)
	}

	// TODO: Implement proper cleanup for the table's resources (closing DBs, etc.)
	// For example, closing VectorDB and InvertedIndex if they have Close methods
	if table.VectorDB != nil {
		table.VectorDB.Close()
	}
	if table.InvertedIndex != nil {
		table.InvertedIndex.Close()
	}

	delete(mts.tables, tableName)

	fmt.Printf("Table '%s' deleted successfully.\n", tableName)

	return nil
}

// CloseTable 关闭指定表的资源
func (mts *MultiTableSearchService) CloseTable(tableName string) error {
	mts.mu.Lock()
	defer mts.mu.Unlock()

	table, exists := mts.tables[tableName]
	if !exists {
		return fmt.Errorf("table '%s' not found", tableName)
	}

	var err error
	// Close VectorDB if it exists and has a Close method
	if table.VectorDB != nil {
		if closer, ok := interface{}(table.VectorDB).(interface{ Close() error }); ok {
			err = closer.Close()
			if err != nil {
				// Log the error but continue to try closing the inverted index
				fmt.Printf("Error closing VectorDB for table '%s': %v\n", tableName, err)
			}
		}
	}

	// Close InvertedIndex if it exists and has a Close method
	if table.InvertedIndex != nil {
		if closer, ok := interface{}(table.InvertedIndex).(interface{ Close() error }); ok {
			indexErr := closer.Close()
			if indexErr != nil {
				// If VectorDB close also had an error, return the VectorDB error first
				if err == nil {
					err = indexErr
				} else {
					// If both failed, combine errors or return a new error indicating multiple failures
					fmt.Printf("Error closing InvertedIndex for table '%s': %v\n", tableName, indexErr)
				}
			}
		}
	}

	// Remove the table from the map regardless of close errors
	delete(mts.tables, tableName)

	fmt.Printf("Table '%s' closed and removed.\n", tableName)

	return err // Return the first error encountered during closing
}

// AddDocument 使用重试机制的 AddDocument 方法
func (mts *MultiTableSearchService) AddDocument(tableName string, doc messages.Document, vectorizedType int) error {
	table, err := mts.GetTable(tableName)
	if err != nil {
		return err
	}

	// 使用重试机制执行事务
	err = mts.withRetry(func(tx *tree.Transaction) error {
		// 将文档添加到倒排索引
		err := table.InvertedIndex.Add(tx, doc)
		if err != nil {
			return err
		}
		return nil
	}, 3) // 最多重试3次

	if err != nil {
		return err
	}

	// 将文档添加到向量数据库
	if table.VectorDB != nil {
		err = table.VectorDB.AddDocument(doc.Id, string(doc.Bytes), vectorizedType)
		if err != nil {
			// 如果向量数据库添加失败，尝试回滚倒排索引的添加
			// 注意：这里可能无法保证原子性，因为倒排索引的事务已经提交
			mts.logError("AddDocument to VectorDB", err)
			return fmt.Errorf("failed to add document %s to VectorDB in table '%s': %w", doc.Id, tableName, err)
		}
	}

	return nil
}

// DeleteDocument 从指定表中删除文档
func (mts *MultiTableSearchService) DeleteDocument(tableName string, docId string, keywords []*messages.KeyWord) (err error) {
	table, err := mts.GetTable(tableName)
	if err != nil {
		return err
	}

	// 开启事务 (使用 MultiTableSearchService 的 TxMgr)
	tx := mts.TxMgr.Begin(tree.Serializable)
	defer func() {
		if err != nil {
			mts.TxMgr.Rollback(tx)
		} else {
			err = mts.TxMgr.Commit(tx)
			if err != nil {
				// 如果提交失败，也需要回滚，并记录错误
				mts.TxMgr.Rollback(tx)
			}
		}
	}()

	// 从倒排索引中删除文档
	err = table.InvertedIndex.Delete(tx, docId, keywords)
	if err != nil {
		return err
	}

	// 从向量数据库中删除文档
	// 注意：这里将VectorDB的删除放在B+Tree之后，以便在VectorDB删除失败时，
	// B+Tree的事务可以回滚。如果VectorDB删除成功，B+Tree的事务再提交。
	// 这种顺序有助于维护数据一致性，尽管VectorDB本身可能不支持事务回滚。
	if table.VectorDB != nil {
		err = table.VectorDB.DeleteVector(docId)
		if err != nil {
			// 如果VectorDB删除失败，B+Tree的事务会在defer中回滚
			return fmt.Errorf("failed to delete document %s from VectorDB in table '%s': %w", docId, tableName, err)
		}
	}

	return nil
}

// Search 在指定表中执行搜索
func (mts *MultiTableSearchService) searchBa(tableName string, query *messages.TermQuery, vectorizedType int, k int, probe int, onFlag uint64, offFlag uint64, orFlags []uint64, useANN bool) ([]string, error) {
	table, err := mts.GetTable(tableName)
	if err != nil {
		return nil, err
	}

	var candidateIDs []entity.Result
	// 使用向量搜索获取候选文件 ID
	if table.VectorDB != nil && (useANN || table.VectorDB.GetDataSize() > 1000) { // 根据useANN或数据量决定是否使用ANN
		// 使用 HybridSearch 或 FindNearestWithScores
		// 假设 HybridSearch 接受与 FileSystemSearch 类似的参数并返回 []string
		// 如果 HybridSearch 不存在或接口不同，可能需要调整或使用 FindNearestWithScores
		// 注意：HybridSearch 和 FindNearestWithScores 返回的可能是带分数的结构体，需要提取 ID
		// 这里简化处理，假设返回 []string
		// TODO: Adjust based on actual VectorDB search method signatures and return types
		// For now, let's assume a method that returns doc IDs based on vector search
		// Example using FindNearestWithScores (assuming it returns []entity.SearchResult or similar)
		// results, err := table.VectorDB.FindNearestWithScores(query.Keyword.ToString(), k, probe)
		// if err == nil { for _, res := range results { candidateIDs = append(candidateIDs, res.ID) } }

		// Using FileSystemSearch as a placeholder based on previous context
		candidateIDs, err = table.VectorDB.FileSystemSearch(query.Keyword.ToString(), vectorizedType, k, probe)
		if err != nil {
			return nil, fmt.Errorf("vector search failed in table '%s': %w", tableName, err)
		}

	} else if table.VectorDB != nil { // 如果不使用ANN或数据量不大，可以使用FindNearestWithScores
		// TODO: Use FindNearestWithScores and process results to get IDs
		// Example:
		// results, err := table.VectorDB.FindNearestWithScores(query.Keyword.ToString(), k, probe)
		// if err == nil { for _, res := range results { candidateIDs = append(candidateIDs, res.ID) } }

		// Using FileSystemSearch as a placeholder
		candidateIDs, err = table.VectorDB.FileSystemSearch(query.Keyword.ToString(), vectorizedType, k, probe)
		if err != nil {
			return nil, fmt.Errorf("vector search failed in table '%s': %w", tableName, err)
		}
	}

	// 开启事务 (使用 MultiTableSearchService 的 TxMgr)
	tx := mts.TxMgr.Begin(tree.Serializable)
	defer mts.TxMgr.Commit(tx)

	// 使用倒排索引进行表达式查询
	indexResults, err := table.InvertedIndex.Search(tx, query, onFlag, offFlag, orFlags, query.Keyword.ToString(), k, useANN)
	if err != nil {
		return nil, fmt.Errorf("inverted index search failed in table '%s': %w", tableName, err)
	}

	// 合并结果：取倒排索引结果和向量搜索候选 ID 的交集
	finalResults := make([]string, 0)
	indexResultSet := make(map[string]struct{})
	for _, id := range indexResults {
		indexResultSet[id] = struct{}{}
	}

	for _, id := range candidateIDs {
		if _, ok := indexResultSet[id.Id]; ok {
			finalResults = append(finalResults, id.Id)
		}
	}

	// TODO: Implement more sophisticated result merging and scoring based on both indexResults and candidateIDs (with scores)

	return finalResults, nil
}

// Search 在指定表中执行搜索 - 优化版本
func (mts *MultiTableSearchService) Search(tableName string, query *messages.TermQuery, vectorizedType int, k int, probe int, onFlag uint64, offFlag uint64, orFlags []uint64, useANN bool) ([]string, error) {
	table, err := mts.GetTable(tableName)
	if err != nil {
		return nil, err
	}

	// 生成缓存键并检查缓存
	cacheKey := generateCacheKey(tableName, query, vectorizedType, k, probe, onFlag, offFlag, orFlags, useANN)
	if cachedResults, found := mts.getCachedResults(cacheKey); found {
		return cachedResults, nil
	}

	var candidateIDs []string
	var vectorResults []entity.Result
	var vectorHash uint64

	// 使用向量搜索获取候选文件 ID
	if table.VectorDB != nil && query.Keyword != nil && query.Keyword.ToString() != "" {
		// 获取查询向量
		queryVector, err := table.VectorDB.GetVectorForTextWithCache(query.Keyword.ToString(), vectorizedType)
		if err != nil {
			return nil, fmt.Errorf("failed to vectorize query text: %w", err)
		}

		// 计算向量哈希用于缓存
		vectorHash = acceler.ComputeVectorHash(queryVector)

		// 根据数据规模和用户设置决定使用哪种搜索方法
		dataSize := table.VectorDB.GetDataSize()
		if useANN || dataSize > 1000 {
			// 使用 HybridSearch 进行近似最近邻搜索
			options := vector.SearchOptions{
				Nprobe:        probe,
				UseANN:        true,
				SearchTimeout: 5 * time.Second, // 可配置的超时时间
			}
			vectorResults, err = table.VectorDB.HybridSearch(queryVector, k, options, probe)
		} else {
			// 使用 FindNearestWithScores 进行精确搜索
			vectorResults, err = table.VectorDB.FindNearestWithScores(queryVector, k, probe)
		}

		if err != nil {
			return nil, fmt.Errorf("vector search failed in table '%s': %w", tableName, err)
		}

		// 提取ID
		for _, result := range vectorResults {
			candidateIDs = append(candidateIDs, result.Id)
		}
	}

	// 开启事务 (使用 MultiTableSearchService 的 TxMgr)
	tx := mts.TxMgr.Begin(tree.Serializable)
	defer func(TxMgr *tree.TransactionManager, tx *tree.Transaction) {
		err := TxMgr.Commit(tx)
		if err != nil {
			log.Error("tx commit has error:%v", err.Error())
		}
	}(mts.TxMgr, tx)

	// 使用倒排索引进行表达式查询
	indexResults, err := table.InvertedIndex.Search(tx, query, onFlag, offFlag, orFlags, query.Keyword.ToString(), k, useANN)
	if err != nil {
		return nil, fmt.Errorf("inverted index search failed in table '%s': %w", tableName, err)
	}

	// 合并结果：如果有向量搜索结果，取交集；否则直接使用倒排索引结果
	var finalResults []string
	if len(candidateIDs) > 0 {
		// 创建倒排索引结果集
		indexResultSet := make(map[string]struct{})
		for _, id := range indexResults {
			indexResultSet[id] = struct{}{}
		}

		// 取交集并保持向量搜索的排序
		for _, id := range candidateIDs {
			if _, ok := indexResultSet[id]; ok {
				finalResults = append(finalResults, id)
			}
		}

		// 如果交集为空但两个结果集都不为空，可能是因为过滤条件太严格
		// 在这种情况下，可以考虑放宽条件或使用其中一个结果集
		if len(finalResults) == 0 && len(indexResults) > 0 && len(candidateIDs) > 0 {
			// 策略1：使用倒排索引结果（更精确的关键词匹配）
			finalResults = indexResults
			// 策略2：使用向量搜索结果（更好的语义相似性）
			// finalResults = candidateIDs
		}
	} else {
		// 如果没有向量搜索结果，直接使用倒排索引结果
		finalResults = indexResults
	}

	// 缓存结果
	mts.cacheResults(cacheKey, finalResults, vectorHash)

	return finalResults, nil
}

// ExecuteQuery 执行类SQL查询 - 优化版本
func (mts *MultiTableSearchService) ExecuteQuery(query string) ([]string, error) {
	// 1. 查询解析器 - 使用更强大的正则表达式
	re := regexp.MustCompile(`(?i)SELECT\s+(.+?)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?(?:\s+ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?)?(?:\s+LIMIT\s+(\d+))?(?:\s+OFFSET\s+(\d+))?`)
	matches := re.FindStringSubmatch(query)

	if len(matches) == 0 {
		return nil, fmt.Errorf("invalid query format: %s", query)
	}

	_ = strings.TrimSpace(matches[1])
	tableName := strings.TrimSpace(matches[2])
	whereClause := strings.TrimSpace(matches[3])
	orderByField := strings.TrimSpace(matches[4])
	_ = strings.TrimSpace(matches[5])
	limitStr := strings.TrimSpace(matches[6])
	offsetStr := strings.TrimSpace(matches[7])

	// 验证表是否存在
	if _, err := mts.GetTable(tableName); err != nil {
		return nil, fmt.Errorf("table '%s' not found: %w", tableName, err)
	}

	// 解析 LIMIT 和 OFFSET
	limit := -1 // -1 表示无限制
	if limitStr != "" {
		l, err := strconv.Atoi(limitStr)
		if err != nil {
			return nil, fmt.Errorf("invalid LIMIT value: %w", err)
		}
		if l < 0 {
			return nil, fmt.Errorf("LIMIT cannot be negative")
		}
		limit = l
	}

	offset := 0
	if offsetStr != "" {
		o, err := strconv.Atoi(offsetStr)
		if err != nil {
			return nil, fmt.Errorf("invalid OFFSET value: %w", err)
		}
		if o < 0 {
			return nil, fmt.Errorf("OFFSET cannot be negative")
		}
		offset = o
	}

	// 解析 WHERE 子句
	var keywordQuery string
	var vectorQueryText string
	var vectorizedType int = 0 // 默认向量化类型
	var k int = 100            // 默认返回结果数量
	var probe int = 10         // 默认探测簇数量
	var onFlag uint64 = 0
	var offFlag uint64 = 0
	var orFlags []uint64
	var useANN bool = false

	if whereClause != "" {
		// 解析关键词查询
		keywordRe := regexp.MustCompile(`keyword\s*=\s*'([^']+)'`)
		keywordMatch := keywordRe.FindStringSubmatch(whereClause)
		if len(keywordMatch) > 1 {
			keywordQuery = keywordMatch[1]
		}

		// 解析向量查询
		vectorRe := regexp.MustCompile(`vector_query\s*=\s*'([^']+)'`)
		vectorMatch := vectorRe.FindStringSubmatch(whereClause)
		if len(vectorMatch) > 1 {
			vectorQueryText = vectorMatch[1]
			useANN = true // 如果有向量查询，默认使用ANN
		}

		// 解析向量化类型
		typeRe := regexp.MustCompile(`vectorized_type\s*=\s*(\d+)`)
		typeMatch := typeRe.FindStringSubmatch(whereClause)
		if len(typeMatch) > 1 {
			vectorizedType, _ = strconv.Atoi(typeMatch[1])
		}

		// 解析k值
		kRe := regexp.MustCompile(`k\s*=\s*(\d+)`)
		kMatch := kRe.FindStringSubmatch(whereClause)
		if len(kMatch) > 1 {
			k, _ = strconv.Atoi(kMatch[1])
		}

		// 解析probe值
		probeRe := regexp.MustCompile(`probe\s*=\s*(\d+)`)
		probeMatch := probeRe.FindStringSubmatch(whereClause)
		if len(probeMatch) > 1 {
			probe, _ = strconv.Atoi(probeMatch[1])
		}

		// 解析useANN标志
		annRe := regexp.MustCompile(`use_ann\s*=\s*(true|false)`)
		annMatch := annRe.FindStringSubmatch(whereClause)
		if len(annMatch) > 1 {
			useANN = annMatch[1] == "true"
		}

		// 解析onFlag
		onFlagRe := regexp.MustCompile(`on_flag\s*=\s*(\d+)`)
		onFlagMatch := onFlagRe.FindStringSubmatch(whereClause)
		if len(onFlagMatch) > 1 {
			onFlag, _ = strconv.ParseUint(onFlagMatch[1], 10, 64)
		}

		// 解析offFlag
		offFlagRe := regexp.MustCompile(`off_flag\s*=\s*(\d+)`)
		offFlagMatch := offFlagRe.FindStringSubmatch(whereClause)
		if len(offFlagMatch) > 1 {
			offFlag, _ = strconv.ParseUint(offFlagMatch[1], 10, 64)
		}

		// 解析orFlags
		orFlagsRe := regexp.MustCompile(`or_flags\s*=\s*\[(\d+(?:,\s*\d+)*)\]`)
		orFlagsMatch := orFlagsRe.FindStringSubmatch(whereClause)
		if len(orFlagsMatch) > 1 {
			flagsStr := strings.Split(orFlagsMatch[1], ",")
			for _, flagStr := range flagsStr {
				flagStr = strings.TrimSpace(flagStr)
				flag, err := strconv.ParseUint(flagStr, 10, 64)
				if err == nil {
					orFlags = append(orFlags, flag)
				}
			}
		}
	}

	// 构建 TermQuery
	termQuery := &messages.TermQuery{
		Keyword: &messages.KeyWord{Word: keywordQuery},
	}

	// 如果提供了向量查询文本，设置到 TermQuery 中
	if vectorQueryText != "" {
		termQuery.Keyword.Word = vectorQueryText
	}

	// 调用底层的 Search 方法
	searchResults, err := mts.Search(tableName, termQuery, vectorizedType, k, probe, onFlag, offFlag, orFlags, useANN)

	if err != nil {
		return nil, fmt.Errorf("error executing search for table '%s': %w", tableName, err)
	}

	// 处理 ORDER BY
	if orderByField != "" {
		// 这里可以实现排序逻辑
		// 例如，如果 orderByField 是 "score"，可以根据相似度分数排序
		// 目前简化处理，假设结果已经按相关性排序
	}

	// 处理 LIMIT 和 OFFSET
	start := offset
	end := len(searchResults)
	if limit >= 0 {
		end = offset + limit
	}

	if end > len(searchResults) {
		end = len(searchResults)
	}

	if start >= len(searchResults) {
		return []string{}, nil // Offset 超出结果范围，返回空
	}

	return searchResults[start:end], nil
}

// BatchAddDocuments 批量添加文档到指定表
func (mts *MultiTableSearchService) BatchAddDocuments(tableName string, docs []messages.Document, vectorizedType int) error {
	table, err := mts.GetTable(tableName)
	if err != nil {
		return err
	}

	// 开启事务
	tx := mts.TxMgr.Begin(tree.Serializable)
	defer func() {
		if err != nil {
			mts.TxMgr.Rollback(tx)
		} else {
			err = mts.TxMgr.Commit(tx)
			if err != nil {
				mts.TxMgr.Rollback(tx)
			}
		}
	}()

	// 批量添加到倒排索引
	for _, doc := range docs {
		err = table.InvertedIndex.Add(tx, doc)
		if err != nil {
			return fmt.Errorf("failed to add document %s to inverted index: %w", doc.Id, err)
		}
	}

	// 批量添加到向量数据库
	if table.VectorDB != nil {
		// 使用工作池并行处理
		numWorkers := runtime.NumCPU()
		workChan := make(chan messages.Document, len(docs))
		errChan := make(chan error, len(docs))
		doneChan := make(chan bool, 1)

		// 启动工作协程
		var wg sync.WaitGroup
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for doc := range workChan {
					err := table.VectorDB.AddDocument(doc.Id, string(doc.Bytes), vectorizedType)
					if err != nil {
						errChan <- fmt.Errorf("failed to add document %s to VectorDB: %w", doc.Id, err)
						return
					}
				}
			}()
		}

		// 发送工作
		go func() {
			for _, doc := range docs {
				workChan <- doc
			}
			close(workChan)
		}()

		// 等待所有工作完成
		go func() {
			wg.Wait()
			close(errChan)
			doneChan <- true
		}()

		// 收集错误
		select {
		case err := <-errChan:
			if err != nil {
				return err
			}
		case <-doneChan:
			// 所有工作完成，无错误
		}
	}

	return nil
}

// BatchDeleteDocuments 批量删除文档
func (mts *MultiTableSearchService) BatchDeleteDocuments(tableName string, docIds []string, keywordsList [][]*messages.KeyWord) error {
	if len(docIds) != len(keywordsList) {
		return fmt.Errorf("docIds and keywordsList must have the same length")
	}

	table, err := mts.GetTable(tableName)
	if err != nil {
		return err
	}

	// 开启事务
	tx := mts.TxMgr.Begin(tree.Serializable)
	defer func() {
		if err != nil {
			mts.TxMgr.Rollback(tx)
		} else {
			err = mts.TxMgr.Commit(tx)
			if err != nil {
				mts.TxMgr.Rollback(tx)
			}
		}
	}()

	// 批量从倒排索引中删除
	for i, docId := range docIds {
		err = table.InvertedIndex.Delete(tx, docId, keywordsList[i])
		if err != nil {
			return fmt.Errorf("failed to delete document %s from inverted index: %w", docId, err)
		}
	}

	// 批量从向量数据库中删除
	if table.VectorDB != nil {
		// 使用工作池并行处理
		numWorkers := runtime.NumCPU()
		workChan := make(chan string, len(docIds))
		errChan := make(chan error, len(docIds))
		doneChan := make(chan bool, 1)

		// 启动工作协程
		var wg sync.WaitGroup
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for docId := range workChan {
					err := table.VectorDB.DeleteVector(docId)
					if err != nil {
						errChan <- fmt.Errorf("failed to delete document %s from VectorDB: %w", docId, err)
						return
					}
				}
			}()
		}

		// 发送工作
		go func() {
			for _, docId := range docIds {
				workChan <- docId
			}
			close(workChan)
		}()

		// 等待所有工作完成
		go func() {
			wg.Wait()
			close(errChan)
			doneChan <- true
		}()

		// 收集错误
		select {
		case err := <-errChan:
			if err != nil {
				return err
			}
		case <-doneChan:
			// 所有工作完成，无错误
		}
	}

	return nil
}

// ServiceStats 服务统计信息结构
type ServiceStats struct {
	TableCount     int
	TotalDocuments int
	CacheSize      int
	CacheHitRate   float64
	AvgQueryTime   time.Duration
	TableStats     map[string]TableStats
}

type TableStats struct {
	DocumentCount int
	VectorDBStats vector.PerformanceStats
	IndexedStatus bool
	LastUpdated   time.Time
}

// GetServiceStats 获取服务统计信息
func (mts *MultiTableSearchService) GetServiceStats() ServiceStats {
	mts.mu.RLock()
	defer mts.mu.RUnlock()

	stats := ServiceStats{
		TableCount: len(mts.tables),
		TableStats: make(map[string]TableStats),
	}

	var totalDocs int

	// 收集每个表的统计信息
	for name, table := range mts.tables {
		tableStats := TableStats{}

		if table.VectorDB != nil {
			tableStats.DocumentCount = table.VectorDB.GetDataSize()
			tableStats.VectorDBStats = table.VectorDB.GetStats()
			tableStats.IndexedStatus = table.VectorDB.IsIndexed()
		}

		stats.TableStats[name] = tableStats
		totalDocs += tableStats.DocumentCount
	}

	stats.TotalDocuments = totalDocs

	// 缓存统计
	mts.queryCacheMu.RLock()
	stats.CacheSize = len(mts.queryCache)
	// 这里可以添加缓存命中率计算
	mts.queryCacheMu.RUnlock()

	return stats
}

// HealthCheck 健康检查方法
func (mts *MultiTableSearchService) HealthCheck() (bool, map[string]string) {
	health := make(map[string]string)
	allHealthy := true

	// 检查事务管理器
	if mts.TxMgr == nil {
		health["TxMgr"] = "Not initialized"
		allHealthy = false
	} else {
		health["TxMgr"] = "OK"
	}

	// 检查锁管理器
	if mts.LockMgr == nil {
		health["LockMgr"] = "Not initialized"
		allHealthy = false
	} else {
		health["LockMgr"] = "OK"
	}

	// 检查WAL管理器
	if mts.WAL == nil {
		health["WAL"] = "Not initialized"
		allHealthy = false
	} else {
		health["WAL"] = "OK"
	}

	// 检查表状态
	mts.mu.RLock()
	tableCount := len(mts.tables)
	mts.mu.RUnlock()

	health["TableCount"] = fmt.Sprintf("%d", tableCount)

	return allHealthy, health
}

// ServiceConfig 服务配置结构
type ServiceConfig struct {
	MaxCacheSize          int
	CacheTTL              time.Duration
	DefaultVectorized     int
	DefaultK              int
	DefaultProbe          int
	UseAdaptiveConfig     bool
	MaxWorkers            int
	IndexRebuildThreshold float64
}

// UpdateConfig 更新服务配置
func (mts *MultiTableSearchService) UpdateConfig(newConfig ServiceConfig) {
	mts.mu.Lock()
	defer mts.mu.Unlock()

	mts.config = newConfig

	// 如果缓存大小减小，可能需要清理一些缓存条目
	if newConfig.MaxCacheSize < mts.maxCacheSize {
		mts.queryCacheMu.Lock()
		for len(mts.queryCache) > newConfig.MaxCacheSize {
			mts.evictOldestCacheEntry()
		}
		mts.queryCacheMu.Unlock()
	}

	mts.maxCacheSize = newConfig.MaxCacheSize
	mts.cacheTTL = newConfig.CacheTTL
}

// AdaptiveOptimize 自适应优化方法
func (mts *MultiTableSearchService) AdaptiveOptimize() {
	if !mts.config.UseAdaptiveConfig {
		return
	}

	// 获取系统资源使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// 根据内存使用情况调整缓存大小
	if m.Alloc > 1024*1024*1024 { // 如果内存使用超过1GB
		// 减小缓存大小
		newMaxCache := mts.maxCacheSize / 2
		if newMaxCache < 100 {
			newMaxCache = 100 // 最小缓存大小
		}

		mts.queryCacheMu.Lock()
		for len(mts.queryCache) > newMaxCache {
			mts.evictOldestCacheEntry()
		}
		mts.maxCacheSize = newMaxCache
		mts.queryCacheMu.Unlock()
	}

	// 对每个表进行优化
	mts.mu.RLock()
	tables := make([]*TableInstance, 0, len(mts.tables))
	for _, table := range mts.tables {
		tables = append(tables, table)
	}
	mts.mu.RUnlock()

	for _, table := range tables {
		if table.VectorDB != nil {
			// 调用VectorDB的自适应配置调整
			table.VectorDB.AdjustConfig()
		}
	}
}

// 封装错误处理函数
func (mts *MultiTableSearchService) logError(operation string, err error) {
	log.Error("[ERROR] %s: %v", operation, err)
}

// 封装日志记录函数
func (mts *MultiTableSearchService) logInfo(format string, args ...interface{}) {
	log.Info("[INFO] "+format, args...)
}

// 事务重试函数
func (mts *MultiTableSearchService) withRetry(operation func(*tree.Transaction) error, maxRetries int) error {
	var err error
	for i := 0; i < maxRetries; i++ {
		tx := mts.TxMgr.Begin(tree.Serializable)
		err = operation(tx)
		if err != nil {
			mts.TxMgr.Rollback(tx)
			if strings.Contains(err.Error(), "transaction conflict") {
				// 如果是事务冲突，等待一小段时间后重试
				time.Sleep(time.Duration(rand.Intn(50)+10) * time.Millisecond)
				continue
			}
			// 其他错误直接返回
			return err
		}

		// 尝试提交事务
		err = mts.TxMgr.Commit(tx)
		if err != nil {
			mts.TxMgr.Rollback(tx)
			if strings.Contains(err.Error(), "transaction conflict") {
				// 如果是事务冲突，等待一小段时间后重试
				time.Sleep(time.Duration(rand.Intn(50)+10) * time.Millisecond)
				continue
			}
			// 其他错误直接返回
			return err
		}

		// 事务成功提交
		return nil
	}

	// 达到最大重试次数
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, err)
}
