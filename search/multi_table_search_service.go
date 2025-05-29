package search

import (
	"fmt"
	"seetaSearch/db"
	"seetaSearch/index"
	tree "seetaSearch/library/tree"
	"seetaSearch/messages"
	"sync"
)

// TableInstance 包含一个表的倒排索引和向量数据库实例
type TableInstance struct {
	InvertedIndex *index.MVCCBPlusTreeInvertedIndex
	VectorDB      *db.VectorDB
}

// MultiTableSearchService 管理多个 TableInstance
type MultiTableSearchService struct {
	tables  map[string]*TableInstance
	mu      sync.RWMutex
	TxMgr   *tree.TransactionManager
	LockMgr *tree.LockManager
	WAL     *tree.WALManager
}

// NewMultiTableSearchService 创建一个新的 MultiTableSearchService 实例
func NewMultiTableSearchService(txMgr *tree.TransactionManager, lockMgr *tree.LockManager, wal *tree.WALManager) *MultiTableSearchService {
	return &MultiTableSearchService{
		tables:  make(map[string]*TableInstance),
		TxMgr:   txMgr,
		LockMgr: lockMgr,
		WAL:     wal,
	}
}

// CreateTable 创建一个新的表
func (mts *MultiTableSearchService) CreateTable(tableName string, vectorDBBasePath string, numClusters int, invertedIndexOrder int) error {
	mts.mu.Lock()
	defer mts.mu.Unlock()

	if _, exists := mts.tables[tableName]; exists {
		return fmt.Errorf("table '%s' already exists", tableName)
	}

	// 为新表创建独立的 VectorDB 和 InvertedIndex 实例
	vectorDBPath := fmt.Sprintf("%s/%s", vectorDBBasePath, tableName) // 假设每个表有独立的 VectorDB 存储路径
	vectorDB := db.NewVectorDB(vectorDBPath, numClusters)
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

// AddDocument 添加文档到指定表
func (mts *MultiTableSearchService) AddDocument(tableName string, doc messages.Document, vectorizedType int) (err error) {
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

	// 将文档添加到倒排索引
	err = table.InvertedIndex.Add(tx, doc)
	if err != nil {
		return err
	}

	// 将文档添加到向量数据库
	// 注意：这里将VectorDB的添加放在B+Tree之后，以便在VectorDB添加失败时，
	// B+Tree的事务可以回滚。如果VectorDB添加成功，B+Tree的事务再提交。
	// 这种顺序有助于维护数据一致性，尽管VectorDB本身可能不支持事务回滚。
	if table.VectorDB != nil {
		err = table.VectorDB.AddDocument(doc.Id, string(doc.Bytes), vectorizedType)
		if err != nil {
			// 如果VectorDB添加失败，B+Tree的事务会在defer中回滚
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
func (mts *MultiTableSearchService) Search(tableName string, query *messages.TermQuery, vectorizedType int, k int, probe int, onFlag uint64, offFlag uint64, orFlags []uint64, useANN bool) ([]string, error) {
	table, err := mts.GetTable(tableName)
	if err != nil {
		return nil, err
	}

	var candidateIDs []string
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
		if _, ok := indexResultSet[id]; ok {
			finalResults = append(finalResults, id)
		}
	}

	// TODO: Implement more sophisticated result merging and scoring based on both indexResults and candidateIDs (with scores)

	return finalResults, nil
}
