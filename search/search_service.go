package search

import (
	"fmt"
	"seetaSearch/db"
	"seetaSearch/index"
	tree "seetaSearch/library/tree"
	"seetaSearch/messages"
)

type SearchService struct {
	VectorDB      *db.VectorDB
	InvertedIndex *index.MVCCBPlusTreeInvertedIndex
	TxMgr         *tree.TransactionManager
	LockMgr       *tree.LockManager
	WAL           *tree.WALManager
}

func NewSearchService(vectorDBPath string, numClusters int, invertedIndexOrder int, txMgr *tree.TransactionManager, lockMgr *tree.LockManager, wal *tree.WALManager) *SearchService {
	vectorDB := db.NewVectorDB(vectorDBPath, numClusters)
	invertedIndex := index.NewMVCCBPlusTreeInvertedIndex(invertedIndexOrder, txMgr, lockMgr, wal)

	return &SearchService{
		VectorDB:      vectorDB,
		InvertedIndex: invertedIndex,
		TxMgr:         txMgr,
		LockMgr:       lockMgr,
		WAL:           wal,
	}
}

// AddDocument 添加文档到搜索系统
func (ss *SearchService) AddDocument(doc messages.Document, vectorizedType int) error {
	// 将文档添加到向量数据库
	err := ss.VectorDB.AddDocument(doc.Id, string(doc.Bytes), vectorizedType)
	if err != nil {
		return err
	}

	// 开启事务
	tx := ss.TxMgr.Begin(tree.Serializable)
	defer func() {
		if err != nil {
			ss.TxMgr.Rollback(tx)
		} else {
			ss.TxMgr.Commit(tx)
		}
	}()

	// 将文档添加到倒排索引
	err = ss.InvertedIndex.Add(tx, doc)
	if err != nil {
		return err
	}

	return nil
}

// DeleteDocument 从搜索系统中删除文档
func (ss *SearchService) DeleteDocument(docId string, keywords []*messages.KeyWord) (err error) {
	// 开启事务
	tx := ss.TxMgr.Begin(tree.Serializable)
	defer func() {
		if err != nil {
			ss.TxMgr.Rollback(tx)
		} else {
			err = ss.TxMgr.Commit(tx)
			if err != nil {
				// 如果提交失败，也需要回滚，并记录错误
				ss.TxMgr.Rollback(tx)
			}
		}
	}()

	// 从倒排索引中删除文档
	err = ss.InvertedIndex.Delete(tx, docId, keywords)
	if err != nil {
		return err
	}

	// 从向量数据库中删除文档
	// 注意：这里将VectorDB的删除放在B+Tree之后，以便在VectorDB删除失败时，
	// B+Tree的事务可以回滚。如果VectorDB删除成功，B+Tree的事务再提交。
	// 这种顺序有助于维护数据一致性，尽管VectorDB本身可能不支持事务回滚。
	if ss.VectorDB != nil { // 添加对VectorDB实例的判断
		err = ss.VectorDB.DeleteVector(docId)
		if err != nil {
			// 如果VectorDB删除失败，B+Tree的事务会在defer中回滚
			return fmt.Errorf("failed to delete document %s from VectorDB: %w", docId, err)
		}
	}

	return nil
}

// Search 支持表达式查询的搜索接口
func (ss *SearchService) Search(query *messages.TermQuery, vectorizedType int, k int, probe int, onFlag uint64, offFlag uint64, orFlags []uint64) ([]string, error) {
	// 使用向量搜索获取候选文件 ID
	candidateIDs, err := ss.VectorDB.FileSystemSearch(query.Keyword.ToString(), vectorizedType, k, probe)
	if err != nil {
		return nil, err
	}

	// 开启事务
	tx := ss.TxMgr.Begin(tree.Serializable)
	defer ss.TxMgr.Commit(tx)

	// 使用倒排索引进行表达式查询
	indexResults, _ := ss.InvertedIndex.Search(tx, query, onFlag, offFlag, orFlags)

	// 合并结果
	var finalResults []string
	for _, id := range candidateIDs {
		for _, resultId := range indexResults {
			if id == resultId {
				finalResults = append(finalResults, id)
				break
			}
		}
	}

	return finalResults, nil
}

//func main() {
//	// 初始化事务管理器、锁管理器和 WAL 管理器
//	txMgr := bplus.NewTransactionManager()
//	lockMgr := bplus.NewLockManager()
//	wal, _ := bplus.NewWALManager()
//
//	// 初始化搜索服务
//	searchService := NewSearchService("path/to/vector.db", 10, 4, txMgr, lockMgr, wal)
//
//	// 添加文档
//	doc := messages.Document{
//		Id:      "1",
//		Content: "这是一个测试文档",
//		// 其他字段需要根据实际情况填充
//	}
//	err := searchService.AddDocument(doc, 2)
//	if err != nil {
//		panic(err)
//	}
//
//	// 查询文档
//	query := &messages.TermQuery{
//		Content: "测试",
//		// 其他字段需要根据实际情况填充
//	}
//	results, err := searchService.Search(query, 2, 5, 1, 0, 0, nil)
//	if err != nil {
//		panic(err)
//	}
//}
