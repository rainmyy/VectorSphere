package search

import (
	"seetaSearch/db"
	"seetaSearch/index"
	bplus "seetaSearch/library/BPlus"
	"seetaSearch/messages"
)

type SearchService struct {
	VectorDB      *db.VectorDB
	InvertedIndex *index.MVCCBPlusTreeInvertedIndex
	TxMgr         *bplus.TransactionManager
	LockMgr       *bplus.LockManager
	WAL           *bplus.WALManager
}

func NewSearchService(vectorDBPath string, numClusters int, invertedIndexOrder int, txMgr *bplus.TransactionManager, lockMgr *bplus.LockManager, wal *bplus.WALManager) *SearchService {
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
	err := ss.VectorDB.AddDocument(doc.Id, doc.Content, vectorizedType)
	if err != nil {
		return err
	}

	// 开启事务
	tx := ss.TxMgr.Begin(bplus.Serializable)
	defer ss.TxMgr.Commit(tx)

	// 将文档添加到倒排索引
	err = ss.InvertedIndex.Add(tx, doc)
	if err != nil {
		return err
	}

	return nil
}

// DeleteDocument 从搜索系统中删除文档
func (ss *SearchService) DeleteDocument(docId string, scoreId int64, keyword *messages.KeyWord) error {
	// 从向量数据库中删除文档
	err := ss.VectorDB.Delete(docId)
	if err != nil {
		return err
	}

	// 开启事务
	tx := ss.TxMgr.Begin(bplus.Serializable)
	defer func(TxMgr *bplus.TransactionManager, tx *bplus.Transaction) {
		err := TxMgr.Commit(tx)
		if err != nil {

		}
	}(ss.TxMgr, tx)

	// 从倒排索引中删除文档
	err = ss.InvertedIndex.Delete(tx, scoreId, keyword)
	if err != nil {
		return err
	}

	return nil
}

// Search 支持表达式查询的搜索接口
func (ss *SearchService) Search(query *messages.TermQuery, vectorizedType int, k int, probe int, onFlag uint64, offFlag uint64, orFlags []uint64) ([]string, error) {
	// 使用向量搜索获取候选文件 ID
	candidateIDs, err := ss.VectorDB.FileSystemSearch(query.Content, vectorizedType, k, probe)
	if err != nil {
		return nil, err
	}

	// 开启事务
	tx := ss.TxMgr.Begin(bplus.Serializable)
	defer ss.TxMgr.Commit(tx)

	// 使用倒排索引进行表达式查询
	indexResults := ss.InvertedIndex.Search(tx, query, onFlag, offFlag, orFlags)

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
