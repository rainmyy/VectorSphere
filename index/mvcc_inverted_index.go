package index

import (
	"fmt"
	"hash/fnv"
	"seetaSearch/db"
	"seetaSearch/library/entity"
	tree "seetaSearch/library/tree"
	"seetaSearch/messages"
	"sort"
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
}
type MVCCBPlusTreeInvertedIndex struct {
	tree     *tree.MVCCBPlusTree
	vectorDB *db.VectorDB // Add VectorDB field
	order    int
	mu       sync.RWMutex
}

func NewMVCCBPlusTreeInvertedIndex(order int, txMgr *tree.TransactionManager, lockMgr *tree.LockManager, wal *tree.WALManager, vectorDB *db.VectorDB) *MVCCBPlusTreeInvertedIndex {
	return &MVCCBPlusTreeInvertedIndex{
		tree:     tree.NewMVCCBPlusTree(order, txMgr, lockMgr, wal),
		order:    order,
		vectorDB: vectorDB, // Initialize vectorDB
	}
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

}

// Add 插入文档到倒排索引
func (idx *MVCCBPlusTreeInvertedIndex) Add(tx *tree.Transaction, doc messages.Document) error {
	idx.mu.Lock() // Consider lock granularity; this locks the whole Add operation
	defer idx.mu.Unlock()
	scoreId := GenerateScoreId(doc)

	successfullyAddedKeywords := []*messages.KeyWord{} // Track keywords successfully added to B+Tree

	// 1. Add to B+Tree (Inverted List for keywords)
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
				})
			}
		} else {
			list = InvertedList{
				MvccSkipListValue{
					Id:          doc.Id,
					BitsFeature: doc.BitsFeature,
					ScoreId:     scoreId,
				},
			}
		}
		if err := idx.tree.Put(tx, keyword, list); err != nil { // Put within transaction
			// If B+Tree Put fails, we stop and return error. No VectorDB operation was attempted yet.
			return fmt.Errorf("failed to put keyword %v to B+Tree: %w", keyword, err)
		}
		successfullyAddedKeywords = append(successfullyAddedKeywords, keyword) // Mark as successfully added
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
				rollbackErr := idx.rollbackBTreeAdd(tx, doc.Id, successfullyAddedKeywords)
				if rollbackErr != nil {
					fmt.Printf("CRITICAL ERROR: Failed to rollback B+Tree changes for document %s after VectorDB add failure: %v. System might be in an inconsistent state.\n", doc.Id, rollbackErr)
					// A more robust system would require a transaction manager coordinating both DBs
					// or a background process to reconcile inconsistencies.
				}
				return fmt.Errorf("failed to add document %s to VectorDB: %w", doc.Id, err)
			}
		}
	}
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
	// 假设存在一个获取第一个和最后一个叶子节点的方法
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
		keys := firstLeaf.GetKeys()
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
// The 'query' parameter is for keyword-based search (B+Tree).
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
	idx.mu.RLock() // Read lock for search
	defer idx.mu.RUnlock()

	var bTreeDocIds map[string]MvccSkipListValue
	performBTreeSearch := query != nil && (query.Keyword != nil || len(query.Must) > 0 || len(query.Should) > 0)

	if performBTreeSearch {
		bTreeResults := idx.searchQuery(tx, query, onFlag, offFlag, orFlags)
		bTreeDocIds = make(map[string]MvccSkipListValue)
		for _, v := range bTreeResults {
			bTreeDocIds[v.Id] = v
		}
	}

	var vectorSearchDocs map[string]float64 // Map docId to similarity score
	performVectorSearch := vectorQueryText != "" && idx.vectorDB != nil && kNearest > 0

	if performVectorSearch {
		queryVec, errVec := idx.vectorDB.GetVectorForText(vectorQueryText, db.DefaultVectorized) // Or a configurable type
		if errVec != nil {
			return nil, fmt.Errorf("error vectorizing query text for vector search: %w", errVec)
		}
		if queryVec != nil {
			vectorSearchDocs = make(map[string]float64)
			var ids []entity.Result
			var err error
			if useANN || len(idx.vectorDB.GetVectors()) > 1000 { // 触发混合索引条件
				options := db.SearchOptions{Nprobe: 8, NumHashTables: 4, UseANN: true}
				ids, err = idx.vectorDB.HybridSearch(queryVec, kNearest, options)
				if err != nil {
					return nil, fmt.Errorf("vector hybrid search failed: %w", err)
				}
			} else {
				// Assuming SearchSimilar returns []string (doc IDs), []float64 (distances), error
				ids, err = idx.vectorDB.FindNearestWithScores(queryVec, kNearest, 0.0) // 0.0 for no distance threshold, or make it a param
				if err != nil {
					return nil, fmt.Errorf("vector search failed: %w", err)
				}
			}
			for i, id := range ids {
				vectorSearchDocs[id.Id] = ids[i].Similarity
			}
		}
	}

	// Combine results and apply scoring
	finalResultCandidates := make(map[string]float64) // docId -> combined_score

	if performBTreeSearch && performVectorSearch {
		// Intersection: only docs present in both keyword and vector search
		for vecId, vecScore := range vectorSearchDocs {
			if _, exists := bTreeDocIds[vecId]; exists {
				// Simple combination: BTree relevance (binary 1.0) + Vector similarity
				// TODO: Implement a more sophisticated scoring function if needed.
				// For example, you might want to normalize scores or use a weighted sum.
				finalResultCandidates[vecId] = 1.0 + vecScore // Assuming BTree match gives a base score
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
