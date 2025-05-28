package index

import (
	"fmt"
	"hash/fnv"
	BPlus "seetaSearch/library/bplus"
	"seetaSearch/messages"
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
	tree  *BPlus.MVCCBPlusTree
	order int
	mu    sync.RWMutex
}

func NewMVCCBPlusTreeInvertedIndex(order int, txMgr *BPlus.TransactionManager, lockMgr *BPlus.LockManager, wal *BPlus.WALManager) *MVCCBPlusTreeInvertedIndex {
	return &MVCCBPlusTreeInvertedIndex{
		tree:  BPlus.NewMVCCBPlusTree(order, txMgr, lockMgr, wal),
		order: order,
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

// Add 插入文档到倒排索引
func (idx *MVCCBPlusTreeInvertedIndex) Add(tx *BPlus.Transaction, doc messages.Document) error {
	scoreId := GenerateScoreId(doc)
	for _, keyword := range doc.KeWords {
		val, ok := idx.tree.Get(tx, keyword)
		var list InvertedList
		if ok && val != nil {
			list = val.(InvertedList)
			// 检查是否已存在
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
		if err := idx.tree.Put(tx, keyword, list); err != nil {
			return err
		}
	}
	return nil
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
	current := idx.getFirstLeaf()

	for current != nil {
		for _, key := range current.GetKeys() {
			if kw, ok := key.(messages.KeyWord); ok {
				keywords = append(keywords, kw)
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
		minKey = firstLeaf.GetKeys()[0].(messages.KeyWord)
	}
	if lastLeaf != nil && len(lastLeaf.GetKeys()) > 0 {
		maxKey = lastLeaf.GetKeys()[len(lastLeaf.GetKeys())-1].(messages.KeyWord)
	}
	return minKey, maxKey
}

// getFirstLeaf 获取第一个叶子节点，需要根据实际情况实现
func (idx *MVCCBPlusTreeInvertedIndex) getFirstLeaf() *BPlus.MVCCNode {
	// 这里需要实现获取第一个叶子节点的逻辑
	// 可以从根节点开始遍历到最左边的叶子节点
	current := idx.tree.Root
	for current != nil && !current.IsLeaf {
		current = current.Children[0].(*BPlus.MVCCNode)
	}
	return current
}

// getLastLeaf 获取最后一个叶子节点，需要根据实际情况实现
func (idx *MVCCBPlusTreeInvertedIndex) getLastLeaf() *BPlus.MVCCNode {
	// 这里需要实现获取最后一个叶子节点的逻辑
	// 可以从根节点开始遍历到最右边的叶子节点
	current := idx.tree.Root
	for current != nil && !current.IsLeaf {
		current = current.Children[len(current.Children)-1].(*BPlus.MVCCNode)
	}
	return current
}

// Delete 从倒排索引中删除文档
func (idx *MVCCBPlusTreeInvertedIndex) Delete(tx *BPlus.Transaction, scoreId int64, keyword *messages.KeyWord) error {
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

// Search 查询倒排索引
func (idx *MVCCBPlusTreeInvertedIndex) Search(tx *BPlus.Transaction, query *messages.TermQuery, onFlag uint64, offFlag uint64, orFlags []uint64) []string {
	result := idx.searchQuery(tx, query, onFlag, offFlag, orFlags)
	if result == nil {
		return nil
	}
	arr := make([]string, 0, len(result))
	for _, v := range result {
		arr = append(arr, v.Id)
	}
	return arr
}

// 内部递归查询
func (idx *MVCCBPlusTreeInvertedIndex) searchQuery(tx *BPlus.Transaction, query *messages.TermQuery, onFlag uint64, offFlag uint64, orFlags []uint64) InvertedList {
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
