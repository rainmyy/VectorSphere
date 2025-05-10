package index

import (
	farmhash "github.com/leemcloughlin/gofarmhash"
	"github.com/rainmyy/easyDB/library/collect"
	"github.com/rainmyy/easyDB/library/strategy"
	"runtime"
	"sync"
)

type Keyword struct {
	Field string `protobuf:"bytes,1,opt,name=Field,proto3" json:"Field,omitempty"`
	Word  string `protobuf:"bytes,2,opt,name=Word,proto3" json:"Word,omitempty"`
}

func (kw Keyword) ToString() string {
	if len(kw.Word) > 0 {
		return kw.Field + "\001" + kw.Word
	}

	return ""
}

type Document struct {
	Id          string     `protobuf:"bytes,1,opt,name=Id,proto3" json:"id,omitempty"`
	FloatId     float64    `protobuf:"varint,2,opt,name=IntId,proto3" json:"intId,omitempty"`
	BitsFeature uint64     `protobuf:"varint,3,opt,name=IntId,proto3" json:"bitsFeature,omitempty"`
	KeyWords    []*Keyword `protobuf:"bytes,4,rep,name=KeyWords,proto3" json:"keyWords,omitempty"`
	Bytes       []byte     `protobuf:"byte,5,opt,name=Bytes,proto3" json:"bytes,omitempty"`
}

type TermQuery struct {
	Keyword *Keyword     `protobuf:"bytes,1,opt,name=Keyword,proto3" json:"Keyword,omitempty"`
	Must    []*TermQuery `protobuf:"bytes,2,rep,name=Must,proto3" json:"Must,omitempty"`
	Should  []*TermQuery `protobuf:"bytes,3,rep,name=Should,proto3" json:"Should,omitempty"`
}
type SkipListValue struct {
	Id          string
	BitsFeature uint64
}
type IRverseIndex interface {
	Add(doc Document)
	Delete(floatId float64, keyword *Keyword)
	Search(query *TermQuery, onFlag uint64, offFlag uint64, orFlags []uint64) []string
}

var _ IRverseIndex = (*SkipListInvertedIndex)(nil)

type SkipListInvertedIndex struct {
	table *collect.HashMap
	locks []sync.RWMutex
}

func NewSkipListInvertedIndex(docNumEstimate int) *SkipListInvertedIndex {
	return &SkipListInvertedIndex{
		table: collect.NewHashMap(runtime.NumCPU(), docNumEstimate),
		locks: make([]sync.RWMutex, 1000),
	}
}

func (index *SkipListInvertedIndex) Add(doc Document) {
	for _, keyword := range doc.KeyWords {
		key := keyword.ToString()
		lock := index.getLock(key)
		skipListValue := SkipListValue{
			Id:          doc.Id,
			BitsFeature: doc.BitsFeature,
		}
		lock.Lock()
		if value, exist := index.table.Get(key); exist {
			list := value.(*strategy.SkipList)
			list.Insert(doc.FloatId, skipListValue)
		} else {
			list := strategy.NewSkipList()
			list.Insert(doc.FloatId, skipListValue)
			index.table.Set(key, list)
		}
		lock.Unlock()
	}
}

func (index *SkipListInvertedIndex) getLock(key string) *sync.RWMutex {
	n := int(farmhash.Hash32WithSeed([]byte(key), 0))
	return &index.locks[n%len(index.locks)]
}

func (index *SkipListInvertedIndex) Delete(floatId float64, keyword *Keyword) {
	key := keyword.ToString()
	lock := index.getLock(key)
	lock.Lock()
	defer lock.Unlock()

	if value, ok := index.table.Get(key); ok {
		list := value.(*strategy.SkipList)
		list.Delete(floatId)
	}
}

func (index *SkipListInvertedIndex) Search(query *TermQuery, onFlag uint64, offFlag uint64, orFlag []uint64) []string {
	result := index.searchQuery(query, onFlag, offFlag, orFlag)
	if result == nil {
		return nil
	}

	arr := make([]string, 0, result.Len())
	node := result.Front()
	for node != nil {
		skipListValue := node.Value.(SkipListValue)
		arr = append(arr, skipListValue.Id)
		node = node.Next()
	}
	return arr
}

func (index *SkipListInvertedIndex) searchQuery(query *TermQuery, onFlag uint64, offFlag uint64, orFlags []uint64) *strategy.SkipList {
	switch {
	case query.Keyword != nil:
		keyWord := query.Keyword.ToString()
		if value, ok := index.table.Get(keyWord); ok {
			list := value.(*strategy.SkipList)
			result := strategy.NewSkipList()
			node := list.Front()
			for node != nil {
				intId := node.Key().(float64)
				skipListValue := node.Value.(SkipListValue)
				flag := skipListValue.BitsFeature
				if intId > 0 && index.FilterBits(flag, onFlag, offFlag, orFlags) {
					result.Insert(intId, skipListValue)
				}

				node = node.Next()
			}
		}
	case len(query.Must) > 0:
		results := make([]*strategy.SkipList, 0, len(query.Must))
		for _, q := range query.Must {
			results = append(results, index.searchQuery(q, offFlag, offFlag, orFlags))
		}
		return index.IntersectionList(results...)
	case len(query.Should) > 0:
		results := make([]*strategy.SkipList, 0, len(query.Should))
		for _, q := range query.Should {
			results = append(results, index.searchQuery(q, offFlag, offFlag, orFlags))
		}

		return index.IntersectionList(results...)
	}

	return nil
}

func (index *SkipListInvertedIndex) FilterBits(bits, onFlag, offFlag uint64, orFlags []uint64) bool {
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

func (index *SkipListInvertedIndex) IntersectionList(lists ...*strategy.SkipList) *strategy.SkipList {
	if len(lists) == 0 {
		return nil
	}
	if len(lists) == 1 {
		return lists[0]
	}
	result := strategy.NewSkipList()
	currNodes := make([]*strategy.Element, len(lists))
	for i, list := range lists {
		if list == nil || list.Len() == 0 {
			return nil
		}
		currNodes[i] = list.Front()
	}
	for {
		maxList := make(map[int]struct{}, len(currNodes))
		var maxValue uint64 = 0
		for i, node := range currNodes {
			if node.Value.(uint64) > maxValue {
				maxValue = node.Value.(uint64)
				maxList = make(map[int]struct{})
				maxList[i] = struct{}{}
			} else if node.Value.(uint64) == maxValue {
				maxList[i] = struct{}{}
			}
		}
		if len(maxList) == len(currNodes) {
			result.Insert(currNodes[0].Key().(float64), currNodes[0].Value)
		}

		for i, node := range currNodes {
			currNodes[i] = node.Next()
			if currNodes[i] == nil {
				return result
			}
		}
	}
}
