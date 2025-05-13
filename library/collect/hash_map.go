package collect

import (
	farmhash "github.com/leemcloughlin/gofarmhash"
	"golang.org/x/exp/maps"
	"sync"
)

type HashMap struct {
	maps []map[string]any
	seg  int
	mus  []sync.RWMutex
	seed uint32
}

func NewHashMap(seg int, cap int) *HashMap {
	mps := make([]map[string]any, seg)
	mus := make([]sync.RWMutex, seg)
	mcp := cap / seg
	for i := 0; i < seg; i++ {
		mps[i] = make(map[string]any, mcp)
	}

	return &HashMap{
		maps: mps,
		seg:  seg,
		mus:  mus,
		seed: 0,
	}
}

func (h *HashMap) Set(key string, value any) {
	segIndex := h.getSegIndex(key)
	h.mus[segIndex].Lock()
	defer h.mus[segIndex].Unlock()
	h.maps[segIndex][key] = value
}

func (h *HashMap) Get(key string) (any, bool) {
	segIndex := h.getSegIndex(key)
	h.mus[segIndex].RLock()
	defer h.mus[segIndex].RUnlock()
	value, ok := h.maps[segIndex][key]
	return value, ok
}
func (h *HashMap) getSegIndex(key string) int {
	hash := int(farmhash.Hash32WithSeed([]byte(key), h.seed))
	return hash % h.seg
}

func (h *HashMap) Iterator() *HashMapIterator {
	keys := make([][]string, h.seg)
	for _, mp := range h.maps {
		row := maps.Keys(mp)
		keys = append(keys, row)
	}
	return &HashMapIterator{
		cm:       h,
		keys:     keys,
		rowIndex: 0,
		colIndex: 0,
	}
}

type MapEntry struct {
	key   string
	value any
}

type MapIterator interface {
	Next() *MapEntry
}

type HashMapIterator struct {
	cm       *HashMap
	keys     [][]string
	rowIndex int
	colIndex int
}

func (iter *HashMapIterator) Next() *MapEntry {
	if iter.rowIndex >= len(iter.keys) {
		return nil
	}
	row := iter.keys[iter.rowIndex]
	if len(row) == 0 {
		iter.rowIndex++
		return iter.Next()
	}
	key := row[iter.colIndex]
	value, _ := iter.cm.Get(key)
	if iter.colIndex < len(row)-1 {
		iter.colIndex++
	} else {
		iter.rowIndex++
		iter.colIndex = 0
	}
	return &MapEntry{
		key:   key,
		value: value,
	}
}
