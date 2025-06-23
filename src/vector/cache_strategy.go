package vector

import (
	"container/list"
	"sync"
)

// CacheStrategy 缓存策略接口
type CacheStrategy interface {
	Get(key string) (interface{}, bool)
	Put(key string, value interface{})
	Delete(key string)
	Clear()
	Size() int
	Stats() CacheStrategyStats
}

// CacheStrategyStats 缓存策略统计信息
type CacheStrategyStats struct {
	Hits        int64
	Misses      int64
	Evictions   int64
	Size        int
	Capacity    int
	HitRate     float64
}

// LRUCache LRU缓存实现
type LRUCache struct {
	capacity int
	cache    map[string]*list.Element
	list     *list.List
	mutex    sync.RWMutex
	stats    CacheStrategyStats
}

// LRUItem LRU缓存项
type LRUItem struct {
	key   string
	value interface{}
}

// NewLRUCache 创建新的LRU缓存
func NewLRUCache(capacity int) *LRUCache {
	return &LRUCache{
		capacity: capacity,
		cache:    make(map[string]*list.Element),
		list:     list.New(),
		stats:    CacheStrategyStats{Capacity: capacity},
	}
}

// Get 获取缓存项
func (c *LRUCache) Get(key string) (interface{}, bool) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if elem, exists := c.cache[key]; exists {
		c.list.MoveToFront(elem)
		c.stats.Hits++
		c.updateHitRate()
		return elem.Value.(*LRUItem).value, true
	}
	c.stats.Misses++
	c.updateHitRate()
	return nil, false
}

// Put 添加缓存项
func (c *LRUCache) Put(key string, value interface{}) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if elem, exists := c.cache[key]; exists {
		c.list.MoveToFront(elem)
		elem.Value.(*LRUItem).value = value
		return
	}

	if c.list.Len() >= c.capacity {
		c.evictOldest()
	}

	item := &LRUItem{key: key, value: value}
	elem := c.list.PushFront(item)
	c.cache[key] = elem
	c.stats.Size = len(c.cache)
}

// Delete 删除缓存项
func (c *LRUCache) Delete(key string) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if elem, exists := c.cache[key]; exists {
		c.list.Remove(elem)
		delete(c.cache, key)
		c.stats.Size = len(c.cache)
	}
}

// Clear 清空缓存
func (c *LRUCache) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.cache = make(map[string]*list.Element)
	c.list = list.New()
	c.stats.Size = 0
}

// Size 获取缓存大小
func (c *LRUCache) Size() int {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return len(c.cache)
}

// Stats 获取缓存统计信息
func (c *LRUCache) Stats() CacheStrategyStats {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.stats
}

// evictOldest 淘汰最旧的缓存项
func (c *LRUCache) evictOldest() {
	elem := c.list.Back()
	if elem != nil {
		c.list.Remove(elem)
		item := elem.Value.(*LRUItem)
		delete(c.cache, item.key)
		c.stats.Evictions++
		c.stats.Size = len(c.cache)
	}
}

// updateHitRate 更新命中率
func (c *LRUCache) updateHitRate() {
	total := c.stats.Hits + c.stats.Misses
	if total > 0 {
		c.stats.HitRate = float64(c.stats.Hits) / float64(total)
	}
}

// LFUCache LFU缓存实现
type LFUCache struct {
	capacity  int
	cache     map[string]*LFUItem
	freqMap   map[int]*list.List
	minFreq   int
	mutex     sync.RWMutex
	stats     CacheStrategyStats
}

// LFUItem LFU缓存项
type LFUItem struct {
	key       string
	value     interface{}
	frequency int
	elem      *list.Element
}

// NewLFUCache 创建新的LFU缓存
func NewLFUCache(capacity int) *LFUCache {
	return &LFUCache{
		capacity: capacity,
		cache:    make(map[string]*LFUItem),
		freqMap:  make(map[int]*list.List),
		minFreq:  0,
		stats:    CacheStrategyStats{Capacity: capacity},
	}
}

// Get 获取缓存项
func (c *LFUCache) Get(key string) (interface{}, bool) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if item, exists := c.cache[key]; exists {
		c.updateFrequency(item)
		c.stats.Hits++
		c.updateHitRate()
		return item.value, true
	}
	c.stats.Misses++
	c.updateHitRate()
	return nil, false
}

// Put 添加缓存项
func (c *LFUCache) Put(key string, value interface{}) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if item, exists := c.cache[key]; exists {
		item.value = value
		c.updateFrequency(item)
		return
	}

	if len(c.cache) >= c.capacity {
		c.evictLFU()
	}

	item := &LFUItem{
		key:       key,
		value:     value,
		frequency: 1,
	}

	if c.freqMap[1] == nil {
		c.freqMap[1] = list.New()
	}
	item.elem = c.freqMap[1].PushFront(item)
	c.cache[key] = item
	c.minFreq = 1
	c.stats.Size = len(c.cache)
}

// Delete 删除缓存项
func (c *LFUCache) Delete(key string) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if item, exists := c.cache[key]; exists {
		c.freqMap[item.frequency].Remove(item.elem)
		delete(c.cache, key)
		c.stats.Size = len(c.cache)
	}
}

// Clear 清空缓存
func (c *LFUCache) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.cache = make(map[string]*LFUItem)
	c.freqMap = make(map[int]*list.List)
	c.minFreq = 0
	c.stats.Size = 0
}

// Size 获取缓存大小
func (c *LFUCache) Size() int {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return len(c.cache)
}

// Stats 获取缓存统计信息
func (c *LFUCache) Stats() CacheStrategyStats {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.stats
}

// updateFrequency 更新访问频率
func (c *LFUCache) updateFrequency(item *LFUItem) {
	oldFreq := item.frequency
	newFreq := oldFreq + 1

	// 从旧频率列表中移除
	c.freqMap[oldFreq].Remove(item.elem)
	if c.freqMap[oldFreq].Len() == 0 && oldFreq == c.minFreq {
		c.minFreq++
	}

	// 添加到新频率列表
	if c.freqMap[newFreq] == nil {
		c.freqMap[newFreq] = list.New()
	}
	item.frequency = newFreq
	item.elem = c.freqMap[newFreq].PushFront(item)
}

// evictLFU 淘汰最少使用的缓存项
func (c *LFUCache) evictLFU() {
	minFreqList := c.freqMap[c.minFreq]
	if minFreqList != nil && minFreqList.Len() > 0 {
		elem := minFreqList.Back()
		item := elem.Value.(*LFUItem)
		minFreqList.Remove(elem)
		delete(c.cache, item.key)
		c.stats.Evictions++
		c.stats.Size = len(c.cache)
	}
}

// updateHitRate 更新命中率
func (c *LFUCache) updateHitRate() {
	total := c.stats.Hits + c.stats.Misses
	if total > 0 {
		c.stats.HitRate = float64(c.stats.Hits) / float64(total)
	}
}

// ARCCache ARC (Adaptive Replacement Cache) 缓存实现
type ARCCache struct {
	capacity int
	p        int // 目标大小
	t1       map[string]*ARCItem
	t2       map[string]*ARCItem
	b1       map[string]*ARCItem
	b2       map[string]*ARCItem
	t1List   *list.List
	t2List   *list.List
	b1List   *list.List
	b2List   *list.List
	mutex    sync.RWMutex
	stats    CacheStrategyStats
}

// ARCItem ARC缓存项
type ARCItem struct {
	key   string
	value interface{}
	elem  *list.Element
}

// NewARCCache 创建新的ARC缓存
func NewARCCache(capacity int) *ARCCache {
	return &ARCCache{
		capacity: capacity,
		p:        0,
		t1:       make(map[string]*ARCItem),
		t2:       make(map[string]*ARCItem),
		b1:       make(map[string]*ARCItem),
		b2:       make(map[string]*ARCItem),
		t1List:   list.New(),
		t2List:   list.New(),
		b1List:   list.New(),
		b2List:   list.New(),
		stats:    CacheStrategyStats{Capacity: capacity},
	}
}

// Get 获取缓存项
func (c *ARCCache) Get(key string) (interface{}, bool) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// 在T1中查找
	if item, exists := c.t1[key]; exists {
		// 移动到T2
		c.t1List.Remove(item.elem)
		delete(c.t1, key)
		item.elem = c.t2List.PushFront(item)
		c.t2[key] = item
		c.stats.Hits++
		c.updateHitRate()
		return item.value, true
	}

	// 在T2中查找
	if item, exists := c.t2[key]; exists {
		c.t2List.MoveToFront(item.elem)
		c.stats.Hits++
		c.updateHitRate()
		return item.value, true
	}

	c.stats.Misses++
	c.updateHitRate()
	return nil, false
}

// Put 添加缓存项
func (c *ARCCache) Put(key string, value interface{}) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// 如果在T1或T2中，更新值
	if item, exists := c.t1[key]; exists {
		item.value = value
		return
	}
	if item, exists := c.t2[key]; exists {
		item.value = value
		return
	}

	// 如果在B1中
	if _, exists := c.b1[key]; exists {
		c.p = min(c.capacity, c.p+max(len(c.b2)/len(c.b1), 1))
		c.replace(key)
		c.b1List.Remove(c.b1[key].elem)
		delete(c.b1, key)
		item := &ARCItem{key: key, value: value}
		item.elem = c.t2List.PushFront(item)
		c.t2[key] = item
		c.updateSize()
		return
	}

	// 如果在B2中
	if _, exists := c.b2[key]; exists {
		c.p = max(0, c.p-max(len(c.b1)/len(c.b2), 1))
		c.replace(key)
		c.b2List.Remove(c.b2[key].elem)
		delete(c.b2, key)
		item := &ARCItem{key: key, value: value}
		item.elem = c.t2List.PushFront(item)
		c.t2[key] = item
		c.updateSize()
		return
	}

	// 新项目
	if len(c.t1)+len(c.b1) == c.capacity {
		if len(c.t1) < c.capacity {
			// 从B1中删除LRU
			elem := c.b1List.Back()
			if elem != nil {
				item := elem.Value.(*ARCItem)
				c.b1List.Remove(elem)
				delete(c.b1, item.key)
			}
			c.replace(key)
		} else {
			// 从T1中删除LRU
			elem := c.t1List.Back()
			if elem != nil {
				item := elem.Value.(*ARCItem)
				c.t1List.Remove(elem)
				delete(c.t1, item.key)
			}
		}
	} else if len(c.t1)+len(c.b1) < c.capacity && len(c.t1)+len(c.t2)+len(c.b1)+len(c.b2) >= c.capacity {
		if len(c.t1)+len(c.t2)+len(c.b1)+len(c.b2) == 2*c.capacity {
			// 从B2中删除LRU
			elem := c.b2List.Back()
			if elem != nil {
				item := elem.Value.(*ARCItem)
				c.b2List.Remove(elem)
				delete(c.b2, item.key)
			}
		}
		c.replace(key)
	}

	// 添加到T1
	item := &ARCItem{key: key, value: value}
	item.elem = c.t1List.PushFront(item)
	c.t1[key] = item
	c.updateSize()
}

// Delete 删除缓存项
func (c *ARCCache) Delete(key string) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if item, exists := c.t1[key]; exists {
		c.t1List.Remove(item.elem)
		delete(c.t1, key)
		c.updateSize()
		return
	}
	if item, exists := c.t2[key]; exists {
		c.t2List.Remove(item.elem)
		delete(c.t2, key)
		c.updateSize()
		return
	}
	if item, exists := c.b1[key]; exists {
		c.b1List.Remove(item.elem)
		delete(c.b1, key)
		return
	}
	if item, exists := c.b2[key]; exists {
		c.b2List.Remove(item.elem)
		delete(c.b2, key)
		return
	}
}

// Clear 清空缓存
func (c *ARCCache) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.t1 = make(map[string]*ARCItem)
	c.t2 = make(map[string]*ARCItem)
	c.b1 = make(map[string]*ARCItem)
	c.b2 = make(map[string]*ARCItem)
	c.t1List = list.New()
	c.t2List = list.New()
	c.b1List = list.New()
	c.b2List = list.New()
	c.p = 0
	c.stats.Size = 0
}

// Size 获取缓存大小
func (c *ARCCache) Size() int {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return len(c.t1) + len(c.t2)
}

// Stats 获取缓存统计信息
func (c *ARCCache) Stats() CacheStrategyStats {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	return c.stats
}

// replace ARC替换算法
func (c *ARCCache) replace(key string) {
	if len(c.t1) != 0 && ((len(c.t1) > c.p) || (len(c.t1) == c.p && c.b2[key] != nil)) {
		// 从T1移动到B1
		elem := c.t1List.Back()
		if elem != nil {
			item := elem.Value.(*ARCItem)
			c.t1List.Remove(elem)
			delete(c.t1, item.key)
			item.elem = c.b1List.PushFront(item)
			c.b1[item.key] = item
			c.stats.Evictions++
		}
	} else {
		// 从T2移动到B2
		elem := c.t2List.Back()
		if elem != nil {
			item := elem.Value.(*ARCItem)
			c.t2List.Remove(elem)
			delete(c.t2, item.key)
			item.elem = c.b2List.PushFront(item)
			c.b2[item.key] = item
			c.stats.Evictions++
		}
	}
}

// updateSize 更新缓存大小统计
func (c *ARCCache) updateSize() {
	c.stats.Size = len(c.t1) + len(c.t2)
}

// updateHitRate 更新命中率
func (c *ARCCache) updateHitRate() {
	total := c.stats.Hits + c.stats.Misses
	if total > 0 {
		c.stats.HitRate = float64(c.stats.Hits) / float64(total)
	}
}

// 辅助函数
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// CacheManager 缓存管理器
type CacheManager struct {
	strategies map[string]CacheStrategy
	mutex      sync.RWMutex
}

// NewCacheManager 创建新的缓存管理器
func NewCacheManager() *CacheManager {
	return &CacheManager{
		strategies: make(map[string]CacheStrategy),
	}
}

// RegisterStrategy 注册缓存策略
func (cm *CacheManager) RegisterStrategy(name string, strategy CacheStrategy) {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()
	cm.strategies[name] = strategy
}

// GetStrategy 获取缓存策略
func (cm *CacheManager) GetStrategy(name string) (CacheStrategy, bool) {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()
	strategy, exists := cm.strategies[name]
	return strategy, exists
}

// CreateStrategy 创建缓存策略
func CreateCacheStrategy(strategyType string, capacity int) CacheStrategy {
	switch strategyType {
	case "lru":
		return NewLRUCache(capacity)
	case "lfu":
		return NewLFUCache(capacity)
	case "arc":
		return NewARCCache(capacity)
	default:
		return NewLRUCache(capacity) // 默认使用LRU
	}
}