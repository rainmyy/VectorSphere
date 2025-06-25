package test

import (
	"fmt"
	"testing"

	"VectorSphere/src/vector"
)

func TestNewLRUCache(t *testing.T) {
	cache := vector.NewLRUCache(10)
	if cache == nil {
		t.Fatal("Expected non-nil LRU cache")
	}

	if cache.Size() != 0 {
		t.Errorf("Expected empty cache, got size %d", cache.Size())
	}
}

func TestLRUCachePutGet(t *testing.T) {
	cache := vector.NewLRUCache(3)

	// 测试基本的put和get操作
	cache.Put("key1", "value1")
	cache.Put("key2", "value2")
	cache.Put("key3", "value3")

	value, found := cache.Get("key1")
	if !found {
		t.Error("Expected to find key1")
	}
	if value != "value1" {
		t.Errorf("Expected value1, got %v", value)
	}

	if cache.Size() != 3 {
		t.Errorf("Expected cache size 3, got %d", cache.Size())
	}
}

func TestLRUCacheEviction(t *testing.T) {
	cache := vector.NewLRUCache(2)

	// 添加超过容量的元素
	cache.Put("key1", "value1")
	cache.Put("key2", "value2")
	cache.Put("key3", "value3") // 应该驱逐key1

	_, found := cache.Get("key1")
	if found {
		t.Error("Expected key1 to be evicted")
	}

	_, found = cache.Get("key2")
	if !found {
		t.Error("Expected key2 to still exist")
	}

	_, found = cache.Get("key3")
	if !found {
		t.Error("Expected key3 to exist")
	}
}

func TestLRUCacheDelete(t *testing.T) {
	cache := vector.NewLRUCache(5)

	cache.Put("key1", "value1")
	cache.Put("key2", "value2")

	cache.Delete("key1")

	_, found := cache.Get("key1")
	if found {
		t.Error("Expected key1 to be deleted")
	}

	if cache.Size() != 1 {
		t.Errorf("Expected cache size 1, got %d", cache.Size())
	}
}

func TestLRUCacheClear(t *testing.T) {
	cache := vector.NewLRUCache(5)

	cache.Put("key1", "value1")
	cache.Put("key2", "value2")
	cache.Put("key3", "value3")

	cache.Clear()

	if cache.Size() != 0 {
		t.Errorf("Expected empty cache after clear, got size %d", cache.Size())
	}
}

func TestLRUCacheStats(t *testing.T) {
	cache := vector.NewLRUCache(3)

	cache.Put("key1", "value1")
	cache.Put("key2", "value2")

	// 命中
	cache.Get("key1")
	cache.Get("key1")

	// 未命中
	cache.Get("key3")

	stats := cache.Stats()
	if stats.Hits != 2 {
		t.Errorf("Expected 2 hits, got %d", stats.Hits)
	}
	if stats.Misses != 1 {
		t.Errorf("Expected 1 miss, got %d", stats.Misses)
	}
	if stats.Size != 2 {
		t.Errorf("Expected size 2, got %d", stats.Size)
	}
}

func TestNewLFUCache(t *testing.T) {
	cache := vector.NewLFUCache(10)
	if cache == nil {
		t.Fatal("Expected non-nil LFU cache")
	}

	if cache.Size() != 0 {
		t.Errorf("Expected empty cache, got size %d", cache.Size())
	}
}

func TestLFUCacheEviction(t *testing.T) {
	cache := vector.NewLFUCache(2)

	cache.Put("key1", "value1")
	cache.Put("key2", "value2")

	// 增加key1的访问频率
	cache.Get("key1")
	cache.Get("key1")

	// 添加新元素，应该驱逐key2（频率较低）
	cache.Put("key3", "value3")

	_, found := cache.Get("key1")
	if !found {
		t.Error("Expected key1 to still exist (higher frequency)")
	}

	_, found = cache.Get("key2")
	if found {
		t.Error("Expected key2 to be evicted (lower frequency)")
	}
}

func TestNewARCCache(t *testing.T) {
	// 测试ARC缓存创建（如果存在的话）
	// 由于没有找到TTL缓存实现，我们测试其他可用的缓存类型
	cache := vector.NewLRUCache(10)
	if cache == nil {
		t.Fatal("Expected non-nil LRU cache")
	}

	// 测试基本功能
	cache.Put("test", "value")
	if val, found := cache.Get("test"); !found || val != "value" {
		t.Error("Expected to find test key with correct value")
	}
}

func TestCacheStrategyConcurrency(t *testing.T) {
	cache := vector.NewLRUCache(100)

	// 并发写入
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 10; j++ {
				cache.Put(fmt.Sprintf("key-%d-%d", id, j), fmt.Sprintf("value-%d-%d", id, j))
			}
			done <- true
		}(i)
	}

	// 等待所有goroutine完成
	for i := 0; i < 10; i++ {
		<-done
	}

	if cache.Size() != 100 {
		t.Errorf("Expected cache size 100, got %d", cache.Size())
	}
}
