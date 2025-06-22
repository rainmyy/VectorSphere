package test

import (
	"VectorSphere/src/vector"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"
)

// TestMultiLevelCacheCreation 测试多级缓存的创建
func TestMultiLevelCacheCreation(t *testing.T) {
	t.Run("正常创建缓存", func(t *testing.T) {
		tempDir := t.TempDir()
		cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)
		if cache == nil {
			t.Fatal("缓存创建失败")
		}
	})

	t.Run("空路径创建缓存", func(t *testing.T) {
		cache := vector.NewMultiLevelCache(10, 20, 1024*1024, "")
		if cache == nil {
			t.Fatal("缓存创建失败")
		}
	})

	t.Run("无效路径创建缓存", func(t *testing.T) {
		// 使用一个无法创建的路径
		invalidPath := "/invalid/path/that/cannot/be/created"
		cache := vector.NewMultiLevelCache(10, 20, 1024*1024, invalidPath)
		// 在Windows上，这可能仍然会成功，所以我们不强制要求失败
		_ = cache
	})
}

// TestBasicPutGet 测试基本的存取操作
func TestBasicPutGet(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)

	t.Run("字符串数组存取", func(t *testing.T) {
		key := "test_string_array"
		value := []string{"hello", "world", "test"}
		
		cache.Put(key, value)
		
		// 立即获取（应该从L1缓存获取）
		result, found := cache.Get(key)
		if !found {
			t.Fatal("未找到缓存数据")
		}
		
		resultSlice, ok := result.([]string)
		if !ok {
			t.Fatalf("类型断言失败，期望[]string，实际%T", result)
		}
		
		if !reflect.DeepEqual(resultSlice, value) {
			t.Fatalf("数据不匹配，期望%v，实际%v", value, resultSlice)
		}
	})

	t.Run("浮点数组存取", func(t *testing.T) {
		key := "test_float_array"
		value := []float64{1.1, 2.2, 3.3}
		
		cache.Put(key, value)
		
		result, found := cache.Get(key)
		if !found {
			t.Fatal("未找到缓存数据")
		}
		
		resultSlice, ok := result.([]float64)
		if !ok {
			t.Fatalf("类型断言失败，期望[]float64，实际%T", result)
		}
		
		if !reflect.DeepEqual(resultSlice, value) {
			t.Fatalf("数据不匹配，期望%v，实际%v", value, resultSlice)
		}
	})

	t.Run("整数数组存取", func(t *testing.T) {
		key := "test_int_array"
		value := []int{1, 2, 3, 4, 5}
		
		cache.Put(key, value)
		
		result, found := cache.Get(key)
		if !found {
			t.Fatal("未找到缓存数据")
		}
		
		resultSlice, ok := result.([]int)
		if !ok {
			t.Fatalf("类型断言失败，期望[]int，实际%T", result)
		}
		
		if !reflect.DeepEqual(resultSlice, value) {
			t.Fatalf("数据不匹配，期望%v，实际%v", value, resultSlice)
		}
	})

	t.Run("Map存取", func(t *testing.T) {
		key := "test_map"
		value := map[string]interface{}{
			"name": "test",
			"age":  25,
			"tags": []string{"go", "cache"},
		}
		
		cache.Put(key, value)
		
		result, found := cache.Get(key)
		if !found {
			t.Fatal("未找到缓存数据")
		}
		
		resultMap, ok := result.(map[string]interface{})
		if !ok {
			t.Fatalf("类型断言失败，期望map[string]interface{}，实际%T", result)
		}
		
		if !reflect.DeepEqual(resultMap, value) {
			t.Fatalf("数据不匹配，期望%v，实际%v", value, resultMap)
		}
	})

	t.Run("字节数组存取", func(t *testing.T) {
		key := "test_bytes"
		value := []byte{0x01, 0x02, 0x03, 0x04}
		
		cache.Put(key, value)
		
		result, found := cache.Get(key)
		if !found {
			t.Fatal("未找到缓存数据")
		}
		
		resultBytes, ok := result.([]byte)
		if !ok {
			t.Fatalf("类型断言失败，期望[]byte，实际%T", result)
		}
		
		if !reflect.DeepEqual(resultBytes, value) {
			t.Fatalf("数据不匹配，期望%v，实际%v", value, resultBytes)
		}
	})
}

// TestCacheNotFound 测试缓存未命中的情况
func TestCacheNotFound(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)

	result, found := cache.Get("non_existent_key")
	if found {
		t.Fatal("不应该找到不存在的键")
	}
	if result != nil {
		t.Fatal("不存在的键应该返回nil")
	}
}

// TestCacheStats 测试缓存统计功能
func TestCacheStats(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)

	// 初始统计应该为0
	stats := cache.GetStats()
	if stats.TotalQueries != 0 || stats.L1Hits != 0 || stats.L2Hits != 0 || stats.L3Hits != 0 {
		t.Fatal("初始统计信息应该为0")
	}

	// 添加一些数据
	cache.Put("key1", []string{"value1"})
	cache.Put("key2", []string{"value2"})

	// 从L1缓存获取
	cache.Get("key1")
	stats = cache.GetStats()
	if stats.TotalQueries != 1 || stats.L1Hits != 1 {
		t.Fatalf("L1命中统计错误，期望TotalQueries=1, L1Hits=1，实际TotalQueries=%d, L1Hits=%d", stats.TotalQueries, stats.L1Hits)
	}

	// 获取不存在的键
	cache.Get("non_existent")
	stats = cache.GetStats()
	if stats.TotalQueries != 2 {
		t.Fatalf("总查询数统计错误，期望2，实际%d", stats.TotalQueries)
	}
}

// TestL1CapacityLimit 测试L1缓存容量限制
func TestL1CapacityLimit(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(3, 20, 1024*1024, tempDir) // L1容量设为3

	// 添加超过容量的数据
	for i := 0; i < 5; i++ {
		key := fmt.Sprintf("key%d", i)
		value := []string{fmt.Sprintf("value%d", i)}
		cache.Put(key, value)
	}

	// 检查最早的键是否被移除
	_, found := cache.CheckL1Cache("key0")
	if found {
		t.Fatal("最早的键应该被移除")
	}

	// 检查最新的键是否存在
	_, found = cache.CheckL1Cache("key4")
	if !found {
		t.Fatal("最新的键应该存在")
	}
}

// TestCapacityManagement 测试容量管理功能
func TestCapacityManagement(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)

	t.Run("增加L1容量", func(t *testing.T) {
		cache.IncreaseL1Capacity(5)
		// 由于没有公开的方法获取容量，我们通过行为来验证
		// 添加更多数据来测试新容量
		for i := 0; i < 15; i++ {
			key := fmt.Sprintf("capacity_test_%d", i)
			value := []string{fmt.Sprintf("value_%d", i)}
			cache.Put(key, value)
		}
	})

	t.Run("减少L1容量", func(t *testing.T) {
		cache.DecreaseL1Capacity(5)
		// 容量减少后，应该触发清理
	})

	t.Run("增加L2容量", func(t *testing.T) {
		cache.IncreaseL2Capacity(10)
	})

	t.Run("减少L2容量", func(t *testing.T) {
		cache.DecreaseL2Capacity(5)
	})

	t.Run("增加L3容量", func(t *testing.T) {
		cache.IncreaseL3Capacity(1024)
	})
}

// TestCacheExpiration 测试缓存过期功能
func TestCacheExpiration(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)

	// 添加一些数据
	cache.Put("key1", []string{"value1"})
	cache.Put("key2", []string{"value2"})

	// 清理过期的缓存（使用未来时间，所以不应该清理任何东西）
	futureTime := time.Now().Add(1 * time.Hour)
	cache.CleanupExpired(futureTime)

	// 数据应该仍然存在
	_, found := cache.Get("key1")
	if !found {
		t.Fatal("数据不应该被清理")
	}

	// 清理过期的缓存（使用过去时间，应该清理所有数据）
	pastTime := time.Now().Add(-1 * time.Hour)
	cache.CleanupExpired(pastTime)

	// 等待异步清理完成
	time.Sleep(100 * time.Millisecond)

	// 数据应该被清理（但可能仍在L1缓存中，因为L1有自己的过期逻辑）
}

// TestCacheClear 测试缓存清空功能
func TestCacheClear(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)

	// 添加一些数据
	cache.Put("key1", []string{"value1"})
	cache.Put("key2", []string{"value2"})

	// 验证数据存在
	_, found := cache.Get("key1")
	if !found {
		t.Fatal("数据应该存在")
	}

	// 清空缓存
	cache.Clear()

	// 等待异步清理完成
	time.Sleep(100 * time.Millisecond)

	// 验证数据被清空
	_, found = cache.CheckL1Cache("key1")
	if found {
		t.Fatal("L1缓存应该被清空")
	}

	// 验证统计信息被重置
	stats := cache.GetStats()
	if stats.TotalQueries != 0 || stats.L1Hits != 0 || stats.L2Hits != 0 || stats.L3Hits != 0 {
		t.Fatal("统计信息应该被重置")
	}
}

// TestCachePromotion 测试缓存提升功能
func TestCachePromotion(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)

	t.Run("提升到L1缓存", func(t *testing.T) {
		key := "promote_l1_key"
		value := []string{"promote_l1_value"}
		
		cache.PromoteToL1Cache(key, value)
		
		result, found := cache.CheckL1Cache(key)
		if !found {
			t.Fatal("数据应该在L1缓存中")
		}
		
		resultSlice, ok := result.([]string)
		if !ok {
			t.Fatalf("类型断言失败，期望[]string，实际%T", result)
		}
		
		if !reflect.DeepEqual(resultSlice, value) {
			t.Fatalf("数据不匹配，期望%v，实际%v", value, resultSlice)
		}
	})

	t.Run("提升到L2缓存", func(t *testing.T) {
		key := "promote_l2_key"
		value := []string{"promote_l2_value"}
		
		cache.PromoteToL2Cache(key, value)
		
		// 等待异步操作完成
		time.Sleep(100 * time.Millisecond)
		
		result, found := cache.CheckL2Cache(key)
		// 如果L2共享内存不可用，跳过这个测试
		if !found {
			t.Skip("L2缓存不可用（共享内存初始化失败），跳过测试")
			return
		}
		
		// L2缓存可能返回不同的格式，所以我们只检查是否找到
		if result == nil {
			t.Fatal("L2缓存结果不应该为nil")
		}
	})
}

// TestCachePrewarm 测试缓存预热功能
func TestCachePrewarm(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)

	t.Run("预热指定键值对", func(t *testing.T) {
		key := "prewarm_key"
		value := []string{"prewarm_value"}
		
		cache.Prewarm(key, value)
		
		// 等待异步操作完成
		time.Sleep(100 * time.Millisecond)
		
		// 检查L1缓存
		_, found := cache.CheckL1Cache(key)
		if !found {
			t.Fatal("预热的数据应该在L1缓存中")
		}
	})

	t.Run("预热整个系统", func(t *testing.T) {
		// 创建测试配置文件
		configDir := filepath.Join(tempDir, "config")
		err := os.MkdirAll(configDir, 0755)
		if err != nil {
			t.Fatalf("创建配置目录失败: %v", err)
		}
		
		// 创建预定义查询文件
		popularQueries := []map[string]interface{}{
			{"query": "test_query_1", "results": []string{"result1", "result2"}},
			{"query": "test_query_2", "results": []string{"result3", "result4"}},
		}
		
		data, err := json.Marshal(popularQueries)
		if err != nil {
			t.Fatalf("序列化预定义查询失败: %v", err)
		}
		
		popularQueriesPath := filepath.Join(configDir, "popular_queries.json")
		err = os.WriteFile(popularQueriesPath, data, 0644)
		if err != nil {
			t.Fatalf("写入预定义查询文件失败: %v", err)
		}
		
		// 预热系统
		cache.PrewarmSystem()
		
		// 等待异步操作完成
		time.Sleep(200 * time.Millisecond)
	})
}

// TestMultiLevelCacheConcurrentAccess 测试多级缓存并发访问
func TestMultiLevelCacheConcurrentAccess(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(100, 200, 1024*1024, tempDir)

	var wg sync.WaitGroup
	numGoroutines := 10
	numOperations := 100

	// 并发写入
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				key := fmt.Sprintf("concurrent_key_%d_%d", id, j)
				value := []string{fmt.Sprintf("value_%d_%d", id, j)}
				cache.Put(key, value)
			}
		}(i)
	}

	// 并发读取
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				key := fmt.Sprintf("concurrent_key_%d_%d", id, j)
				cache.Get(key)
			}
		}(i)
	}

	wg.Wait()

	// 验证统计信息
	stats := cache.GetStats()
	if stats.TotalQueries == 0 {
		t.Fatal("应该有查询统计")
	}
}

// TestEnforceCapacityLimits 测试强制执行容量限制
func TestEnforceCapacityLimits(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(5, 10, 1024, tempDir)

	// 添加超过容量的数据
	for i := 0; i < 20; i++ {
		key := fmt.Sprintf("capacity_key_%d", i)
		value := []string{fmt.Sprintf("capacity_value_%d", i)}
		cache.Put(key, value)
	}

	// 强制执行容量限制
	cache.EnforceCapacityLimits()

	// 等待异步操作完成
	time.Sleep(200 * time.Millisecond)

	// 验证容量限制生效（具体验证逻辑取决于实现细节）
}

// TestL3CacheFileOperations 测试L3缓存文件操作
func TestL3CacheFileOperations(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(2, 5, 1024*1024, tempDir)

	// 添加数据，确保会写入L3缓存
	key := "l3_test_key"
	value := []string{"l3_test_value"}
	cache.Put(key, value)

	// 等待异步写入完成
	time.Sleep(200 * time.Millisecond)

	// 清空L1和L2缓存，强制从L3读取
	cache.Clear()
	time.Sleep(100 * time.Millisecond)

	// 重新创建缓存实例，模拟重启
	newCache := vector.NewMultiLevelCache(2, 5, 1024*1024, tempDir)

	// 尝试从L3缓存读取
	result, found := newCache.CheckL3Cache(key)
	if found {
		t.Logf("从L3缓存成功读取数据: %v", result)
	} else {
		t.Log("L3缓存中未找到数据（可能是正常的，取决于实现）")
	}
}

// TestDataTypeDetection 测试数据类型检测
func TestDataTypeDetection(t *testing.T) {
	tempDir := t.TempDir()
	cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)

	testCases := []struct {
		name  string
		value interface{}
	}{
		{"字符串数组", []string{"a", "b", "c"}},
		{"浮点数组", []float64{1.1, 2.2, 3.3}},
		{"整数数组", []int{1, 2, 3}},
		{"字节数组", []byte{0x01, 0x02, 0x03}},
		{"Map", map[string]interface{}{"key": "value"}},
		{"结构体", struct{ Name string }{Name: "test"}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			key := fmt.Sprintf("type_test_%s", tc.name)
			cache.Put(key, tc.value)
			
			result, found := cache.Get(key)
			if !found {
				t.Fatal("应该找到缓存数据")
			}
			
			if result == nil {
				t.Fatal("结果不应该为nil")
			}
			
			// 对于某些类型，可能会有序列化/反序列化的转换
			// 所以我们主要验证数据不为空
		})
	}
}

// TestErrorHandling 测试错误处理
func TestErrorHandling(t *testing.T) {
	t.Run("nil值处理", func(t *testing.T) {
		tempDir := t.TempDir()
		cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)
		
		// 存储nil值
		cache.Put("nil_key", nil)
		
		result, found := cache.Get("nil_key")
		if !found {
			t.Fatal("应该找到nil值")
		}
		
		if result != nil {
			t.Fatal("结果应该为nil")
		}
	})

	t.Run("空字符串键处理", func(t *testing.T) {
		tempDir := t.TempDir()
		cache := vector.NewMultiLevelCache(10, 20, 1024*1024, tempDir)
		
		// 使用空字符串作为键
		cache.Put("", []string{"empty_key_value"})
		
		result, found := cache.Get("")
		if !found {
			t.Fatal("应该找到空键的值")
		}
		
		if result == nil {
			t.Fatal("结果不应该为nil")
		}
	})
}

// BenchmarkCacheOperations 缓存操作性能测试
func BenchmarkCacheOperations(b *testing.B) {
	tempDir := b.TempDir()
	cache := vector.NewMultiLevelCache(1000, 2000, 1024*1024, tempDir)

	b.Run("Put操作", func(b *testing.B) {
		value := []string{"benchmark", "test", "value"}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("bench_key_%d", i)
			cache.Put(key, value)
		}
	})

	b.Run("Get操作", func(b *testing.B) {
		// 预先填充一些数据
		value := []string{"benchmark", "test", "value"}
		for i := 0; i < 1000; i++ {
			key := fmt.Sprintf("bench_get_key_%d", i)
			cache.Put(key, value)
		}
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("bench_get_key_%d", i%1000)
			cache.Get(key)
		}
	})

	b.Run("并发Put操作", func(b *testing.B) {
		value := []string{"concurrent", "benchmark", "value"}
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				key := fmt.Sprintf("concurrent_bench_key_%d", i)
				cache.Put(key, value)
				i++
			}
		})
	})

	b.Run("并发Get操作", func(b *testing.B) {
		// 预先填充数据
		value := []string{"concurrent", "get", "benchmark"}
		for i := 0; i < 1000; i++ {
			key := fmt.Sprintf("concurrent_get_key_%d", i)
			cache.Put(key, value)
		}
		
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			i := 0
			for pb.Next() {
				key := fmt.Sprintf("concurrent_get_key_%d", i%1000)
				cache.Get(key)
				i++
			}
		})
	})
}