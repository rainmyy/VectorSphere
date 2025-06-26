package test

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/vector"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func init() {
	// 初始化随机数生成器
	rand.Seed(time.Now().UnixNano())
}

// TestHardwareAcceleratedSearch 测试硬件加速搜索功能
func TestHardwareAcceleratedSearch(t *testing.T) {
	// 创建向量数据库
	db, err := vector.NewVectorDBWithDimension(128, "cosine")
	if err != nil {
		t.Fatalf("创建向量数据库失败: %v", err)
	}

	// 创建硬件管理器
	hardwareManager := acceler.NewHardwareManager()

	// 应用硬件管理器到向量数据库
	if err := db.ApplyHardwareManager(hardwareManager); err != nil {
		t.Fatalf("应用硬件管理器失败: %v", err)
	}

	// 生成测试数据
	generateTestData(t, db, 10000)

	// 生成查询向量
	query := generateRandomVector(128)

	// 测试不同工作负载类型的搜索
	testWorkloads := []struct {
		name    string
		options entity.SearchOptions
	}{
		{
			name: "低延迟工作负载",
			options: entity.SearchOptions{
				K:             10,
				SearchTimeout: 5 * time.Millisecond,
				QualityLevel:  0.9,
			},
		},
		{
			name: "高吞吐量工作负载",
			options: entity.SearchOptions{
				K:         10,
				BatchSize: 200,
			},
		},
		{
			name: "分布式工作负载",
			options: entity.SearchOptions{
				K:                10,
				DistributedSearch: true,
				ParallelSearch:    true,
			},
		},
		{
			name: "内存优化工作负载",
			options: entity.SearchOptions{
				K:               10,
				MemoryOptimized: true,
			},
		},
		{
			name: "持久化存储工作负载",
			options: entity.SearchOptions{
				K:                 10,
				PersistentStorage: true,
			},
		},
		{
			name: "显式GPU工作负载",
			options: entity.SearchOptions{
				K:      10,
				UseGPU: true,
			},
		},
		{
			name: "显式FPGA工作负载",
			options: entity.SearchOptions{
				K:       10,
				UseFPGA: true,
			},
		},
	}

	// 执行测试
	for _, tc := range testWorkloads {
		t.Run(tc.name, func(t *testing.T) {
			startTime := time.Now()
			results, err := db.OptimizedSearch(query, tc.options.K, tc.options)
			elapsed := time.Since(startTime)

			if err != nil {
				t.Errorf("%s 搜索失败: %v", tc.name, err)
				return
			}

			t.Logf("%s 搜索成功, 结果数量: %d, 耗时: %v", tc.name, len(results), elapsed)

			// 验证结果数量
			if len(results) > tc.options.K {
				t.Errorf("%s 结果数量 %d 超过请求的 K=%d", tc.name, len(results), tc.options.K)
			}
		})
	}

	// 测试硬件加速器性能指标
	t.Run("硬件加速器性能指标", func(t *testing.T) {
		if hardwareManager == nil {
			t.Skip("硬件管理器未初始化，跳过测试")
		}

		// 获取所有加速器的性能指标
		metrics := hardwareManager.GetAllPerformanceMetrics()
		for accType, metric := range metrics {
			t.Logf("加速器 %s 性能指标:", accType)
			t.Logf("  当前延迟: %v", metric.LatencyCurrent)
			t.Logf("  当前吞吐量: %.2f ops/sec", metric.ThroughputCurrent)
			t.Logf("  资源利用率: %v", metric.ResourceUtilization)
		}

		// 获取所有加速器的统计信息
		stats := hardwareManager.GetAllStats()
		for accType, stat := range stats {
			t.Logf("加速器 %s 统计信息:", accType)
			t.Logf("  总操作数: %d", stat.TotalOperations)
			t.Logf("  成功操作数: %d", stat.SuccessfulOps)
			t.Logf("  失败操作数: %d", stat.FailedOps)
			t.Logf("  平均延迟: %v", stat.AverageLatency)
			t.Logf("  吞吐量: %.2f ops/sec", stat.Throughput)
		}
	})
}

// 生成随机向量
func generateRandomVector(dim int) []float64 {
	vector := make([]float64, dim)
	for i := 0; i < dim; i++ {
		vector[i] = rand.Float64()*2 - 1 // 生成 [-1, 1] 范围内的随机数
	}
	return vector
}

// 生成测试数据
func generateTestData(t *testing.T, db *vector.VectorDB, count int) {
	t.Logf("生成 %d 条测试数据...", count)
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("vec_%d", i)
		vec := generateRandomVector(128)
		db.Add(id, vec)
	}
}