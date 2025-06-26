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

// BenchmarkHardwareAcceleratedSearch 基准测试硬件加速搜索性能
func BenchmarkHardwareAcceleratedSearch(b *testing.B) {
	// 创建向量数据库
	db, err := vector.NewVectorDBWithDimension(128, "cosine")
	if err != nil {
		b.Fatalf("创建向量数据库失败: %v", err)
	}

	// 创建硬件管理器
	hardwareManager := acceler.NewHardwareManager()

	// 应用硬件管理器到向量数据库
	if err := db.ApplyHardwareManager(hardwareManager); err != nil {
		b.Fatalf("应用硬件管理器失败: %v", err)
	}

	// 生成测试数据
	generateBenchmarkData(b, db, 50000)

	// 生成查询向量
	query := generateRandomVector(128)

	// 基准测试不同工作负载类型
	workloads := []struct {
		name    string
		options entity.SearchOptions
	}{
		{
			name: "CPU_Baseline",
			options: entity.SearchOptions{
				K: 10,
				// 不指定任何硬件加速
			},
		},
		{
			name: "GPU_Explicit",
			options: entity.SearchOptions{
				K:      10,
				UseGPU: true,
			},
		},
		{
			name: "FPGA_Explicit",
			options: entity.SearchOptions{
				K:       10,
				UseFPGA: true,
			},
		},
		{
			name: "Auto_LowLatency",
			options: entity.SearchOptions{
				K:             10,
				SearchTimeout: 5 * time.Millisecond,
			},
		},
		{
			name: "Auto_HighThroughput",
			options: entity.SearchOptions{
				K:         10,
				BatchSize: 200,
			},
		},
		{
			name: "Auto_Distributed",
			options: entity.SearchOptions{
				K:                 10,
				DistributedSearch: true,
			},
		},
		{
			name: "Auto_MemoryOptimized",
			options: entity.SearchOptions{
				K:               10,
				MemoryOptimized: true,
			},
		},
		{
			name: "Auto_PersistentStorage",
			options: entity.SearchOptions{
				K:                 10,
				PersistentStorage: true,
			},
		},
	}

	// 执行基准测试
	for _, wl := range workloads {
		b.Run(wl.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := db.OptimizedSearch(query, wl.options.K, wl.options)
				if err != nil {
					b.Fatalf("%s 搜索失败: %v", wl.name, err)
				}
			}
		})
	}

	// 基准测试不同数据规模
	dataSizes := []int{1000, 10000, 100000}
	for _, size := range dataSizes {
		// 为每个数据规模创建新的数据库
		testDB, err := vector.NewVectorDBWithDimension(128, "cosine")
		if err != nil {
			b.Fatalf("创建向量数据库失败: %v", err)
		}
		testDB.ApplyHardwareManager(hardwareManager)
		generateBenchmarkData(b, testDB, size)

		b.Run(fmt.Sprintf("DataSize_%d", size), func(b *testing.B) {
			options := entity.SearchOptions{K: 10} // 使用自动选择
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := testDB.OptimizedSearch(query, options.K, options)
				if err != nil {
					b.Fatalf("搜索失败: %v", err)
				}
			}
		})
	}

	// 基准测试不同向量维度
	dimensions := []int{64, 128, 256, 512}
	for _, dim := range dimensions {
		// 为每个维度创建新的数据库
		testDB, err := vector.NewVectorDBWithDimension(dim, "cosine")
		if err != nil {
			b.Fatalf("创建向量数据库失败: %v", err)
		}
		testDB.ApplyHardwareManager(hardwareManager)

		// 生成特定维度的测试数据
		for i := 0; i < 10000; i++ {
			id := fmt.Sprintf("vec_dim%d_%d", dim, i)
			vec := generateRandomVector(dim)
			testDB.Add(id, vec)
		}

		// 生成特定维度的查询向量
		dimQuery := generateRandomVector(dim)

		b.Run(fmt.Sprintf("Dimension_%d", dim), func(b *testing.B) {
			options := entity.SearchOptions{K: 10} // 使用自动选择
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := testDB.OptimizedSearch(dimQuery, options.K, options)
				if err != nil {
					b.Fatalf("搜索失败: %v", err)
				}
			}
		})
	}
}

// 生成基准测试数据
func generateBenchmarkData(b *testing.B, db *vector.VectorDB, count int) {
	b.Helper()
	b.Logf("生成 %d 条基准测试数据...", count)
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("bench_vec_%d", i)
		vec := generateRandomVector(128)
		db.Add(id, vec)
	}
}
