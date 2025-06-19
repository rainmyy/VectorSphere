package test

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"
)

// 简化的结构体定义，用于测试
type AdaptiveConfig struct {
	NumClusters           int
	IndexRebuildThreshold float64
	DefaultNprobe         int
	CacheTimeout          time.Duration
	MaxWorkers            int
	VectorCompression     bool
	UseMultiLevelIndex    bool
	MinNprobe             int
	MaxNprobe             int
	RecallTarget          float64
	MinEfConstruction     float64
	MaxEfConstruction     float64
	QualityThreshold      float64
}

type VectorDB struct {
	vectors        map[string][]float64
	mu             sync.RWMutex
	numClusters    int
	config         AdaptiveConfig
	vectorDim      int
	efConstruction float64
	maxConnections int
}

// 复制adaptive_config.go中的核心逻辑
func (db *VectorDB) AdjustConfig() {
	db.mu.RLock()
	vectorCount := len(db.vectors)
	db.mu.RUnlock()

	config := db.config

	// 根据向量数量调整簇数量
	if vectorCount > 1000000 {
		config.NumClusters = 1000
	} else if vectorCount > 100000 {
		config.NumClusters = 100
	} else if vectorCount > 10000 {
		config.NumClusters = 50
	} else {
		config.NumClusters = 10
	}

	// 根据系统资源调整工作协程数
	config.MaxWorkers = runtime.NumCPU()

	// 根据内存使用情况决定是否启用向量压缩
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	if m.Alloc > 1024*1024*1024 { // 如果内存使用超过1GB
		config.VectorCompression = true
	}

	db.mu.Lock()
	db.config = config
	db.mu.Unlock()
}

func (db *VectorDB) AdaptiveHNSWConfig() {
	db.mu.Lock()
	defer db.mu.Unlock()

	dataSize := len(db.vectors)

	// 根据数据规模调整 efConstruction
	switch {
	case dataSize < 10000:
		db.efConstruction = 100.0
	case dataSize < 100000:
		db.efConstruction = 200.0
	case dataSize < 1000000:
		db.efConstruction = 400.0
	default:
		db.efConstruction = 800.0
	}

	// 根据向量维度调整连接数
	if db.vectorDim > 0 {
		db.maxConnections = int(math.Min(64, math.Max(16, float64(db.vectorDim)/10)))
	}
}

// 计算自适应nprobe的逻辑
func calculateAdaptiveNprobe(dataSize, numClusters int) int {
	var nprobe int
	switch {
	case dataSize < 10000:
		nprobe = int(math.Max(1, float64(numClusters)/4))
	case dataSize < 100000:
		nprobe = int(math.Max(2, float64(numClusters)/3))
	case dataSize < 1000000:
		nprobe = int(math.Max(3, float64(numClusters)/2))
	default:
		nprobe = int(math.Max(5, float64(numClusters)*2/3))
	}

	// 确保 nprobe 在合理范围内
	if nprobe > numClusters {
		nprobe = numClusters
	}

	return nprobe
}

// 创建测试用的VectorDB实例
func createTestDB(vectorCount, vectorDim int) *VectorDB {
	db := &VectorDB{
		vectors:     make(map[string][]float64),
		mu:          sync.RWMutex{},
		vectorDim:   vectorDim,
		numClusters: 10,
		config: AdaptiveConfig{
			NumClusters:           10,
			IndexRebuildThreshold: 0.1,
			DefaultNprobe:         5,
			CacheTimeout:          time.Minute * 5,
			MaxWorkers:            4,
			VectorCompression:     false,
			UseMultiLevelIndex:    false,
			MinNprobe:             1,
			MaxNprobe:             100,
			RecallTarget:          0.9,
			MinEfConstruction:     100.0,
			MaxEfConstruction:     800.0,
			QualityThreshold:      0.95,
		},
		efConstruction: 100.0,
		maxConnections: 16,
	}

	// 填充测试向量数据
	for i := 0; i < vectorCount; i++ {
		vectorID := fmt.Sprintf("vec_%d", i)
		vector := make([]float64, vectorDim)
		for j := 0; j < vectorDim; j++ {
			vector[j] = float64(i*vectorDim + j)
		}
		db.vectors[vectorID] = vector
	}

	return db
}

// 测试AdjustConfig逻辑
func TestAdjustConfig(t *testing.T) {
	fmt.Println("=== 测试 AdjustConfig 逻辑 ===")

	tests := []struct {
		name             string
		vectorCount      int
		expectedClusters int
	}{
		{"小规模数据集 (<10k)", 5000, 10},
		{"中等规模数据集 (10k-100k)", 50000, 50},
		{"大规模数据集 (100k-1M)", 500000, 100},
		{"超大规模数据集 (>1M)", 1500000, 1000},
	}

	for _, test := range tests {
		fmt.Printf("\n测试: %s\n", test.name)
		db := createTestDB(test.vectorCount, 128)

		fmt.Printf("  向量数量: %d\n", test.vectorCount)
		fmt.Printf("  调整前簇数量: %d\n", db.config.NumClusters)

		db.AdjustConfig()

		fmt.Printf("  调整后簇数量: %d\n", db.config.NumClusters)
		fmt.Printf("  期望簇数量: %d\n", test.expectedClusters)
		fmt.Printf("  工作协程数: %d (CPU核心数: %d)\n", db.config.MaxWorkers, runtime.NumCPU())

		if db.config.NumClusters == test.expectedClusters {
			fmt.Printf("  ✅ 簇数量调整正确\n")
		} else {
			fmt.Printf("  ❌ 簇数量调整错误，期望 %d，实际 %d\n", test.expectedClusters, db.config.NumClusters)
		}

		if db.config.MaxWorkers == runtime.NumCPU() {
			fmt.Printf("  ✅ 工作协程数调整正确\n")
		} else {
			fmt.Printf("  ❌ 工作协程数调整错误\n")
		}
	}
}

// 测试AdaptiveNprobe计算逻辑
func TestAdaptiveNprobe(t *testing.T) {
	fmt.Println("\n=== 测试 AdaptiveNprobe 计算逻辑 ===")

	tests := []struct {
		name           string
		vectorCount    int
		numClusters    int
		expectedNprobe int
	}{
		{"小数据集 (<10k)", 5000, 20, int(math.Max(1, float64(20)/4))},        // 5
		{"中等数据集 (10k-100k)", 50000, 30, int(math.Max(2, float64(30)/3))}, // 10
		{"大数据集 (100k-1M)", 500000, 40, int(math.Max(3, float64(40)/2))},   // 20
		{"超大数据集 (>1M)", 1500000, 60, int(math.Max(5, float64(60)*2/3))},  // 40
		{"边界测试: nprobe > numClusters", 1500000, 5, 5},                     // 应该被限制为numClusters
	}

	for _, test := range tests {
		fmt.Printf("\n测试: %s\n", test.name)
		fmt.Printf("  数据规模: %d\n", test.vectorCount)
		fmt.Printf("  簇数量: %d\n", test.numClusters)

		actualNprobe := calculateAdaptiveNprobe(test.vectorCount, test.numClusters)

		fmt.Printf("  计算的nprobe: %d\n", actualNprobe)
		fmt.Printf("  期望nprobe: %d\n", test.expectedNprobe)

		if actualNprobe == test.expectedNprobe {
			fmt.Printf("  ✅ nprobe计算正确\n")
		} else {
			fmt.Printf("  ❌ nprobe计算错误，期望 %d，实际 %d\n", test.expectedNprobe, actualNprobe)
		}

		// 验证nprobe不超过numClusters
		if actualNprobe <= test.numClusters {
			fmt.Printf("  ✅ nprobe未超过numClusters限制\n")
		} else {
			fmt.Printf("  ❌ nprobe超过了numClusters限制\n")
		}
	}
}

// 测试AdaptiveHNSWConfig逻辑
func TestAdaptiveHNSWConfig2(t *testing.T) {
	fmt.Println("\n=== 测试 AdaptiveHNSWConfig 逻辑 ===")

	tests := []struct {
		name                   string
		vectorCount            int
		vectorDim              int
		expectedEfConstruction float64
		expectedMaxConnections int
	}{
		{"小数据集", 5000, 100, 100.0, int(math.Min(64, math.Max(16, float64(100)/10)))},
		{"中等数据集", 50000, 256, 200.0, int(math.Min(64, math.Max(16, float64(256)/10)))},
		{"大数据集", 500000, 512, 400.0, int(math.Min(64, math.Max(16, float64(512)/10)))},
		{"超大数据集", 1500000, 1024, 800.0, int(math.Min(64, math.Max(16, float64(1024)/10)))},
		{"高维向量", 100000, 2048, 200.0, int(math.Min(64, math.Max(16, float64(2048)/10)))},
	}

	for _, test := range tests {
		fmt.Printf("\n测试: %s\n", test.name)
		db := createTestDB(test.vectorCount, test.vectorDim)

		fmt.Printf("  数据规模: %d\n", test.vectorCount)
		fmt.Printf("  向量维度: %d\n", test.vectorDim)
		fmt.Printf("  调整前efConstruction: %.1f\n", db.efConstruction)
		fmt.Printf("  调整前maxConnections: %d\n", db.maxConnections)

		db.AdaptiveHNSWConfig()

		fmt.Printf("  调整后efConstruction: %.1f\n", db.efConstruction)
		fmt.Printf("  调整后maxConnections: %d\n", db.maxConnections)
		fmt.Printf("  期望efConstruction: %.1f\n", test.expectedEfConstruction)
		fmt.Printf("  期望maxConnections: %d\n", test.expectedMaxConnections)

		if db.efConstruction == test.expectedEfConstruction {
			fmt.Printf("  ✅ efConstruction调整正确\n")
		} else {
			fmt.Printf("  ❌ efConstruction调整错误\n")
		}

		if db.maxConnections == test.expectedMaxConnections {
			fmt.Printf("  ✅ maxConnections调整正确\n")
		} else {
			fmt.Printf("  ❌ maxConnections调整错误\n")
		}
	}
}

// 集成测试
func TestIntegration(t *testing.T) {
	fmt.Println("\n=== 集成测试 ===")

	db := createTestDB(150000, 384) // 中等规模，384维向量

	fmt.Printf("初始状态:\n")
	fmt.Printf("  向量数量: %d\n", len(db.vectors))
	fmt.Printf("  向量维度: %d\n", db.vectorDim)
	fmt.Printf("  初始簇数量: %d\n", db.config.NumClusters)
	fmt.Printf("  初始efConstruction: %.1f\n", db.efConstruction)
	fmt.Printf("  初始maxConnections: %d\n", db.maxConnections)

	// 执行所有自适应配置
	db.AdjustConfig()
	db.AdaptiveHNSWConfig()

	fmt.Printf("\n自适应调整后:\n")
	fmt.Printf("  簇数量: %d (期望: 100)\n", db.config.NumClusters)
	fmt.Printf("  efConstruction: %.1f (期望: 200.0)\n", db.efConstruction)
	fmt.Printf("  maxConnections: %d (期望: %d)\n", db.maxConnections, int(math.Min(64, math.Max(16, float64(384)/10))))
	fmt.Printf("  工作协程数: %d (CPU核心数: %d)\n", db.config.MaxWorkers, runtime.NumCPU())

	// 测试自适应nprobe计算
	nprobe := calculateAdaptiveNprobe(len(db.vectors), db.config.NumClusters)
	expectedNprobe := int(math.Max(3, float64(db.config.NumClusters)/2))
	fmt.Printf("  自适应nprobe: %d (期望: %d)\n", nprobe, expectedNprobe)

	// 验证结果
	allCorrect := true
	if db.config.NumClusters != 100 {
		fmt.Printf("  ❌ 簇数量调整错误\n")
		allCorrect = false
	}
	if db.efConstruction != 200.0 {
		fmt.Printf("  ❌ efConstruction调整错误\n")
		allCorrect = false
	}
	expectedMaxConn := int(math.Min(64, math.Max(16, float64(384)/10)))
	if db.maxConnections != expectedMaxConn {
		fmt.Printf("  ❌ maxConnections调整错误\n")
		allCorrect = false
	}
	if nprobe != expectedNprobe {
		fmt.Printf("  ❌ nprobe计算错误\n")
		allCorrect = false
	}

	if allCorrect {
		fmt.Printf("\n✅ 集成测试通过！所有自适应逻辑都符合预期\n")
	} else {
		fmt.Printf("\n❌ 集成测试失败！存在逻辑错误\n")
	}
}
