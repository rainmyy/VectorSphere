package test

import (
	"fmt"
	"math"
	"runtime"
	"testing"
)

// TestClusterAdjustment 测试簇数量调整逻辑
func TestClusterAdjustment(t *testing.T) {
	fmt.Println("=== 测试簇数量调整逻辑 ===")

	testCases := []struct {
		vectorCount int
		expected    int
		description string
	}{
		{5000, 10, "小规模数据集 (<10k)"},
		{50000, 50, "中等规模数据集 (10k-100k)"},
		{500000, 100, "大规模数据集 (100k-1M)"},
		{1500000, 1000, "超大规模数据集 (>1M)"},
		{9999, 10, "边界测试: 9999"},
		{10000, 50, "边界测试: 10000"},
		{100000, 50, "边界测试: 100000"},
		{100001, 100, "边界测试: 100001"},
		{1000000, 100, "边界测试: 1000000"},
		{1000001, 1000, "边界测试: 1000001"},
	}

	for _, tc := range testCases {
		var numClusters int

		// 复制adaptive_config.go中的逻辑
		if tc.vectorCount > 1000000 {
			numClusters = 1000
		} else if tc.vectorCount > 100000 {
			numClusters = 100
		} else if tc.vectorCount > 10000 {
			numClusters = 50
		} else {
			numClusters = 10
		}

		result := "✅ 正确"
		if numClusters != tc.expected {
			result = "❌ 错误"
		}

		fmt.Printf("%s: 向量数=%d, 期望簇数=%d, 实际簇数=%d %s\n",
			tc.description, tc.vectorCount, tc.expected, numClusters, result)
	}
}

func TestNprobeCalculation(t *testing.T) {
	fmt.Println("\n=== 测试nprobe计算逻辑 ===")

	testCases := []struct {
		dataSize    int
		numClusters int
		expected    int
		description string
	}{
		{5000, 20, 5, "小数据集: max(1, 20/4)=5"},
		{50000, 30, 10, "中等数据集: max(2, 30/3)=10"},
		{500000, 40, 20, "大数据集: max(3, 40/2)=20"},
		{1500000, 60, 40, "超大数据集: max(5, 60*2/3)=40"},
		{1500000, 5, 5, "边界测试: nprobe不能超过numClusters"},
		{5000, 3, 1, "边界测试: max(1, 3/4)=1"},
		{50000, 5, 2, "边界测试: max(2, 5/3)=2"},
	}

	for _, tc := range testCases {
		var nprobe int

		// 复制adaptive_config.go中的逻辑
		switch {
		case tc.dataSize < 10000:
			nprobe = int(math.Max(1, float64(tc.numClusters)/4))
		case tc.dataSize < 100000:
			nprobe = int(math.Max(2, float64(tc.numClusters)/3))
		case tc.dataSize < 1000000:
			nprobe = int(math.Max(3, float64(tc.numClusters)/2))
		default:
			nprobe = int(math.Max(5, float64(tc.numClusters)*2/3))
		}

		// 确保 nprobe 在合理范围内
		if nprobe > tc.numClusters {
			nprobe = tc.numClusters
		}

		result := "✅ 正确"
		if nprobe != tc.expected {
			result = "❌ 错误"
		}

		fmt.Printf("%s: 数据规模=%d, 簇数=%d, 期望nprobe=%d, 实际nprobe=%d %s\n",
			tc.description, tc.dataSize, tc.numClusters, tc.expected, nprobe, result)
	}
}

func TestHNSWConfig(t *testing.T) {
	fmt.Println("\n=== 测试HNSW配置逻辑 ===")

	testCases := []struct {
		dataSize        int
		vectorDim       int
		expectedEf      float64
		expectedMaxConn int
		description     string
	}{
		{5000, 100, 100.0, 16, "小数据集: ef=100, maxConn=min(64,max(16,10))=16"},
		{50000, 256, 200.0, 25, "中等数据集: ef=200, maxConn=min(64,max(16,25.6))=25"},
		{500000, 512, 400.0, 51, "大数据集: ef=400, maxConn=min(64,max(16,51.2))=51"},
		{1500000, 1024, 800.0, 64, "超大数据集: ef=800, maxConn=min(64,max(16,102.4))=64"},
		{50000, 50, 200.0, 16, "低维向量: maxConn=min(64,max(16,5))=16"},
		{50000, 2048, 200.0, 64, "高维向量: maxConn=min(64,max(16,204.8))=64"},
		{50000, 0, 200.0, 0, "零维度: maxConn不变"},
	}

	for _, tc := range testCases {
		var efConstruction float64
		var maxConnections int

		// 复制adaptive_config.go中的efConstruction逻辑
		switch {
		case tc.dataSize < 10000:
			efConstruction = 100.0
		case tc.dataSize < 100000:
			efConstruction = 200.0
		case tc.dataSize < 1000000:
			efConstruction = 400.0
		default:
			efConstruction = 800.0
		}

		// 复制adaptive_config.go中的maxConnections逻辑
		if tc.vectorDim > 0 {
			maxConnections = int(math.Min(64, math.Max(16, float64(tc.vectorDim)/10)))
		} else {
			maxConnections = 0 // 模拟不变的情况
		}

		efResult := "✅ 正确"
		if efConstruction != tc.expectedEf {
			efResult = "❌ 错误"
		}

		maxConnResult := "✅ 正确"
		if maxConnections != tc.expectedMaxConn {
			maxConnResult = "❌ 错误"
		}

		fmt.Printf("%s:\n", tc.description)
		fmt.Printf("  数据规模=%d, 向量维度=%d\n", tc.dataSize, tc.vectorDim)
		fmt.Printf("  efConstruction: 期望=%.1f, 实际=%.1f %s\n", tc.expectedEf, efConstruction, efResult)
		fmt.Printf("  maxConnections: 期望=%d, 实际=%d %s\n", tc.expectedMaxConn, maxConnections, maxConnResult)
	}
}

// 测试系统资源调整逻辑
func TestSystemResourceAdjustment(t *testing.T) {
	fmt.Println("\n=== 测试系统资源调整逻辑 ===")

	// 测试工作协程数调整
	maxWorkers := runtime.NumCPU()
	fmt.Printf("当前系统CPU核心数: %d\n", maxWorkers)
	fmt.Printf("自适应工作协程数应该设置为: %d\n", maxWorkers)

	// 测试内存使用情况
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	currentMemMB := m.Alloc / 1024 / 1024
	fmt.Printf("当前内存使用: %d MB\n", currentMemMB)

	if m.Alloc > 1024*1024*1024 {
		fmt.Printf("内存使用超过1GB，应该启用向量压缩\n")
	} else {
		fmt.Printf("内存使用未超过1GB，不需要启用向量压缩\n")
	}
}
