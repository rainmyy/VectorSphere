package acceler

import (
	"VectorSphere/src/library/entity"
	"fmt"
	"time"
)

// VectorDBAdapter 向量数据库硬件加速适配器
// 提供向量数据库调用硬件加速的统一接口
type VectorDBAdapter struct {
	hardwareManager *HardwareManager
	defaultWorkload WorkloadProfile
	enabled         bool
}

// NewVectorDBAdapter 创建新的向量数据库适配器
func NewVectorDBAdapter() *VectorDBAdapter {
	adapter := &VectorDBAdapter{
		hardwareManager: NewHardwareManager(),
		defaultWorkload: WorkloadProfile{
			BatchSize:        1000,
			VectorDimension:  128,
			DataType:         "float32",
			LatencyTarget:    10 * time.Millisecond,
			ThroughputTarget: 10000,
			AccuracyTarget:   0.95,
			MemoryBudget:     1024 * 1024 * 1024, // 1GB
			Concurrency:      4,
		},
		enabled: true,
	}

	// 初始化所有硬件加速器
	if err := adapter.hardwareManager.InitializeAll(); err != nil {
		fmt.Printf("警告: 硬件加速器初始化失败: %v\n", err)
		adapter.enabled = false
	}

	return adapter
}

// IsEnabled 检查硬件加速是否启用
func (vda *VectorDBAdapter) IsEnabled() bool {
	return vda.enabled && len(vda.hardwareManager.GetAvailableAccelerators()) > 0
}

// Enable 启用硬件加速
func (vda *VectorDBAdapter) Enable() error {
	if err := vda.hardwareManager.InitializeAll(); err != nil {
		return fmt.Errorf("启用硬件加速失败: %v", err)
	}
	vda.enabled = true
	return nil
}

// Disable 禁用硬件加速
func (vda *VectorDBAdapter) Disable() error {
	vda.enabled = false
	return vda.hardwareManager.ShutdownAll()
}

// ComputeSimilarity 计算相似度（单查询）
func (vda *VectorDBAdapter) ComputeSimilarity(query []float64, vectors [][]float64, workloadType string) ([]float64, error) {
	if !vda.IsEnabled() {
		return vda.computeSimilarityCPU(query, vectors), nil
	}

	acc := vda.hardwareManager.GetBestAccelerator(workloadType)
	if acc == nil {
		return vda.computeSimilarityCPU(query, vectors), nil
	}

	return acc.ComputeDistance(query, vectors)
}

// BatchComputeSimilarity 批量计算相似度
func (vda *VectorDBAdapter) BatchComputeSimilarity(queries [][]float64, vectors [][]float64, workloadType string) ([][]float64, error) {
	if !vda.IsEnabled() {
		return vda.batchComputeSimilarityCPU(queries, vectors), nil
	}

	return vda.hardwareManager.BatchComputeDistance(queries, vectors, workloadType)
}

// Search 向量搜索
func (vda *VectorDBAdapter) Search(query []float64, database [][]float64, k int, workloadType string) ([]AccelResult, error) {
	if !vda.IsEnabled() {
		return vda.searchCPU(query, database, k), nil
	}

	options := entity.SearchOptions{K: k}
	return vda.hardwareManager.AccelerateSearch(query, database, options, workloadType)
}

// BatchSearch 批量向量搜索
func (vda *VectorDBAdapter) BatchSearch(queries [][]float64, database [][]float64, k int, workloadType string) ([][]AccelResult, error) {
	if !vda.IsEnabled() {
		return vda.batchSearchCPU(queries, database, k), nil
	}

	return vda.hardwareManager.BatchSearch(queries, database, k, workloadType)
}

// OptimizeSearch 优化搜索结果
func (vda *VectorDBAdapter) OptimizeSearch(query []float64, database [][]float64, options entity.SearchOptions, workloadType string) ([]AccelResult, error) {
	if !vda.IsEnabled() {
		return vda.searchCPU(query, database, options.K), nil
	}

	return vda.hardwareManager.AccelerateSearch(query, database, options, workloadType)
}

// GetRecommendedAccelerator 获取推荐的加速器
func (vda *VectorDBAdapter) GetRecommendedAccelerator(workload WorkloadProfile) string {
	if !vda.IsEnabled() {
		return "cpu"
	}

	acc := vda.hardwareManager.GetOptimalAccelerator(workload)
	if acc == nil {
		return "cpu"
	}

	return string(acc.GetType())
}

// AutoTune 自动调优
func (vda *VectorDBAdapter) AutoTune(workload WorkloadProfile) error {
	if !vda.IsEnabled() {
		return nil
	}

	vda.defaultWorkload = workload
	return vda.hardwareManager.AutoTuneAll(workload)
}

// GetPerformanceReport 获取性能报告
func (vda *VectorDBAdapter) GetPerformanceReport() map[string]interface{} {
	report := map[string]interface{}{
		"enabled":                vda.enabled,
		"available_accelerators": vda.hardwareManager.GetAvailableAccelerators(),
		"system_info":            vda.hardwareManager.GetSystemInfo(),
		"timestamp":              time.Now(),
	}

	if vda.IsEnabled() {
		report["capabilities"] = vda.hardwareManager.GetAllCapabilities()
		report["stats"] = vda.hardwareManager.GetAllStats()
		report["metrics"] = vda.hardwareManager.GetAllPerformanceMetrics()
		report["health"] = vda.hardwareManager.HealthCheck()
	}

	return report
}

// SetWorkloadProfile 设置工作负载配置
func (vda *VectorDBAdapter) SetWorkloadProfile(workload WorkloadProfile) {
	vda.defaultWorkload = workload
}

// GetWorkloadProfile 获取工作负载配置
func (vda *VectorDBAdapter) GetWorkloadProfile() WorkloadProfile {
	return vda.defaultWorkload
}

// GetHardwareManager 获取硬件管理器（用于高级操作）
func (vda *VectorDBAdapter) GetHardwareManager() *HardwareManager {
	return vda.hardwareManager
}

// Shutdown 关闭适配器
func (vda *VectorDBAdapter) Shutdown() error {
	vda.enabled = false
	return vda.hardwareManager.ShutdownAll()
}

// CPU备用实现

// computeSimilarityCPU CPU计算相似度
func (vda *VectorDBAdapter) computeSimilarityCPU(query []float64, vectors [][]float64) []float64 {
	results := make([]float64, len(vectors))
	for i, vec := range vectors {
		// 计算欧几里得距离
		dist := 0.0
		for j := range query {
			if j < len(vec) {
				diff := query[j] - vec[j]
				dist += diff * diff
			}
		}
		results[i] = dist
	}
	return results
}

// batchComputeSimilarityCPU CPU批量计算相似度
func (vda *VectorDBAdapter) batchComputeSimilarityCPU(queries [][]float64, vectors [][]float64) [][]float64 {
	results := make([][]float64, len(queries))
	for i, query := range queries {
		results[i] = vda.computeSimilarityCPU(query, vectors)
	}
	return results
}

// searchCPU CPU搜索
func (vda *VectorDBAdapter) searchCPU(query []float64, database [][]float64, k int) []AccelResult {
	// 计算距离
	distances := vda.computeSimilarityCPU(query, database)

	// 创建索引-距离对
	type indexDistance struct {
		index    int
		distance float64
	}

	indexDistances := make([]indexDistance, len(distances))
	for i, dist := range distances {
		indexDistances[i] = indexDistance{index: i, distance: dist}
	}

	// 部分排序，只需要前k个
	for i := 0; i < k && i < len(indexDistances); i++ {
		minIdx := i
		for j := i + 1; j < len(indexDistances); j++ {
			if indexDistances[j].distance < indexDistances[minIdx].distance {
				minIdx = j
			}
		}
		indexDistances[i], indexDistances[minIdx] = indexDistances[minIdx], indexDistances[i]
	}

	// 构建结果
	results := make([]AccelResult, k)
	for i := 0; i < k && i < len(indexDistances); i++ {
		results[i] = AccelResult{
			ID:         fmt.Sprintf("vec_%d", indexDistances[i].index),
			Similarity: 1.0 / (1.0 + indexDistances[i].distance), // 转换为相似度
			Metadata:   map[string]interface{}{"index": indexDistances[i].index, "cpu_fallback": true},
		}
	}

	return results
}

// batchSearchCPU CPU批量搜索
func (vda *VectorDBAdapter) batchSearchCPU(queries [][]float64, database [][]float64, k int) [][]AccelResult {
	results := make([][]AccelResult, len(queries))
	for i, query := range queries {
		results[i] = vda.searchCPU(query, database, k)
	}
	return results
}

// 工作负载类型常量
const (
	WorkloadLowLatency     = "low_latency"
	WorkloadHighThroughput = "high_throughput"
	WorkloadDistributed    = "distributed"
	WorkloadPersistent     = "persistent"
	WorkloadBalanced       = "balanced"
	WorkloadMemoryOptimal  = "memory_optimal"
)

// CreateWorkloadProfile 创建工作负载配置
func CreateWorkloadProfile(workloadType string, batchSize int, dimension int) WorkloadProfile {
	base := WorkloadProfile{
		BatchSize:       batchSize,
		VectorDimension: dimension,
		DataType:        "float32",
		Concurrency:     4,
		MemoryBudget:    1024 * 1024 * 1024, // 1GB
	}

	switch workloadType {
	case WorkloadLowLatency:
		base.LatencyTarget = 1 * time.Millisecond
		base.ThroughputTarget = 1000
		base.AccuracyTarget = 0.90
	case WorkloadHighThroughput:
		base.LatencyTarget = 50 * time.Millisecond
		base.ThroughputTarget = 100000
		base.AccuracyTarget = 0.95
	case WorkloadDistributed:
		base.LatencyTarget = 10 * time.Millisecond
		base.ThroughputTarget = 50000
		base.AccuracyTarget = 0.95
		base.Concurrency = 16
	case WorkloadPersistent:
		base.LatencyTarget = 20 * time.Millisecond
		base.ThroughputTarget = 10000
		base.AccuracyTarget = 0.98
		base.MemoryBudget = 8 * 1024 * 1024 * 1024 // 8GB
	case WorkloadMemoryOptimal:
		base.LatencyTarget = 15 * time.Millisecond
		base.ThroughputTarget = 20000
		base.AccuracyTarget = 0.95
		base.MemoryBudget = 512 * 1024 * 1024 // 512MB
	default: // WorkloadBalanced
		base.LatencyTarget = 10 * time.Millisecond
		base.ThroughputTarget = 10000
		base.AccuracyTarget = 0.95
	}

	return base
}

// OptimizeForWorkload 根据工作负载优化适配器
func (vda *VectorDBAdapter) OptimizeForWorkload(workloadType string, batchSize int, dimension int) error {
	workload := CreateWorkloadProfile(workloadType, batchSize, dimension)
	vda.SetWorkloadProfile(workload)
	return vda.AutoTune(workload)
}

// GetOptimalBatchSize 获取最优批处理大小
func (vda *VectorDBAdapter) GetOptimalBatchSize(workloadType string) int {
	if !vda.IsEnabled() {
		return 100 // CPU默认批处理大小
	}

	acc := vda.hardwareManager.GetBestAccelerator(workloadType)
	if acc == nil {
		return 100
	}

	caps := acc.GetCapabilities()
	return caps.MaxBatchSize / 10 // 使用最大批处理大小的10%作为推荐值
}

// EstimatePerformance 估算性能
func (vda *VectorDBAdapter) EstimatePerformance(workload WorkloadProfile) map[string]interface{} {
	estimate := map[string]interface{}{
		"workload":  workload,
		"timestamp": time.Now(),
	}

	if !vda.IsEnabled() {
		estimate["accelerator"] = "cpu"
		estimate["estimated_latency"] = 100 * time.Millisecond
		estimate["estimated_throughput"] = 1000.0
		return estimate
	}

	acc := vda.hardwareManager.GetOptimalAccelerator(workload)
	if acc == nil {
		estimate["accelerator"] = "cpu"
		estimate["estimated_latency"] = 100 * time.Millisecond
		estimate["estimated_throughput"] = 1000.0
		return estimate
	}

	caps := acc.GetCapabilities()
	metrics := acc.GetPerformanceMetrics()

	estimate["accelerator"] = caps.Type
	estimate["estimated_latency"] = caps.Latency
	estimate["estimated_throughput"] = metrics.ThroughputCurrent
	estimate["capabilities"] = caps
	estimate["current_metrics"] = metrics

	return estimate
}
