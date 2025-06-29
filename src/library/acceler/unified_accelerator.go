package acceler

import (
	"VectorSphere/src/library/entity"
	"fmt"
	"math"
	"sync"
	"time"
)

const (
	AcceleratorCPU  string = "CPU"
	AcceleratorGPU  string = "GPU"
	AcceleratorFPGA string = "FPGA"
	AcceleratorPMem string = "PMem"
	AcceleratorRDMA string = "RDMA"
	AcceleratorNPU  string = "NPU"
)

// UnifiedAccelerator 统一硬件加速器接口
type UnifiedAccelerator interface {
	// GetType 基础生命周期管理
	GetType() string
	IsAvailable() bool
	Initialize() error
	Shutdown() error
	//Start() error
	//Stop() error

	// 核心计算功能
	ComputeDistance(query []float64, vectors [][]float64) ([]float64, error)
	BatchComputeDistance(queries [][]float64, vectors [][]float64) ([][]float64, error)
	BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error)
	BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error)

	// 高级功能
	AccelerateSearch(query []float64, database [][]float64, options entity.SearchOptions) ([]AccelResult, error)
	//OptimizeMemoryLayout(vectors [][]float64) error
	//PrefetchData(vectors [][]float64) error

	// 能力和统计信息
	GetCapabilities() HardwareCapabilities
	GetStats() HardwareStats
	GetPerformanceMetrics() PerformanceMetrics

	// 配置和调优
	AutoTune(workload WorkloadProfile) error
	//UpdateConfig(config interface{}) error
	//Cleanup() error
}

// HardwareCapabilities 硬件能力信息
type HardwareCapabilities struct {
	HasAVX2           bool          `json:"has_avx2"`
	HasAVX512         bool          `json:"has_avx512"`
	HasGPU            bool          `json:"has_gpu"`
	CPUCores          int           `json:"cpu_cores"`
	GPUDevices        int           `json:"gpu_devices"`
	Type              string        `json:"type"`
	MemorySize        int64         `json:"memory_size"`        // 内存大小(字节)
	ComputeUnits      int           `json:"compute_units"`      // 计算单元数量
	MaxBatchSize      int           `json:"max_batch_size"`     // 最大批处理大小
	SupportedOps      []string      `json:"supported_ops"`      // 支持的操作类型
	PerformanceRating float64       `json:"performance_rating"` // 性能评级
	Bandwidth         int64         `json:"bandwidth"`          // 内存带宽(bytes/sec)
	Latency           time.Duration `json:"latency"`            // 访问延迟
	PowerConsumption  float64       `json:"power_consumption"`  // 功耗(瓦特)
	SpecialFeatures   []string      `json:"special_features"`   // 特殊功能
}

// HardwareStats 硬件统计信息
type HardwareStats struct {
	TotalOperations   int64         `json:"total_operations"`
	SuccessfulOps     int64         `json:"successful_ops"`
	FailedOps         int64         `json:"failed_ops"`
	ErrorCount        int64         `json:"error_count"`        // 错误计数
	AverageLatency    time.Duration `json:"average_latency"`
	Throughput        float64       `json:"throughput"`         // 操作/秒
	MemoryUtilization float64       `json:"memory_utilization"` // 内存利用率
	Temperature       float64       `json:"temperature"`        // 温度(如果支持)
	PowerConsumption  float64       `json:"power_consumption"`  // 功耗(如果支持)
	ErrorRate         float64       `json:"error_rate"`         // 错误率
	LastUsed          time.Time     `json:"uptime"`             // 运行时间
}

// PerformanceMetrics 性能指标
type PerformanceMetrics struct {
	LatencyCurrent      time.Duration      `json:"latency_current"`
	LatencyMin          time.Duration      `json:"latency_min"`
	LatencyMax          time.Duration      `json:"latency_max"`
	LatencyP50          float64            `json:"latency_p50"`
	LatencyP95          float64            `json:"latency_p95"`
	LatencyP99          float64            `json:"latency_p99"`
	ThroughputCurrent   float64            `json:"throughput_current"`
	ThroughputPeak      float64            `json:"throughput_peak"`
	CacheHitRate        float64            `json:"cache_hit_rate"`
	ResourceUtilization map[string]float64 `json:"resource_utilization"`

	MemoryUsage                 float64
	CPUUsage                    float64
	Throughput                  float64
	LastSuccessfulOperationTime *time.Time
}

// WorkloadProfile 工作负载配置文件
type WorkloadProfile struct {
	VectorDimension  int           `json:"vector_dimension"`
	BatchSize        int           `json:"batch_size"`
	QueryFrequency   float64       `json:"query_frequency"`
	DataSize         int64         `json:"data_size"`
	AccessPattern    string        `json:"access_pattern"` // "sequential", "random", "mixed"
	LatencyTarget    time.Duration `json:"latency_target"`
	ThroughputTarget float64       `json:"throughput_target"`
	Type             string        `json:"type"`
	AccuracyTarget   float64       `json:"accuracy_target"`
	Concurrency      int           `json:"concurrency"`
	MemoryBudget     int64         `json:"memory_budget"`
	DataType         string        `json:"date_type"`
}

// AcceleratorManager 加速器管理器
type AcceleratorManager struct {
	accelerators map[string]UnifiedAccelerator
	mutex        sync.RWMutex
	strategies   map[string]string // 工作负载到加速器的映射
}

// NewAcceleratorManager 创建新的加速器管理器
func NewAcceleratorManager() *AcceleratorManager {
	return &AcceleratorManager{
		accelerators: make(map[string]UnifiedAccelerator),
		strategies:   make(map[string]string),
	}
}

// RegisterAccelerator 注册加速器
func (am *AcceleratorManager) RegisterAccelerator(accel UnifiedAccelerator) error {
	am.mutex.Lock()
	defer am.mutex.Unlock()

	if !accel.IsAvailable() {
		return fmt.Errorf("加速器 %s 不可用", accel.GetType())
	}

	am.accelerators[accel.GetType()] = accel
	return nil
}

// GetAccelerator 获取指定类型的加速器
func (am *AcceleratorManager) GetAccelerator(accelType string) (UnifiedAccelerator, error) {
	am.mutex.RLock()
	defer am.mutex.RUnlock()

	accel, exists := am.accelerators[accelType]
	if !exists {
		return nil, fmt.Errorf("加速器 %s 未注册", accelType)
	}

	return accel, nil
}

// GetBestAccelerator 根据工作负载获取最佳加速器
func (am *AcceleratorManager) GetBestAccelerator(workload WorkloadProfile) (UnifiedAccelerator, error) {
	am.mutex.RLock()
	defer am.mutex.RUnlock()

	// 根据工作负载特征选择最佳加速器
	var bestAccel UnifiedAccelerator
	var bestScore float64

	for _, accel := range am.accelerators {
		if !accel.IsAvailable() {
			continue
		}

		caps := accel.GetCapabilities()
		score := am.calculateAcceleratorScore(caps, workload)

		if score > bestScore {
			bestScore = score
			bestAccel = accel
		}
	}

	if bestAccel == nil {
		return nil, fmt.Errorf("没有可用的加速器")
	}

	return bestAccel, nil
}

// calculateAcceleratorScore 计算加速器评分
func (am *AcceleratorManager) calculateAcceleratorScore(caps HardwareCapabilities, workload WorkloadProfile) float64 {
	score := caps.PerformanceRating

	// 根据内存需求调整评分
	memoryNeeded := int64(workload.VectorDimension * workload.BatchSize * 8) // float64 = 8 bytes
	if caps.MemorySize > 0 {
		memoryRatio := float64(memoryNeeded) / float64(caps.MemorySize)
		if memoryRatio > 1.0 {
			score *= 0.5 // 内存不足，降低评分
		} else if memoryRatio < 0.5 {
			score *= 1.2 // 内存充足，提高评分
		}
	}

	// 根据批处理大小调整评分
	if caps.MaxBatchSize > 0 && workload.BatchSize > caps.MaxBatchSize {
		score *= 0.7 // 批处理大小超限，降低评分
	}

	// 根据延迟要求调整评分
	if workload.LatencyTarget > 0 && caps.Latency > workload.LatencyTarget {
		score *= 0.8 // 延迟不满足要求，降低评分
	}

	return score
}

// InitializeAll 初始化所有注册的加速器
func (am *AcceleratorManager) InitializeAll() error {
	am.mutex.Lock()
	defer am.mutex.Unlock()

	for accelType, accel := range am.accelerators {
		if err := accel.Initialize(); err != nil {
			return fmt.Errorf("初始化加速器 %s 失败: %v", accelType, err)
		}
	}

	return nil
}

// ShutdownAll 关闭所有加速器
func (am *AcceleratorManager) ShutdownAll() error {
	am.mutex.Lock()
	defer am.mutex.Unlock()

	var errors []error
	for accelType, accel := range am.accelerators {
		if err := accel.Shutdown(); err != nil {
			errors = append(errors, fmt.Errorf("关闭加速器 %s 失败: %v", accelType, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("关闭加速器时发生错误: %v", errors)
	}

	return nil
}

// GetAllCapabilities 获取所有加速器的能力信息
func (am *AcceleratorManager) GetAllCapabilities() map[string]HardwareCapabilities {
	am.mutex.RLock()
	defer am.mutex.RUnlock()

	caps := make(map[string]HardwareCapabilities)
	for accelType, accel := range am.accelerators {
		caps[accelType] = accel.GetCapabilities()
	}

	return caps
}

// GetAllStats 获取所有加速器的统计信息
func (am *AcceleratorManager) GetAllStats() map[string]HardwareStats {
	am.mutex.RLock()
	defer am.mutex.RUnlock()

	stats := make(map[string]HardwareStats)
	for accelType, accel := range am.accelerators {
		stats[accelType] = accel.GetStats()
	}

	return stats
}

// GetOptimalAccelerator 通过策略选择最优加速器
func (am *AcceleratorManager) GetOptimalAccelerator(workload WorkloadProfile) UnifiedAccelerator {
	am.mutex.RLock()
	defer am.mutex.RUnlock()

	// 首先检查是否有预定义的策略
	workloadKey := am.getWorkloadKey(workload)
	if preferredType, exists := am.strategies[workloadKey]; exists {
		if accel, found := am.accelerators[preferredType]; found && accel.IsAvailable() {
			return accel
		}
	}

	// 如果没有预定义策略，使用GetBestAccelerator的逻辑
	bestAccel, err := am.GetBestAccelerator(workload)
	if err != nil {
		// 如果GetBestAccelerator失败，返回第一个可用的加速器
		for _, accel := range am.accelerators {
			if accel.IsAvailable() {
				return accel
			}
		}
		return nil
	}

	return bestAccel
}

// SetStrategy 设置工作负载策略
func (am *AcceleratorManager) SetStrategy(workloadType string, acceleratorType string) {
	am.mutex.Lock()
	defer am.mutex.Unlock()
	am.strategies[workloadType] = acceleratorType
}

// getWorkloadKey 根据工作负载特征生成键
func (am *AcceleratorManager) getWorkloadKey(workload WorkloadProfile) string {
	// 根据工作负载特征生成策略键
	if workload.BatchSize > 1000 {
		return "high_throughput"
	}
	if workload.LatencyTarget > 0 && workload.LatencyTarget < 10*time.Millisecond {
		return "low_latency"
	}
	if workload.VectorDimension > 1024 {
		return "high_dimension"
	}
	return "default"
}

func AccelerateSearch(query []float64, database [][]float64, options entity.SearchOptions) ([]AccelResult, error) {
	if len(query) == 0 || len(database) == 0 {
		return nil, fmt.Errorf("empty query or database")
	}
	k := options.K
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive")
	}

	distances := make([]struct {
		index    int
		distance float64
	}, len(database))
	for j, dbVector := range database {
		if len(dbVector) != len(query) {
			return nil, fmt.Errorf("dimension mismatch: query %d, database %d", len(query), len(dbVector))
		}
		dist := 0.0
		for d := 0; d < len(query); d++ {
			diff := query[d] - dbVector[d]
			dist += diff * diff
		}
		distances[j] = struct {
			index    int
			distance float64
		}{j, math.Sqrt(dist)}
	}

	// TopK 选择
	for p := 0; p < k && p < len(distances); p++ {
		minIdx := p
		for q := p + 1; q < len(distances); q++ {
			if distances[q].distance < distances[minIdx].distance {
				minIdx = q
			}
		}
		if minIdx != p {
			distances[p], distances[minIdx] = distances[minIdx], distances[p]
		}
	}

	results := make([]AccelResult, 0, k)
	for j := 0; j < k && j < len(distances); j++ {
		results = append(results, AccelResult{
			ID:         fmt.Sprintf("vec_%d", distances[j].index),
			Similarity: 1.0 / (1.0 + distances[j].distance),
			Distance:   distances[j].distance,
			Index:      distances[j].index,
		})
	}
	return results, nil
}
