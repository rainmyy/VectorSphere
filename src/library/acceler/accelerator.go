package acceler

import (
	"sync"
	"time"
	"unsafe"
)

// Accelerator 加速器接口
//type Accelerator interface {
//	Initialize() error
//	BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error)
//	BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error)
//	Cleanup() error
//}

// CPUConfig CPU配置结构体
type CPUConfig struct {
	Enable      bool   `json:"enable" yaml:"enable"`
	IndexType   string `json:"index_type" yaml:"index_type"`
	DeviceID    int    `json:"device_id" yaml:"device_id"`
	Threads     int    `json:"threads" yaml:"threads"`
	VectorWidth int    `json:"vector_width" yaml:"vector_width"`
}

// CPUAccelerator CPU加速器实现
type CPUAccelerator struct {
	*BaseAccelerator
}

// CpuAccelerator 为了兼容性，定义一个别名
type CpuAccelerator = CPUAccelerator

// AccelResult 搜索结果结构体
type AccelResult struct {
	ID         string
	Similarity float64
	Distance   float64
	Metadata   map[string]interface{}
	DocIds     []string
	Vector     []float64
	Index      int
}

// GPUAccelerator GPU加速器实现
type GPUAccelerator struct {
	*BaseAccelerator
	// GPU特定字段
	operationCount int64
	errorCount     int64
	batchSize      int
	streamCount    int
	gpuWrapper     unsafe.Pointer // 用于存储C结构体指针
	gpuResources   unsafe.Pointer // FAISS GPU资源

	// GPU特定统计信息
	gpuStats struct {
		ComputeTime     time.Duration
		KernelLaunches  int64
		MemoryTransfers int64
	}

	// 内存管理
	memoryUsed  int64
	memoryTotal int64
	deviceCount int
}

// PMemAccelerator 持久内存加速器实现
type PMemAccelerator struct {
	*BaseAccelerator
	// PMem特定字段
	devicePaths   []string
	deviceHandles []unsafe.Pointer
	config        *PMemConfig
	vectorCache   map[string][]float64 // 向量缓存
	cacheMutex    sync.RWMutex
	devicePath    string
	deviceSize    uint64
	memoryPool    map[string][]float64 // 模拟持久内存存储
	namespaces    map[string]*PMemNamespace
}

// PMemConfig 持久内存配置
type PMemConfig struct {
	Enable       bool                   `json:"enable"`
	DevicePaths  []string               `json:"device_paths"`
	Mode         string                 `json:"mode"` // "app_direct", "memory_mode", "mixed"
	Namespaces   []PMemNamespace        `json:"namespaces"`
	Interleaving PMemInterleavingConfig `json:"interleaving"`
	Persistence  PMemPersistenceConfig  `json:"persistence"`
	Performance  PMemPerformanceConfig  `json:"performance"`
	Reliability  PMemReliabilityConfig  `json:"reliability"`

	DevicePath        string `json:"device_path"`
	PoolSize          uint64 `json:"pool_size"`
	EnableCompression bool   `json:"enable_compression"`
	EnableEncryption  bool   `json:"enable_encryption"`
}

// PMemNamespace 持久内存命名空间
type PMemNamespace struct {
	Name       string `json:"name"`
	Size       int64  `json:"size"` // bytes
	Mode       string `json:"mode"` // "fsdax", "devdax", "sector"
	Alignment  int    `json:"alignment"`
	SectorSize int    `json:"sector_size"`

	MapSync bool `json:"map_sync"`
}

// PMemInterleavingConfig 持久内存交错配置
type PMemInterleavingConfig struct {
	Enable      bool `json:"enable"`
	Ways        int  `json:"ways"`
	Granularity int  `json:"granularity"` // bytes
	Alignment   int  `json:"alignment"`
}

// PMemPersistenceConfig 持久内存持久化配置
type PMemPersistenceConfig struct {
	FlushStrategy      string        `json:"flush_strategy"` // "sync", "async", "lazy"
	FlushInterval      time.Duration `json:"flush_interval"`
	Checkpointing      bool          `json:"checkpointing"`
	CheckpointInterval time.Duration `json:"checkpoint_interval"`
	RecoveryMode       string        `json:"recovery_mode"` // "fast", "safe", "auto"

	// stub
	FlushMode       string        `json:"flush_mode"` // auto, manual, async
	SyncOnWrite     bool          `json:"sync_on_write"`
	ChecksumEnabled bool          `json:"checksum_enabled"`
	BackupEnabled   bool          `json:"backup_enabled"`
	BackupInterval  time.Duration `json:"backup_interval"`
}

// PMemPerformanceConfig 持久内存性能配置
type PMemPerformanceConfig struct {
	ReadAhead        bool `json:"read_ahead"`
	WriteBehind      bool `json:"write_behind"`
	BatchSize        int  `json:"batch_size"`
	QueueDepth       int  `json:"queue_depth"`
	NUMAOptimization bool `json:"numa_optimization"`

	// stub
	PrefetchEnabled  bool   `json:"prefetch_enabled"`
	CacheSize        uint64 `json:"cache_size"`
	CompressionLevel int    `json:"compression_level"`
}

// PMemReliabilityConfig 持久内存可靠性配置
type PMemReliabilityConfig struct {
	ECC                bool          `json:"ecc"`
	Scrubbing          bool          `json:"scrubbing"`
	ScrubbingInterval  time.Duration `json:"scrubbing_interval"`
	BadBlockManagement bool          `json:"bad_block_management"`
	WearLeveling       bool          `json:"wear_leveling"`

	ECCEnabled     bool          `json:"ecc_enabled"`
	ScrubInterval  time.Duration `json:"scrub_interval"`
	ErrorThreshold int           `json:"error_threshold"`
	RepairEnabled  bool          `json:"repair_enabled"`
	MirrorEnabled  bool          `json:"mirror_enabled"`
	MirrorDevices  []string      `json:"mirror_devices"`
}

// FPGAAccelerator FPGA加速器实现
type FPGAAccelerator struct {
	*BaseAccelerator
	// FPGA特定字段
	deviceHandle  unsafe.Pointer
	config        *FPGAConfig
	bitstream     string
}

// FPGAConfig FPGA配置
type FPGAConfig struct {
	Enable          bool                      `json:"enable"`
	DeviceIDs       []int                     `json:"device_ids"`
	Bitstream       string                    `json:"bitstream"`
	ClockFrequency  int                       `json:"clock_frequency"`  // MHz
	MemoryBandwidth int64                     `json:"memory_bandwidth"` // bytes/sec
	PipelineDepth   int                       `json:"pipeline_depth"`
	Parallelism     FPGAParallelismConfig     `json:"parallelism"`
	Optimization    FPGAOptimizationConfig    `json:"optimization"`
	Reconfiguration FPGAReconfigurationConfig `json:"reconfiguration"`
}

// FPGAParallelismConfig FPGA并行配置
type FPGAParallelismConfig struct {
	ComputeUnits   int `json:"compute_units"`
	VectorWidth    int `json:"vector_width"`
	UnrollFactor   int `json:"unroll_factor"`
	PipelineStages int `json:"pipeline_stages"`
}

// FPGAOptimizationConfig FPGA优化配置
type FPGAOptimizationConfig struct {
	ResourceSharing    bool `json:"resource_sharing"`
	MemoryOptimization bool `json:"memory_optimization"`
	TimingOptimization bool `json:"timing_optimization"`
	PowerOptimization  bool `json:"power_optimization"`
	AreaOptimization   bool `json:"area_optimization"`
}

// FPGAReconfigurationConfig FPGA重配置
type FPGAReconfigurationConfig struct {
	Enable                 bool          `json:"enable"`
	PartialReconfiguration bool          `json:"partial_reconfiguration"`
	ReconfigurationTime    time.Duration `json:"reconfiguration_time"`
	BitstreamCache         bool          `json:"bitstream_cache"`
	HotSwap                bool          `json:"hot_swap"`
}

// RDMAConfig RDMA配置
type RDMAConfig struct {
	Enable    bool   `json:"enable" yaml:"enable"`
	DeviceID  int    `json:"device_id" yaml:"device_id"`
	PortNum   int    `json:"port_num" yaml:"port_num"`
	QueueSize int    `json:"queue_size" yaml:"queue_size"`
	Protocol  string `json:"protocol" yaml:"protocol"`

	// 为了兼容RDMA加速器，添加必要的字段
	Devices            []RDMADevice                  `json:"devices,omitempty"`
	TransportType      string                        `json:"transport_type,omitempty"`
	QueuePairs         *RDMAQueuePairConfig          `json:"queue_pairs,omitempty"`
	MemoryRegistration *RDMAMemoryRegistrationConfig `json:"memory_registration,omitempty"`
	CongestionControl  *RDMACongestionControlConfig  `json:"congestion_control,omitempty"`
	PerformanceTuning  *RDMAPerformanceTuningConfig  `json:"performance_tuning,omitempty"`
	ClusterNodes       []string                      `json:"cluster_nodes,omitempty"`
}

// RDMADevice RDMA设备
type RDMADevice struct {
	Name      string `json:"name"`
	Port      int    `json:"port"`
	GID       string `json:"gid"`
	LID       int    `json:"lid"`
	MTU       int    `json:"mtu"`
	LinkLayer string `json:"link_layer"`
}

// RDMAQueuePairConfig RDMA队列对配置
type RDMAQueuePairConfig struct {
	MaxQPs           int `json:"max_qps"`
	SendQueueSize    int `json:"send_queue_size"`
	ReceiveQueueSize int `json:"receive_queue_size"`
	MaxInlineData    int `json:"max_inline_data"`
	MaxSGE           int `json:"max_sge"`
	RetryCount       int `json:"retry_count"`
	RNRRetryCount    int `json:"rnr_retry_count"`
	Timeout          int `json:"timeout"`
}

// RDMAMemoryRegistrationConfig RDMA内存注册配置
type RDMAMemoryRegistrationConfig struct {
	Strategy          string `json:"strategy"`
	CacheSize         int64  `json:"cache_size"`
	HugePagesEnable   bool   `json:"huge_pages_enable"`
	MemoryPinning     bool   `json:"memory_pinning"`
	RegistrationCache bool   `json:"registration_cache"`
}

// RDMACongestionControlConfig RDMA拥塞控制配置
type RDMACongestionControlConfig struct {
	Algorithm    string  `json:"algorithm"`
	ECNThreshold float64 `json:"ecn_threshold"`
	RateIncrease float64 `json:"rate_increase"`
	RateDecrease float64 `json:"rate_decrease"`
	MinRate      int64   `json:"min_rate"`
	MaxRate      int64   `json:"max_rate"`
}

// RDMAPerformanceTuningConfig RDMA性能调优配置
type RDMAPerformanceTuningConfig struct {
	BatchSize           int   `json:"batch_size"`
	PollingMode         bool  `json:"polling_mode"`
	InterruptCoalescing bool  `json:"interrupt_coalescing"`
	CPUAffinity         []int `json:"cpu_affinity"`
	NUMAOptimization    bool  `json:"numa_optimization"`
	ZeroCopy            bool  `json:"zero_copy"`
}

//// 新增 EnableHybridMode 方法
//func (c *FAISSAccelerator) EnableHybridMode() error {
//	c.mu.Lock()
//	defer c.mu.Unlock()
//
//	if !c.initialized {
//		return fmt.Errorf("GPU 加速器未初始化")
//	}
//
//	// 检测 CPU 硬件能力
//	cpuDetector := &HardwareDetector{}
//	cpuCaps := cpuDetector.GetHardwareCapabilities()
//
//	// 设置混合模式
//	c.hybridMode = true
//	c.cpuCapabilities = cpuCaps
//
//	log.Info("已启用混合计算模式 - CPU: %d 核心, AVX2: %v, AVX512: %v",
//		cpuCaps.CPUCores, cpuCaps.HasAVX2, cpuCaps.HasAVX512)
//
//	return nil
//}
//
//// 新增 hybridBatchSearch 方法
//func (c *FAISSAccelerator) hybridBatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
//	// 根据数据特性决定处理方式
//	if len(queries) < 10 || len(database) < 1000 {
//		// 小数据量使用 CPU
//		return c.batchSearchCPUFallback(queries, database, k)
//	}
//
//	// 大数据量使用 GPU
//	return c.batchSearchGPU(queries, database, k)
//}
