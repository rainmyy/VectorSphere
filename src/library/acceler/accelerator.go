package acceler

import (
	"sync"
	"time"
	"unsafe"
)

// Accelerator GPU 加速器接口
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

// CPUAccelerator FAISS 加速器实现
type CPUAccelerator struct {
	deviceID        int
	initialized     bool
	available       bool
	mu              sync.RWMutex
	indexType       string
	dimension       int
	strategy        *ComputeStrategySelector
	currentStrategy ComputeStrategy
	dataSize        int
}

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
	deviceID        int
	initialized     bool
	available       bool
	mu              sync.RWMutex
	indexType       string
	dimension       int
	strategy        *ComputeStrategySelector
	currentStrategy ComputeStrategy
	dataSize        int
	// GPU特定字段
	operationCount int64
	errorCount     int64
	batchSize      int
	streamCount    int
	gpuWrapper     unsafe.Pointer // 用于存储C结构体指针
	gpuResources   unsafe.Pointer // FAISS GPU资源

	// 统计信息
	stats struct {
		TotalOperations int64
		SuccessfulOps   int64
		FailedOps       int64
		ComputeTime     time.Duration
		KernelLaunches  int64
		MemoryTransfers int64
		LastUsed        time.Time
	}

	// 性能指标
	performanceMetrics PerformanceMetrics

	// 内存管理
	memoryUsed  int64
	memoryTotal int64
	deviceCount int
}

// PMemAccelerator 持久内存加速器实现
type PMemAccelerator struct {
	devicePaths   []string
	deviceHandles []unsafe.Pointer
	initialized   bool
	available     bool
	capabilities  HardwareCapabilities
	stats         HardwareStats
	mutex         sync.RWMutex
	config        *PMemConfig
	lastStatsTime time.Time
	startTime     time.Time
	vectorCache   map[string][]float64 // 向量缓存
	cacheMutex    sync.RWMutex

	devicePath  string
	deviceSize  uint64
	memoryPool  map[string][]float64 // 模拟持久内存存储
	namespaces  map[string]*PMemNamespace
	performance PerformanceMetrics
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
