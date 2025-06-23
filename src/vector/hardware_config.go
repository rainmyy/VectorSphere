package vector

import (
	"runtime"
	"time"
)

// HardwareAccelerationConfig 硬件加速配置
type HardwareAccelerationConfig struct {
	GPU  GPUConfig             `json:"gpu" yaml:"gpu"`
	FPGA FPGAConfig            `json:"fpga" yaml:"fpga"`
	PMem PMemConfig            `json:"pmem" yaml:"pmem"`
	RDMA RDMANetworkConfig     `json:"rdma" yaml:"rdma"`
	CPU  CPUOptimizationConfig `json:"cpu" yaml:"cpu"`
	NPU  NPUConfig             `json:"npu" yaml:"npu"`
}

// GPUConfig GPU配置
type GPUConfig struct {
	Enable            bool                 `json:"enable" yaml:"enable"`
	DeviceIDs         []int                `json:"device_ids" yaml:"device_ids"`
	MemoryLimit       int64                `json:"memory_limit" yaml:"memory_limit"` // bytes
	BatchSize         int                  `json:"batch_size" yaml:"batch_size"`
	ComputeCapability string               `json:"compute_capability" yaml:"compute_capability"`
	CUDAConfig        CUDAConfig           `json:"cuda_config" yaml:"cuda_config"`
	OpenCLConfig      OpenCLConfig         `json:"opencl_config" yaml:"opencl_config"`
	TensorRTConfig    TensorRTConfig       `json:"tensorrt_config" yaml:"tensorrt_config"`
	MultiGPU          MultiGPUConfig       `json:"multi_gpu" yaml:"multi_gpu"`
	MemoryManagement  GPUMemoryManagement  `json:"memory_management" yaml:"memory_management"`
	PerformanceTuning GPUPerformanceTuning `json:"performance_tuning" yaml:"performance_tuning"`
}

// CUDAConfig CUDA配置
type CUDAConfig struct {
	Enable            bool   `json:"enable" yaml:"enable"`
	Version           string `json:"version" yaml:"version"`
	Streams           int    `json:"streams" yaml:"streams"`
	BlockSize         int    `json:"block_size" yaml:"block_size"`
	GridSize          int    `json:"grid_size" yaml:"grid_size"`
	SharedMemorySize  int    `json:"shared_memory_size" yaml:"shared_memory_size"`
	OptimizationLevel int    `json:"optimization_level" yaml:"optimization_level"`
}

// OpenCLConfig OpenCL配置
type OpenCLConfig struct {
	Enable          bool     `json:"enable" yaml:"enable"`
	PlatformID      int      `json:"platform_id" yaml:"platform_id"`
	DeviceType      string   `json:"device_type" yaml:"device_type"` // "GPU", "CPU", "ACCELERATOR"
	WorkGroupSize   int      `json:"work_group_size" yaml:"work_group_size"`
	LocalMemorySize int      `json:"local_memory_size" yaml:"local_memory_size"`
	CompilerOptions []string `json:"compiler_options" yaml:"compiler_options"`
}

// TensorRTConfig TensorRT配置
type TensorRTConfig struct {
	Enable            bool   `json:"enable" yaml:"enable"`
	Precision         string `json:"precision" yaml:"precision"` // "FP32", "FP16", "INT8"
	MaxBatchSize      int    `json:"max_batch_size" yaml:"max_batch_size"`
	MaxWorkspaceSize  int64  `json:"max_workspace_size" yaml:"max_workspace_size"`
	OptimizationLevel int    `json:"optimization_level" yaml:"optimization_level"`
	CalibrationData   string `json:"calibration_data" yaml:"calibration_data"`
}

// MultiGPUConfig 多GPU配置
type MultiGPUConfig struct {
	Enable          bool                     `json:"enable" yaml:"enable"`
	Strategy        string                   `json:"strategy" yaml:"strategy"` // "data_parallel", "model_parallel", "pipeline_parallel"
	Communication   GPUCommunicationConfig   `json:"communication" yaml:"communication"`
	LoadBalancing   GPULoadBalancingConfig   `json:"load_balancing" yaml:"load_balancing"`
	Synchronization GPUSynchronizationConfig `json:"synchronization" yaml:"synchronization"`
}

// GPUCommunicationConfig GPU通信配置
type GPUCommunicationConfig struct {
	Backend           string `json:"backend" yaml:"backend"`   // "NCCL", "MPI", "GLOO"
	Topology          string `json:"topology" yaml:"topology"` // "ring", "tree", "mesh"
	CompressionEnable bool   `json:"compression_enable" yaml:"compression_enable"`
	BandwidthLimit    int64  `json:"bandwidth_limit" yaml:"bandwidth_limit"` // bytes/sec
}

// GPULoadBalancingConfig GPU负载均衡配置
type GPULoadBalancingConfig struct {
	Strategy           string        `json:"strategy" yaml:"strategy"` // "round_robin", "load_aware", "memory_aware"
	MonitoringInterval time.Duration `json:"monitoring_interval" yaml:"monitoring_interval"`
	RebalanceThreshold float64       `json:"rebalance_threshold" yaml:"rebalance_threshold"`
	MigrationCost      float64       `json:"migration_cost" yaml:"migration_cost"`
}

// GPUSynchronizationConfig GPU同步配置
type GPUSynchronizationConfig struct {
	Mode             string        `json:"mode" yaml:"mode"` // "sync", "async", "bulk_sync"
	SyncInterval     time.Duration `json:"sync_interval" yaml:"sync_interval"`
	TimeoutThreshold time.Duration `json:"timeout_threshold" yaml:"timeout_threshold"`
	RetryCount       int           `json:"retry_count" yaml:"retry_count"`
}

// GPUMemoryManagement GPU内存管理
type GPUMemoryManagement struct {
	Pooling                bool    `json:"pooling" yaml:"pooling"`
	Preallocation          bool    `json:"preallocation" yaml:"preallocation"`
	FragmentationThreshold float64 `json:"fragmentation_threshold" yaml:"fragmentation_threshold"`
	GCThreshold            float64 `json:"gc_threshold" yaml:"gc_threshold"`
	UnifiedMemory          bool    `json:"unified_memory" yaml:"unified_memory"`
	MemoryMapping          bool    `json:"memory_mapping" yaml:"memory_mapping"`
}

// GPUPerformanceTuning GPU性能调优
type GPUPerformanceTuning struct {
	AutoTuning            bool                 `json:"auto_tuning" yaml:"auto_tuning"`
	KernelFusion          bool                 `json:"kernel_fusion" yaml:"kernel_fusion"`
	MemoryCoalescing      bool                 `json:"memory_coalescing" yaml:"memory_coalescing"`
	OccupancyOptimization bool                 `json:"occupancy_optimization" yaml:"occupancy_optimization"`
	PowerManagement       GPUPowerManagement   `json:"power_management" yaml:"power_management"`
	ThermalManagement     GPUThermalManagement `json:"thermal_management" yaml:"thermal_management"`
}

// GPUPowerManagement GPU功耗管理
type GPUPowerManagement struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	PowerLimit       int           `json:"power_limit" yaml:"power_limit"` // watts
	FrequencyScaling bool          `json:"frequency_scaling" yaml:"frequency_scaling"`
	VoltageScaling   bool          `json:"voltage_scaling" yaml:"voltage_scaling"`
	IdleTimeout      time.Duration `json:"idle_timeout" yaml:"idle_timeout"`
}

// GPUThermalManagement GPU热管理
type GPUThermalManagement struct {
	Enable           bool   `json:"enable" yaml:"enable"`
	TemperatureLimit int    `json:"temperature_limit" yaml:"temperature_limit"` // celsius
	FanControl       bool   `json:"fan_control" yaml:"fan_control"`
	Throttling       bool   `json:"throttling" yaml:"throttling"`
	CoolingStrategy  string `json:"cooling_strategy" yaml:"cooling_strategy"` // "passive", "active", "hybrid"
}

// FPGAConfig FPGA配置
type FPGAConfig struct {
	Enable          bool                      `json:"enable" yaml:"enable"`
	DeviceIDs       []int                     `json:"device_ids" yaml:"device_ids"`
	Bitstream       string                    `json:"bitstream" yaml:"bitstream"`
	ClockFrequency  int                       `json:"clock_frequency" yaml:"clock_frequency"`   // MHz
	MemoryBandwidth int64                     `json:"memory_bandwidth" yaml:"memory_bandwidth"` // bytes/sec
	PipelineDepth   int                       `json:"pipeline_depth" yaml:"pipeline_depth"`
	Parallelism     FPGAParallelismConfig     `json:"parallelism" yaml:"parallelism"`
	Optimization    FPGAOptimizationConfig    `json:"optimization" yaml:"optimization"`
	Reconfiguration FPGAReconfigurationConfig `json:"reconfiguration" yaml:"reconfiguration"`
}

// FPGAParallelismConfig FPGA并行配置
type FPGAParallelismConfig struct {
	ComputeUnits   int `json:"compute_units" yaml:"compute_units"`
	VectorWidth    int `json:"vector_width" yaml:"vector_width"`
	UnrollFactor   int `json:"unroll_factor" yaml:"unroll_factor"`
	PipelineStages int `json:"pipeline_stages" yaml:"pipeline_stages"`
}

// FPGAOptimizationConfig FPGA优化配置
type FPGAOptimizationConfig struct {
	ResourceSharing    bool `json:"resource_sharing" yaml:"resource_sharing"`
	MemoryOptimization bool `json:"memory_optimization" yaml:"memory_optimization"`
	TimingOptimization bool `json:"timing_optimization" yaml:"timing_optimization"`
	PowerOptimization  bool `json:"power_optimization" yaml:"power_optimization"`
	AreaOptimization   bool `json:"area_optimization" yaml:"area_optimization"`
}

// FPGAReconfigurationConfig FPGA重配置
type FPGAReconfigurationConfig struct {
	Enable                 bool          `json:"enable" yaml:"enable"`
	PartialReconfiguration bool          `json:"partial_reconfiguration" yaml:"partial_reconfiguration"`
	ReconfigurationTime    time.Duration `json:"reconfiguration_time" yaml:"reconfiguration_time"`
	BitstreamCache         bool          `json:"bitstream_cache" yaml:"bitstream_cache"`
	HotSwap                bool          `json:"hot_swap" yaml:"hot_swap"`
}

// PMemConfig 持久内存配置
type PMemConfig struct {
	Enable       bool                   `json:"enable" yaml:"enable"`
	DevicePaths  []string               `json:"device_paths" yaml:"device_paths"`
	Mode         string                 `json:"mode" yaml:"mode"` // "app_direct", "memory_mode", "mixed"
	Namespaces   []PMemNamespace        `json:"namespaces" yaml:"namespaces"`
	Interleaving PMemInterleavingConfig `json:"interleaving" yaml:"interleaving"`
	Persistence  PMemPersistenceConfig  `json:"persistence" yaml:"persistence"`
	Performance  PMemPerformanceConfig  `json:"performance" yaml:"performance"`
	Reliability  PMemReliabilityConfig  `json:"reliability" yaml:"reliability"`
}

// PMemNamespace 持久内存命名空间
type PMemNamespace struct {
	Name       string `json:"name" yaml:"name"`
	Size       int64  `json:"size" yaml:"size"` // bytes
	Mode       string `json:"mode" yaml:"mode"` // "fsdax", "devdax", "sector"
	Alignment  int    `json:"alignment" yaml:"alignment"`
	SectorSize int    `json:"sector_size" yaml:"sector_size"`
}

// PMemInterleavingConfig 持久内存交错配置
type PMemInterleavingConfig struct {
	Enable      bool `json:"enable" yaml:"enable"`
	Ways        int  `json:"ways" yaml:"ways"`
	Granularity int  `json:"granularity" yaml:"granularity"` // bytes
	Alignment   int  `json:"alignment" yaml:"alignment"`
}

// PMemPersistenceConfig 持久内存持久化配置
type PMemPersistenceConfig struct {
	FlushStrategy      string        `json:"flush_strategy" yaml:"flush_strategy"` // "sync", "async", "lazy"
	FlushInterval      time.Duration `json:"flush_interval" yaml:"flush_interval"`
	Checkpointing      bool          `json:"checkpointing" yaml:"checkpointing"`
	CheckpointInterval time.Duration `json:"checkpoint_interval" yaml:"checkpoint_interval"`
	RecoveryMode       string        `json:"recovery_mode" yaml:"recovery_mode"` // "fast", "safe", "auto"
}

// PMemPerformanceConfig 持久内存性能配置
type PMemPerformanceConfig struct {
	ReadAhead        bool `json:"read_ahead" yaml:"read_ahead"`
	WriteBehind      bool `json:"write_behind" yaml:"write_behind"`
	BatchSize        int  `json:"batch_size" yaml:"batch_size"`
	QueueDepth       int  `json:"queue_depth" yaml:"queue_depth"`
	NUMAOptimization bool `json:"numa_optimization" yaml:"numa_optimization"`
}

// PMemReliabilityConfig 持久内存可靠性配置
type PMemReliabilityConfig struct {
	ECC                bool          `json:"ecc" yaml:"ecc"`
	Scrubbing          bool          `json:"scrubbing" yaml:"scrubbing"`
	ScrubbingInterval  time.Duration `json:"scrubbing_interval" yaml:"scrubbing_interval"`
	BadBlockManagement bool          `json:"bad_block_management" yaml:"bad_block_management"`
	WearLeveling       bool          `json:"wear_leveling" yaml:"wear_leveling"`
}

// RDMANetworkConfig RDMA网络配置
type RDMANetworkConfig struct {
	Enable             bool                         `json:"enable" yaml:"enable"`
	Devices            []RDMADevice                 `json:"devices" yaml:"devices"`
	Protocol           string                       `json:"protocol" yaml:"protocol"`             // "IB", "RoCE", "iWARP"
	TransportType      string                       `json:"transport_type" yaml:"transport_type"` // "RC", "UC", "UD"
	QueuePairs         RDMAQueuePairConfig          `json:"queue_pairs" yaml:"queue_pairs"`
	MemoryRegistration RDMAMemoryRegistrationConfig `json:"memory_registration" yaml:"memory_registration"`
	CongestionControl  RDMACongestionControlConfig  `json:"congestion_control" yaml:"congestion_control"`
	PerformanceTuning  RDMAPerformanceTuningConfig  `json:"performance_tuning" yaml:"performance_tuning"`
}

// RDMADevice RDMA设备
type RDMADevice struct {
	Name      string `json:"name" yaml:"name"`
	Port      int    `json:"port" yaml:"port"`
	GID       string `json:"gid" yaml:"gid"`
	LID       int    `json:"lid" yaml:"lid"`
	MTU       int    `json:"mtu" yaml:"mtu"`
	LinkLayer string `json:"link_layer" yaml:"link_layer"`
}

// RDMAQueuePairConfig RDMA队列对配置
type RDMAQueuePairConfig struct {
	MaxQPs           int `json:"max_qps" yaml:"max_qps"`
	SendQueueSize    int `json:"send_queue_size" yaml:"send_queue_size"`
	ReceiveQueueSize int `json:"receive_queue_size" yaml:"receive_queue_size"`
	MaxInlineData    int `json:"max_inline_data" yaml:"max_inline_data"`
	MaxSGE           int `json:"max_sge" yaml:"max_sge"`
	RetryCount       int `json:"retry_count" yaml:"retry_count"`
	RNRRetryCount    int `json:"rnr_retry_count" yaml:"rnr_retry_count"`
	Timeout          int `json:"timeout" yaml:"timeout"`
}

// RDMAMemoryRegistrationConfig RDMA内存注册配置
type RDMAMemoryRegistrationConfig struct {
	Strategy          string `json:"strategy" yaml:"strategy"`     // "eager", "lazy", "on_demand"
	CacheSize         int64  `json:"cache_size" yaml:"cache_size"` // bytes
	HugePagesEnable   bool   `json:"huge_pages_enable" yaml:"huge_pages_enable"`
	MemoryPinning     bool   `json:"memory_pinning" yaml:"memory_pinning"`
	RegistrationCache bool   `json:"registration_cache" yaml:"registration_cache"`
}

// RDMACongestionControlConfig RDMA拥塞控制配置
type RDMACongestionControlConfig struct {
	Algorithm    string  `json:"algorithm" yaml:"algorithm"` // "DCQCN", "TIMELY", "HPCC"
	ECNThreshold float64 `json:"ecn_threshold" yaml:"ecn_threshold"`
	RateIncrease float64 `json:"rate_increase" yaml:"rate_increase"`
	RateDecrease float64 `json:"rate_decrease" yaml:"rate_decrease"`
	MinRate      int64   `json:"min_rate" yaml:"min_rate"` // bytes/sec
	MaxRate      int64   `json:"max_rate" yaml:"max_rate"` // bytes/sec
}

// RDMAPerformanceTuningConfig RDMA性能调优配置
type RDMAPerformanceTuningConfig struct {
	BatchSize           int   `json:"batch_size" yaml:"batch_size"`
	PollingMode         bool  `json:"polling_mode" yaml:"polling_mode"`
	InterruptCoalescing bool  `json:"interrupt_coalescing" yaml:"interrupt_coalescing"`
	CPUAffinity         []int `json:"cpu_affinity" yaml:"cpu_affinity"`
	NUMAOptimization    bool  `json:"numa_optimization" yaml:"numa_optimization"`
	ZeroCopy            bool  `json:"zero_copy" yaml:"zero_copy"`
}

// CPUOptimizationConfig CPU优化配置
type CPUOptimizationConfig struct {
	Enable            bool                     `json:"enable" yaml:"enable"`
	Vectorization     VectorizationConfig      `json:"vectorization" yaml:"vectorization"`
	CacheOptimization CacheOptimizationConfig  `json:"cache_optimization" yaml:"cache_optimization"`
	BranchPrediction  BranchPredictionConfig   `json:"branch_prediction" yaml:"branch_prediction"`
	Prefetching       CPUPrefetchingConfig     `json:"prefetching" yaml:"prefetching"`
	Parallelization   CPUParallelizationConfig `json:"parallelization" yaml:"parallelization"`
	PowerManagement   CPUPowerManagementConfig `json:"power_management" yaml:"power_management"`
}

// VectorizationConfig 向量化配置
type VectorizationConfig struct {
	Enable            bool     `json:"enable" yaml:"enable"`
	InstructionSets   []string `json:"instruction_sets" yaml:"instruction_sets"` // "SSE", "AVX", "AVX2", "AVX512", "NEON", "SVE"
	VectorWidth       int      `json:"vector_width" yaml:"vector_width"`
	AutoVectorization bool     `json:"auto_vectorization" yaml:"auto_vectorization"`
	UnrollFactor      int      `json:"unroll_factor" yaml:"unroll_factor"`
}

// CacheOptimizationConfig 缓存优化配置
type CacheOptimizationConfig struct {
	Enable         bool                   `json:"enable" yaml:"enable"`
	L1Optimization CacheLevelOptimization `json:"l1_optimization" yaml:"l1_optimization"`
	L2Optimization CacheLevelOptimization `json:"l2_optimization" yaml:"l2_optimization"`
	L3Optimization CacheLevelOptimization `json:"l3_optimization" yaml:"l3_optimization"`
	CacheBlocking  CacheBlockingConfig    `json:"cache_blocking" yaml:"cache_blocking"`
	DataLayout     DataLayoutOptimization `json:"data_layout" yaml:"data_layout"`
}

// CacheLevelOptimization 缓存级别优化
type CacheLevelOptimization struct {
	BlockSize         int    `json:"block_size" yaml:"block_size"`
	Associativity     int    `json:"associativity" yaml:"associativity"`
	ReplacementPolicy string `json:"replacement_policy" yaml:"replacement_policy"` // "LRU", "LFU", "RANDOM"
	PrefetchDistance  int    `json:"prefetch_distance" yaml:"prefetch_distance"`
}

// CacheBlockingConfig 缓存分块配置
type CacheBlockingConfig struct {
	Enable         bool   `json:"enable" yaml:"enable"`
	BlockSizeL1    int    `json:"block_size_l1" yaml:"block_size_l1"`
	BlockSizeL2    int    `json:"block_size_l2" yaml:"block_size_l2"`
	BlockSizeL3    int    `json:"block_size_l3" yaml:"block_size_l3"`
	TilingStrategy string `json:"tiling_strategy" yaml:"tiling_strategy"` // "square", "rectangular", "adaptive"
}

// DataLayoutOptimization 数据布局优化
type DataLayoutOptimization struct {
	Enable         bool `json:"enable" yaml:"enable"`
	Alignment      int  `json:"alignment" yaml:"alignment"`
	Padding        bool `json:"padding" yaml:"padding"`
	StructPacking  bool `json:"struct_packing" yaml:"struct_packing"`
	ArrayOfStructs bool `json:"array_of_structs" yaml:"array_of_structs"`
	StructOfArrays bool `json:"struct_of_arrays" yaml:"struct_of_arrays"`
}

// BranchPredictionConfig 分支预测配置
type BranchPredictionConfig struct {
	Enable            bool   `json:"enable" yaml:"enable"`
	PredictorType     string `json:"predictor_type" yaml:"predictor_type"` // "static", "dynamic", "hybrid"
	BranchHints       bool   `json:"branch_hints" yaml:"branch_hints"`
	ProfileGuided     bool   `json:"profile_guided" yaml:"profile_guided"`
	BranchElimination bool   `json:"branch_elimination" yaml:"branch_elimination"`
}

// CPUPrefetchingConfig CPU预取配置
type CPUPrefetchingConfig struct {
	Enable           bool   `json:"enable" yaml:"enable"`
	HardwarePrefetch bool   `json:"hardware_prefetch" yaml:"hardware_prefetch"`
	SoftwarePrefetch bool   `json:"software_prefetch" yaml:"software_prefetch"`
	PrefetchDistance int    `json:"prefetch_distance" yaml:"prefetch_distance"`
	PrefetchStrategy string `json:"prefetch_strategy" yaml:"prefetch_strategy"` // "sequential", "strided", "adaptive"
}

// CPUParallelizationConfig CPU并行化配置
type CPUParallelizationConfig struct {
	Enable          bool                     `json:"enable" yaml:"enable"`
	Threads         int                      `json:"threads" yaml:"threads"`
	Affinity        CPUAffinityConfig        `json:"affinity" yaml:"affinity"`
	Scheduling      CPUSchedulingConfig      `json:"scheduling" yaml:"scheduling"`
	Synchronization CPUSynchronizationConfig `json:"synchronization" yaml:"synchronization"`
	LoadBalancing   CPULoadBalancingConfig   `json:"load_balancing" yaml:"load_balancing"`
}

// CPUAffinityConfig CPU亲和性配置
type CPUAffinityConfig struct {
	Enable         bool  `json:"enable" yaml:"enable"`
	CPUSet         []int `json:"cpu_set" yaml:"cpu_set"`
	NUMANodes      []int `json:"numa_nodes" yaml:"numa_nodes"`
	IsolatedCPUs   []int `json:"isolated_cpus" yaml:"isolated_cpus"`
	HyperThreading bool  `json:"hyper_threading" yaml:"hyper_threading"`
}

// CPUSchedulingConfig CPU调度配置
type CPUSchedulingConfig struct {
	Policy     string        `json:"policy" yaml:"policy"` // "SCHED_NORMAL", "SCHED_FIFO", "SCHED_RR"
	Priority   int           `json:"priority" yaml:"priority"`
	NiceValue  int           `json:"nice_value" yaml:"nice_value"`
	TimeSlice  time.Duration `json:"time_slice" yaml:"time_slice"`
	Preemption bool          `json:"preemption" yaml:"preemption"`
}

// CPUSynchronizationConfig CPU同步配置
type CPUSynchronizationConfig struct {
	Primitive          string `json:"primitive" yaml:"primitive"`               // "mutex", "spinlock", "rwlock", "atomic"
	BackoffStrategy    string `json:"backoff_strategy" yaml:"backoff_strategy"` // "exponential", "linear", "adaptive"
	LockFreeAlgorithms bool   `json:"lock_free_algorithms" yaml:"lock_free_algorithms"`
	WaitFreeAlgorithms bool   `json:"wait_free_algorithms" yaml:"wait_free_algorithms"`
}

// CPULoadBalancingConfig CPU负载均衡配置
type CPULoadBalancingConfig struct {
	Strategy        string        `json:"strategy" yaml:"strategy"` // "work_stealing", "work_sharing", "static"
	MigrationCost   float64       `json:"migration_cost" yaml:"migration_cost"`
	BalanceInterval time.Duration `json:"balance_interval" yaml:"balance_interval"`
	LoadThreshold   float64       `json:"load_threshold" yaml:"load_threshold"`
}

// CPUPowerManagementConfig CPU功耗管理配置
type CPUPowerManagementConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	Governor         string        `json:"governor" yaml:"governor"` // "performance", "powersave", "ondemand", "conservative"
	FrequencyScaling bool          `json:"frequency_scaling" yaml:"frequency_scaling"`
	VoltageScaling   bool          `json:"voltage_scaling" yaml:"voltage_scaling"`
	CStates          bool          `json:"c_states" yaml:"c_states"`
	PStates          bool          `json:"p_states" yaml:"p_states"`
	TurboBoost       bool          `json:"turbo_boost" yaml:"turbo_boost"`
	IdleTimeout      time.Duration `json:"idle_timeout" yaml:"idle_timeout"`
}

// NPUConfig NPU(神经处理单元)配置
type NPUConfig struct {
	Enable          bool                  `json:"enable" yaml:"enable"`
	DeviceIDs       []int                 `json:"device_ids" yaml:"device_ids"`
	ModelFormat     string                `json:"model_format" yaml:"model_format"` // "ONNX", "TensorFlow", "PyTorch", "Caffe"
	Precision       string                `json:"precision" yaml:"precision"`       // "FP32", "FP16", "INT8", "INT4"
	BatchSize       int                   `json:"batch_size" yaml:"batch_size"`
	MemoryLimit     int64                 `json:"memory_limit" yaml:"memory_limit"` // bytes
	CompilerOptions NPUCompilerOptions    `json:"compiler_options" yaml:"compiler_options"`
	RuntimeOptions  NPURuntimeOptions     `json:"runtime_options" yaml:"runtime_options"`
	Optimization    NPUOptimizationConfig `json:"optimization" yaml:"optimization"`
}

// NPUCompilerOptions NPU编译器选项
type NPUCompilerOptions struct {
	OptimizationLevel int      `json:"optimization_level" yaml:"optimization_level"`
	TargetDevice      string   `json:"target_device" yaml:"target_device"`
	CompilerFlags     []string `json:"compiler_flags" yaml:"compiler_flags"`
	DebugMode         bool     `json:"debug_mode" yaml:"debug_mode"`
	ProfilingEnable   bool     `json:"profiling_enable" yaml:"profiling_enable"`
}

// NPURuntimeOptions NPU运行时选项
type NPURuntimeOptions struct {
	Threads          int    `json:"threads" yaml:"threads"`
	MemoryStrategy   string `json:"memory_strategy" yaml:"memory_strategy"`     // "static", "dynamic", "hybrid"
	SchedulingPolicy string `json:"scheduling_policy" yaml:"scheduling_policy"` // "fifo", "priority", "fair"
	ErrorHandling    string `json:"error_handling" yaml:"error_handling"`       // "strict", "lenient", "ignore"
}

// NPUOptimizationConfig NPU优化配置
type NPUOptimizationConfig struct {
	GraphOptimization     bool                           `json:"graph_optimization" yaml:"graph_optimization"`
	KernelFusion          bool                           `json:"kernel_fusion" yaml:"kernel_fusion"`
	MemoryOptimization    bool                           `json:"memory_optimization" yaml:"memory_optimization"`
	Quantization          NPUQuantizationConfig          `json:"quantization" yaml:"quantization"`
	Pruning               NPUPruningConfig               `json:"pruning" yaml:"pruning"`
	KnowledgeDistillation NPUKnowledgeDistillationConfig `json:"knowledge_distillation" yaml:"knowledge_distillation"`
}

// NPUQuantizationConfig NPU量化配置
type NPUQuantizationConfig struct {
	Enable            bool    `json:"enable" yaml:"enable"`
	Method            string  `json:"method" yaml:"method"` // "post_training", "quantization_aware", "dynamic"
	CalibrationData   string  `json:"calibration_data" yaml:"calibration_data"`
	AccuracyThreshold float64 `json:"accuracy_threshold" yaml:"accuracy_threshold"`
	MixedPrecision    bool    `json:"mixed_precision" yaml:"mixed_precision"`
}

// NPUPruningConfig NPU剪枝配置
type NPUPruningConfig struct {
	Enable         bool    `json:"enable" yaml:"enable"`
	Method         string  `json:"method" yaml:"method"` // "magnitude", "structured", "unstructured"
	SparsityRatio  float64 `json:"sparsity_ratio" yaml:"sparsity_ratio"`
	GradualPruning bool    `json:"gradual_pruning" yaml:"gradual_pruning"`
	Finetuning     bool    `json:"finetuning" yaml:"finetuning"`
}

// NPUKnowledgeDistillationConfig NPU知识蒸馏配置
type NPUKnowledgeDistillationConfig struct {
	Enable           bool    `json:"enable" yaml:"enable"`
	TeacherModel     string  `json:"teacher_model" yaml:"teacher_model"`
	Temperature      float64 `json:"temperature" yaml:"temperature"`
	Alpha            float64 `json:"alpha" yaml:"alpha"`
	DistillationLoss string  `json:"distillation_loss" yaml:"distillation_loss"` // "kl_div", "mse", "cosine"
}

// GetDefaultHardwareConfig 获取默认硬件配置
func GetDefaultHardwareConfig() *HardwareAccelerationConfig {
	return &HardwareAccelerationConfig{
		GPU: GPUConfig{
			Enable:      false,
			DeviceIDs:   []int{0},
			MemoryLimit: 8 * 1024 * 1024 * 1024, // 8GB
			BatchSize:   1000,
			CUDAConfig: CUDAConfig{
				Enable:            true,
				Streams:           4,
				BlockSize:         256,
				GridSize:          65535,
				SharedMemorySize:  48 * 1024, // 48KB
				OptimizationLevel: 3,
			},
			MultiGPU: MultiGPUConfig{
				Enable:   false,
				Strategy: "data_parallel",
				Communication: GPUCommunicationConfig{
					Backend:           "NCCL",
					Topology:          "ring",
					CompressionEnable: false,
					BandwidthLimit:    10 * 1024 * 1024 * 1024, // 10GB/s
				},
			},
			MemoryManagement: GPUMemoryManagement{
				Pooling:                true,
				Preallocation:          true,
				FragmentationThreshold: 0.8,
				GCThreshold:            0.9,
				UnifiedMemory:          false,
				MemoryMapping:          true,
			},
			PerformanceTuning: GPUPerformanceTuning{
				AutoTuning:            true,
				KernelFusion:          true,
				MemoryCoalescing:      true,
				OccupancyOptimization: true,
				PowerManagement: GPUPowerManagement{
					Enable:           false,
					PowerLimit:       300, // watts
					FrequencyScaling: false,
					VoltageScaling:   false,
					IdleTimeout:      30 * time.Second,
				},
				ThermalManagement: GPUThermalManagement{
					Enable:           true,
					TemperatureLimit: 85, // celsius
					FanControl:       true,
					Throttling:       true,
					CoolingStrategy:  "active",
				},
			},
		},
		FPGA: FPGAConfig{
			Enable:          false,
			DeviceIDs:       []int{0},
			ClockFrequency:  200,                      // MHz
			MemoryBandwidth: 100 * 1024 * 1024 * 1024, // 100GB/s
			PipelineDepth:   8,
			Parallelism: FPGAParallelismConfig{
				ComputeUnits:   16,
				VectorWidth:    512,
				UnrollFactor:   4,
				PipelineStages: 8,
			},
			Optimization: FPGAOptimizationConfig{
				ResourceSharing:    true,
				MemoryOptimization: true,
				TimingOptimization: true,
				PowerOptimization:  false,
				AreaOptimization:   false,
			},
		},
		PMem: PMemConfig{
			Enable: false,
			Mode:   "app_direct",
			Interleaving: PMemInterleavingConfig{
				Enable:      true,
				Ways:        2,
				Granularity: 4096, // 4KB
				Alignment:   4096,
			},
			Persistence: PMemPersistenceConfig{
				FlushStrategy:      "async",
				FlushInterval:      100 * time.Millisecond,
				Checkpointing:      true,
				CheckpointInterval: 5 * time.Minute,
				RecoveryMode:       "auto",
			},
			Performance: PMemPerformanceConfig{
				ReadAhead:        true,
				WriteBehind:      true,
				BatchSize:        64,
				QueueDepth:       32,
				NUMAOptimization: true,
			},
		},
		RDMA: RDMANetworkConfig{
			Enable:        false,
			Protocol:      "RoCE",
			TransportType: "RC",
			QueuePairs: RDMAQueuePairConfig{
				MaxQPs:           1000,
				SendQueueSize:    1024,
				ReceiveQueueSize: 1024,
				MaxInlineData:    64,
				MaxSGE:           16,
				RetryCount:       7,
				RNRRetryCount:    7,
				Timeout:          14,
			},
			MemoryRegistration: RDMAMemoryRegistrationConfig{
				Strategy:          "lazy",
				CacheSize:         1024 * 1024 * 1024, // 1GB
				HugePagesEnable:   true,
				MemoryPinning:     true,
				RegistrationCache: true,
			},
			PerformanceTuning: RDMAPerformanceTuningConfig{
				BatchSize:           32,
				PollingMode:         true,
				InterruptCoalescing: true,
				NUMAOptimization:    true,
				ZeroCopy:            true,
			},
		},
		CPU: CPUOptimizationConfig{
			Enable: true,
			Vectorization: VectorizationConfig{
				Enable:            true,
				InstructionSets:   []string{"AVX2", "SSE4.2"},
				VectorWidth:       256,
				AutoVectorization: true,
				UnrollFactor:      4,
			},
			CacheOptimization: CacheOptimizationConfig{
				Enable: true,
				L1Optimization: CacheLevelOptimization{
					BlockSize:         64,
					Associativity:     8,
					ReplacementPolicy: "LRU",
					PrefetchDistance:  1,
				},
				L2Optimization: CacheLevelOptimization{
					BlockSize:         64,
					Associativity:     8,
					ReplacementPolicy: "LRU",
					PrefetchDistance:  2,
				},
				L3Optimization: CacheLevelOptimization{
					BlockSize:         64,
					Associativity:     16,
					ReplacementPolicy: "LRU",
					PrefetchDistance:  4,
				},
				CacheBlocking: CacheBlockingConfig{
					Enable:         true,
					BlockSizeL1:    32 * 1024,       // 32KB
					BlockSizeL2:    256 * 1024,      // 256KB
					BlockSizeL3:    8 * 1024 * 1024, // 8MB
					TilingStrategy: "adaptive",
				},
				DataLayout: DataLayoutOptimization{
					Enable:         true,
					Alignment:      64,
					Padding:        true,
					StructPacking:  true,
					ArrayOfStructs: false,
					StructOfArrays: true,
				},
			},
			Parallelization: CPUParallelizationConfig{
				Enable:  true,
				Threads: runtime.NumCPU(),
				Affinity: CPUAffinityConfig{
					Enable:         true,
					HyperThreading: false,
				},
				Scheduling: CPUSchedulingConfig{
					Policy:     "SCHED_NORMAL",
					Priority:   0,
					NiceValue:  0,
					TimeSlice:  10 * time.Millisecond,
					Preemption: true,
				},
				Synchronization: CPUSynchronizationConfig{
					Primitive:          "atomic",
					BackoffStrategy:    "exponential",
					LockFreeAlgorithms: true,
					WaitFreeAlgorithms: false,
				},
				LoadBalancing: CPULoadBalancingConfig{
					Strategy:        "work_stealing",
					MigrationCost:   0.1,
					BalanceInterval: 100 * time.Millisecond,
					LoadThreshold:   0.8,
				},
			},
		},
		NPU: NPUConfig{
			Enable:      false,
			DeviceIDs:   []int{0},
			ModelFormat: "ONNX",
			Precision:   "FP16",
			BatchSize:   1000,
			MemoryLimit: 4 * 1024 * 1024 * 1024, // 4GB
			CompilerOptions: NPUCompilerOptions{
				OptimizationLevel: 3,
				DebugMode:         false,
				ProfilingEnable:   false,
			},
			RuntimeOptions: NPURuntimeOptions{
				Threads:          4,
				MemoryStrategy:   "dynamic",
				SchedulingPolicy: "fair",
				ErrorHandling:    "strict",
			},
			Optimization: NPUOptimizationConfig{
				GraphOptimization:  true,
				KernelFusion:       true,
				MemoryOptimization: true,
				Quantization: NPUQuantizationConfig{
					Enable:            false,
					Method:            "post_training",
					AccuracyThreshold: 0.95,
					MixedPrecision:    true,
				},
			},
		},
	}
}
