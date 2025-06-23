package vector

import (
	"runtime"
	"time"
)

// PerformanceOptimizationConfig 性能优化配置
type PerformanceOptimizationConfig struct {
	// 查询加速配置
	QueryAcceleration QueryAccelerationConfig `json:"query_acceleration" yaml:"query_acceleration"`

	// 并发控制配置
	ConcurrencyControl ConcurrencyControlConfig `json:"concurrency_control" yaml:"concurrency_control"`

	// 内存管理配置
	MemoryManagement MemoryManagementConfig `json:"memory_management" yaml:"memory_management"`

	// 批处理配置
	BatchProcessing BatchProcessingConfig `json:"batch_processing" yaml:"batch_processing"`

	// 预取配置
	Prefetching PrefetchingConfig `json:"prefetching" yaml:"prefetching"`
}

// QueryAccelerationConfig 查询加速配置
type QueryAccelerationConfig struct {
	Enable            bool                    `json:"enable" yaml:"enable"`
	Preprocessing     PreprocessingConfig     `json:"preprocessing" yaml:"preprocessing"`
	MultiStageSearch  MultiStageSearchConfig  `json:"multi_stage_search" yaml:"multi_stage_search"`
	ParallelSearch    ParallelSearchConfig    `json:"parallel_search" yaml:"parallel_search"`
	EarlyTermination  EarlyTerminationConfig  `json:"early_termination" yaml:"early_termination"`
	QueryOptimization QueryOptimizationConfig `json:"query_optimization" yaml:"query_optimization"`
}

// PreprocessingConfig 预处理配置
type PreprocessingConfig struct {
	Normalization      bool                     `json:"normalization" yaml:"normalization"`
	DimensionReduction DimensionReductionConfig `json:"dimension_reduction" yaml:"dimension_reduction"`
	Quantization       QuantizationConfig       `json:"quantization" yaml:"quantization"`
	Filtering          FilteringConfig          `json:"filtering" yaml:"filtering"`
}

// DimensionReductionConfig 降维配置
type DimensionReductionConfig struct {
	Enable        bool    `json:"enable" yaml:"enable"`
	Method        string  `json:"method" yaml:"method"` // "pca", "autoencoder", "umap", "tsne"
	TargetDim     int     `json:"target_dim" yaml:"target_dim"`
	VarianceRatio float64 `json:"variance_ratio" yaml:"variance_ratio"` // PCA保留的方差比例
	TrainingSize  int     `json:"training_size" yaml:"training_size"`   // 训练样本数量
}

// QuantizationConfig 量化配置
type QuantizationConfig struct {
	Enable       bool   `json:"enable" yaml:"enable"`
	Method       string `json:"method" yaml:"method"` // "scalar", "vector", "product"
	Bits         int    `json:"bits" yaml:"bits"`     // 量化位数
	CodebookSize int    `json:"codebook_size" yaml:"codebook_size"`
}

// FilteringConfig 过滤配置
type FilteringConfig struct {
	Enable        bool                `json:"enable" yaml:"enable"`
	PreFiltering  PreFilteringConfig  `json:"pre_filtering" yaml:"pre_filtering"`
	PostFiltering PostFilteringConfig `json:"post_filtering" yaml:"post_filtering"`
	BloomFilter   BloomFilterConfig   `json:"bloom_filter" yaml:"bloom_filter"`
}

// PreFilteringConfig 预过滤配置
type PreFilteringConfig struct {
	Enable      bool      `json:"enable" yaml:"enable"`
	FilterRatio float64   `json:"filter_ratio" yaml:"filter_ratio"` // 过滤比例
	Criteria    []string  `json:"criteria" yaml:"criteria"`         // 过滤条件
	Thresholds  []float64 `json:"thresholds" yaml:"thresholds"`     // 阈值
}

// PostFilteringConfig 后过滤配置
type PostFilteringConfig struct {
	Enable              bool    `json:"enable" yaml:"enable"`
	SimilarityThreshold float64 `json:"similarity_threshold" yaml:"similarity_threshold"`
	DiversityFilter     bool    `json:"diversity_filter" yaml:"diversity_filter"`
	DiversityRatio      float64 `json:"diversity_ratio" yaml:"diversity_ratio"`
}

// BloomFilterConfig 布隆过滤器配置
type BloomFilterConfig struct {
	Enable            bool    `json:"enable" yaml:"enable"`
	ExpectedItems     int     `json:"expected_items" yaml:"expected_items"`
	FalsePositiveRate float64 `json:"false_positive_rate" yaml:"false_positive_rate"`
	HashFunctions     int     `json:"hash_functions" yaml:"hash_functions"`
}

// MultiStageSearchConfig 多阶段搜索配置
type MultiStageSearchConfig struct {
	Enable           bool                  `json:"enable" yaml:"enable"`
	Stages           []SearchStageConfig   `json:"stages" yaml:"stages"`
	CoarseCandidates int                   `json:"coarse_candidates" yaml:"coarse_candidates"`
	RefinementRatio  float64               `json:"refinement_ratio" yaml:"refinement_ratio"`
	AdaptiveStaging  AdaptiveStagingConfig `json:"adaptive_staging" yaml:"adaptive_staging"`
}

// SearchStageConfig 搜索阶段配置
type SearchStageConfig struct {
	Name          string        `json:"name" yaml:"name"`
	Method        string        `json:"method" yaml:"method"` // "coarse", "fine", "rerank"
	Candidates    int           `json:"candidates" yaml:"candidates"`
	Accuracy      float64       `json:"accuracy" yaml:"accuracy"`
	LatencyBudget time.Duration `json:"latency_budget" yaml:"latency_budget"`
}

// AdaptiveStagingConfig 自适应分阶段配置
type AdaptiveStagingConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	LatencyThreshold   time.Duration `json:"latency_threshold" yaml:"latency_threshold"`
	AccuracyThreshold  float64       `json:"accuracy_threshold" yaml:"accuracy_threshold"`
	LoadThreshold      float64       `json:"load_threshold" yaml:"load_threshold"`
	AdjustmentInterval time.Duration `json:"adjustment_interval" yaml:"adjustment_interval"`
}

// ParallelSearchConfig 并行搜索配置
type ParallelSearchConfig struct {
	Enable            bool                    `json:"enable" yaml:"enable"`
	MaxWorkers        int                     `json:"max_workers" yaml:"max_workers"`
	WorkloadSplitting WorkloadSplittingConfig `json:"workload_splitting" yaml:"workload_splitting"`
	ResultMerging     ResultMergingConfig     `json:"result_merging" yaml:"result_merging"`
}

// WorkloadSplittingConfig 工作负载分割配置
type WorkloadSplittingConfig struct {
	Strategy      string  `json:"strategy" yaml:"strategy"` // "data_parallel", "query_parallel", "hybrid"
	ChunkSize     int     `json:"chunk_size" yaml:"chunk_size"`
	OverlapRatio  float64 `json:"overlap_ratio" yaml:"overlap_ratio"`
	LoadBalancing bool    `json:"load_balancing" yaml:"load_balancing"`
}

// ResultMergingConfig 结果合并配置
type ResultMergingConfig struct {
	Strategy            string  `json:"strategy" yaml:"strategy"` // "merge_sort", "heap_merge", "priority_queue"
	MergeThreshold      int     `json:"merge_threshold" yaml:"merge_threshold"`
	DuplicateRemoval    bool    `json:"duplicate_removal" yaml:"duplicate_removal"`
	SimilarityThreshold float64 `json:"similarity_threshold" yaml:"similarity_threshold"`
}

// EarlyTerminationConfig 早期终止配置
type EarlyTerminationConfig struct {
	Enable              bool                    `json:"enable" yaml:"enable"`
	ConfidenceThreshold float64                 `json:"confidence_threshold" yaml:"confidence_threshold"`
	TimeoutThreshold    time.Duration           `json:"timeout_threshold" yaml:"timeout_threshold"`
	QualityThreshold    float64                 `json:"quality_threshold" yaml:"quality_threshold"`
	ProgressiveSearch   ProgressiveSearchConfig `json:"progressive_search" yaml:"progressive_search"`
}

// ProgressiveSearchConfig 渐进式搜索配置
type ProgressiveSearchConfig struct {
	Enable               bool    `json:"enable" yaml:"enable"`
	InitialCandidates    int     `json:"initial_candidates" yaml:"initial_candidates"`
	ExpansionFactor      float64 `json:"expansion_factor" yaml:"expansion_factor"`
	MaxIterations        int     `json:"max_iterations" yaml:"max_iterations"`
	ConvergenceThreshold float64 `json:"convergence_threshold" yaml:"convergence_threshold"`
}

// QueryOptimizationConfig 查询优化配置
type QueryOptimizationConfig struct {
	Enable         bool                 `json:"enable" yaml:"enable"`
	QueryRewriting QueryRewritingConfig `json:"query_rewriting" yaml:"query_rewriting"`
	QueryCaching   QueryCachingConfig   `json:"query_caching" yaml:"query_caching"`
	QueryPlanning  QueryPlanningConfig  `json:"query_planning" yaml:"query_planning"`
	QueryProfiling QueryProfilingConfig `json:"query_profiling" yaml:"query_profiling"`
}

// QueryRewritingConfig 查询重写配置
type QueryRewritingConfig struct {
	Enable             bool `json:"enable" yaml:"enable"`
	ExpansionTerms     int  `json:"expansion_terms" yaml:"expansion_terms"`
	SynonymExpansion   bool `json:"synonym_expansion" yaml:"synonym_expansion"`
	QueryNormalization bool `json:"query_normalization" yaml:"query_normalization"`
	StopwordRemoval    bool `json:"stopword_removal" yaml:"stopword_removal"`
}

// QueryCachingConfig 查询缓存配置
type QueryCachingConfig struct {
	Enable              bool          `json:"enable" yaml:"enable"`
	CacheSize           int           `json:"cache_size" yaml:"cache_size"`
	TTL                 time.Duration `json:"ttl" yaml:"ttl"`
	SimilarityThreshold float64       `json:"similarity_threshold" yaml:"similarity_threshold"`
	CompressionRatio    float64       `json:"compression_ratio" yaml:"compression_ratio"`
}

// QueryPlanningConfig 查询规划配置
type QueryPlanningConfig struct {
	Enable           bool                   `json:"enable" yaml:"enable"`
	CostModel        CostModelConfig        `json:"cost_model" yaml:"cost_model"`
	PlanOptimization PlanOptimizationConfig `json:"plan_optimization" yaml:"plan_optimization"`
	AdaptivePlanning AdaptivePlanningConfig `json:"adaptive_planning" yaml:"adaptive_planning"`
}

// CostModelConfig 成本模型配置
type CostModelConfig struct {
	CPUWeight     float64 `json:"cpu_weight" yaml:"cpu_weight"`
	MemoryWeight  float64 `json:"memory_weight" yaml:"memory_weight"`
	IOWeight      float64 `json:"io_weight" yaml:"io_weight"`
	NetworkWeight float64 `json:"network_weight" yaml:"network_weight"`
	LatencyWeight float64 `json:"latency_weight" yaml:"latency_weight"`
}

// PlanOptimizationConfig 计划优化配置
type PlanOptimizationConfig struct {
	Algorithm            string  `json:"algorithm" yaml:"algorithm"`                 // "greedy", "dynamic_programming", "genetic"
	OptimizationGoal     string  `json:"optimization_goal" yaml:"optimization_goal"` // "latency", "throughput", "cost"
	MaxIterations        int     `json:"max_iterations" yaml:"max_iterations"`
	ConvergenceThreshold float64 `json:"convergence_threshold" yaml:"convergence_threshold"`
}

// AdaptivePlanningConfig 自适应规划配置
type AdaptivePlanningConfig struct {
	Enable         bool          `json:"enable" yaml:"enable"`
	LearningRate   float64       `json:"learning_rate" yaml:"learning_rate"`
	UpdateInterval time.Duration `json:"update_interval" yaml:"update_interval"`
	HistoryWindow  int           `json:"history_window" yaml:"history_window"`
	FeedbackWeight float64       `json:"feedback_weight" yaml:"feedback_weight"`
}

// QueryProfilingConfig 查询分析配置
type QueryProfilingConfig struct {
	Enable              bool                      `json:"enable" yaml:"enable"`
	SamplingRate        float64                   `json:"sampling_rate" yaml:"sampling_rate"`
	ProfilingInterval   time.Duration             `json:"profiling_interval" yaml:"profiling_interval"`
	MetricsCollection   []string                  `json:"metrics_collection" yaml:"metrics_collection"`
	PerformanceAnalysis PerformanceAnalysisConfig `json:"performance_analysis" yaml:"performance_analysis"`
}

// PerformanceAnalysisConfig 性能分析配置
type PerformanceAnalysisConfig struct {
	BottleneckDetection bool             `json:"bottleneck_detection" yaml:"bottleneck_detection"`
	TrendAnalysis       bool             `json:"trend_analysis" yaml:"trend_analysis"`
	AnomalyDetection    bool             `json:"anomaly_detection" yaml:"anomaly_detection"`
	Recommendations     bool             `json:"recommendations" yaml:"recommendations"`
	AlertThresholds     []AlertThreshold `json:"alert_thresholds" yaml:"alert_thresholds"`
}

// AlertThreshold 告警阈值
type AlertThreshold struct {
	Metric    string  `json:"metric" yaml:"metric"`
	Threshold float64 `json:"threshold" yaml:"threshold"`
	Severity  string  `json:"severity" yaml:"severity"`
	Action    string  `json:"action" yaml:"action"`
}

// ConcurrencyControlConfig 并发控制配置
type ConcurrencyControlConfig struct {
	MaxConcurrentQueries int                     `json:"max_concurrent_queries" yaml:"max_concurrent_queries"`
	QueryQueueSize       int                     `json:"query_queue_size" yaml:"query_queue_size"`
	ThreadPoolConfig     ThreadPoolConfig        `json:"thread_pool_config" yaml:"thread_pool_config"`
	RateLimiting         RateLimitingConfig      `json:"rate_limiting" yaml:"rate_limiting"`
	ResourceIsolation    ResourceIsolationConfig `json:"resource_isolation" yaml:"resource_isolation"`
}

// ThreadPoolConfig 线程池配置
type ThreadPoolConfig struct {
	CoreThreads     int           `json:"core_threads" yaml:"core_threads"`
	MaxThreads      int           `json:"max_threads" yaml:"max_threads"`
	QueueCapacity   int           `json:"queue_capacity" yaml:"queue_capacity"`
	KeepAliveTime   time.Duration `json:"keep_alive_time" yaml:"keep_alive_time"`
	RejectionPolicy string        `json:"rejection_policy" yaml:"rejection_policy"` // "abort", "caller_runs", "discard", "discard_oldest"
}

// RateLimitingConfig 限流配置
type RateLimitingConfig struct {
	Enable            bool                    `json:"enable" yaml:"enable"`
	GlobalLimit       RateLimitRule           `json:"global_limit" yaml:"global_limit"`
	PerUserLimit      RateLimitRule           `json:"per_user_limit" yaml:"per_user_limit"`
	PerIPLimit        RateLimitRule           `json:"per_ip_limit" yaml:"per_ip_limit"`
	AdaptiveRateLimit AdaptiveRateLimitConfig `json:"adaptive_rate_limit" yaml:"adaptive_rate_limit"`
}

// RateLimitRule 限流规则
type RateLimitRule struct {
	RequestsPerSecond int           `json:"requests_per_second" yaml:"requests_per_second"`
	BurstSize         int           `json:"burst_size" yaml:"burst_size"`
	WindowSize        time.Duration `json:"window_size" yaml:"window_size"`
	PenaltyDuration   time.Duration `json:"penalty_duration" yaml:"penalty_duration"`
}

// AdaptiveRateLimitConfig 自适应限流配置
type AdaptiveRateLimitConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	BaseRate         int           `json:"base_rate" yaml:"base_rate"`
	MaxRate          int           `json:"max_rate" yaml:"max_rate"`
	AdjustmentFactor float64       `json:"adjustment_factor" yaml:"adjustment_factor"`
	MonitoringWindow time.Duration `json:"monitoring_window" yaml:"monitoring_window"`
}

// ResourceIsolationConfig 资源隔离配置
type ResourceIsolationConfig struct {
	Enable          bool                  `json:"enable" yaml:"enable"`
	CPUIsolation    CPUIsolationConfig    `json:"cpu_isolation" yaml:"cpu_isolation"`
	MemoryIsolation MemoryIsolationConfig `json:"memory_isolation" yaml:"memory_isolation"`
	IOIsolation     IOIsolationConfig     `json:"io_isolation" yaml:"io_isolation"`
	PriorityQueues  PriorityQueuesConfig  `json:"priority_queues" yaml:"priority_queues"`
}

// CPUIsolationConfig CPU隔离配置
type CPUIsolationConfig struct {
	Enable      bool    `json:"enable" yaml:"enable"`
	CPUQuota    float64 `json:"cpu_quota" yaml:"cpu_quota"`
	CPUShares   int     `json:"cpu_shares" yaml:"cpu_shares"`
	CPUAffinity []int   `json:"cpu_affinity" yaml:"cpu_affinity"`
	NUMABinding bool    `json:"numa_binding" yaml:"numa_binding"`
}

// MemoryIsolationConfig 内存隔离配置
type MemoryIsolationConfig struct {
	Enable            bool  `json:"enable" yaml:"enable"`
	MemoryLimit       int64 `json:"memory_limit" yaml:"memory_limit"` // bytes
	SwapLimit         int64 `json:"swap_limit" yaml:"swap_limit"`     // bytes
	OOMKillDisable    bool  `json:"oom_kill_disable" yaml:"oom_kill_disable"`
	MemoryReservation int64 `json:"memory_reservation" yaml:"memory_reservation"`
}

// IOIsolationConfig IO隔离配置
type IOIsolationConfig struct {
	Enable         bool  `json:"enable" yaml:"enable"`
	ReadBandwidth  int64 `json:"read_bandwidth" yaml:"read_bandwidth"`   // bytes/sec
	WriteBandwidth int64 `json:"write_bandwidth" yaml:"write_bandwidth"` // bytes/sec
	ReadIOPS       int   `json:"read_iops" yaml:"read_iops"`
	WriteIOPS      int   `json:"write_iops" yaml:"write_iops"`
	IOWeight       int   `json:"io_weight" yaml:"io_weight"`
}

// PriorityQueuesConfig 优先级队列配置
type PriorityQueuesConfig struct {
	Enable            bool                   `json:"enable" yaml:"enable"`
	NumPriorityLevels int                    `json:"num_priority_levels" yaml:"num_priority_levels"`
	PriorityRules     []PriorityRule         `json:"priority_rules" yaml:"priority_rules"`
	SchedulingPolicy  SchedulingPolicyConfig `json:"scheduling_policy" yaml:"scheduling_policy"`
}

// PriorityRule 优先级规则
type PriorityRule struct {
	Condition   string  `json:"condition" yaml:"condition"`
	Priority    int     `json:"priority" yaml:"priority"`
	Weight      float64 `json:"weight" yaml:"weight"`
	Description string  `json:"description" yaml:"description"`
}

// SchedulingPolicyConfig 调度策略配置
type SchedulingPolicyConfig struct {
	Algorithm            string        `json:"algorithm" yaml:"algorithm"` // "fifo", "priority", "fair", "weighted_fair"
	TimeSlice            time.Duration `json:"time_slice" yaml:"time_slice"`
	Preemption           bool          `json:"preemption" yaml:"preemption"`
	StarvationPrevention bool          `json:"starvation_prevention" yaml:"starvation_prevention"`
}

// MemoryManagementConfig 内存管理配置
type MemoryManagementConfig struct {
	MemoryPool        MemoryPoolConfig        `json:"memory_pool" yaml:"memory_pool"`
	GarbageCollection GarbageCollectionConfig `json:"garbage_collection" yaml:"garbage_collection"`
	MemoryMapping     MemoryMappingConfig     `json:"memory_mapping" yaml:"memory_mapping"`
	SwapManagement    SwapManagementConfig    `json:"swap_management" yaml:"swap_management"`
}

// MemoryPoolConfig 内存池配置
type MemoryPoolConfig struct {
	Enable          bool    `json:"enable" yaml:"enable"`
	PoolSize        int64   `json:"pool_size" yaml:"pool_size"` // bytes
	ChunkSize       int     `json:"chunk_size" yaml:"chunk_size"`
	Preallocation   bool    `json:"preallocation" yaml:"preallocation"`
	GrowthStrategy  string  `json:"growth_strategy" yaml:"growth_strategy"` // "fixed", "linear", "exponential"
	ShrinkThreshold float64 `json:"shrink_threshold" yaml:"shrink_threshold"`
}

// GarbageCollectionConfig 垃圾回收配置
type GarbageCollectionConfig struct {
	Enable         bool          `json:"enable" yaml:"enable"`
	GCInterval     time.Duration `json:"gc_interval" yaml:"gc_interval"`
	GCThreshold    float64       `json:"gc_threshold" yaml:"gc_threshold"`
	ConcurrentGC   bool          `json:"concurrent_gc" yaml:"concurrent_gc"`
	GenerationalGC bool          `json:"generational_gc" yaml:"generational_gc"`
}

// MemoryMappingConfig 内存映射配置
type MemoryMappingConfig struct {
	Enable          bool   `json:"enable" yaml:"enable"`
	MmapThreshold   int64  `json:"mmap_threshold" yaml:"mmap_threshold"`   // bytes
	AdviseStrategy  string `json:"advise_strategy" yaml:"advise_strategy"` // "normal", "random", "sequential", "willneed", "dontneed"
	HugePagesEnable bool   `json:"huge_pages_enable" yaml:"huge_pages_enable"`
	LockMemory      bool   `json:"lock_memory" yaml:"lock_memory"`
}

// SwapManagementConfig 交换管理配置
type SwapManagementConfig struct {
	Enable         bool    `json:"enable" yaml:"enable"`
	Swappiness     int     `json:"swappiness" yaml:"swappiness"` // 0-100
	SwapThreshold  float64 `json:"swap_threshold" yaml:"swap_threshold"`
	SwapPriority   int     `json:"swap_priority" yaml:"swap_priority"`
	CompressedSwap bool    `json:"compressed_swap" yaml:"compressed_swap"`
}

// BatchProcessingConfig 批处理配置
type BatchProcessingConfig struct {
	Enable           bool                   `json:"enable" yaml:"enable"`
	BatchSize        int                    `json:"batch_size" yaml:"batch_size"`
	BatchTimeout     time.Duration          `json:"batch_timeout" yaml:"batch_timeout"`
	AdaptiveBatching AdaptiveBatchingConfig `json:"adaptive_batching" yaml:"adaptive_batching"`
	Pipelining       PipeliningConfig       `json:"pipelining" yaml:"pipelining"`
}

// AdaptiveBatchingConfig 自适应批处理配置
type AdaptiveBatchingConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	MinBatchSize     int           `json:"min_batch_size" yaml:"min_batch_size"`
	MaxBatchSize     int           `json:"max_batch_size" yaml:"max_batch_size"`
	LatencyTarget    time.Duration `json:"latency_target" yaml:"latency_target"`
	ThroughputTarget int           `json:"throughput_target" yaml:"throughput_target"`
	AdjustmentFactor float64       `json:"adjustment_factor" yaml:"adjustment_factor"`
}

// PipeliningConfig 流水线配置
type PipeliningConfig struct {
	Enable         bool `json:"enable" yaml:"enable"`
	Stages         int  `json:"stages" yaml:"stages"`
	BufferSize     int  `json:"buffer_size" yaml:"buffer_size"`
	ParallelStages bool `json:"parallel_stages" yaml:"parallel_stages"`
	Backpressure   bool `json:"backpressure" yaml:"backpressure"`
}

// PrefetchingConfig 预取配置
type PrefetchingConfig struct {
	Enable           bool                   `json:"enable" yaml:"enable"`
	DataPrefetching  DataPrefetchingConfig  `json:"data_prefetching" yaml:"data_prefetching"`
	IndexPrefetching IndexPrefetchingConfig `json:"index_prefetching" yaml:"index_prefetching"`
	QueryPrefetching QueryPrefetchingConfig `json:"query_prefetching" yaml:"query_prefetching"`
}

// DataPrefetchingConfig 数据预取配置
type DataPrefetchingConfig struct {
	Enable           bool    `json:"enable" yaml:"enable"`
	PrefetchDistance int     `json:"prefetch_distance" yaml:"prefetch_distance"`
	PrefetchRatio    float64 `json:"prefetch_ratio" yaml:"prefetch_ratio"`
	AccessPattern    string  `json:"access_pattern" yaml:"access_pattern"` // "sequential", "random", "adaptive"
	CacheWarmup      bool    `json:"cache_warmup" yaml:"cache_warmup"`
}

// IndexPrefetchingConfig 索引预取配置
type IndexPrefetchingConfig struct {
	Enable         bool `json:"enable" yaml:"enable"`
	PrefetchDepth  int  `json:"prefetch_depth" yaml:"prefetch_depth"`
	PrefetchWidth  int  `json:"prefetch_width" yaml:"prefetch_width"`
	PredictiveLoad bool `json:"predictive_load" yaml:"predictive_load"`
	LazyLoading    bool `json:"lazy_loading" yaml:"lazy_loading"`
}

// QueryPrefetchingConfig 查询预取配置
type QueryPrefetchingConfig struct {
	Enable              bool          `json:"enable" yaml:"enable"`
	PredictionModel     string        `json:"prediction_model" yaml:"prediction_model"` // "markov", "lstm", "transformer"
	HistoryWindow       int           `json:"history_window" yaml:"history_window"`
	PrefetchCount       int           `json:"prefetch_count" yaml:"prefetch_count"`
	ConfidenceThreshold float64       `json:"confidence_threshold" yaml:"confidence_threshold"`
	UpdateInterval      time.Duration `json:"update_interval" yaml:"update_interval"`
}

// GetDefaultPerformanceConfig 获取默认性能配置
func GetDefaultPerformanceConfig() *PerformanceOptimizationConfig {
	return &PerformanceOptimizationConfig{
		QueryAcceleration: QueryAccelerationConfig{
			Enable: true,
			Preprocessing: PreprocessingConfig{
				Normalization: true,
				DimensionReduction: DimensionReductionConfig{
					Enable:        false,
					Method:        "pca",
					TargetDim:     128,
					VarianceRatio: 0.95,
					TrainingSize:  10000,
				},
				Quantization: QuantizationConfig{
					Enable:       false,
					Method:       "scalar",
					Bits:         8,
					CodebookSize: 256,
				},
			},
			MultiStageSearch: MultiStageSearchConfig{
				Enable:           true,
				CoarseCandidates: 1000,
				RefinementRatio:  0.1,
				Stages: []SearchStageConfig{
					{
						Name:          "coarse",
						Method:        "coarse",
						Candidates:    1000,
						Accuracy:      0.8,
						LatencyBudget: 50 * time.Millisecond,
					},
					{
						Name:          "fine",
						Method:        "fine",
						Candidates:    100,
						Accuracy:      0.95,
						LatencyBudget: 100 * time.Millisecond,
					},
				},
			},
			ParallelSearch: ParallelSearchConfig{
				Enable:     true,
				MaxWorkers: runtime.NumCPU(),
				WorkloadSplitting: WorkloadSplittingConfig{
					Strategy:      "data_parallel",
					ChunkSize:     1000,
					OverlapRatio:  0.1,
					LoadBalancing: true,
				},
				ResultMerging: ResultMergingConfig{
					Strategy:            "heap_merge",
					MergeThreshold:      100,
					DuplicateRemoval:    true,
					SimilarityThreshold: 0.99,
				},
			},
			EarlyTermination: EarlyTerminationConfig{
				Enable:              true,
				ConfidenceThreshold: 0.95,
				TimeoutThreshold:    200 * time.Millisecond,
				QualityThreshold:    0.9,
			},
		},
		ConcurrencyControl: ConcurrencyControlConfig{
			MaxConcurrentQueries: 100,
			QueryQueueSize:       1000,
			ThreadPoolConfig: ThreadPoolConfig{
				CoreThreads:     runtime.NumCPU(),
				MaxThreads:      runtime.NumCPU() * 2,
				QueueCapacity:   1000,
				KeepAliveTime:   60 * time.Second,
				RejectionPolicy: "caller_runs",
			},
			RateLimiting: RateLimitingConfig{
				Enable: true,
				GlobalLimit: RateLimitRule{
					RequestsPerSecond: 1000,
					BurstSize:         100,
					WindowSize:        time.Second,
					PenaltyDuration:   5 * time.Second,
				},
				PerUserLimit: RateLimitRule{
					RequestsPerSecond: 100,
					BurstSize:         10,
					WindowSize:        time.Second,
					PenaltyDuration:   10 * time.Second,
				},
			},
		},
		MemoryManagement: MemoryManagementConfig{
			MemoryPool: MemoryPoolConfig{
				Enable:          true,
				PoolSize:        1024 * 1024 * 1024, // 1GB
				ChunkSize:       4096,
				Preallocation:   true,
				GrowthStrategy:  "exponential",
				ShrinkThreshold: 0.5,
			},
			GarbageCollection: GarbageCollectionConfig{
				Enable:         true,
				GCInterval:     5 * time.Minute,
				GCThreshold:    0.8,
				ConcurrentGC:   true,
				GenerationalGC: true,
			},
		},
		BatchProcessing: BatchProcessingConfig{
			Enable:       true,
			BatchSize:    100,
			BatchTimeout: 10 * time.Millisecond,
			AdaptiveBatching: AdaptiveBatchingConfig{
				Enable:           true,
				MinBatchSize:     10,
				MaxBatchSize:     1000,
				LatencyTarget:    50 * time.Millisecond,
				ThroughputTarget: 10000,
				AdjustmentFactor: 0.1,
			},
		},
		Prefetching: PrefetchingConfig{
			Enable: true,
			DataPrefetching: DataPrefetchingConfig{
				Enable:           true,
				PrefetchDistance: 10,
				PrefetchRatio:    0.2,
				AccessPattern:    "adaptive",
				CacheWarmup:      true,
			},
			IndexPrefetching: IndexPrefetchingConfig{
				Enable:         true,
				PrefetchDepth:  3,
				PrefetchWidth:  5,
				PredictiveLoad: true,
				LazyLoading:    false,
			},
		},
	}
}
