package entity

import "time"

// SearchOptions 搜索选项结构
type SearchOptions struct {
	// 基础选项
	NumHashTables int  // LSH哈希表数量
	UseANN        bool // 是否使用近似最近邻

	// IVF相关选项
	Nprobe        int           `json:"nprobe"`         // IVF搜索的探测簇数量
	SearchTimeout time.Duration `json:"search_timeout"` // 搜索超时时间
	QualityLevel  float64       `json:"quality_level"`  // 质量要求等级 0.0-1.0
	UseCache      bool          `json:"use_cache"`      // 是否使用缓存
	MaxCandidates int           `json:"max_candidates"` // 最大候选数量

	// 索引策略选择
	ForceStrategy  string `json:"force_strategy,omitempty"` // 强制使用指定策略: "ivf", "hnsw", "pq", "lsh", "ivf_hnsw"
	EnableHybrid   bool   `json:"enable_hybrid"`            // 启用混合索引策略
	PreferAccuracy bool   `json:"prefer_accuracy"`          // 优先精度还是速度

	// HNSW相关选项
	EfSearch       int `json:"ef_search,omitempty"`       // HNSW搜索时的扩展因子
	MaxConnections int `json:"max_connections,omitempty"` // HNSW最大连接数

	// PQ相关选项
	UsePQCompression bool `json:"use_pq_compression"`       // 启用PQ压缩
	PQSubVectors     int  `json:"pq_sub_vectors,omitempty"` // PQ子向量数量

	// 多阶段搜索选项
	EnableMultiStage bool    `json:"enable_multi_stage"`         // 启用多阶段搜索
	CoarseK          int     `json:"coarse_k,omitempty"`         // 粗搜索返回的候选数量
	RefinementRatio  float64 `json:"refinement_ratio,omitempty"` // 精搜索的候选比例

	// 硬件加速选项
	UseGPU  bool `json:"use_gpu"`  // 启用GPU加速
	UseFPGA bool `json:"use_fpga"` // 启用FPGA加速
	UseRDMA bool `json:"use_rdma"` // 启用RDMA网络
	UsePMem bool `json:"use_pmem"` // 启用持久内存

	// 缓存策略选项
	CacheStrategy   string `json:"cache_strategy,omitempty"`    // 缓存策略: "lru", "lfu", "arc"
	ResultCacheSize int    `json:"result_cache_size,omitempty"` // 结果缓存大小
	VectorCacheSize int    `json:"vector_cache_size,omitempty"` // 向量缓存大小
	IndexCacheSize  int    `json:"index_cache_size,omitempty"`  // 索引缓存大小

	// 数据预处理选项
	NormalizeVectors bool `json:"normalize_vectors"`          // 向量归一化
	UsePCA           bool `json:"use_pca"`                    // 启用PCA降维
	TargetDimension  int  `json:"target_dimension,omitempty"` // PCA目标维度

	// 分布式搜索选项
	ShardingStrategy  string `json:"sharding_strategy,omitempty"` // 分片策略: "range", "hash", "cluster"
	ParallelSearch    bool   `json:"parallel_search"`             // 并行搜索
	MaxShards         int    `json:"max_shards,omitempty"`        // 最大分片数
	DistributedSearch bool   `json:"distributed_search"`          // 启用分布式搜索
	BatchSize         int    `json:"batch_size,omitempty"`

	// 混合查询选项
	EnableHybridQuery bool                   `json:"enable_hybrid_query"`      // 启用向量+标量混合查询
	ScalarFilters     map[string]interface{} `json:"scalar_filters,omitempty"` // 标量过滤条件
	FilterFirst       bool                   `json:"filter_first"`             // 是否先执行标量过滤

	// 性能监控选项
	EnableMetrics    bool `json:"enable_metrics"`     // 启用性能指标收集
	TrackLatency     bool `json:"track_latency"`      // 跟踪延迟
	TrackThroughput  bool `json:"track_throughput"`   // 跟踪吞吐量
	TrackMemoryUsage bool `json:"track_memory_usage"` // 跟踪内存使用

	// 其他选项
	EnablePQRefinement  bool  `json:"enable_pq_refinement"` // 启用PQ精化
	EnableDeduplication bool  `json:"enable_deduplication"` // 启用结果去重
	CacheTTL            int64 `json:"cache_ttl,omitempty"`  // 缓存TTL（秒）

	// 工作负载优化选项
	MemoryOptimized   bool `json:"memory_optimized"`   // 内存优化模式
	PersistentStorage bool `json:"persistent_storage"` // 持久化存储模式

	K                int           `json:"k"`
	Threshold        float64       `json:"threshold"`
	UseApproximation bool          `json:"use_approximation"`
	Timeout          time.Duration `json:"timeout"`
	Parallel         bool          `json:"parallel"`
}
