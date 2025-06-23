package vector

import (
	"time"
)

// CacheStrategyConfig 缓存策略配置
type CacheStrategyConfig struct {
	ResultCache   ResultCacheConfig   `json:"result_cache" yaml:"result_cache"`
	VectorCache   VectorCacheConfig   `json:"vector_cache" yaml:"vector_cache"`
	IndexCache    IndexCacheConfig    `json:"index_cache" yaml:"index_cache"`
	QueryCache    QueryCacheConfig    `json:"query_cache" yaml:"query_cache"`
	MetadataCache MetadataCacheConfig `json:"metadata_cache" yaml:"metadata_cache"`
	GlobalCache   GlobalCacheConfig   `json:"global_cache" yaml:"global_cache"`
}

// ResultCacheConfig 结果缓存配置
type ResultCacheConfig struct {
	Enable               bool                    `json:"enable" yaml:"enable"`
	MaxSize              int64                   `json:"max_size" yaml:"max_size"` // bytes
	MaxEntries           int                     `json:"max_entries" yaml:"max_entries"`
	TTL                  time.Duration           `json:"ttl" yaml:"ttl"`
	EvictionPolicy       string                  `json:"eviction_policy" yaml:"eviction_policy"` // "LRU", "LFU", "FIFO", "K-LRU"
	CompressionEnable    bool                    `json:"compression_enable" yaml:"compression_enable"`
	CompressionAlgorithm string                  `json:"compression_algorithm" yaml:"compression_algorithm"` // "gzip", "lz4", "snappy", "zstd"
	Persistence          ResultCachePersistence  `json:"persistence" yaml:"persistence"`
	Distribution         ResultCacheDistribution `json:"distribution" yaml:"distribution"`
	Optimization         ResultCacheOptimization `json:"optimization" yaml:"optimization"`
	Monitoring           CacheMonitoringConfig   `json:"monitoring" yaml:"monitoring"`
}

// ResultCachePersistence 结果缓存持久化
type ResultCachePersistence struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	StoragePath      string        `json:"storage_path" yaml:"storage_path"`
	SyncInterval     time.Duration `json:"sync_interval" yaml:"sync_interval"`
	CompressionLevel int           `json:"compression_level" yaml:"compression_level"`
	BackupEnable     bool          `json:"backup_enable" yaml:"backup_enable"`
	BackupInterval   time.Duration `json:"backup_interval" yaml:"backup_interval"`
	RecoveryMode     string        `json:"recovery_mode" yaml:"recovery_mode"` // "auto", "manual", "disabled"
}

// ResultCacheDistribution 结果缓存分布式配置
type ResultCacheDistribution struct {
	Enable            bool                       `json:"enable" yaml:"enable"`
	Strategy          string                     `json:"strategy" yaml:"strategy"` // "replicated", "partitioned", "hybrid"
	ReplicationFactor int                        `json:"replication_factor" yaml:"replication_factor"`
	ConsistencyLevel  string                     `json:"consistency_level" yaml:"consistency_level"` // "strong", "eventual", "weak"
	Partitioning      ResultCachePartitioning    `json:"partitioning" yaml:"partitioning"`
	Synchronization   ResultCacheSynchronization `json:"synchronization" yaml:"synchronization"`
	Failover          ResultCacheFailover        `json:"failover" yaml:"failover"`
}

// ResultCachePartitioning 结果缓存分区配置
type ResultCachePartitioning struct {
	Method             string  `json:"method" yaml:"method"` // "hash", "range", "consistent_hash"
	Partitions         int     `json:"partitions" yaml:"partitions"`
	HashFunction       string  `json:"hash_function" yaml:"hash_function"` // "md5", "sha1", "murmur3", "xxhash"
	VirtualNodes       int     `json:"virtual_nodes" yaml:"virtual_nodes"`
	RebalanceEnable    bool    `json:"rebalance_enable" yaml:"rebalance_enable"`
	RebalanceThreshold float64 `json:"rebalance_threshold" yaml:"rebalance_threshold"`
}

// ResultCacheSynchronization 结果缓存同步配置
type ResultCacheSynchronization struct {
	Protocol           string        `json:"protocol" yaml:"protocol"` // "gossip", "raft", "paxos"
	SyncInterval       time.Duration `json:"sync_interval" yaml:"sync_interval"`
	BatchSize          int           `json:"batch_size" yaml:"batch_size"`
	ConflictResolution string        `json:"conflict_resolution" yaml:"conflict_resolution"` // "timestamp", "version", "manual"
	MergeStrategy      string        `json:"merge_strategy" yaml:"merge_strategy"`           // "union", "intersection", "priority"
}

// ResultCacheFailover 结果缓存故障转移
type ResultCacheFailover struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	DetectionInterval time.Duration `json:"detection_interval" yaml:"detection_interval"`
	FailureThreshold  int           `json:"failure_threshold" yaml:"failure_threshold"`
	RecoveryTimeout   time.Duration `json:"recovery_timeout" yaml:"recovery_timeout"`
	BackupNodes       []string      `json:"backup_nodes" yaml:"backup_nodes"`
	AutoFailback      bool          `json:"auto_failback" yaml:"auto_failback"`
}

// ResultCacheOptimization 结果缓存优化
type ResultCacheOptimization struct {
	Prefetching        CachePrefetchingConfig        `json:"prefetching" yaml:"prefetching"`
	Compaction         CacheCompactionConfig         `json:"compaction" yaml:"compaction"`
	Deduplication      CacheDeduplicationConfig      `json:"deduplication" yaml:"deduplication"`
	LoadBalancing      CacheLoadBalancingConfig      `json:"load_balancing" yaml:"load_balancing"`
	MemoryOptimization CacheMemoryOptimizationConfig `json:"memory_optimization" yaml:"memory_optimization"`
}

// CachePrefetchingConfig 缓存预取配置
type CachePrefetchingConfig struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	Strategy          string        `json:"strategy" yaml:"strategy"` // "sequential", "pattern_based", "ml_based"
	LookaheadSize     int           `json:"lookahead_size" yaml:"lookahead_size"`
	PrefetchRatio     float64       `json:"prefetch_ratio" yaml:"prefetch_ratio"`
	AdaptiveEnable    bool          `json:"adaptive_enable" yaml:"adaptive_enable"`
	LearningWindow    time.Duration `json:"learning_window" yaml:"learning_window"`
	AccuracyThreshold float64       `json:"accuracy_threshold" yaml:"accuracy_threshold"`
}

// CacheCompactionConfig 缓存压缩配置
type CacheCompactionConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	TriggerThreshold   float64       `json:"trigger_threshold" yaml:"trigger_threshold"`
	CompactionInterval time.Duration `json:"compaction_interval" yaml:"compaction_interval"`
	Strategy           string        `json:"strategy" yaml:"strategy"` // "size_based", "time_based", "hybrid"
	MaxCompactionTime  time.Duration `json:"max_compaction_time" yaml:"max_compaction_time"`
	BackgroundEnable   bool          `json:"background_enable" yaml:"background_enable"`
}

// CacheDeduplicationConfig 缓存去重配置
type CacheDeduplicationConfig struct {
	Enable              bool          `json:"enable" yaml:"enable"`
	Method              string        `json:"method" yaml:"method"`               // "hash_based", "content_based", "semantic_based"
	HashFunction        string        `json:"hash_function" yaml:"hash_function"` // "md5", "sha256", "xxhash"
	SimilarityThreshold float64       `json:"similarity_threshold" yaml:"similarity_threshold"`
	WindowSize          int           `json:"window_size" yaml:"window_size"`
	CleanupInterval     time.Duration `json:"cleanup_interval" yaml:"cleanup_interval"`
}

// CacheLoadBalancingConfig 缓存负载均衡配置
type CacheLoadBalancingConfig struct {
	Enable          bool                   `json:"enable" yaml:"enable"`
	Strategy        string                 `json:"strategy" yaml:"strategy"` // "round_robin", "least_loaded", "consistent_hash"
	HealthCheck     CacheHealthCheckConfig `json:"health_check" yaml:"health_check"`
	WeightedRouting bool                   `json:"weighted_routing" yaml:"weighted_routing"`
	StickySession   bool                   `json:"sticky_session" yaml:"sticky_session"`
	FailoverEnable  bool                   `json:"failover_enable" yaml:"failover_enable"`
}

// CacheHealthCheckConfig 缓存健康检查配置
type CacheHealthCheckConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	Interval           time.Duration `json:"interval" yaml:"interval"`
	Timeout            time.Duration `json:"timeout" yaml:"timeout"`
	RetryCount         int           `json:"retry_count" yaml:"retry_count"`
	UnhealthyThreshold int           `json:"unhealthy_threshold" yaml:"unhealthy_threshold"`
	HealthyThreshold   int           `json:"healthy_threshold" yaml:"healthy_threshold"`
}

// CacheMemoryOptimizationConfig 缓存内存优化配置
type CacheMemoryOptimizationConfig struct {
	Enable                 bool    `json:"enable" yaml:"enable"`
	MemoryPooling          bool    `json:"memory_pooling" yaml:"memory_pooling"`
	ObjectPooling          bool    `json:"object_pooling" yaml:"object_pooling"`
	GCOptimization         bool    `json:"gc_optimization" yaml:"gc_optimization"`
	MemoryMapping          bool    `json:"memory_mapping" yaml:"memory_mapping"`
	CompressionRatio       float64 `json:"compression_ratio" yaml:"compression_ratio"`
	FragmentationThreshold float64 `json:"fragmentation_threshold" yaml:"fragmentation_threshold"`
}

// VectorCacheConfig 向量缓存配置
type VectorCacheConfig struct {
	Enable            bool                    `json:"enable" yaml:"enable"`
	MaxSize           int64                   `json:"max_size" yaml:"max_size"` // bytes
	MaxVectors        int                     `json:"max_vectors" yaml:"max_vectors"`
	TTL               time.Duration           `json:"ttl" yaml:"ttl"`
	EvictionPolicy    string                  `json:"eviction_policy" yaml:"eviction_policy"` // "LRU", "LFU", "FIFO", "frequency_based"
	HotDataStrategy   VectorHotDataStrategy   `json:"hot_data_strategy" yaml:"hot_data_strategy"`
	CompressionConfig VectorCompressionConfig `json:"compression_config" yaml:"compression_config"`
	Tiering           VectorCacheTiering      `json:"tiering" yaml:"tiering"`
	Prefetching       VectorPrefetchingConfig `json:"prefetching" yaml:"prefetching"`
	Monitoring        CacheMonitoringConfig   `json:"monitoring" yaml:"monitoring"`
}

// VectorHotDataStrategy 向量热数据策略
type VectorHotDataStrategy struct {
	Enable          bool          `json:"enable" yaml:"enable"`
	DetectionMethod string        `json:"detection_method" yaml:"detection_method"` // "access_frequency", "access_pattern", "ml_based"
	HotThreshold    float64       `json:"hot_threshold" yaml:"hot_threshold"`
	ColdThreshold   float64       `json:"cold_threshold" yaml:"cold_threshold"`
	AnalysisWindow  time.Duration `json:"analysis_window" yaml:"analysis_window"`
	PromotionPolicy string        `json:"promotion_policy" yaml:"promotion_policy"` // "immediate", "batch", "scheduled"
	DemotionPolicy  string        `json:"demotion_policy" yaml:"demotion_policy"`   // "immediate", "batch", "scheduled"
	ResidentMemory  bool          `json:"resident_memory" yaml:"resident_memory"`
}

// VectorCompressionConfig 向量压缩配置
type VectorCompressionConfig struct {
	Enable           bool    `json:"enable" yaml:"enable"`
	Algorithm        string  `json:"algorithm" yaml:"algorithm"` // "pq", "sq", "opq", "ivf_pq"
	CompressionRatio float64 `json:"compression_ratio" yaml:"compression_ratio"`
	QualityThreshold float64 `json:"quality_threshold" yaml:"quality_threshold"`
	AdaptiveEnable   bool    `json:"adaptive_enable" yaml:"adaptive_enable"`
	CodebookSize     int     `json:"codebook_size" yaml:"codebook_size"`
	SubvectorCount   int     `json:"subvector_count" yaml:"subvector_count"`
}

// VectorCacheTiering 向量缓存分层
type VectorCacheTiering struct {
	Enable          bool                  `json:"enable" yaml:"enable"`
	Levels          []VectorCacheLevel    `json:"levels" yaml:"levels"`
	PromotionPolicy VectorPromotionPolicy `json:"promotion_policy" yaml:"promotion_policy"`
	EvictionPolicy  VectorEvictionPolicy  `json:"eviction_policy" yaml:"eviction_policy"`
	MigrationConfig VectorMigrationConfig `json:"migration_config" yaml:"migration_config"`
}

// VectorCacheLevel 向量缓存级别
type VectorCacheLevel struct {
	Level             int           `json:"level" yaml:"level"`
	Name              string        `json:"name" yaml:"name"`
	StorageType       string        `json:"storage_type" yaml:"storage_type"` // "memory", "ssd", "hdd", "pmem"
	MaxSize           int64         `json:"max_size" yaml:"max_size"`         // bytes
	MaxVectors        int           `json:"max_vectors" yaml:"max_vectors"`
	AccessLatency     time.Duration `json:"access_latency" yaml:"access_latency"`
	Bandwidth         int64         `json:"bandwidth" yaml:"bandwidth"` // bytes/sec
	CompressionEnable bool          `json:"compression_enable" yaml:"compression_enable"`
	ReplicationFactor int           `json:"replication_factor" yaml:"replication_factor"`
}

// VectorPromotionPolicy 向量提升策略
type VectorPromotionPolicy struct {
	Strategy          string        `json:"strategy" yaml:"strategy"` // "access_based", "time_based", "hybrid"
	AccessThreshold   int           `json:"access_threshold" yaml:"access_threshold"`
	TimeThreshold     time.Duration `json:"time_threshold" yaml:"time_threshold"`
	BatchSize         int           `json:"batch_size" yaml:"batch_size"`
	PromotionInterval time.Duration `json:"promotion_interval" yaml:"promotion_interval"`
	CostModel         string        `json:"cost_model" yaml:"cost_model"` // "simple", "weighted", "ml_based"
}

// VectorEvictionPolicy 向量驱逐策略
type VectorEvictionPolicy struct {
	Strategy         string        `json:"strategy" yaml:"strategy"` // "lru", "lfu", "cost_based", "ml_based"
	EvictionRatio    float64       `json:"eviction_ratio" yaml:"eviction_ratio"`
	BatchSize        int           `json:"batch_size" yaml:"batch_size"`
	EvictionInterval time.Duration `json:"eviction_interval" yaml:"eviction_interval"`
	ProtectedRatio   float64       `json:"protected_ratio" yaml:"protected_ratio"`
	CostModel        string        `json:"cost_model" yaml:"cost_model"` // "simple", "weighted", "ml_based"
}

// VectorMigrationConfig 向量迁移配置
type VectorMigrationConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	Strategy           string        `json:"strategy" yaml:"strategy"` // "background", "on_demand", "scheduled"
	BatchSize          int           `json:"batch_size" yaml:"batch_size"`
	MigrationRate      int64         `json:"migration_rate" yaml:"migration_rate"` // vectors/sec
	Parallelism        int           `json:"parallelism" yaml:"parallelism"`
	VerificationEnable bool          `json:"verification_enable" yaml:"verification_enable"`
	RollbackEnable     bool          `json:"rollback_enable" yaml:"rollback_enable"`
	Timeout            time.Duration `json:"timeout" yaml:"timeout"`
}

// VectorPrefetchingConfig 向量预取配置
type VectorPrefetchingConfig struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	Strategy          string        `json:"strategy" yaml:"strategy"` // "neighbor_based", "pattern_based", "ml_based"
	NeighborCount     int           `json:"neighbor_count" yaml:"neighbor_count"`
	PrefetchDepth     int           `json:"prefetch_depth" yaml:"prefetch_depth"`
	AccuracyThreshold float64       `json:"accuracy_threshold" yaml:"accuracy_threshold"`
	LearningWindow    time.Duration `json:"learning_window" yaml:"learning_window"`
	AdaptiveEnable    bool          `json:"adaptive_enable" yaml:"adaptive_enable"`
}

// IndexCacheConfig 索引缓存配置
type IndexCacheConfig struct {
	Enable         bool                      `json:"enable" yaml:"enable"`
	MaxSize        int64                     `json:"max_size" yaml:"max_size"` // bytes
	MaxIndices     int                       `json:"max_indices" yaml:"max_indices"`
	TTL            time.Duration             `json:"ttl" yaml:"ttl"`
	EvictionPolicy string                    `json:"eviction_policy" yaml:"eviction_policy"` // "LRU", "LFU", "usage_based"
	Preloading     IndexPreloadingConfig     `json:"preloading" yaml:"preloading"`
	PartialCaching IndexPartialCachingConfig `json:"partial_caching" yaml:"partial_caching"`
	Compression    IndexCompressionConfig    `json:"compression" yaml:"compression"`
	Versioning     IndexVersioningConfig     `json:"versioning" yaml:"versioning"`
	Monitoring     CacheMonitoringConfig     `json:"monitoring" yaml:"monitoring"`
}

// IndexPreloadingConfig 索引预加载配置
type IndexPreloadingConfig struct {
	Enable             bool               `json:"enable" yaml:"enable"`
	Strategy           string             `json:"strategy" yaml:"strategy"` // "all", "frequent", "priority_based"
	CommonIndices      []string           `json:"common_indices" yaml:"common_indices"`
	PriorityWeights    map[string]float64 `json:"priority_weights" yaml:"priority_weights"`
	LoadingParallelism int                `json:"loading_parallelism" yaml:"loading_parallelism"`
	WarmupEnable       bool               `json:"warmup_enable" yaml:"warmup_enable"`
	WarmupQueries      []string           `json:"warmup_queries" yaml:"warmup_queries"`
}

// IndexPartialCachingConfig 索引部分缓存配置
type IndexPartialCachingConfig struct {
	Enable             bool    `json:"enable" yaml:"enable"`
	Strategy           string  `json:"strategy" yaml:"strategy"` // "level_based", "region_based", "adaptive"
	CacheRatio         float64 `json:"cache_ratio" yaml:"cache_ratio"`
	HotRegionDetection bool    `json:"hot_region_detection" yaml:"hot_region_detection"`
	RegionSize         int     `json:"region_size" yaml:"region_size"`
	OverlapRatio       float64 `json:"overlap_ratio" yaml:"overlap_ratio"`
	DynamicAdjustment  bool    `json:"dynamic_adjustment" yaml:"dynamic_adjustment"`
}

// IndexCompressionConfig 索引压缩配置
type IndexCompressionConfig struct {
	Enable           bool    `json:"enable" yaml:"enable"`
	Algorithm        string  `json:"algorithm" yaml:"algorithm"` // "gzip", "lz4", "snappy", "zstd"
	CompressionLevel int     `json:"compression_level" yaml:"compression_level"`
	CompressionRatio float64 `json:"compression_ratio" yaml:"compression_ratio"`
	AdaptiveEnable   bool    `json:"adaptive_enable" yaml:"adaptive_enable"`
	QualityThreshold float64 `json:"quality_threshold" yaml:"quality_threshold"`
}

// IndexVersioningConfig 索引版本控制配置
type IndexVersioningConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	MaxVersions        int           `json:"max_versions" yaml:"max_versions"`
	VersionTTL         time.Duration `json:"version_ttl" yaml:"version_ttl"`
	IncrementalUpdate  bool          `json:"incremental_update" yaml:"incremental_update"`
	DeltaCompression   bool          `json:"delta_compression" yaml:"delta_compression"`
	RollbackEnable     bool          `json:"rollback_enable" yaml:"rollback_enable"`
	ConflictResolution string        `json:"conflict_resolution" yaml:"conflict_resolution"` // "latest", "merge", "manual"
}

// QueryCacheConfig 查询缓存配置
type QueryCacheConfig struct {
	Enable         bool                     `json:"enable" yaml:"enable"`
	MaxSize        int64                    `json:"max_size" yaml:"max_size"` // bytes
	MaxQueries     int                      `json:"max_queries" yaml:"max_queries"`
	TTL            time.Duration            `json:"ttl" yaml:"ttl"`
	EvictionPolicy string                   `json:"eviction_policy" yaml:"eviction_policy"` // "LRU", "LFU", "cost_based"
	KeyGeneration  QueryKeyGenerationConfig `json:"key_generation" yaml:"key_generation"`
	Invalidation   QueryInvalidationConfig  `json:"invalidation" yaml:"invalidation"`
	Optimization   QueryOptimizationConfig  `json:"optimization" yaml:"optimization"`
	Monitoring     CacheMonitoringConfig    `json:"monitoring" yaml:"monitoring"`
}

// QueryKeyGenerationConfig 查询键生成配置
type QueryKeyGenerationConfig struct {
	Strategy            string   `json:"strategy" yaml:"strategy"`           // "hash_based", "semantic_based", "hybrid"
	HashFunction        string   `json:"hash_function" yaml:"hash_function"` // "md5", "sha256", "xxhash"
	Normalization       bool     `json:"normalization" yaml:"normalization"`
	ParameterFiltering  bool     `json:"parameter_filtering" yaml:"parameter_filtering"`
	IgnoredParameters   []string `json:"ignored_parameters" yaml:"ignored_parameters"`
	SimilarityThreshold float64  `json:"similarity_threshold" yaml:"similarity_threshold"`
}

// QueryInvalidationConfig 查询失效配置
type QueryInvalidationConfig struct {
	Strategy           string        `json:"strategy" yaml:"strategy"` // "time_based", "event_based", "hybrid"
	InvalidationEvents []string      `json:"invalidation_events" yaml:"invalidation_events"`
	DependencyTracking bool          `json:"dependency_tracking" yaml:"dependency_tracking"`
	BatchInvalidation  bool          `json:"batch_invalidation" yaml:"batch_invalidation"`
	GracePeriod        time.Duration `json:"grace_period" yaml:"grace_period"`
	NotificationEnable bool          `json:"notification_enable" yaml:"notification_enable"`
}

// MetadataCacheConfig 元数据缓存配置
type MetadataCacheConfig struct {
	Enable          bool                          `json:"enable" yaml:"enable"`
	MaxSize         int64                         `json:"max_size" yaml:"max_size"` // bytes
	MaxEntries      int                           `json:"max_entries" yaml:"max_entries"`
	TTL             time.Duration                 `json:"ttl" yaml:"ttl"`
	EvictionPolicy  string                        `json:"eviction_policy" yaml:"eviction_policy"` // "LRU", "LFU", "priority_based"
	Consistency     MetadataConsistencyConfig     `json:"consistency" yaml:"consistency"`
	Synchronization MetadataSynchronizationConfig `json:"synchronization" yaml:"synchronization"`
	Versioning      MetadataVersioningConfig      `json:"versioning" yaml:"versioning"`
	Monitoring      CacheMonitoringConfig         `json:"monitoring" yaml:"monitoring"`
}

// MetadataConsistencyConfig 元数据一致性配置
type MetadataConsistencyConfig struct {
	Level              string        `json:"level" yaml:"level"` // "strong", "eventual", "weak"
	ValidationEnable   bool          `json:"validation_enable" yaml:"validation_enable"`
	ChecksumEnable     bool          `json:"checksum_enable" yaml:"checksum_enable"`
	RefreshInterval    time.Duration `json:"refresh_interval" yaml:"refresh_interval"`
	ConflictResolution string        `json:"conflict_resolution" yaml:"conflict_resolution"` // "latest", "merge", "manual"
}

// MetadataSynchronizationConfig 元数据同步配置
type MetadataSynchronizationConfig struct {
	Strategy          string          `json:"strategy" yaml:"strategy"` // "push", "pull", "hybrid"
	SyncInterval      time.Duration   `json:"sync_interval" yaml:"sync_interval"`
	BatchSize         int             `json:"batch_size" yaml:"batch_size"`
	CompressionEnable bool            `json:"compression_enable" yaml:"compression_enable"`
	RetryPolicy       SyncRetryPolicy `json:"retry_policy" yaml:"retry_policy"`
	ConflictDetection bool            `json:"conflict_detection" yaml:"conflict_detection"`
}

// SyncRetryPolicy 同步重试策略
type SyncRetryPolicy struct {
	MaxRetries      int           `json:"max_retries" yaml:"max_retries"`
	RetryInterval   time.Duration `json:"retry_interval" yaml:"retry_interval"`
	BackoffStrategy string        `json:"backoff_strategy" yaml:"backoff_strategy"` // "exponential", "linear", "fixed"
	MaxBackoffTime  time.Duration `json:"max_backoff_time" yaml:"max_backoff_time"`
	JitterEnable    bool          `json:"jitter_enable" yaml:"jitter_enable"`
}

// MetadataVersioningConfig 元数据版本控制配置
type MetadataVersioningConfig struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	MaxVersions       int           `json:"max_versions" yaml:"max_versions"`
	VersionTTL        time.Duration `json:"version_ttl" yaml:"version_ttl"`
	DeltaStorage      bool          `json:"delta_storage" yaml:"delta_storage"`
	CompressionEnable bool          `json:"compression_enable" yaml:"compression_enable"`
	RollbackEnable    bool          `json:"rollback_enable" yaml:"rollback_enable"`
}

// GlobalCacheConfig 全局缓存配置
type GlobalCacheConfig struct {
	MemoryLimit     int64                 `json:"memory_limit" yaml:"memory_limit"` // bytes
	MemoryThreshold float64               `json:"memory_threshold" yaml:"memory_threshold"`
	GCStrategy      string                `json:"gc_strategy" yaml:"gc_strategy"` // "aggressive", "conservative", "adaptive"
	GCInterval      time.Duration         `json:"gc_interval" yaml:"gc_interval"`
	Statistics      CacheStatisticsConfig `json:"statistics" yaml:"statistics"`
	Warmup          CacheWarmupConfig     `json:"warmup" yaml:"warmup"`
	Backup          CacheBackupConfig     `json:"backup" yaml:"backup"`
	Security        CacheSecurityConfig   `json:"security" yaml:"security"`
}

// CacheStatisticsConfig 缓存统计配置
type CacheStatisticsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
	RetentionPeriod    time.Duration `json:"retention_period" yaml:"retention_period"`
	Metrics            []string      `json:"metrics" yaml:"metrics"`
	ExportFormat       string        `json:"export_format" yaml:"export_format"` // "prometheus", "json", "csv"
	ExportPath         string        `json:"export_path" yaml:"export_path"`
	AggregationEnable  bool          `json:"aggregation_enable" yaml:"aggregation_enable"`
}

// CacheWarmupConfig 缓存预热配置
type CacheWarmupConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	Strategy         string        `json:"strategy" yaml:"strategy"` // "preload", "background", "on_demand"
	WarmupData       []string      `json:"warmup_data" yaml:"warmup_data"`
	WarmupQueries    []string      `json:"warmup_queries" yaml:"warmup_queries"`
	Parallelism      int           `json:"parallelism" yaml:"parallelism"`
	Timeout          time.Duration `json:"timeout" yaml:"timeout"`
	ProgressTracking bool          `json:"progress_tracking" yaml:"progress_tracking"`
}

// CacheBackupConfig 缓存备份配置
type CacheBackupConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	Strategy           string        `json:"strategy" yaml:"strategy"` // "full", "incremental", "differential"
	BackupInterval     time.Duration `json:"backup_interval" yaml:"backup_interval"`
	BackupPath         string        `json:"backup_path" yaml:"backup_path"`
	RetentionCount     int           `json:"retention_count" yaml:"retention_count"`
	CompressionEnable  bool          `json:"compression_enable" yaml:"compression_enable"`
	EncryptionEnable   bool          `json:"encryption_enable" yaml:"encryption_enable"`
	VerificationEnable bool          `json:"verification_enable" yaml:"verification_enable"`
}

// CacheSecurityConfig 缓存安全配置
type CacheSecurityConfig struct {
	EncryptionEnable    bool                     `json:"encryption_enable" yaml:"encryption_enable"`
	EncryptionAlgorithm string                   `json:"encryption_algorithm" yaml:"encryption_algorithm"` // "AES-256", "ChaCha20"
	KeyRotationEnable   bool                     `json:"key_rotation_enable" yaml:"key_rotation_enable"`
	KeyRotationInterval time.Duration            `json:"key_rotation_interval" yaml:"key_rotation_interval"`
	AccessControl       CacheAccessControlConfig `json:"access_control" yaml:"access_control"`
	AuditLogging        CacheAuditLoggingConfig  `json:"audit_logging" yaml:"audit_logging"`
}

// CacheAccessControlConfig 缓存访问控制配置
type CacheAccessControlConfig struct {
	Enable               bool                `json:"enable" yaml:"enable"`
	AuthenticationEnable bool                `json:"authentication_enable" yaml:"authentication_enable"`
	AuthorizationEnable  bool                `json:"authorization_enable" yaml:"authorization_enable"`
	Roles                []string            `json:"roles" yaml:"roles"`
	Permissions          map[string][]string `json:"permissions" yaml:"permissions"`
	SessionTimeout       time.Duration       `json:"session_timeout" yaml:"session_timeout"`
}

// CacheAuditLoggingConfig 缓存审计日志配置
type CacheAuditLoggingConfig struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	LogLevel          string        `json:"log_level" yaml:"log_level"` // "DEBUG", "INFO", "WARN", "ERROR"
	LogPath           string        `json:"log_path" yaml:"log_path"`
	LogFormat         string        `json:"log_format" yaml:"log_format"` // "json", "text"
	LoggedOperations  []string      `json:"logged_operations" yaml:"logged_operations"`
	RetentionPeriod   time.Duration `json:"retention_period" yaml:"retention_period"`
	CompressionEnable bool          `json:"compression_enable" yaml:"compression_enable"`
}

// CacheMonitoringConfig 缓存监控配置
type CacheMonitoringConfig struct {
	Enable    bool                 `json:"enable" yaml:"enable"`
	Metrics   CacheMetricsConfig   `json:"metrics" yaml:"metrics"`
	Alerting  CacheAlertingConfig  `json:"alerting" yaml:"alerting"`
	Dashboard CacheDashboardConfig `json:"dashboard" yaml:"dashboard"`
	Profiling CacheProfilingConfig `json:"profiling" yaml:"profiling"`
}

// CacheMetricsConfig 缓存指标配置
type CacheMetricsConfig struct {
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
	RetentionPeriod    time.Duration `json:"retention_period" yaml:"retention_period"`
	HitRateTracking    bool          `json:"hit_rate_tracking" yaml:"hit_rate_tracking"`
	LatencyTracking    bool          `json:"latency_tracking" yaml:"latency_tracking"`
	MemoryTracking     bool          `json:"memory_tracking" yaml:"memory_tracking"`
	ThroughputTracking bool          `json:"throughput_tracking" yaml:"throughput_tracking"`
	ErrorTracking      bool          `json:"error_tracking" yaml:"error_tracking"`
	CustomMetrics      []string      `json:"custom_metrics" yaml:"custom_metrics"`
}

// CacheAlertingConfig 缓存告警配置
type CacheAlertingConfig struct {
	Enable       bool                    `json:"enable" yaml:"enable"`
	Rules        []CacheAlertRule        `json:"rules" yaml:"rules"`
	Notification CacheNotificationConfig `json:"notification" yaml:"notification"`
	Escalation   CacheEscalationConfig   `json:"escalation" yaml:"escalation"`
}

// CacheAlertRule 缓存告警规则
type CacheAlertRule struct {
	Name        string        `json:"name" yaml:"name"`
	Metric      string        `json:"metric" yaml:"metric"`
	Operator    string        `json:"operator" yaml:"operator"` // ">", "<", ">=", "<=", "==", "!="
	Threshold   float64       `json:"threshold" yaml:"threshold"`
	Duration    time.Duration `json:"duration" yaml:"duration"`
	Severity    string        `json:"severity" yaml:"severity"` // "critical", "warning", "info"
	Description string        `json:"description" yaml:"description"`
	Enabled     bool          `json:"enabled" yaml:"enabled"`
}

// CacheNotificationConfig 缓存通知配置
type CacheNotificationConfig struct {
	Channels      []string                       `json:"channels" yaml:"channels"` // "email", "slack", "webhook", "sms"
	EmailConfig   EmailNotificationConfig        `json:"email_config" yaml:"email_config"`
	SlackConfig   SlackNotificationConfig        `json:"slack_config" yaml:"slack_config"`
	WebhookConfig WebhookNotificationConfig      `json:"webhook_config" yaml:"webhook_config"`
	RateLimiting  NotificationRateLimitingConfig `json:"rate_limiting" yaml:"rate_limiting"`
}

// EmailNotificationConfig 邮件通知配置
type EmailNotificationConfig struct {
	SMTPServer      string   `json:"smtp_server" yaml:"smtp_server"`
	SMTPPort        int      `json:"smtp_port" yaml:"smtp_port"`
	Username        string   `json:"username" yaml:"username"`
	Password        string   `json:"password" yaml:"password"`
	FromAddress     string   `json:"from_address" yaml:"from_address"`
	ToAddresses     []string `json:"to_addresses" yaml:"to_addresses"`
	SubjectTemplate string   `json:"subject_template" yaml:"subject_template"`
	BodyTemplate    string   `json:"body_template" yaml:"body_template"`
	TLSEnable       bool     `json:"tls_enable" yaml:"tls_enable"`
}

// SlackNotificationConfig Slack通知配置
type SlackNotificationConfig struct {
	WebhookURL      string `json:"webhook_url" yaml:"webhook_url"`
	Channel         string `json:"channel" yaml:"channel"`
	Username        string `json:"username" yaml:"username"`
	IconEmoji       string `json:"icon_emoji" yaml:"icon_emoji"`
	MessageTemplate string `json:"message_template" yaml:"message_template"`
}

// WebhookNotificationConfig Webhook通知配置
type WebhookNotificationConfig struct {
	URL             string            `json:"url" yaml:"url"`
	Method          string            `json:"method" yaml:"method"` // "POST", "PUT", "PATCH"
	Headers         map[string]string `json:"headers" yaml:"headers"`
	PayloadTemplate string            `json:"payload_template" yaml:"payload_template"`
	Timeout         time.Duration     `json:"timeout" yaml:"timeout"`
	RetryCount      int               `json:"retry_count" yaml:"retry_count"`
}

// NotificationRateLimitingConfig 通知限流配置
type NotificationRateLimitingConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	MaxNotifications int           `json:"max_notifications" yaml:"max_notifications"`
	TimeWindow       time.Duration `json:"time_window" yaml:"time_window"`
	CooldownPeriod   time.Duration `json:"cooldown_period" yaml:"cooldown_period"`
	GroupingEnable   bool          `json:"grouping_enable" yaml:"grouping_enable"`
	GroupingWindow   time.Duration `json:"grouping_window" yaml:"grouping_window"`
}

// CacheEscalationConfig 缓存升级配置
type CacheEscalationConfig struct {
	Enable          bool                   `json:"enable" yaml:"enable"`
	Levels          []CacheEscalationLevel `json:"levels" yaml:"levels"`
	AutoEscalation  bool                   `json:"auto_escalation" yaml:"auto_escalation"`
	EscalationDelay time.Duration          `json:"escalation_delay" yaml:"escalation_delay"`
}

// CacheEscalationLevel 缓存升级级别
type CacheEscalationLevel struct {
	Level      int           `json:"level" yaml:"level"`
	Name       string        `json:"name" yaml:"name"`
	Contacts   []string      `json:"contacts" yaml:"contacts"`
	Channels   []string      `json:"channels" yaml:"channels"`
	Delay      time.Duration `json:"delay" yaml:"delay"`
	Conditions []string      `json:"conditions" yaml:"conditions"`
}

// CacheDashboardConfig 缓存仪表板配置
type CacheDashboardConfig struct {
	Enable               bool          `json:"enable" yaml:"enable"`
	Port                 int           `json:"port" yaml:"port"`
	RefreshInterval      time.Duration `json:"refresh_interval" yaml:"refresh_interval"`
	Charts               []string      `json:"charts" yaml:"charts"`
	CustomDashboards     []string      `json:"custom_dashboards" yaml:"custom_dashboards"`
	AuthenticationEnable bool          `json:"authentication_enable" yaml:"authentication_enable"`
}

// CacheProfilingConfig 缓存性能分析配置
type CacheProfilingConfig struct {
	Enable          bool          `json:"enable" yaml:"enable"`
	SamplingRate    float64       `json:"sampling_rate" yaml:"sampling_rate"`
	ProfileTypes    []string      `json:"profile_types" yaml:"profile_types"` // "cpu", "memory", "goroutine", "block"
	OutputPath      string        `json:"output_path" yaml:"output_path"`
	RetentionPeriod time.Duration `json:"retention_period" yaml:"retention_period"`
	AutoAnalysis    bool          `json:"auto_analysis" yaml:"auto_analysis"`
}

// GetDefaultCacheConfig 获取默认缓存配置
func GetDefaultCacheConfig() *CacheStrategyConfig {
	return &CacheStrategyConfig{
		ResultCache: ResultCacheConfig{
			Enable:               true,
			MaxSize:              1024 * 1024 * 1024, // 1GB
			MaxEntries:           100000,
			TTL:                  30 * time.Minute,
			EvictionPolicy:       "LRU",
			CompressionEnable:    true,
			CompressionAlgorithm: "lz4",
			Persistence: ResultCachePersistence{
				Enable:           false,
				SyncInterval:     5 * time.Minute,
				CompressionLevel: 6,
				BackupEnable:     false,
				RecoveryMode:     "auto",
			},
			Distribution: ResultCacheDistribution{
				Enable:            false,
				Strategy:          "replicated",
				ReplicationFactor: 2,
				ConsistencyLevel:  "eventual",
			},
			Optimization: ResultCacheOptimization{
				Prefetching: CachePrefetchingConfig{
					Enable:            true,
					Strategy:          "pattern_based",
					LookaheadSize:     10,
					PrefetchRatio:     0.2,
					AdaptiveEnable:    true,
					LearningWindow:    1 * time.Hour,
					AccuracyThreshold: 0.7,
				},
				Compaction: CacheCompactionConfig{
					Enable:             true,
					TriggerThreshold:   0.8,
					CompactionInterval: 1 * time.Hour,
					Strategy:           "hybrid",
					BackgroundEnable:   true,
				},
				Deduplication: CacheDeduplicationConfig{
					Enable:              true,
					Method:              "hash_based",
					HashFunction:        "xxhash",
					SimilarityThreshold: 0.95,
					WindowSize:          1000,
					CleanupInterval:     30 * time.Minute,
				},
			},
			Monitoring: CacheMonitoringConfig{
				Enable: true,
				Metrics: CacheMetricsConfig{
					CollectionInterval: 10 * time.Second,
					RetentionPeriod:    24 * time.Hour,
					HitRateTracking:    true,
					LatencyTracking:    true,
					MemoryTracking:     true,
					ThroughputTracking: true,
					ErrorTracking:      true,
				},
			},
		},
		VectorCache: VectorCacheConfig{
			Enable:         true,
			MaxSize:        2048 * 1024 * 1024, // 2GB
			MaxVectors:     1000000,
			TTL:            1 * time.Hour,
			EvictionPolicy: "LRU",
			HotDataStrategy: VectorHotDataStrategy{
				Enable:          true,
				DetectionMethod: "access_frequency",
				HotThreshold:    0.8,
				ColdThreshold:   0.2,
				AnalysisWindow:  1 * time.Hour,
				PromotionPolicy: "batch",
				DemotionPolicy:  "batch",
				ResidentMemory:  true,
			},
			CompressionConfig: VectorCompressionConfig{
				Enable:           true,
				Algorithm:        "pq",
				CompressionRatio: 0.25,
				QualityThreshold: 0.9,
				AdaptiveEnable:   true,
				CodebookSize:     256,
				SubvectorCount:   8,
			},
			Tiering: VectorCacheTiering{
				Enable: true,
				Levels: []VectorCacheLevel{
					{
						Level:             1,
						Name:              "L1_Memory",
						StorageType:       "memory",
						MaxSize:           512 * 1024 * 1024, // 512MB
						MaxVectors:        100000,
						AccessLatency:     1 * time.Microsecond,
						Bandwidth:         100 * 1024 * 1024 * 1024, // 100GB/s
						CompressionEnable: false,
						ReplicationFactor: 1,
					},
					{
						Level:             2,
						Name:              "L2_SSD",
						StorageType:       "ssd",
						MaxSize:           10 * 1024 * 1024 * 1024, // 10GB
						MaxVectors:        1000000,
						AccessLatency:     100 * time.Microsecond,
						Bandwidth:         3 * 1024 * 1024 * 1024, // 3GB/s
						CompressionEnable: true,
						ReplicationFactor: 1,
					},
				},
				PromotionPolicy: VectorPromotionPolicy{
					Strategy:          "hybrid",
					AccessThreshold:   5,
					TimeThreshold:     10 * time.Minute,
					BatchSize:         100,
					PromotionInterval: 5 * time.Minute,
					CostModel:         "weighted",
				},
				EvictionPolicy: VectorEvictionPolicy{
					Strategy:         "lru",
					EvictionRatio:    0.1,
					BatchSize:        50,
					EvictionInterval: 10 * time.Minute,
					ProtectedRatio:   0.1,
					CostModel:        "simple",
				},
			},
			Prefetching: VectorPrefetchingConfig{
				Enable:            true,
				Strategy:          "neighbor_based",
				NeighborCount:     10,
				PrefetchDepth:     2,
				AccuracyThreshold: 0.7,
				LearningWindow:    1 * time.Hour,
				AdaptiveEnable:    true,
			},
			Monitoring: CacheMonitoringConfig{
				Enable: true,
				Metrics: CacheMetricsConfig{
					CollectionInterval: 10 * time.Second,
					RetentionPeriod:    24 * time.Hour,
					HitRateTracking:    true,
					LatencyTracking:    true,
					MemoryTracking:     true,
					ThroughputTracking: true,
					ErrorTracking:      true,
				},
			},
		},
		IndexCache: IndexCacheConfig{
			Enable:         true,
			MaxSize:        4096 * 1024 * 1024, // 4GB
			MaxIndices:     100,
			TTL:            2 * time.Hour,
			EvictionPolicy: "usage_based",
			Preloading: IndexPreloadingConfig{
				Enable:             true,
				Strategy:           "frequent",
				LoadingParallelism: 4,
				WarmupEnable:       true,
			},
			PartialCaching: IndexPartialCachingConfig{
				Enable:             true,
				Strategy:           "adaptive",
				CacheRatio:         0.3,
				HotRegionDetection: true,
				RegionSize:         1000,
				OverlapRatio:       0.1,
			},
		},
	}
}
