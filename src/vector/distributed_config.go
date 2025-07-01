package vector

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/logger"
	"encoding/json"
	"fmt"
	"gopkg.in/yaml.v2"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"
)

// DistributedConfig 分布式配置结构
type DistributedConfig struct {
	// 索引结构选择配置
	IndexConfig IndexSelectionConfig `json:"index_config" yaml:"index_config"`

	// 分布式架构配置
	ArchitectureConfig DistributedArchitectureConfig `json:"architecture_config" yaml:"architecture_config"`

	// 性能优化配置
	PerformanceConfig PerformanceOptimizationConfig `json:"performance_config" yaml:"performance_config"`

	// 硬件加速配置
	HardwareConfig HardwareAccelerationConfig `json:"hardware_config" yaml:"hardware_config"`

	// 缓存策略配置
	CacheConfig CacheStrategyConfig `json:"cache_config" yaml:"cache_config"`

	// 监控配置
	MonitoringConfig MonitoringConfig `json:"monitoring_config" yaml:"monitoring_config"`

	// 自动扩缩容配置
	AutoScalingConfig AutoScalingConfig `json:"auto_scaling_config" yaml:"auto_scaling_config"`
}

// IndexSelectionConfig 索引结构选择配置
type IndexSelectionConfig struct {
	// ANN算法配置
	HNSWConfig HNSWIndexConfig `json:"hnsw_config" yaml:"hnsw_config"`
	IVFConfig  IVFIndexConfig  `json:"ivf_config" yaml:"ivf_config"`
	PQConfig   PQIndexConfig   `json:"pq_config" yaml:"pq_config"`
	LSHConfig  LSHIndexConfig  `json:"lsh_config" yaml:"lsh_config"`

	// 自适应索引选择
	AdaptiveSelection AdaptiveIndexSelection `json:"adaptive_selection" yaml:"adaptive_selection"`

	// 混合索引策略
	HybridStrategy HybridIndexStrategy `json:"hybrid_strategy" yaml:"hybrid_strategy"`
}

// HNSWIndexConfig HNSW索引配置
type HNSWIndexConfig struct {
	Enable          bool    `json:"enable" yaml:"enable"`
	MaxConnections  int     `json:"max_connections" yaml:"max_connections"`
	EfConstruction  int     `json:"ef_construction" yaml:"ef_construction"`
	EfSearch        int     `json:"ef_search" yaml:"ef_search"`
	MaxLevel        int     `json:"max_level" yaml:"max_level"`
	RecallThreshold float64 `json:"recall_threshold" yaml:"recall_threshold"`
	UsageScenario   string  `json:"usage_scenario" yaml:"usage_scenario"` // "high_recall", "balanced", "fast"
}

// IVFIndexConfig IVF索引配置
type IVFIndexConfig struct {
	Enable             bool    `json:"enable" yaml:"enable"`
	NumClusters        int     `json:"num_clusters" yaml:"num_clusters"`
	Nprobe             int     `json:"nprobe" yaml:"nprobe"`
	TrainingRatio      float64 `json:"training_ratio" yaml:"training_ratio"`
	RebalanceThreshold int     `json:"rebalance_threshold" yaml:"rebalance_threshold"`
	UsageScenario      string  `json:"usage_scenario" yaml:"usage_scenario"` // "large_scale", "balanced", "memory_efficient"
}

// PQIndexConfig PQ索引配置
type PQIndexConfig struct {
	Enable           bool    `json:"enable" yaml:"enable"`
	NumSubVectors    int     `json:"num_sub_vectors" yaml:"num_sub_vectors"`
	NumCentroids     int     `json:"num_centroids" yaml:"num_centroids"`
	CompressionRatio float64 `json:"compression_ratio" yaml:"compression_ratio"`
	UsageScenario    string  `json:"usage_scenario" yaml:"usage_scenario"` // "memory_critical", "balanced", "quality_first"
}

// LSHIndexConfig LSH索引配置
type LSHIndexConfig struct {
	Enable           bool   `json:"enable" yaml:"enable"`
	NumTables        int    `json:"num_tables" yaml:"num_tables"`
	NumHashFunctions int    `json:"num_hash_functions" yaml:"num_hash_functions"`
	BucketSize       int    `json:"bucket_size" yaml:"bucket_size"`
	ProbeRadius      int    `json:"probe_radius" yaml:"probe_radius"`
	UsageScenario    string `json:"usage_scenario" yaml:"usage_scenario"` // "ultra_large", "fast_approximate", "balanced"
}

// AdaptiveIndexSelection 自适应索引选择
type AdaptiveIndexSelection struct {
	Enable                bool                  `json:"enable" yaml:"enable"`
	DataSizeThresholds    DataSizeThresholds    `json:"data_size_thresholds" yaml:"data_size_thresholds"`
	QualityThresholds     QualityThresholds     `json:"quality_thresholds" yaml:"quality_thresholds"`
	PerformanceThresholds PerformanceThresholds `json:"performance_thresholds" yaml:"performance_thresholds"`
	SelectionRules        []IndexSelectionRule  `json:"selection_rules" yaml:"selection_rules"`
}

// DataSizeThresholds 数据规模阈值
type DataSizeThresholds struct {
	SmallDataset  int `json:"small_dataset" yaml:"small_dataset"`   // < 10K
	MediumDataset int `json:"medium_dataset" yaml:"medium_dataset"` // 10K - 1M
	LargeDataset  int `json:"large_dataset" yaml:"large_dataset"`   // 1M - 100M
	UltraLarge    int `json:"ultra_large" yaml:"ultra_large"`       // > 100M
}

// QualityThresholds 质量阈值
type QualityThresholds struct {
	HighQuality       float64 `json:"high_quality" yaml:"high_quality"`             // > 0.95
	MediumQuality     float64 `json:"medium_quality" yaml:"medium_quality"`         // 0.85 - 0.95
	AcceptableQuality float64 `json:"acceptable_quality" yaml:"acceptable_quality"` // 0.7 - 0.85
}

// PerformanceThresholds 性能阈值
type PerformanceThresholds struct {
	LowLatency    time.Duration `json:"low_latency" yaml:"low_latency"`       // < 10ms
	MediumLatency time.Duration `json:"medium_latency" yaml:"medium_latency"` // 10ms - 100ms
	HighLatency   time.Duration `json:"high_latency" yaml:"high_latency"`     // > 100ms
}

// IndexSelectionRule 索引选择规则
type IndexSelectionRule struct {
	Condition        IndexCondition `json:"condition" yaml:"condition"`
	RecommendedIndex string         `json:"recommended_index" yaml:"recommended_index"`
	Priority         int            `json:"priority" yaml:"priority"`
	Description      string         `json:"description" yaml:"description"`
}

// IndexCondition 索引条件
type IndexCondition struct {
	DataSizeRange    []int           `json:"data_size_range" yaml:"data_size_range"`
	DimensionRange   []int           `json:"dimension_range" yaml:"dimension_range"`
	QualityRange     []float64       `json:"quality_range" yaml:"quality_range"`
	LatencyRange     []time.Duration `json:"latency_range" yaml:"latency_range"`
	MemoryConstraint int64           `json:"memory_constraint" yaml:"memory_constraint"`
}

// HybridIndexStrategy 混合索引策略
type HybridIndexStrategy struct {
	Enable              bool                      `json:"enable" yaml:"enable"`
	PrimaryIndex        string                    `json:"primary_index" yaml:"primary_index"`
	SecondaryIndex      string                    `json:"secondary_index" yaml:"secondary_index"`
	SwitchingThresholds HybridSwitchingThresholds `json:"switching_thresholds" yaml:"switching_thresholds"`
	LoadBalancing       HybridLoadBalancing       `json:"load_balancing" yaml:"load_balancing"`
}

// HybridSwitchingThresholds 混合索引切换阈值
type HybridSwitchingThresholds struct {
	CPUUsageThreshold    float64       `json:"cpu_usage_threshold" yaml:"cpu_usage_threshold"`
	MemoryUsageThreshold float64       `json:"memory_usage_threshold" yaml:"memory_usage_threshold"`
	LatencyThreshold     time.Duration `json:"latency_threshold" yaml:"latency_threshold"`
	QPSThreshold         int           `json:"qps_threshold" yaml:"qps_threshold"`
}

// HybridLoadBalancing 混合索引负载均衡
type HybridLoadBalancing struct {
	Strategy        string         `json:"strategy" yaml:"strategy"` // "round_robin", "weighted", "adaptive"
	PrimaryWeight   float64        `json:"primary_weight" yaml:"primary_weight"`
	SecondaryWeight float64        `json:"secondary_weight" yaml:"secondary_weight"`
	AdaptiveRules   []AdaptiveRule `json:"adaptive_rules" yaml:"adaptive_rules"`
}

// AdaptiveRule 自适应规则
type AdaptiveRule struct {
	Condition string  `json:"condition" yaml:"condition"`
	Action    string  `json:"action" yaml:"action"`
	Weight    float64 `json:"weight" yaml:"weight"`
}

// DistributedArchitectureConfig 分布式架构配置
type DistributedArchitectureConfig struct {
	// 分片策略配置
	ShardingConfig ShardingConfig `json:"sharding_config" yaml:"sharding_config"`

	// 计算存储分离配置
	ComputeStorageSeparation ComputeStorageSeparationConfig `json:"compute_storage_separation" yaml:"compute_storage_separation"`

	// 节点管理配置
	NodeManagement NodeManagementConfig `json:"node_management" yaml:"node_management"`

	// 负载均衡配置
	LoadBalancing LoadBalancingConfig `json:"load_balancing" yaml:"load_balancing"`

	// 一致性哈希配置
	ConsistentHashing ConsistentHashingConfig `json:"consistent_hashing" yaml:"consistent_hashing"`
}

// ShardingConfig 分片配置
type ShardingConfig struct {
	Strategy          string                 `json:"strategy" yaml:"strategy"` // "range", "hash", "cluster", "adaptive"
	NumShards         int                    `json:"num_shards" yaml:"num_shards"`
	ReplicationFactor int                    `json:"replication_factor" yaml:"replication_factor"`
	RebalanceConfig   RebalanceConfig        `json:"rebalance_config" yaml:"rebalance_config"`
	RangeSharding     RangeShardingConfig    `json:"range_sharding" yaml:"range_sharding"`
	HashSharding      HashShardingConfig     `json:"hash_sharding" yaml:"hash_sharding"`
	ClusterSharding   ClusterShardingConfig  `json:"cluster_sharding" yaml:"cluster_sharding"`
	AdaptiveSharding  AdaptiveShardingConfig `json:"adaptive_sharding" yaml:"adaptive_sharding"`
}

// RebalanceConfig 重平衡配置
type RebalanceConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	TriggerThreshold   float64       `json:"trigger_threshold" yaml:"trigger_threshold"`
	CheckInterval      time.Duration `json:"check_interval" yaml:"check_interval"`
	MaxConcurrentMoves int           `json:"max_concurrent_moves" yaml:"max_concurrent_moves"`
	DataMigrationRate  int64         `json:"data_migration_rate" yaml:"data_migration_rate"` // bytes/sec
}

// RangeShardingConfig 范围分片配置
type RangeShardingConfig struct {
	KeyField       string   `json:"key_field" yaml:"key_field"`
	Ranges         []string `json:"ranges" yaml:"ranges"`
	AutoSplit      bool     `json:"auto_split" yaml:"auto_split"`
	SplitThreshold int      `json:"split_threshold" yaml:"split_threshold"`
}

// HashShardingConfig 哈希分片配置
type HashShardingConfig struct {
	HashFunction   string `json:"hash_function" yaml:"hash_function"` // "md5", "sha1", "fnv", "murmur3"
	VirtualNodes   int    `json:"virtual_nodes" yaml:"virtual_nodes"`
	ConsistentHash bool   `json:"consistent_hash" yaml:"consistent_hash"`
}

// ClusterShardingConfig 聚类分片配置
type ClusterShardingConfig struct {
	ClusteringAlgorithm   string        `json:"clustering_algorithm" yaml:"clustering_algorithm"` // "kmeans", "hierarchical", "dbscan"
	NumClusters           int           `json:"num_clusters" yaml:"num_clusters"`
	ReClusterThreshold    float64       `json:"re_cluster_threshold" yaml:"re_cluster_threshold"`
	ClusterUpdateInterval time.Duration `json:"cluster_update_interval" yaml:"cluster_update_interval"`
}

// AdaptiveShardingConfig 自适应分片配置
type AdaptiveShardingConfig struct {
	Enable           bool                   `json:"enable" yaml:"enable"`
	Metrics          []string               `json:"metrics" yaml:"metrics"` // "load", "latency", "throughput", "memory"
	DecisionInterval time.Duration          `json:"decision_interval" yaml:"decision_interval"`
	ShardingRules    []AdaptiveShardingRule `json:"sharding_rules" yaml:"sharding_rules"`
	FallbackStrategy string                 `json:"fallback_strategy" yaml:"fallback_strategy"`
}

// AdaptiveShardingRule 自适应分片规则
type AdaptiveShardingRule struct {
	Condition      string        `json:"condition" yaml:"condition"`
	TargetStrategy string        `json:"target_strategy" yaml:"target_strategy"`
	Priority       int           `json:"priority" yaml:"priority"`
	CooldownPeriod time.Duration `json:"cooldown_period" yaml:"cooldown_period"`
}

// ComputeStorageSeparationConfig 计算存储分离配置
type ComputeStorageSeparationConfig struct {
	Enable           bool                   `json:"enable" yaml:"enable"`
	ComputeNodes     ComputeNodesConfig     `json:"compute_nodes" yaml:"compute_nodes"`
	StorageNodes     StorageNodesConfig     `json:"storage_nodes" yaml:"storage_nodes"`
	CoordinatorNodes CoordinatorNodesConfig `json:"coordinator_nodes" yaml:"coordinator_nodes"`
	MetadataService  MetadataServiceConfig  `json:"metadata_service" yaml:"metadata_service"`
	Networking       NetworkingConfig       `json:"networking" yaml:"networking"`
}

// ComputeNodesConfig 计算节点配置
type ComputeNodesConfig struct {
	MinNodes      int             `json:"min_nodes" yaml:"min_nodes"`
	MaxNodes      int             `json:"max_nodes" yaml:"max_nodes"`
	NodeSpecs     NodeSpecs       `json:"node_specs" yaml:"node_specs"`
	ScalingPolicy ScalingPolicy   `json:"scaling_policy" yaml:"scaling_policy"`
	CacheConfig   NodeCacheConfig `json:"cache_config" yaml:"cache_config"`
}

// StorageNodesConfig 存储节点配置
type StorageNodesConfig struct {
	MinNodes          int               `json:"min_nodes" yaml:"min_nodes"`
	MaxNodes          int               `json:"max_nodes" yaml:"max_nodes"`
	NodeSpecs         NodeSpecs         `json:"node_specs" yaml:"node_specs"`
	ReplicationFactor int               `json:"replication_factor" yaml:"replication_factor"`
	StorageType       string            `json:"storage_type" yaml:"storage_type"` // "ssd", "nvme", "pmem", "hybrid"
	CompressionConfig CompressionConfig `json:"compression_config" yaml:"compression_config"`
}

// CoordinatorNodesConfig 协调节点配置
type CoordinatorNodesConfig struct {
	NumNodes       int               `json:"num_nodes" yaml:"num_nodes"`
	NodeSpecs      NodeSpecs         `json:"node_specs" yaml:"node_specs"`
	ElectionConfig ElectionConfig    `json:"election_config" yaml:"election_config"`
	HealthCheck    HealthCheckConfig `json:"health_check" yaml:"health_check"`
}

// MetadataServiceConfig 元数据服务配置
type MetadataServiceConfig struct {
	Type             string       `json:"type" yaml:"type"` // "etcd", "consul", "zookeeper", "embedded"
	Endpoints        []string     `json:"endpoints" yaml:"endpoints"`
	BackupConfig     BackupConfig `json:"backup_config" yaml:"backup_config"`
	ConsistencyLevel string       `json:"consistency_level" yaml:"consistency_level"`
}

// NetworkingConfig 网络配置
type NetworkingConfig struct {
	Protocol       string               `json:"protocol" yaml:"protocol"` // "tcp", "rdma", "infiniband"
	Compression    bool                 `json:"compression" yaml:"compression"`
	Encryption     bool                 `json:"encryption" yaml:"encryption"`
	BandwidthLimit int64                `json:"bandwidth_limit" yaml:"bandwidth_limit"` // bytes/sec
	ConnectionPool ConnectionPoolConfig `json:"connection_pool" yaml:"connection_pool"`
	RDMAConfig     RDMAConfig           `json:"rdma_config" yaml:"rdma_config"`
}

// NodeSpecs 节点规格
type NodeSpecs struct {
	CPUCores    int `json:"cpu_cores" yaml:"cpu_cores"`
	MemoryGB    int `json:"memory_gb" yaml:"memory_gb"`
	StorageGB   int `json:"storage_gb" yaml:"storage_gb"`
	NetworkGbps int `json:"network_gbps" yaml:"network_gbps"`
	GPUCount    int `json:"gpu_count" yaml:"gpu_count"`
}

// NodeCacheConfig 节点缓存配置
type NodeCacheConfig struct {
	Enable   bool          `json:"enable" yaml:"enable"`
	SizeGB   int           `json:"size_gb" yaml:"size_gb"`
	Strategy string        `json:"strategy" yaml:"strategy"` // "lru", "lfu", "arc"
	TTL      time.Duration `json:"ttl" yaml:"ttl"`
}

// CompressionConfig 压缩配置
type CompressionConfig struct {
	Enable    bool   `json:"enable" yaml:"enable"`
	Algorithm string `json:"algorithm" yaml:"algorithm"` // "gzip", "lz4", "snappy", "zstd"
	Level     int    `json:"level" yaml:"level"`
	Threshold int    `json:"threshold" yaml:"threshold"` // 压缩阈值(bytes)
}

// ElectionConfig 选举配置
type ElectionConfig struct {
	Algorithm         string        `json:"algorithm" yaml:"algorithm"` // "raft", "bully", "ring"
	ElectionTimeout   time.Duration `json:"election_timeout" yaml:"election_timeout"`
	HeartbeatInterval time.Duration `json:"heartbeat_interval" yaml:"heartbeat_interval"`
}

// BackupConfig 备份配置
type BackupConfig struct {
	Enable          bool          `json:"enable" yaml:"enable"`
	Interval        time.Duration `json:"interval" yaml:"interval"`
	RetentionPeriod time.Duration `json:"retention_period" yaml:"retention_period"`
	StoragePath     string        `json:"storage_path" yaml:"storage_path"`
	Compression     bool          `json:"compression" yaml:"compression"`
}

// ConnectionPoolConfig 连接池配置
type ConnectionPoolConfig struct {
	MaxConnections int           `json:"max_connections" yaml:"max_connections"`
	MinConnections int           `json:"min_connections" yaml:"min_connections"`
	IdleTimeout    time.Duration `json:"idle_timeout" yaml:"idle_timeout"`
	ConnectTimeout time.Duration `json:"connect_timeout" yaml:"connect_timeout"`
	Keepalive      bool          `json:"keepalive" yaml:"keepalive"`
}

// RDMAConfig RDMA配置
type RDMAConfig struct {
	Enable        bool   `json:"enable" yaml:"enable"`
	DeviceName    string `json:"device_name" yaml:"device_name"`
	PortNum       int    `json:"port_num" yaml:"port_num"`
	QueueDepth    int    `json:"queue_depth" yaml:"queue_depth"`
	MaxInlineData int    `json:"max_inline_data" yaml:"max_inline_data"`
}

// NodeManagementConfig 节点管理配置
type NodeManagementConfig struct {
	Discovery    ServiceDiscoveryConfig `json:"discovery" yaml:"discovery"`
	Registration NodeRegistrationConfig `json:"registration" yaml:"registration"`
	Monitoring   NodeMonitoringConfig   `json:"monitoring" yaml:"monitoring"`
	Failover     FailoverConfig         `json:"failover" yaml:"failover"`
}

// ServiceDiscoveryConfig 服务发现配置
type ServiceDiscoveryConfig struct {
	Type            string        `json:"type" yaml:"type"` // "dns", "consul", "etcd", "kubernetes"
	Endpoints       []string      `json:"endpoints" yaml:"endpoints"`
	RefreshInterval time.Duration `json:"refresh_interval" yaml:"refresh_interval"`
	HealthCheck     bool          `json:"health_check" yaml:"health_check"`
}

// NodeRegistrationConfig 节点注册配置
type NodeRegistrationConfig struct {
	AutoRegister    bool              `json:"auto_register" yaml:"auto_register"`
	RegistryTTL     time.Duration     `json:"registry_ttl" yaml:"registry_ttl"`
	RenewalInterval time.Duration     `json:"renewal_interval" yaml:"renewal_interval"`
	Metadata        map[string]string `json:"metadata" yaml:"metadata"`
}

// NodeMonitoringConfig 节点监控配置
type NodeMonitoringConfig struct {
	MetricsInterval time.Duration `json:"metrics_interval" yaml:"metrics_interval"`
	MetricsEndpoint string        `json:"metrics_endpoint" yaml:"metrics_endpoint"`
	AlertRules      []AlertRule   `json:"alert_rules" yaml:"alert_rules"`
}

// FailoverConfig 故障转移配置
type FailoverConfig struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	DetectionInterval time.Duration `json:"detection_interval" yaml:"detection_interval"`
	FailoverTimeout   time.Duration `json:"failover_timeout" yaml:"failover_timeout"`
	MaxRetries        int           `json:"max_retries" yaml:"max_retries"`
	BackoffStrategy   string        `json:"backoff_strategy" yaml:"backoff_strategy"` // "linear", "exponential", "fixed"
}

// LoadBalancingConfig 负载均衡配置
type LoadBalancingConfig struct {
	Strategy            string                    `json:"strategy" yaml:"strategy"` // "round_robin", "weighted", "least_load", "latency_based", "consistent_hash"
	HealthCheckInterval time.Duration             `json:"health_check_interval" yaml:"health_check_interval"`
	WeightedConfig      WeightedLoadBalancing     `json:"weighted_config" yaml:"weighted_config"`
	LatencyConfig       LatencyBasedLoadBalancing `json:"latency_config" yaml:"latency_config"`
	StickySession       StickySessionConfig       `json:"sticky_session" yaml:"sticky_session"`
}

// WeightedLoadBalancing 加权负载均衡配置
type WeightedLoadBalancing struct {
	WeightUpdateInterval time.Duration  `json:"weight_update_interval" yaml:"weight_update_interval"`
	WeightFactors        []WeightFactor `json:"weight_factors" yaml:"weight_factors"`
}

// WeightFactor 权重因子
type WeightFactor struct {
	Metric string  `json:"metric" yaml:"metric"`
	Weight float64 `json:"weight" yaml:"weight"`
}

// LatencyBasedLoadBalancing 基于延迟的负载均衡配置
type LatencyBasedLoadBalancing struct {
	LatencyWindow    time.Duration `json:"latency_window" yaml:"latency_window"`
	LatencyThreshold time.Duration `json:"latency_threshold" yaml:"latency_threshold"`
	PenaltyFactor    float64       `json:"penalty_factor" yaml:"penalty_factor"`
	RecoveryFactor   float64       `json:"recovery_factor" yaml:"recovery_factor"`
}

// StickySessionConfig 粘性会话配置
type StickySessionConfig struct {
	Enable     bool          `json:"enable" yaml:"enable"`
	Method     string        `json:"method" yaml:"method"` // "cookie", "ip_hash", "header"
	TTL        time.Duration `json:"ttl" yaml:"ttl"`
	CookieName string        `json:"cookie_name" yaml:"cookie_name"`
	HeaderName string        `json:"header_name" yaml:"header_name"`
}

// ConsistentHashingConfig 一致性哈希配置
type ConsistentHashingConfig struct {
	Enable            bool   `json:"enable" yaml:"enable"`
	VirtualNodes      int    `json:"virtual_nodes" yaml:"virtual_nodes"`
	HashFunction      string `json:"hash_function" yaml:"hash_function"`
	ReplicationFactor int    `json:"replication_factor" yaml:"replication_factor"`
}

// GetDefaultDistributedConfig 获取默认分布式配置
func GetDefaultDistributedConfig() *DistributedConfig {
	return &DistributedConfig{
		IndexConfig: IndexSelectionConfig{
			HNSWConfig: HNSWIndexConfig{
				Enable:          true,
				MaxConnections:  32,
				EfConstruction:  200,
				EfSearch:        100,
				MaxLevel:        6,
				RecallThreshold: 0.95,
				UsageScenario:   "balanced",
			},
			IVFConfig: IVFIndexConfig{
				Enable:             true,
				NumClusters:        100,
				Nprobe:             10,
				TrainingRatio:      0.1,
				RebalanceThreshold: 1000,
				UsageScenario:      "large_scale",
			},
			PQConfig: PQIndexConfig{
				Enable:           false,
				NumSubVectors:    8,
				NumCentroids:     256,
				CompressionRatio: 0.25,
				UsageScenario:    "memory_critical",
			},
			LSHConfig: LSHIndexConfig{
				Enable:           false,
				NumTables:        10,
				NumHashFunctions: 8,
				BucketSize:       100,
				ProbeRadius:      2,
				UsageScenario:    "ultra_large",
			},
			AdaptiveSelection: AdaptiveIndexSelection{
				Enable: true,
				DataSizeThresholds: DataSizeThresholds{
					SmallDataset:  10000,
					MediumDataset: 1000000,
					LargeDataset:  100000000,
					UltraLarge:    1000000000,
				},
				QualityThresholds: QualityThresholds{
					HighQuality:       0.95,
					MediumQuality:     0.85,
					AcceptableQuality: 0.7,
				},
				PerformanceThresholds: PerformanceThresholds{
					LowLatency:    10 * time.Millisecond,
					MediumLatency: 100 * time.Millisecond,
					HighLatency:   1000 * time.Millisecond,
				},
			},
			HybridStrategy: HybridIndexStrategy{
				Enable:         false,
				PrimaryIndex:   "hnsw",
				SecondaryIndex: "ivf",
			},
		},
		ArchitectureConfig: DistributedArchitectureConfig{
			ShardingConfig: ShardingConfig{
				Strategy:          "hash",
				NumShards:         8,
				ReplicationFactor: 3,
				RebalanceConfig: RebalanceConfig{
					Enable:             true,
					TriggerThreshold:   0.8,
					CheckInterval:      5 * time.Minute,
					MaxConcurrentMoves: 2,
					DataMigrationRate:  100 * 1024 * 1024, // 100MB/s
				},
			},
			ComputeStorageSeparation: ComputeStorageSeparationConfig{
				Enable: true,
				ComputeNodes: ComputeNodesConfig{
					MinNodes: 2,
					MaxNodes: 10,
					NodeSpecs: NodeSpecs{
						CPUCores:    8,
						MemoryGB:    32,
						StorageGB:   100,
						NetworkGbps: 10,
						GPUCount:    1,
					},
				},
				StorageNodes: StorageNodesConfig{
					MinNodes:          3,
					MaxNodes:          20,
					ReplicationFactor: 3,
					StorageType:       "nvme",
				},
			},
			LoadBalancing: LoadBalancingConfig{
				Strategy:            "least_load",
				HealthCheckInterval: 30 * time.Second,
			},
		},
		PerformanceConfig: PerformanceOptimizationConfig{
			QueryAcceleration: QueryAccelerationConfig{
				Enable: true,
				Preprocessing: PreprocessingConfig{
					Normalization: true,
					DimensionReduction: DimensionReductionConfig{
						Enable:    false,
						Method:    "pca",
						TargetDim: 128,
					},
				},
				MultiStageSearch: MultiStageSearchConfig{
					Enable:           true,
					CoarseCandidates: 1000,
					RefinementRatio:  0.1,
				},
			},
		},
		HardwareConfig: HardwareAccelerationConfig{
			GPU: GPUConfig{
				Enable:      false,
				BatchSize:   1000,
				MemoryLimit: 8 * 1024 * 1024 * 1024, // 8GB
			},
			FPGA: FPGAConfig{
				Enable: false,
			},
			PMem: PMemConfig{
				Enable: false,
			},
			RDMA: RDMANetworkConfig{
				Enable: false,
			},
		},
		CacheConfig: CacheStrategyConfig{
			ResultCache: ResultCacheConfig{
				Enable:         true,
				MaxSize:        1024 * 1024 * 1024,
				MaxEntries:     10000,
				TTL:            30 * time.Minute,
				EvictionPolicy: "LRU",
			},
			VectorCache: VectorCacheConfig{
				Enable:         true,
				MaxSize:        2048 * 1024 * 1024,
				MaxVectors:     100000,
				TTL:            60 * time.Minute,
				EvictionPolicy: "LFU",
			},
			IndexCache: IndexCacheConfig{
				Enable:         true,
				MaxSize:        512 * 1024 * 1024,
				MaxIndices:     1000,
				TTL:            120 * time.Minute,
				EvictionPolicy: "LRU",
			},
		},
		MonitoringConfig: MonitoringConfig{
			Metrics: MetricsConfig{
				Enable:             true,
				CollectionInterval: 10 * time.Second,
				RetentionPeriod:    24 * time.Hour,
				Exporter: MetricsExporterConfig{
					Prometheus: PrometheusConfig{
						Enable:         true,
						Endpoint:       ":9090",
						Port:           9090,
						Path:           "/metrics",
						ScrapeInterval: 15 * time.Second,
						Timeout:        10 * time.Second,
					},
				},
			},
		},
		AutoScalingConfig: AutoScalingConfig{
			Enable:   true,
			Strategy: "reactive",
			Metrics: ScalingMetricsConfig{
				CPUUtilization: CPUScalingConfig{
					Enable:             true,
					ScaleUpThreshold:   70.0,
					ScaleDownThreshold: 30.0,
					Weight:             1.0,
				},
				MemoryUtilization: MemoryScalingConfig{
					Enable:             true,
					ScaleUpThreshold:   80.0,
					ScaleDownThreshold: 40.0,
					Weight:             1.0,
				},
			},
			Cooldown: CooldownConfig{
				ScaleUpCooldown:   300 * time.Second,
				ScaleDownCooldown: 600 * time.Second,
			},
		},
	}
}

// ApplyDistributedConfig 应用分布式配置
func (db *VectorDB) ApplyDistributedConfig(config *DistributedConfig) error {
	if config == nil {
		config = GetDefaultDistributedConfig()
	}

	// 应用索引配置
	if err := db.applyIndexConfig(&config.IndexConfig); err != nil {
		return fmt.Errorf("应用索引配置失败: %v", err)
	}

	// 应用架构配置
	if err := db.applyArchitectureConfig(&config.ArchitectureConfig); err != nil {
		return fmt.Errorf("应用架构配置失败: %v", err)
	}

	// 应用性能配置
	if err := db.applyPerformanceConfig(&config.PerformanceConfig); err != nil {
		return fmt.Errorf("应用性能配置失败: %v", err)
	}

	// 应用硬件配置
	if err := db.applyHardwareConfig(&config.HardwareConfig); err != nil {
		return fmt.Errorf("应用硬件配置失败: %v", err)
	}

	// 应用缓存配置
	if err := db.applyCacheConfig(&config.CacheConfig); err != nil {
		return fmt.Errorf("应用缓存配置失败: %v", err)
	}

	// 应用监控配置
	if err := db.applyMonitoringConfig(&config.MonitoringConfig); err != nil {
		return fmt.Errorf("应用监控配置失败: %v", err)
	}

	return nil
}

// applyIndexConfig 应用索引配置
func (db *VectorDB) applyIndexConfig(config *IndexSelectionConfig) error {
	// 配置HNSW
	if config.HNSWConfig.Enable {
		db.useHNSWIndex = true
		db.maxConnections = config.HNSWConfig.MaxConnections
		db.efConstruction = float64(config.HNSWConfig.EfConstruction)
		db.efSearch = float64(config.HNSWConfig.EfSearch)
	}

	// 配置IVF
	if config.IVFConfig.Enable {
		db.numClusters = config.IVFConfig.NumClusters
		if db.ivfConfig == nil {
			db.ivfConfig = &IVFConfig{}
		}
		db.ivfConfig.NumClusters = config.IVFConfig.NumClusters
		db.ivfConfig.Nprobe = config.IVFConfig.Nprobe
		db.ivfConfig.TrainingRatio = config.IVFConfig.TrainingRatio
	}

	// 配置PQ
	if config.PQConfig.Enable {
		db.usePQCompression = true
		db.numSubVectors = config.PQConfig.NumSubVectors
		db.numCentroidsPerSubVector = config.PQConfig.NumCentroids
	}

	// 配置LSH
	if config.LSHConfig.Enable {
		if db.LshConfig == nil {
			db.LshConfig = &LSHConfig{}
		}
		db.LshConfig.NumTables = config.LSHConfig.NumTables
		db.LshConfig.NumHashFunctions = config.LSHConfig.NumHashFunctions
		db.LshConfig.BucketSize = config.LSHConfig.BucketSize
		db.LshConfig.ProbeRadius = config.LSHConfig.ProbeRadius
	}

	return nil
}

// applyArchitectureConfig 应用架构配置
func (db *VectorDB) applyArchitectureConfig(config *DistributedArchitectureConfig) error {
	// 这里可以配置分布式架构相关参数
	// 由于当前是单机版本，主要是为未来扩展做准备
	return nil
}

// applyPerformanceConfig 应用性能配置
func (db *VectorDB) applyPerformanceConfig(config *PerformanceOptimizationConfig) error {
	// 配置查询加速
	if config.QueryAcceleration.Enable {
		if config.QueryAcceleration.Preprocessing.Normalization {
			db.useNormalization = true
		}
	}
	return nil
}

// applyHardwareConfig 应用硬件配置
func (db *VectorDB) applyHardwareConfig(config *HardwareAccelerationConfig) error {
	// 配置GPU加速
	if config.GPU.Enable && db.hardwareManager != nil {
		gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
		if gpuAccelerator != nil {
			// GPU加速器已通过hardware_manager管理，无需直接设置HardwareCaps
			logger.Info("GPU加速器配置已应用")
		}
	}
	return nil
}

// ApplyHardwareManager 应用硬件管理器
func (db *VectorDB) ApplyHardwareManager(hardwareManager *acceler.HardwareManager) error {
	// 检查参数
	if hardwareManager == nil {
		return fmt.Errorf("硬件管理器不能为空")
	}

	// 设置硬件管理器
	db.hardwareManager = hardwareManager

	// 获取硬件配置
	config := hardwareManager.GetConfig()

	// 更新硬件能力信息
	gpuAcc, hasGPU := hardwareManager.GetAccelerator(acceler.AcceleratorGPU)
	if hasGPU && config.GPU.Enable {
		// 注册GPU加速器到硬件管理器
		err := hardwareManager.RegisterGPUAccelerator(gpuAcc)
		if err != nil {
			return fmt.Errorf("注册GPU加速器失败: %v", err)
		}

		// 更新硬件能力信息（GPU状态由hardware_manager管理）
		db.HardwareCaps.GPUDevices = len(config.GPU.DeviceIDs)

		// 记录GPU配置信息
		logger.Info("已启用GPU加速，设备数量: %d, 批处理大小: %d",
			db.HardwareCaps.GPUDevices, config.GPU.BatchSize)
	}

	// 获取CPU加速器
	cpuAcc, hasCPU := hardwareManager.GetAccelerator(acceler.AcceleratorCPU)
	if hasCPU && config.CPU.Enable {
		// 更新CPU能力信息
		caps := cpuAcc.GetCapabilities()
		db.HardwareCaps.HasAVX2 = caps.HasAVX2
		db.HardwareCaps.HasAVX512 = caps.HasAVX512
		db.HardwareCaps.CPUCores = caps.CPUCores

		// 记录CPU配置信息
		logger.Info("已启用CPU优化，核心数: %d, AVX2: %v, AVX512: %v",
			db.HardwareCaps.CPUCores, db.HardwareCaps.HasAVX2, db.HardwareCaps.HasAVX512)
	}

	// 获取向量数据库的实际大小
	db.mu.RLock()
	actualDataSize := len(db.vectors)
	db.mu.RUnlock()

	// 将硬件管理器设置到策略选择器中
	if db.strategyComputeSelector != nil {
		db.strategyComputeSelector.SetHardwareManager(hardwareManager)
		// 使用向量维度和实际数据量作为参数
		db.currentStrategy = db.strategyComputeSelector.SelectOptimalStrategy(actualDataSize, db.vectorDim)
		logger.Info("已选择最佳计算策略: %v，基于数据量: %d, 向量维度: %d",
			db.currentStrategy, actualDataSize, db.vectorDim)
	}

	// 保存硬件管理器引用，以便后续使用
	db.hardwareManager = hardwareManager

	return nil
}

// applyCacheConfig 应用缓存配置
func (db *VectorDB) applyCacheConfig(config *CacheStrategyConfig) error {
	// 配置多级缓存
	if db.MultiCache != nil {
		// 配置结果缓存
		if config.ResultCache.Enable {
			// 设置L1缓存容量（基于最大条目数）
			if config.ResultCache.MaxEntries > 0 {
				db.MultiCache.l1Capacity = config.ResultCache.MaxEntries / 3 // L1占总容量的1/3
			}

			// 设置L2缓存容量
			if config.ResultCache.MaxEntries > 0 {
				db.MultiCache.l2Capacity = config.ResultCache.MaxEntries / 3 // L2占总容量的1/3
			}

			// 设置L3缓存容量和TTL
			if config.ResultCache.MaxEntries > 0 {
				db.MultiCache.l3Capacity = config.ResultCache.MaxEntries / 3 // L3占总容量的1/3
			}
			if config.ResultCache.TTL > 0 {
				db.MultiCache.l3TTL = config.ResultCache.TTL
			}

			// 配置缓存路径
			if config.ResultCache.Persistence.Enable && config.ResultCache.Persistence.StoragePath != "" {
				db.MultiCache.l3CachePath = config.ResultCache.Persistence.StoragePath
			}
		}

		// 配置向量缓存
		if config.VectorCache.Enable {
			// 根据向量缓存配置调整L1缓存大小
			if config.VectorCache.MaxVectors > 0 {
				vectorCacheSize := config.VectorCache.MaxVectors / 2 // 向量缓存占L1的一半
				if vectorCacheSize > db.MultiCache.l1Capacity {
					db.MultiCache.l1Capacity = vectorCacheSize
				}
			}

			// 配置向量缓存TTL
			if config.VectorCache.TTL > 0 && config.VectorCache.TTL < db.MultiCache.l3TTL {
				// 如果向量缓存TTL更短，使用向量缓存的TTL
				db.MultiCache.l3TTL = config.VectorCache.TTL
			}
		}

		// 配置索引缓存
		if config.IndexCache.Enable {
			// 索引缓存通常需要更大的L2缓存空间
			if config.IndexCache.MaxSize > 0 {
				// 根据索引缓存大小调整L2容量
				indexCacheEntries := int(config.IndexCache.MaxSize / 1024) // 假设每个索引条目平均1KB
				if indexCacheEntries > db.MultiCache.l2Capacity {
					db.MultiCache.l2Capacity = indexCacheEntries
				}
			}
		}

		// 配置查询缓存
		if config.QueryCache.Enable {
			// 查询缓存主要影响L1缓存
			if config.QueryCache.MaxQueries > 0 {
				queryCacheSize := config.QueryCache.MaxQueries
				if queryCacheSize > db.MultiCache.l1Capacity {
					db.MultiCache.l1Capacity = queryCacheSize
				}
			}
		}

		// 配置元数据缓存
		if config.MetadataCache.Enable {
			// 元数据缓存通常放在L2层
			if config.MetadataCache.MaxEntries > 0 {
				metadataCacheSize := config.MetadataCache.MaxEntries
				if metadataCacheSize > db.MultiCache.l2Capacity {
					db.MultiCache.l2Capacity = metadataCacheSize
				}
			}
		}

		// 应用全局缓存配置
		if config.GlobalCache.Enable {
			// 设置全局缓存大小限制
			if config.GlobalCache.MaxMemoryUsage > 0 {
				// 根据内存使用限制调整各级缓存容量
				maxMemoryMB := config.GlobalCache.MaxMemoryUsage / (1024 * 1024)
				// 假设每个缓存条目平均占用1KB内存
				totalEntries := int(maxMemoryMB * 1024)

				// 重新分配各级缓存容量
				db.MultiCache.l1Capacity = totalEntries / 6 // L1占总容量的1/6
				db.MultiCache.l2Capacity = totalEntries / 3 // L2占总容量的1/3
				db.MultiCache.l3Capacity = totalEntries / 2 // L3占总容量的1/2
			}

			// 配置全局TTL
			if config.GlobalCache.DefaultTTL > 0 {
				db.MultiCache.l3TTL = config.GlobalCache.DefaultTTL
			}

			// 配置清理策略
			if config.GlobalCache.CleanupInterval > 0 {
				// 可以启动定期清理任务
				go func() {
					ticker := time.NewTicker(config.GlobalCache.CleanupInterval)
					defer ticker.Stop()
					for {
						select {
						case <-ticker.C:
							if db.MultiCache != nil {
								db.MultiCache.CleanupExpired(time.Now())
								if config.GlobalCache.LowHitRateCleanup {
									db.MultiCache.CleanupLowHitRate(config.GlobalCache.HitRateThreshold)
								}
							}
						}
					}
				}()
			}
		}

		// 确保缓存容量的最小值
		if db.MultiCache.l1Capacity <= 0 {
			db.MultiCache.l1Capacity = 1000 // 默认L1容量
		}
		if db.MultiCache.l2Capacity <= 0 {
			db.MultiCache.l2Capacity = 5000 // 默认L2容量
		}
		if db.MultiCache.l3Capacity <= 0 {
			db.MultiCache.l3Capacity = 10000 // 默认L3容量
		}
		if db.MultiCache.l3TTL <= 0 {
			db.MultiCache.l3TTL = 24 * time.Hour // 默认TTL为24小时
		}

		logger.Info("缓存配置已应用", map[string]interface{}{
			"l1_capacity": db.MultiCache.l1Capacity,
			"l2_capacity": db.MultiCache.l2Capacity,
			"l3_capacity": db.MultiCache.l3Capacity,
			"l3_ttl":      db.MultiCache.l3TTL,
			"l3_path":     db.MultiCache.l3CachePath,
		})
	}
	return nil
}

// applyMonitoringConfig 应用监控配置
func (db *VectorDB) applyMonitoringConfig(config *MonitoringConfig) error {
	logger.Info("开始应用监控配置")

	// 1. 配置指标收集
	if err := db.configureMetrics(&config.Metrics); err != nil {
		return fmt.Errorf("配置指标收集失败: %v", err)
	}

	// 2. 配置告警系统
	if err := db.configureAlerting(&config.Alerting); err != nil {
		return fmt.Errorf("配置告警系统失败: %v", err)
	}

	// 3. 配置日志系统
	if err := db.configureLogging(&config.Logging); err != nil {
		return fmt.Errorf("配置日志系统失败: %v", err)
	}

	// 4. 配置链路追踪
	if err := db.configureTracing(&config.Tracing); err != nil {
		return fmt.Errorf("配置链路追踪失败: %v", err)
	}

	// 5. 配置性能分析
	if err := db.configureProfiling(&config.Profiling); err != nil {
		return fmt.Errorf("配置性能分析失败: %v", err)
	}

	// 6. 配置仪表板
	if err := db.configureDashboard(&config.Dashboard); err != nil {
		return fmt.Errorf("配置仪表板失败: %v", err)
	}

	// 7. 配置健康检查
	if err := db.configureHealthCheck(&config.HealthCheck); err != nil {
		return fmt.Errorf("配置健康检查失败: %v", err)
	}

	// 8. 配置自动扩缩容监控
	if err := db.configureAutoScalingMonitoring(&config.AutoScaling); err != nil {
		return fmt.Errorf("配置自动扩缩容监控失败: %v", err)
	}

	logger.Info("监控配置应用完成")
	return nil
}

// configureMetrics 配置指标收集
func (db *VectorDB) configureMetrics(config *MetricsConfig) error {
	if !config.Enable {
		logger.Info("指标收集已禁用")
		return nil
	}

	// 配置性能监控器
	if db.performanceMonitor != nil {
		// 重置监控器
		db.performanceMonitor.Reset()
		logger.Info("性能监控器已重置")
	}

	// 配置指标导出器
	if err := db.configureMetricsExporter(&config.Exporter); err != nil {
		return fmt.Errorf("配置指标导出器失败: %v", err)
	}

	// 配置性能指标
	if err := db.configurePerformanceMetrics(&config.Performance); err != nil {
		return fmt.Errorf("配置性能指标失败: %v", err)
	}

	// 配置业务指标
	if err := db.configureBusinessMetrics(&config.Business); err != nil {
		return fmt.Errorf("配置业务指标失败: %v", err)
	}

	// 配置系统指标
	if err := db.configureSystemMetrics(&config.System); err != nil {
		return fmt.Errorf("配置系统指标失败: %v", err)
	}

	// 配置自定义指标
	if err := db.configureCustomMetrics(&config.Custom); err != nil {
		return fmt.Errorf("配置自定义指标失败: %v", err)
	}

	logger.Info("指标收集配置完成")
	return nil
}

// configureMetricsExporter 配置指标导出器
func (db *VectorDB) configureMetricsExporter(config *MetricsExporterConfig) error {
	// 配置Prometheus导出器
	if config.Prometheus.Enable {
		logger.Info(fmt.Sprintf("启用Prometheus导出器，端点: %s:%d%s",
			config.Prometheus.Endpoint, config.Prometheus.Port, config.Prometheus.Path))
		// 这里可以初始化Prometheus导出器
	}

	// 配置InfluxDB导出器
	if config.InfluxDB.Enable {
		logger.Info(fmt.Sprintf("启用InfluxDB导出器，URL: %s", config.InfluxDB.URL))
		// 这里可以初始化InfluxDB导出器
	}

	// 配置Elasticsearch导出器
	if config.Elasticsearch.Enable {
		logger.Info(fmt.Sprintf("启用Elasticsearch导出器，URLs: %v", config.Elasticsearch.URLs))
		// 这里可以初始化Elasticsearch导出器
	}

	// 配置CloudWatch导出器
	if config.CloudWatch.Enable {
		logger.Info(fmt.Sprintf("启用CloudWatch导出器，区域: %s", config.CloudWatch.Region))
		// 这里可以初始化CloudWatch导出器
	}

	// 配置Datadog导出器
	if config.Datadog.Enable {
		logger.Info(fmt.Sprintf("启用Datadog导出器，站点: %s", config.Datadog.Site))
		// 这里可以初始化Datadog导出器
	}

	return nil
}

// configurePerformanceMetrics 配置性能指标
func (db *VectorDB) configurePerformanceMetrics(config *PerformanceMetricsConfig) error {
	// 配置搜索延迟指标
	if config.SearchLatency.Enable {
		logger.Info("启用搜索延迟指标收集")
		// 配置延迟指标的百分位数和桶
	}

	// 配置索引构建时间指标
	if config.IndexBuildTime.Enable {
		logger.Info("启用索引构建时间指标收集")
	}

	// 配置吞吐量指标
	if config.Throughput.Enable {
		logger.Info("启用吞吐量指标收集")
	}

	// 配置资源使用指标
	if err := db.configureResourceMetrics(&config.ResourceUsage); err != nil {
		return fmt.Errorf("配置资源使用指标失败: %v", err)
	}

	// 配置缓存性能指标
	if err := db.configureCacheMetrics(&config.CachePerformance); err != nil {
		return fmt.Errorf("配置缓存性能指标失败: %v", err)
	}

	return nil
}

// configureResourceMetrics 配置资源指标
func (db *VectorDB) configureResourceMetrics(config *ResourceMetricsConfig) error {
	// 配置CPU指标
	if config.CPU.Enable {
		logger.Info("启用CPU指标收集")
	}

	// 配置内存指标
	if config.Memory.Enable {
		logger.Info("启用内存指标收集")
	}

	// 配置磁盘指标
	if config.Disk.Enable {
		logger.Info("启用磁盘指标收集")
	}

	// 配置网络指标
	if config.Network.Enable {
		logger.Info("启用网络指标收集")
	}

	// 配置GPU指标
	if config.GPU.Enable {
		logger.Info("启用GPU指标收集")
	}

	return nil
}

// configureCacheMetrics 配置缓存指标
func (db *VectorDB) configureCacheMetrics(config *CacheMetricsConfig) error {
	if db.MultiCache != nil {
		logger.Info("配置多级缓存指标收集")
		// 这里可以配置缓存命中率、缓存大小等指标的收集
	}
	return nil
}

// configureBusinessMetrics 配置业务指标
func (db *VectorDB) configureBusinessMetrics(config *BusinessMetricsConfig) error {
	if !config.Enable {
		return nil
	}

	// 配置搜索准确性指标
	if config.SearchAccuracy.Enable {
		logger.Info("启用搜索准确性指标收集")
	}

	// 配置用户满意度指标
	if config.UserSatisfaction.Enable {
		logger.Info("启用用户满意度指标收集")
	}

	// 配置数据质量指标
	if config.DataQuality.Enable {
		logger.Info("启用数据质量指标收集")
	}

	// 配置成本指标
	if config.CostMetrics.Enable {
		logger.Info("启用成本指标收集")
	}

	return nil
}

// configureSystemMetrics 配置系统指标
func (db *VectorDB) configureSystemMetrics(config *SystemMetricsConfig) error {
	if !config.Enable {
		return nil
	}

	// 配置Goroutine指标
	if config.Goroutines.Enable {
		logger.Info("启用Goroutine指标收集")
	}

	// 配置GC指标
	if config.GC.Enable {
		logger.Info("启用GC指标收集")
	}

	// 配置文件描述符指标
	if config.FileDescriptors.Enable {
		logger.Info("启用文件描述符指标收集")
	}

	// 配置进程指标
	if config.ProcessMetrics.Enable {
		logger.Info("启用进程指标收集")
	}

	return nil
}

// configureCustomMetrics 配置自定义指标
func (db *VectorDB) configureCustomMetrics(config *CustomMetricsConfig) error {
	if !config.Enable {
		return nil
	}

	// 配置自定义指标
	for _, metric := range config.Metrics {
		logger.Info(fmt.Sprintf("配置自定义指标: %s (类型: %s)", metric.Name, metric.Type))
	}

	// 配置指标插件
	for _, plugin := range config.Plugins {
		if plugin.Enable {
			logger.Info(fmt.Sprintf("启用指标插件: %s (路径: %s)", plugin.Name, plugin.Path))
		}
	}

	return nil
}

// configureAlerting 配置告警系统
func (db *VectorDB) configureAlerting(config *AlertingConfig) error {
	if !config.Enable {
		logger.Info("告警系统已禁用")
		return nil
	}

	// 配置告警规则
	for _, rule := range config.Rules {
		if rule.Enabled {
			logger.Info(fmt.Sprintf("配置告警规则: %s (严重级别: %s)", rule.Name, rule.Severity))
		}
	}

	// 配置通知渠道
	for _, channel := range config.Notification.Channels {
		if channel.Enabled {
			logger.Info(fmt.Sprintf("配置通知渠道: %s (类型: %s)", channel.Name, channel.Type))
		}
	}

	// 配置告警升级
	if config.Escalation.Enable {
		logger.Info("启用告警升级机制")
	}

	// 配置告警抑制
	if config.Suppression.Enable {
		logger.Info("启用告警抑制机制")
	}

	// 配置告警抑制规则
	if config.Inhibition.Enable {
		logger.Info("启用告警抑制规则")
	}

	logger.Info("告警系统配置完成")
	return nil
}

// configureLogging 配置日志系统
func (db *VectorDB) configureLogging(config *LoggingConfig) error {
	if !config.Enable {
		logger.Info("日志系统已禁用")
		return nil
	}

	logger.Info(fmt.Sprintf("配置日志级别: %s, 格式: %s", config.Level, config.Format))

	// 配置日志输出
	if config.Output.Console {
		logger.Info("启用控制台日志输出")
	}

	if config.Output.File.Enable {
		logger.Info(fmt.Sprintf("启用文件日志输出，路径: %s", config.Output.File.Path))
	}

	if config.Output.Syslog.Enable {
		logger.Info(fmt.Sprintf("启用Syslog输出，地址: %s", config.Output.Syslog.Address))
	}

	if config.Output.Elasticsearch.Enable {
		logger.Info(fmt.Sprintf("启用Elasticsearch日志输出，URLs: %v", config.Output.Elasticsearch.URLs))
	}

	if config.Output.Kafka.Enable {
		logger.Info(fmt.Sprintf("启用Kafka日志输出，Brokers: %v", config.Output.Kafka.Brokers))
	}

	// 配置日志轮转
	if config.Rotation.Enable {
		logger.Info("启用日志轮转")
	}

	// 配置日志采样
	if config.Sampling.Enable {
		logger.Info("启用日志采样")
	}

	// 配置结构化日志
	if config.Structured.Enable {
		logger.Info("启用结构化日志")
	}

	// 配置审计日志
	if config.Audit.Enable {
		logger.Info(fmt.Sprintf("启用审计日志，路径: %s", config.Audit.Path))
	}

	logger.Info("日志系统配置完成")
	return nil
}

/*
- 支持多种追踪提供商（Jaeger、Zipkin、OpenTelemetry）
- 配置采样率和自定义标签
- 配置服务名和端点
*/
// configureTracing 配置链路追踪
func (db *VectorDB) configureTracing(config *TracingConfig) error {
	if !config.Enable {
		logger.Info("链路追踪已禁用")
		return nil
	}

	logger.Info(fmt.Sprintf("配置链路追踪提供商: %s, 采样率: %.2f", config.Provider, config.SamplingRate))

	// 验证采样率
	if config.SamplingRate < 0 || config.SamplingRate > 1 {
		return fmt.Errorf("采样率必须在0-1之间")
	}

	// 配置Jaeger
	if config.Provider == "jaeger" && config.Jaeger.Endpoint != "" {
		logger.Info(fmt.Sprintf("配置Jaeger追踪，端点: %s, 服务名: %s",
			config.Jaeger.Endpoint, config.Jaeger.ServiceName))

		// 初始化Jaeger追踪器
		if config.Jaeger.ServiceName == "" {
			config.Jaeger.ServiceName = "vectorsphere"
		}

		// 设置Jaeger配置
		if config.Jaeger.AgentPort <= 0 {
			config.Jaeger.AgentPort = 6831
		}
		if config.Jaeger.AgentHost == "" {
			config.Jaeger.AgentHost = "localhost"
		}
	}

	// 配置Zipkin
	if config.Provider == "zipkin" && config.Zipkin.Endpoint != "" {
		logger.Info(fmt.Sprintf("配置Zipkin追踪，端点: %s, 服务名: %s",
			config.Zipkin.Endpoint, config.Zipkin.ServiceName))

		// 初始化Zipkin追踪器
		if config.Zipkin.ServiceName == "" {
			config.Zipkin.ServiceName = "vectorsphere"
		}

		// 设置Zipkin配置
		if config.Zipkin.BatchSize <= 0 {
			config.Zipkin.BatchSize = 100
		}
		if config.Zipkin.Timeout <= 0 {
			config.Zipkin.Timeout = 5 * time.Second
		}
	}

	// 配置OpenTelemetry
	if config.Provider == "opentelemetry" && config.OpenTelemetry.Endpoint != "" {
		logger.Info(fmt.Sprintf("配置OpenTelemetry追踪，端点: %s, 服务名: %s",
			config.OpenTelemetry.Endpoint, config.OpenTelemetry.ServiceName))

		// 初始化OpenTelemetry追踪器
		if config.OpenTelemetry.ServiceName == "" {
			config.OpenTelemetry.ServiceName = "vectorsphere"
		}

		// 设置OpenTelemetry配置
		if config.OpenTelemetry.ServiceVersion == "" {
			config.OpenTelemetry.ServiceVersion = "1.0.0"
		}
		if config.OpenTelemetry.BatchTimeout <= 0 {
			config.OpenTelemetry.BatchTimeout = 5 * time.Second
		}
		if config.OpenTelemetry.ExportTimeout <= 0 {
			config.OpenTelemetry.ExportTimeout = 30 * time.Second
		}
	}

	// 配置自定义标签
	if len(config.CustomTags) > 0 {
		logger.Info(fmt.Sprintf("配置自定义标签: %v", config.CustomTags))
		// 验证标签格式
		for key, value := range config.CustomTags {
			if key == "" || value == "" {
				return fmt.Errorf("自定义标签的键和值不能为空")
			}
		}
	}

	// 配置追踪过滤器
	if len(config.OperationFilters) > 0 {
		logger.Info(fmt.Sprintf("配置追踪操作过滤器: %v", config.OperationFilters))
	}

	// 配置自定义标签
	if len(config.CustomTags) > 0 {
		logger.Info(fmt.Sprintf("配置自定义标签: %v", config.CustomTags))
	}

	logger.Info("链路追踪配置完成")
	return nil
}

// configureProfiling 配置性能分析
func (db *VectorDB) configureProfiling(config *ProfilingConfig) error {
	if !config.Enable {
		logger.Info("性能分析已禁用")
		return nil
	}

	// 创建默认输出目录
	outputDir := "./profiles"
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("创建性能分析输出目录失败: %v", err)
	}

	// 配置CPU性能分析
	if config.CPU.Enable {
		logger.Info(fmt.Sprintf("启用CPU性能分析，采样率: %d Hz, 输出路径: %s",
			config.CPU.SamplingRate, config.CPU.OutputPath))

		// 验证CPU性能分析配置
		if config.CPU.SamplingRate <= 0 {
			config.CPU.SamplingRate = 100 // 默认100Hz
		}
		if config.CPU.OutputPath == "" {
			config.CPU.OutputPath = filepath.Join(outputDir, "cpu.prof")
		}

		// 设置CPU性能分析参数
		runtime.SetCPUProfileRate(config.CPU.SamplingRate)
	}

	// 配置内存性能分析
	if config.Memory.Enable {
		logger.Info(fmt.Sprintf("启用内存性能分析，采样率: %d, 输出路径: %s",
			config.Memory.SamplingRate, config.Memory.OutputPath))

		// 验证内存性能分析配置
		if config.Memory.SamplingRate <= 0 {
			config.Memory.SamplingRate = 512 * 1024 // 默认512KB
		}
		if config.Memory.OutputPath == "" {
			config.Memory.OutputPath = filepath.Join(outputDir, "mem.prof")
		}

		// 设置内存性能分析参数
		runtime.MemProfileRate = config.Memory.SamplingRate
	}

	// 配置Goroutine性能分析
	if config.Goroutine.Enable {
		logger.Info(fmt.Sprintf("启用Goroutine性能分析，输出路径: %s", config.Goroutine.OutputPath))

		if config.Goroutine.OutputPath == "" {
			config.Goroutine.OutputPath = filepath.Join(outputDir, "goroutine.prof")
		}
	}

	// 配置阻塞性能分析
	if config.Block.Enable {
		logger.Info(fmt.Sprintf("启用阻塞性能分析，速率: %d, 输出路径: %s",
			config.Block.Rate, config.Block.OutputPath))

		// 验证阻塞性能分析配置
		if config.Block.Rate <= 0 {
			config.Block.Rate = 1
		}
		if config.Block.OutputPath == "" {
			config.Block.OutputPath = filepath.Join(outputDir, "block.prof")
		}

		// 设置阻塞性能分析参数
		runtime.SetBlockProfileRate(config.Block.Rate)
	}

	// 配置互斥锁性能分析
	if config.Mutex.Enable {
		logger.Info(fmt.Sprintf("启用互斥锁性能分析，速率: %d, 输出路径: %s",
			config.Mutex.Rate, config.Mutex.OutputPath))

		// 验证互斥锁性能分析配置
		if config.Mutex.Rate <= 0 {
			config.Mutex.Rate = 1
		}
		if config.Mutex.OutputPath == "" {
			config.Mutex.OutputPath = filepath.Join(outputDir, "mutex.prof")
		}

		// 设置互斥锁性能分析参数
		runtime.SetMutexProfileFraction(config.Mutex.Rate)
	}

	// 配置自定义性能分析
	if config.Custom.Enable {
		for _, profile := range config.Custom.Profiles {
			if profile.Enabled {
				logger.Info(fmt.Sprintf("启用自定义性能分析: %s (类型: %s)", profile.Name, profile.Type))

				// 验证自定义性能分析配置
				if profile.Name == "" {
					return fmt.Errorf("自定义性能分析名称不能为空")
				}
				if profile.Type == "" {
					profile.Type = "custom"
				}
				if profile.OutputPath == "" {
					profile.OutputPath = filepath.Join(outputDir, fmt.Sprintf("%s.prof", profile.Name))
				}
			}
		}
	}

	// 性能分析配置完成，使用各个子配置的间隔设置
	// CPU、Memory、Goroutine等各自有独立的间隔配置

	logger.Info("性能分析配置完成")
	return nil
}

// configureDashboard 配置仪表板
func (db *VectorDB) configureDashboard(config *DashboardConfig) error {
	if config == nil {
		return fmt.Errorf("dashboard config cannot be nil")
	}

	// 验证基本配置
	if config.Port <= 0 || config.Port > 65535 {
		return fmt.Errorf("invalid dashboard port: %d", config.Port)
	}

	if config.Host == "" {
		config.Host = "localhost"
	}

	if config.Path == "" {
		config.Path = "/dashboard"
	}

	// 验证认证配置
	if config.Authentication.Enable {
		if len(config.Authentication.Users) == 0 {
			return fmt.Errorf("authentication enabled but no users configured")
		}
		for _, user := range config.Authentication.Users {
			if user.Username == "" || user.Password == "" {
				return fmt.Errorf("user credentials cannot be empty")
			}
		}
	}

	// 验证图表配置
	for i, chart := range config.Charts {
		if chart.ID == "" {
			return fmt.Errorf("chart %d: ID cannot be empty", i)
		}
		if chart.Title == "" {
			return fmt.Errorf("chart %d: title cannot be empty", i)
		}
		if chart.Type == "" {
			return fmt.Errorf("chart %d: type cannot be empty", i)
		}
		if chart.Query == "" {
			return fmt.Errorf("chart %d: query cannot be empty", i)
		}
	}

	// 设置默认值
	if config.RefreshInterval == 0 {
		config.RefreshInterval = 30 * time.Second
	}

	if config.Theme == "" {
		config.Theme = "light"
	}

	logger.Info("Dashboard configuration applied successfully")

	if !config.Enable {
		logger.Info("仪表板已禁用")
		return nil
	}

	// 验证仪表板配置
	if config.Host == "" {
		config.Host = "localhost"
	}
	if config.Port <= 0 || config.Port > 65535 {
		return fmt.Errorf("仪表板端口必须在1-65535之间")
	}
	if config.Path == "" {
		config.Path = "/dashboard"
	}
	if config.Theme == "" {
		config.Theme = "default"
	}

	logger.Info(fmt.Sprintf("配置仪表板，地址: %s:%d%s, 主题: %s",
		config.Host, config.Port, config.Path, config.Theme))

	// 配置认证
	if config.Authentication.Enable {
		logger.Info(fmt.Sprintf("启用仪表板认证，类型: %s", config.Authentication.Type))

		// 验证认证配置
		if config.Authentication.Type == "" {
			config.Authentication.Type = "basic"
		}

		// 验证认证凭据
		switch config.Authentication.Type {
		case "basic":
			if config.Authentication.Username == "" || config.Authentication.Password == "" {
				return fmt.Errorf("基础认证需要用户名和密码")
			}
		case "token":
			if config.Authentication.Token == "" {
				return fmt.Errorf("令牌认证需要有效的令牌")
			}
		case "oauth":
			if config.Authentication.ClientID == "" || config.Authentication.ClientSecret == "" {
				return fmt.Errorf("OAuth认证需要客户端ID和密钥")
			}
		}

		// 配置会话超时
		if config.Authentication.SessionTimeout <= 0 {
			config.Authentication.SessionTimeout = 24 * time.Hour
		}
	}

	// 配置图表
	logger.Info(fmt.Sprintf("配置 %d 个仪表板图表", len(config.Charts)))

	// 验证图表配置
	for i, chart := range config.Charts {
		if chart.ID == "" {
			return fmt.Errorf("图表 %d 的ID不能为空", i)
		}
		if chart.Title == "" {
			chart.Title = chart.ID
		}
		if chart.Type == "" {
			chart.Type = "line"
		}
		if chart.RefreshInterval <= 0 {
			chart.RefreshInterval = 30 * time.Second
		}
	}

	// 配置仪表板更新间隔
	if config.RefreshInterval <= 0 {
		config.RefreshInterval = 5 * time.Second
	}

	// 配置仪表板数据保留期
	if config.DataRetention <= 0 {
		config.DataRetention = 7 * 24 * time.Hour // 默认保留7天
	}

	// 配置仪表板静态资源路径
	if config.StaticPath == "" {
		config.StaticPath = "./static"
	}

	// 验证静态资源路径是否存在
	if _, err := os.Stat(config.StaticPath); os.IsNotExist(err) {
		logger.Warning(fmt.Sprintf("仪表板静态资源路径不存在: %s", config.StaticPath))
	}

	logger.Info("仪表板配置完成")
	return nil
}

// configureHealthCheck 配置健康检查
func (db *VectorDB) configureHealthCheck(config *HealthCheckConfig) error {
	if !config.Enable {
		logger.Info("健康检查已禁用")
		return nil
	}

	logger.Info("配置健康检查系统")

	// 验证健康检查配置
	if config.Interval <= 0 {
		config.Interval = 30 * time.Second
	}
	if config.Timeout <= 0 {
		config.Timeout = 5 * time.Second
	}
	// 设置默认就绪检查配置
	if !config.Readiness.Enable {
		config.Readiness.Enable = true
		config.Readiness.FailureThreshold = 3
		config.Readiness.SuccessThreshold = 1
	}

	// 配置默认健康检查端点
	if len(config.Endpoints) == 0 {
		config.Endpoints = []HealthCheckEndpoint{
			{
				Name:           "default",
				Path:           "/health",
				Method:         "GET",
				ExpectedStatus: 200,
				Timeout:        config.Timeout,
				Enabled:        true,
			},
		}
	}

	logger.Info(fmt.Sprintf("健康检查配置完成, 检查间隔: %v, 超时: %v, 端点数量: %d",
		config.Interval, config.Timeout, len(config.Endpoints)))

	// 配置依赖检查项目
	if len(config.Dependencies) == 0 {
		// 添加默认依赖检查项目
		config.Dependencies = []DependencyCheck{
			{
				Name:       "database",
				Type:       "database",
				Address:    "localhost:5432",
				Timeout:    3 * time.Second,
				RetryCount: 3,
				Critical:   true,
				Enabled:    true,
			},
			{
				Name:       "redis",
				Type:       "redis",
				Address:    "localhost:6379",
				Timeout:    1 * time.Second,
				RetryCount: 2,
				Critical:   false,
				Enabled:    true,
			},
		}
	}

	// 验证依赖检查项目配置
	for i, dep := range config.Dependencies {
		if dep.Name == "" {
			return fmt.Errorf("依赖检查项目 %d 的名称不能为空", i)
		}
		if dep.Type == "" {
			return fmt.Errorf("依赖检查项目 %s 的类型不能为空", dep.Name)
		}
		if dep.Timeout <= 0 {
			config.Dependencies[i].Timeout = config.Timeout
		}
		if dep.RetryCount <= 0 {
			config.Dependencies[i].RetryCount = 3 // 默认重试3次
		}

		logger.Info(fmt.Sprintf("配置依赖检查项目: %s (类型: %s, 地址: %s, 启用: %v)",
			dep.Name, dep.Type, dep.Address, dep.Enabled))
	}

	// 配置存活检查
	if !config.Liveness.Enable {
		config.Liveness.Enable = true
		config.Liveness.Path = "/health/live"
		config.Liveness.Period = 10 * time.Second
		config.Liveness.Timeout = 3 * time.Second
		config.Liveness.FailureThreshold = 3
	}

	// 配置启动检查
	if !config.Startup.Enable {
		config.Startup.Enable = true
		config.Startup.Path = "/health/startup"
		config.Startup.Period = 5 * time.Second
		config.Startup.Timeout = 3 * time.Second
		config.Startup.FailureThreshold = 30
	}

	logger.Info("健康检查系统配置完成")
	return nil
}

// configureAutoScalingMonitoring 配置自动扩缩容监控
func (db *VectorDB) configureAutoScalingMonitoring(config *AutoScalingConfig) error {
	if config == nil {
		return fmt.Errorf("auto scaling config cannot be nil")
	}

	// 验证基本配置
	if !config.Enable {
		logger.Info("自动扩缩容已禁用")
		return nil
	}

	// 验证策略
	validStrategies := []string{"reactive", "predictive", "hybrid"}
	validStrategy := false
	for _, strategy := range validStrategies {
		if config.Strategy == strategy {
			validStrategy = true
			break
		}
	}
	if !validStrategy {
		return fmt.Errorf("invalid scaling strategy: %s, must be one of %v", config.Strategy, validStrategies)
	}

	// 验证指标配置
	if config.Metrics.CPU.Enable {
		if config.Metrics.CPU.TargetUtilization <= 0 || config.Metrics.CPU.TargetUtilization > 100 {
			return fmt.Errorf("invalid CPU target utilization: %f, must be between 0 and 100", config.Metrics.CPU.TargetUtilization)
		}
		if config.Metrics.CPU.AggregationWindow <= 0 {
			config.Metrics.CPU.AggregationWindow = 60 * time.Second
		}
	}

	if config.Metrics.Memory.Enable {
		if config.Metrics.Memory.TargetUtilization <= 0 || config.Metrics.Memory.TargetUtilization > 100 {
			return fmt.Errorf("invalid memory target utilization: %f, must be between 0 and 100", config.Metrics.Memory.TargetUtilization)
		}
		if config.Metrics.Memory.AggregationWindow <= 0 {
			config.Metrics.Memory.AggregationWindow = 60 * time.Second
		}
	}

	if config.Metrics.QPS.Enable {
		if config.Metrics.QPS.TargetValue <= 0 {
			return fmt.Errorf("invalid QPS target value: %f, must be greater than 0", config.Metrics.QPS.TargetValue)
		}
		if config.Metrics.QPS.AggregationWindow <= 0 {
			config.Metrics.QPS.AggregationWindow = 60 * time.Second
		}
	}

	if config.Metrics.QueueLength.Enable {
		if config.Metrics.QueueLength.TargetValue <= 0 {
			return fmt.Errorf("invalid queue length target value: %f, must be greater than 0", config.Metrics.QueueLength.TargetValue)
		}
		if config.Metrics.QueueLength.AggregationWindow <= 0 {
			config.Metrics.QueueLength.AggregationWindow = 60 * time.Second
		}
	}

	// 验证自定义指标
	for i, metric := range config.Metrics.Custom {
		if metric.Name == "" {
			return fmt.Errorf("custom metric %d: name cannot be empty", i)
		}
		if metric.Query == "" {
			return fmt.Errorf("custom metric %d: query cannot be empty", i)
		}
		if metric.TargetValue <= 0 {
			return fmt.Errorf("custom metric %d: target value must be greater than 0", i)
		}
		if metric.AggregationWindow <= 0 {
			metric.AggregationWindow = 60 * time.Second
		}
	}

	// 验证扩缩容限制
	if config.Limits.MinComputeNodes <= 0 {
		return fmt.Errorf("minimum compute nodes must be greater than 0")
	}
	if config.Limits.MaxComputeNodes <= config.Limits.MinComputeNodes {
		return fmt.Errorf("maximum compute nodes must be greater than minimum compute nodes")
	}
	if config.Limits.MaxScaleUpRate <= 0 {
		return fmt.Errorf("max scale up rate must be greater than 0")
	}
	if config.Limits.MaxScaleDownRate <= 0 {
		return fmt.Errorf("max scale down rate must be greater than 0")
	}

	// 验证冷却配置
	if config.Cooldown.ScaleUpCooldown <= 0 {
		config.Cooldown.ScaleUpCooldown = 300 * time.Second
	}
	if config.Cooldown.ScaleDownCooldown <= 0 {
		config.Cooldown.ScaleDownCooldown = 300 * time.Second
	}

	// 验证预测配置
	if config.Strategy == "predictive" || config.Strategy == "hybrid" {
		if config.Prediction.PredictionHorizon <= 0 {
			config.Prediction.PredictionHorizon = 300 * time.Second
		}
		if config.Prediction.UpdateInterval <= 0 {
			config.Prediction.UpdateInterval = 60 * time.Second
		}
	}

	// 验证扩缩容条件
	for i, condition := range config.Conditions {
		if condition.Metric == "" {
			return fmt.Errorf("scaling condition %d: metric cannot be empty", i)
		}
		if condition.Operator == "" {
			return fmt.Errorf("scaling condition %d: operator cannot be empty", i)
		}
		if condition.Threshold <= 0 {
			return fmt.Errorf("scaling condition %d: threshold must be greater than 0", i)
		}
		if condition.Duration <= 0 {
			condition.Duration = 60 * time.Second
		}
	}

	// 设置监控间隔默认值
	if config.MonitoringInterval <= 0 {
		config.MonitoringInterval = 30 * time.Second
	}

	logger.Info("自动扩缩容监控配置应用成功")

	if !config.Enable {
		logger.Info("自动扩缩容监控已禁用")
		return nil
	}

	// 验证扩缩容策略
	if config.Strategy == "" {
		config.Strategy = "reactive" // 默认响应式策略
	}
	validStrategies := []string{"reactive", "predictive", "hybrid"}
	validStrategy := false
	for _, strategy := range validStrategies {
		if config.Strategy == strategy {
			validStrategy = true
			break
		}
	}
	if !validStrategy {
		return fmt.Errorf("无效的扩缩容策略: %s，支持的策略: %v", config.Strategy, validStrategies)
	}

	logger.Info(fmt.Sprintf("配置自动扩缩容监控，策略: %s", config.Strategy))

	// 验证扩缩容限制
	if config.Limits.MinComputeNodes <= 0 {
		config.Limits.MinComputeNodes = 1
	}
	if config.Limits.MaxComputeNodes <= config.Limits.MinComputeNodes {
		config.Limits.MaxComputeNodes = config.Limits.MinComputeNodes * 10 // 默认最大节点数为最小节点数的10倍
	}
	if config.Limits.MaxScaleUpRate <= 0 {
		config.Limits.MaxScaleUpRate = 0.5 // 默认每次最多扩容50%
	}
	if config.Limits.MaxScaleDownRate <= 0 {
		config.Limits.MaxScaleDownRate = 0.2 // 默认每次最多缩容20%
	}

	logger.Info(fmt.Sprintf("扩缩容限制 - 最小计算节点: %d, 最大计算节点: %d, 最大扩容率: %.2f, 最大缩容率: %.2f",
		config.Limits.MinComputeNodes, config.Limits.MaxComputeNodes, config.Limits.MaxScaleUpRate, config.Limits.MaxScaleDownRate))

	// 配置冷却期
	if config.Cooldown.ScaleUpCooldown <= 0 {
		config.Cooldown.ScaleUpCooldown = 5 * time.Minute
	}
	if config.Cooldown.ScaleDownCooldown <= 0 {
		config.Cooldown.ScaleDownCooldown = 10 * time.Minute
	}

	logger.Info(fmt.Sprintf("冷却期配置 - 扩容冷却: %v, 缩容冷却: %v",
		config.Cooldown.ScaleUpCooldown, config.Cooldown.ScaleDownCooldown))

	// 配置扩缩容指标
	if config.Metrics.CPUUtilization.Enable {
		logger.Info("启用CPU利用率扩缩容监控")

		// 验证CPU指标配置
		if config.Metrics.CPUUtilization.ScaleUpThreshold <= 0 || config.Metrics.CPUUtilization.ScaleUpThreshold > 1 {
			config.Metrics.CPUUtilization.ScaleUpThreshold = 0.8 // 默认80%
		}
		if config.Metrics.CPUUtilization.ScaleDownThreshold <= 0 || config.Metrics.CPUUtilization.ScaleDownThreshold >= config.Metrics.CPUUtilization.ScaleUpThreshold {
			config.Metrics.CPUUtilization.ScaleDownThreshold = 0.3 // 默认30%
		}
		if config.Metrics.CPUUtilization.AggregationWindow <= 0 {
			config.Metrics.CPUUtilization.AggregationWindow = 5 * time.Minute
		}

		logger.Info(fmt.Sprintf("CPU指标 - 扩容阈值: %.2f, 缩容阈值: %.2f, 聚合窗口: %v",
			config.Metrics.CPUUtilization.ScaleUpThreshold, config.Metrics.CPUUtilization.ScaleDownThreshold, config.Metrics.CPUUtilization.AggregationWindow))
	}

	if config.Metrics.MemoryUtilization.Enable {
		logger.Info("启用内存利用率扩缩容监控")

		// 验证内存指标配置
		if config.Metrics.MemoryUtilization.ScaleUpThreshold <= 0 || config.Metrics.MemoryUtilization.ScaleUpThreshold > 1 {
			config.Metrics.MemoryUtilization.ScaleUpThreshold = 0.85 // 默认85%
		}
		if config.Metrics.MemoryUtilization.ScaleDownThreshold <= 0 || config.Metrics.MemoryUtilization.ScaleDownThreshold >= config.Metrics.MemoryUtilization.ScaleUpThreshold {
			config.Metrics.MemoryUtilization.ScaleDownThreshold = 0.4 // 默认40%
		}
		if config.Metrics.MemoryUtilization.AggregationWindow <= 0 {
			config.Metrics.MemoryUtilization.AggregationWindow = 5 * time.Minute
		}

		logger.Info(fmt.Sprintf("内存指标 - 扩容阈值: %.2f, 缩容阈值: %.2f, 聚合窗口: %v",
			config.Metrics.MemoryUtilization.ScaleUpThreshold, config.Metrics.MemoryUtilization.ScaleDownThreshold, config.Metrics.MemoryUtilization.AggregationWindow))
	}

	if config.Metrics.QueryLatency.Enable {
		logger.Info("启用查询延迟扩缩容监控")

		// 验证延迟指标配置
		if config.Metrics.QueryLatency.ScaleUpThreshold <= 0 {
			config.Metrics.QueryLatency.ScaleUpThreshold = 100 * time.Millisecond // 默认100ms
		}
		if config.Metrics.QueryLatency.ScaleDownThreshold <= 0 || config.Metrics.QueryLatency.ScaleDownThreshold >= config.Metrics.QueryLatency.ScaleUpThreshold {
			config.Metrics.QueryLatency.ScaleDownThreshold = 50 * time.Millisecond // 默认50ms
		}
		if config.Metrics.QueryLatency.AggregationWindow <= 0 {
			config.Metrics.QueryLatency.AggregationWindow = 3 * time.Minute
		}

		logger.Info(fmt.Sprintf("延迟指标 - 扩容阈值: %v, 缩容阈值: %v, 聚合窗口: %v",
			config.Metrics.QueryLatency.ScaleUpThreshold, config.Metrics.QueryLatency.ScaleDownThreshold, config.Metrics.QueryLatency.AggregationWindow))
	}

	if config.Metrics.QPS.Enable {
		logger.Info("启用QPS扩缩容监控")

		// 验证QPS指标配置
		if config.Metrics.QPS.ScaleUpThreshold <= 0 {
			config.Metrics.QPS.ScaleUpThreshold = 1000 // 默认1000 QPS
		}
		if config.Metrics.QPS.ScaleDownThreshold <= 0 || config.Metrics.QPS.ScaleDownThreshold >= config.Metrics.QPS.ScaleUpThreshold {
			config.Metrics.QPS.ScaleDownThreshold = 200 // 默认200 QPS
		}
		if config.Metrics.QPS.AggregationWindow <= 0 {
			config.Metrics.QPS.AggregationWindow = 2 * time.Minute
		}

		logger.Info(fmt.Sprintf("QPS指标 - 扩容阈值: %.0f, 缩容阈值: %.0f, 聚合窗口: %v",
			config.Metrics.QPS.ScaleUpThreshold, config.Metrics.QPS.ScaleDownThreshold, config.Metrics.QPS.AggregationWindow))
	}

	// 配置队列深度监控
	if config.Metrics.QueueLength.Enable {
		logger.Info("启用队列深度扩缩容监控")

		if config.Metrics.QueueLength.ScaleUpThreshold <= 0 {
			config.Metrics.QueueLength.ScaleUpThreshold = 100 // 默认队列深度100
		}
		if config.Metrics.QueueLength.ScaleDownThreshold <= 0 || config.Metrics.QueueLength.ScaleDownThreshold >= config.Metrics.QueueLength.ScaleUpThreshold {
			config.Metrics.QueueLength.ScaleDownThreshold = 10 // 默认队列深度10
		}
		if config.Metrics.QueueLength.AggregationWindow <= 0 {
			config.Metrics.QueueLength.AggregationWindow = 1 * time.Minute
		}
	}

	// 配置自定义指标
	for i, metric := range config.Metrics.CustomMetrics {
		if metric.Name == "" {
			return fmt.Errorf("自定义指标 %d 的名称不能为空", i)
		}
		if metric.Query == "" {
			return fmt.Errorf("自定义指标 %s 的查询语句不能为空", metric.Name)
		}
		if metric.AggregationWindow <= 0 {
			config.Metrics.CustomMetrics[i].AggregationWindow = 5 * time.Minute
		}

		logger.Info(fmt.Sprintf("配置自定义指标: %s (查询: %s)", metric.Name, metric.Query))
	}

	// 配置预测模型
	if config.Prediction.Enable {
		logger.Info(fmt.Sprintf("启用扩缩容预测，算法: %s", config.Prediction.Algorithm))

		// 验证预测配置
		if config.Prediction.Algorithm == "" {
			config.Prediction.Algorithm = "linear_regression" // 默认线性回归
		}
		validAlgorithms := []string{"linear_regression", "arima", "lstm", "prophet"}
		validAlgorithm := false
		for _, algo := range validAlgorithms {
			if config.Prediction.Algorithm == algo {
				validAlgorithm = true
				break
			}
		}
		if !validAlgorithm {
			return fmt.Errorf("无效的预测算法: %s，支持的算法: %v", config.Prediction.Algorithm, validAlgorithms)
		}

		if config.Prediction.WindowSize <= 0 {
			config.Prediction.WindowSize = 60 // 默认60个数据点
		}
		if config.Prediction.PredictionHorizon <= 0 {
			config.Prediction.PredictionHorizon = 10 * time.Minute // 默认预测10分钟
		}
		if config.Prediction.UpdateInterval <= 0 {
			config.Prediction.UpdateInterval = 5 * time.Minute // 默认5分钟更新一次模型
		}

		logger.Info(fmt.Sprintf("预测配置 - 窗口大小: %d, 预测时长: %v, 更新间隔: %v",
			config.Prediction.WindowSize, config.Prediction.PredictionHorizon, config.Prediction.UpdateInterval))
	}

	// 配置扩缩容条件
	for i, condition := range config.Conditions {
		if condition.Metric == "" {
			return fmt.Errorf("扩缩容条件 %d 的指标不能为空", i)
		}
		if condition.Operator == "" {
			return fmt.Errorf("扩缩容条件 %s 的操作符不能为空", condition.Metric)
		}
		if condition.Duration <= 0 {
			config.Conditions[i].Duration = 5 * time.Minute
		}

		logger.Info(fmt.Sprintf("配置扩缩容条件: %s %s %.2f (持续时间: %v)", condition.Metric, condition.Operator, condition.Threshold, condition.Duration))
	}

	// 配置监控间隔
	if config.MonitoringInterval <= 0 {
		config.MonitoringInterval = 30 * time.Second
	}

	logger.Info(fmt.Sprintf("监控间隔: %v", config.MonitoringInterval))
	logger.Info("自动扩缩容监控配置完成")
	return nil
}

// DistributedConfigManager 分布式配置管理器
type DistributedConfigManager struct {
	mu           sync.RWMutex
	config       *DistributedConfig
	configPath   string
	watchers     []ConfigWatcher
	lastModified time.Time
}

// ConfigWatcher 配置监听器接口
type ConfigWatcher interface {
	OnConfigChanged(oldConfig, newConfig *DistributedConfig) error
}

// NewDistributedConfigManager 创建分布式配置管理器
func NewDistributedConfigManager(configPath string) *DistributedConfigManager {
	return &DistributedConfigManager{
		configPath: configPath,
		config:     GetDefaultDistributedConfig(),
		watchers:   make([]ConfigWatcher, 0),
	}
}

// LoadConfig 加载配置文件
func (dcm *DistributedConfigManager) LoadConfig() error {
	dcm.mu.Lock()
	defer dcm.mu.Unlock()

	if dcm.configPath == "" {
		return fmt.Errorf("配置文件路径为空")
	}

	// 检查文件是否存在
	if _, err := os.Stat(dcm.configPath); os.IsNotExist(err) {
		// 如果文件不存在，创建默认配置文件
		return dcm.saveConfigToFile(dcm.config)
	}

	// 读取文件内容
	data, err := os.ReadFile(dcm.configPath)
	if err != nil {
		return fmt.Errorf("读取配置文件失败: %v", err)
	}

	// 根据文件扩展名选择解析方式
	ext := filepath.Ext(dcm.configPath)
	var config DistributedConfig

	switch ext {
	case ".json":
		if err := json.Unmarshal(data, &config); err != nil {
			return fmt.Errorf("解析JSON配置文件失败: %v", err)
		}
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, &config); err != nil {
			return fmt.Errorf("解析YAML配置文件失败: %v", err)
		}
	default:
		return fmt.Errorf("不支持的配置文件格式: %s", ext)
	}

	// 验证配置
	if err := dcm.validateConfig(&config); err != nil {
		return fmt.Errorf("配置验证失败: %v", err)
	}

	oldConfig := dcm.config
	dcm.config = &config

	// 更新文件修改时间
	if stat, err := os.Stat(dcm.configPath); err == nil {
		dcm.lastModified = stat.ModTime()
	}

	// 通知监听器
	for _, watcher := range dcm.watchers {
		if err := watcher.OnConfigChanged(oldConfig, dcm.config); err != nil {
			logger.Error("配置变更通知失败: %v", err)
		}
	}

	return nil
}

// SaveConfig 保存配置到文件
func (dcm *DistributedConfigManager) SaveConfig() error {
	dcm.mu.RLock()
	defer dcm.mu.RUnlock()

	return dcm.saveConfigToFile(dcm.config)
}

// saveConfigToFile 保存配置到文件（内部方法）
func (dcm *DistributedConfigManager) saveConfigToFile(config *DistributedConfig) error {
	if dcm.configPath == "" {
		return fmt.Errorf("配置文件路径为空")
	}

	// 确保目录存在
	dir := filepath.Dir(dcm.configPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("创建配置目录失败: %v", err)
	}

	// 根据文件扩展名选择序列化方式
	ext := filepath.Ext(dcm.configPath)
	var data []byte
	var err error

	switch ext {
	case ".json":
		data, err = json.MarshalIndent(config, "", "  ")
	case ".yaml", ".yml":
		data, err = yaml.Marshal(config)
	default:
		return fmt.Errorf("不支持的配置文件格式: %s", ext)
	}

	if err != nil {
		return fmt.Errorf("序列化配置失败: %v", err)
	}

	// 写入文件
	if err := os.WriteFile(dcm.configPath, data, 0644); err != nil {
		return fmt.Errorf("写入配置文件失败: %v", err)
	}

	return nil
}

// GetConfig 获取当前配置
func (dcm *DistributedConfigManager) GetConfig() *DistributedConfig {
	dcm.mu.RLock()
	defer dcm.mu.RUnlock()

	// 返回配置的深拷贝
	configCopy := *dcm.config
	return &configCopy
}

// UpdateConfig 更新配置
func (dcm *DistributedConfigManager) UpdateConfig(newConfig *DistributedConfig) error {
	dcm.mu.Lock()
	defer dcm.mu.Unlock()

	// 验证新配置
	if err := dcm.validateConfig(newConfig); err != nil {
		return fmt.Errorf("配置验证失败: %v", err)
	}

	oldConfig := dcm.config
	dcm.config = newConfig

	// 保存到文件
	if err := dcm.saveConfigToFile(newConfig); err != nil {
		// 如果保存失败，回滚配置
		dcm.config = oldConfig
		return fmt.Errorf("保存配置失败: %v", err)
	}

	// 通知监听器
	for _, watcher := range dcm.watchers {
		if err := watcher.OnConfigChanged(oldConfig, dcm.config); err != nil {
			logger.Error("配置变更通知失败: %v", err)
		}
	}

	return nil
}

// AddWatcher 添加配置监听器
func (dcm *DistributedConfigManager) AddWatcher(watcher ConfigWatcher) {
	dcm.mu.Lock()
	defer dcm.mu.Unlock()

	dcm.watchers = append(dcm.watchers, watcher)
}

// RemoveWatcher 移除配置监听器
func (dcm *DistributedConfigManager) RemoveWatcher(watcher ConfigWatcher) {
	dcm.mu.Lock()
	defer dcm.mu.Unlock()

	for i, w := range dcm.watchers {
		if w == watcher {
			dcm.watchers = append(dcm.watchers[:i], dcm.watchers[i+1:]...)
			break
		}
	}
}

// WatchConfig 监控配置文件变化
func (dcm *DistributedConfigManager) WatchConfig() error {
	if dcm.configPath == "" {
		return fmt.Errorf("配置文件路径为空")
	}

	go func() {
		for {
			time.Sleep(5 * time.Second) // 每5秒检查一次

			stat, err := os.Stat(dcm.configPath)
			if err != nil {
				continue
			}

			dcm.mu.RLock()
			lastModified := dcm.lastModified
			dcm.mu.RUnlock()

			if stat.ModTime().After(lastModified) {
				logger.Info("检测到配置文件变化，重新加载配置")
				if err := dcm.LoadConfig(); err != nil {
					logger.Error("重新加载配置失败: %v", err)
				}
			}
		}
	}()

	return nil
}

// validateConfig 验证配置
func (dcm *DistributedConfigManager) validateConfig(config *DistributedConfig) error {
	if config == nil {
		return fmt.Errorf("配置不能为空")
	}

	// 验证基本配置
	//if config.ClusterID == "" {
	//	return fmt.Errorf("集群ID不能为空")
	//}
	//if config.NodeID == "" {
	//	return fmt.Errorf("节点ID不能为空")
	//}
	//if config.Version == "" {
	//	config.Version = "1.0.0" // 设置默认版本
	//}

	// 验证索引配置
	if err := dcm.validateIndexConfig(&config.IndexConfig); err != nil {
		return fmt.Errorf("索引配置验证失败: %v", err)
	}

	// 验证架构配置
	if err := dcm.validateArchitectureConfig(&config.ArchitectureConfig); err != nil {
		return fmt.Errorf("架构配置验证失败: %v", err)
	}

	// 验证性能配置
	if err := dcm.validatePerformanceConfig(&config.PerformanceConfig); err != nil {
		return fmt.Errorf("性能配置验证失败: %v", err)
	}

	// 验证监控配置
	if err := dcm.validateMonitoringConfig(&config.MonitoringConfig); err != nil {
		return fmt.Errorf("监控配置验证失败: %v", err)
	}

	// 验证自动扩缩容配置
	if err := dcm.validateAutoScalingConfig(&config.AutoScalingConfig); err != nil {
		return fmt.Errorf("自动扩缩容配置验证失败: %v", err)
	}

	return nil
}

// validateMonitoringConfig 验证监控配置
func (dcm *DistributedConfigManager) validateMonitoringConfig(config *MonitoringConfig) error {
	if config == nil {
		return fmt.Errorf("监控配置不能为空")
	}

	// 验证指标收集配置
	if config.Metrics.Enable {
		if config.Metrics.Interval <= 0 {
			config.Metrics.Interval = 30 * time.Second
		}
		if config.Metrics.RetentionPeriod <= 0 {
			config.Metrics.RetentionPeriod = 7 * 24 * time.Hour // 默认7天
		}
	}

	// 验证告警配置
	if config.Alerting.Enable {
		for i, rule := range config.Alerting.Rules {
			if rule.Name == "" {
				return fmt.Errorf("告警规则 %d 的名称不能为空", i)
			}
			if rule.Expression == "" {
				return fmt.Errorf("告警规则 %s 的表达式不能为空", rule.Name)
			}
			if rule.Severity == "" {
				config.Alerting.Rules[i].Severity = "warning" // 默认警告级别
			}
		}
	}

	// 验证日志配置
	if config.Logging.Enable {
		if config.Logging.Level == "" {
			config.Logging.Level = "info" // 默认info级别
		}
		validLevels := []string{"debug", "info", "warn", "error", "fatal"}
		validLevel := false
		for _, level := range validLevels {
			if config.Logging.Level == level {
				validLevel = true
				break
			}
		}
		if !validLevel {
			return fmt.Errorf("无效的日志级别: %s，支持的级别: %v", config.Logging.Level, validLevels)
		}
	}

	return nil
}

// validateAutoScalingConfig 验证自动扩缩容配置
func (dcm *DistributedConfigManager) validateAutoScalingConfig(config *AutoScalingConfig) error {
	if config == nil {
		return fmt.Errorf("自动扩缩容配置不能为空")
	}

	if !config.Enable {
		return nil // 如果未启用，跳过验证
	}

	// 验证扩缩容策略
	if config.Strategy == "" {
		config.Strategy = "reactive" // 默认响应式策略
	}
	validStrategies := []string{"reactive", "predictive", "hybrid"}
	validStrategy := false
	for _, strategy := range validStrategies {
		if config.Strategy == strategy {
			validStrategy = true
			break
		}
	}
	if !validStrategy {
		return fmt.Errorf("无效的扩缩容策略: %s，支持的策略: %v", config.Strategy, validStrategies)
	}

	// 验证实例限制
	if config.Limits.MinComputeNodes <= 0 {
		config.Limits.MinComputeNodes = 1
	}
	if config.Limits.MaxComputeNodes <= config.Limits.MinComputeNodes {
		config.Limits.MaxComputeNodes = config.Limits.MinComputeNodes * 10
	}

	// 验证指标配置
	if config.Metrics.CPUUtilization.Enable {
		if config.Metrics.CPUUtilization.ScaleUpThreshold <= 0 || config.Metrics.CPUUtilization.ScaleUpThreshold > 1 {
			config.Metrics.CPUUtilization.ScaleUpThreshold = 0.8
		}
		if config.Metrics.CPUUtilization.ScaleDownThreshold <= 0 || config.Metrics.CPUUtilization.ScaleDownThreshold >= config.Metrics.CPUUtilization.ScaleUpThreshold {
			config.Metrics.CPUUtilization.ScaleDownThreshold = 0.3
		}
	}

	return nil
}

// validateIndexConfig 验证索引配置
func (dcm *DistributedConfigManager) validateIndexConfig(config *IndexSelectionConfig) error {
	// 验证HNSW配置
	if config.HNSWConfig.Enable {
		if config.HNSWConfig.MaxConnections <= 0 {
			return fmt.Errorf("HNSW最大连接数必须大于0")
		}
		if config.HNSWConfig.EfConstruction <= 0 {
			return fmt.Errorf("HNSW构建参数ef必须大于0")
		}
		if config.HNSWConfig.EfSearch <= 0 {
			return fmt.Errorf("HNSW搜索参数ef必须大于0")
		}
	}

	// 验证IVF配置
	if config.IVFConfig.Enable {
		if config.IVFConfig.NumClusters <= 0 {
			return fmt.Errorf("IVF聚类数量必须大于0")
		}
		if config.IVFConfig.Nprobe <= 0 {
			return fmt.Errorf("IVF探测数量必须大于0")
		}
		if config.IVFConfig.TrainingRatio <= 0 || config.IVFConfig.TrainingRatio > 1 {
			return fmt.Errorf("IVF训练比例必须在0-1之间")
		}
	}

	// 验证PQ配置
	if config.PQConfig.Enable {
		if config.PQConfig.NumSubVectors <= 0 {
			return fmt.Errorf("PQ子向量数量必须大于0")
		}
		if config.PQConfig.NumCentroids <= 0 {
			return fmt.Errorf("PQ聚类中心数量必须大于0")
		}
	}

	return nil
}

// validateArchitectureConfig 验证架构配置
func (dcm *DistributedConfigManager) validateArchitectureConfig(config *DistributedArchitectureConfig) error {
	// 验证分片配置
	if config.ShardingConfig.NumShards <= 0 {
		return fmt.Errorf("分片数量必须大于0")
	}
	if config.ShardingConfig.ReplicationFactor <= 0 {
		return fmt.Errorf("副本因子必须大于0")
	}

	// 验证计算存储分离配置
	if config.ComputeStorageSeparation.Enable {
		if config.ComputeStorageSeparation.ComputeNodes.MinNodes <= 0 {
			return fmt.Errorf("计算节点最小数量必须大于0")
		}
		if config.ComputeStorageSeparation.ComputeNodes.MaxNodes < config.ComputeStorageSeparation.ComputeNodes.MinNodes {
			return fmt.Errorf("计算节点最大数量不能小于最小数量")
		}
		if config.ComputeStorageSeparation.StorageNodes.MinNodes <= 0 {
			return fmt.Errorf("存储节点最小数量必须大于0")
		}
	}

	return nil
}

// validatePerformanceConfig 验证性能配置
func (dcm *DistributedConfigManager) validatePerformanceConfig(config *PerformanceOptimizationConfig) error {
	if config == nil {
		return fmt.Errorf("性能配置不能为空")
	}

	// 验证查询加速配置
	if config.QueryAcceleration.Enable {
		if config.QueryAcceleration.ParallelSearch.Enable {
			if config.QueryAcceleration.ParallelSearch.MaxWorkers <= 0 {
				return fmt.Errorf("并行搜索最大工作线程数必须大于0")
			}
		}
		if config.QueryAcceleration.MultiStageSearch.Enable {
			if config.QueryAcceleration.MultiStageSearch.CoarseCandidates <= 0 {
				return fmt.Errorf("多阶段搜索粗糙候选数必须大于0")
			}
			if config.QueryAcceleration.MultiStageSearch.RefinementRatio <= 0 || config.QueryAcceleration.MultiStageSearch.RefinementRatio > 1 {
				return fmt.Errorf("多阶段搜索精化比例必须在0-1之间")
			}
		}
	}

	// 验证并发控制配置
	if config.ConcurrencyControl.MaxConcurrentQueries <= 0 {
		return fmt.Errorf("最大并发查询数必须大于0")
	}
	if config.ConcurrencyControl.MaxConcurrentInserts <= 0 {
		return fmt.Errorf("最大并发插入数必须大于0")
	}

	// 验证内存管理配置
	if config.MemoryManagement.MaxMemoryUsage <= 0 {
		return fmt.Errorf("最大内存使用量必须大于0")
	}
	if config.MemoryManagement.GCThreshold <= 0 || config.MemoryManagement.GCThreshold > 1 {
		return fmt.Errorf("GC阈值必须在0-1之间")
	}

	// 验证批处理配置
	if config.BatchProcessing.Enable {
		if config.BatchProcessing.BatchSize <= 0 {
			return fmt.Errorf("批处理大小必须大于0")
		}
		if config.BatchProcessing.MaxBatchDelay <= 0 {
			return fmt.Errorf("最大批处理延迟必须大于0")
		}
	}

	// 验证预取配置
	if config.Prefetching.Enable {
		if config.Prefetching.PrefetchSize <= 0 {
			return fmt.Errorf("预取大小必须大于0")
		}
		if config.Prefetching.PrefetchThreshold <= 0 || config.Prefetching.PrefetchThreshold > 1 {
			return fmt.Errorf("预取阈值必须在0-1之间")
		}
	}

	return nil
}

// SelectOptimalIndexStrategy 选择最优索引策略
func (dcm *DistributedConfigManager) SelectOptimalIndexStrategy(dataSize int, dimension int, qualityRequirement float64, latencyRequirement time.Duration) (string, error) {
	dcm.mu.RLock()
	config := dcm.config
	dcm.mu.RUnlock()

	if !config.IndexConfig.AdaptiveSelection.Enable {
		return "hnsw", nil // 默认使用HNSW
	}

	// 根据数据规模选择
	thresholds := config.IndexConfig.AdaptiveSelection.DataSizeThresholds
	var sizeCategory string
	if dataSize < thresholds.SmallDataset {
		sizeCategory = "small"
	} else if dataSize < thresholds.MediumDataset {
		sizeCategory = "medium"
	} else if dataSize < thresholds.LargeDataset {
		sizeCategory = "large"
	} else {
		sizeCategory = "ultra_large"
	}

	// 根据质量要求选择
	qualityThresholds := config.IndexConfig.AdaptiveSelection.QualityThresholds
	var qualityCategory string
	if qualityRequirement >= qualityThresholds.HighQuality {
		qualityCategory = "high"
	} else if qualityRequirement >= qualityThresholds.MediumQuality {
		qualityCategory = "medium"
	} else {
		qualityCategory = "acceptable"
	}

	// 根据延迟要求选择
	latencyThresholds := config.IndexConfig.AdaptiveSelection.PerformanceThresholds
	var latencyCategory string
	if latencyRequirement <= latencyThresholds.LowLatency {
		latencyCategory = "low"
	} else if latencyRequirement <= latencyThresholds.MediumLatency {
		latencyCategory = "medium"
	} else {
		latencyCategory = "high"
	}

	// 应用选择规则
	for _, rule := range config.IndexConfig.AdaptiveSelection.SelectionRules {
		if dcm.matchesCondition(&rule.Condition, dataSize, dimension, qualityRequirement, latencyRequirement, sizeCategory, qualityCategory, latencyCategory) {
			return rule.RecommendedIndex, nil
		}
	}

	// 如果没有匹配的规则，使用默认策略
	return dcm.getDefaultIndexForCategory(sizeCategory, qualityCategory, latencyCategory), nil
}

// matchesCondition 检查是否匹配条件
func (dcm *DistributedConfigManager) matchesCondition(condition *IndexCondition, dataSize, dimension int, quality float64, latency time.Duration, sizeCategory, qualityCategory, latencyCategory string) bool {
	// 检查数据规模范围
	if len(condition.DataSizeRange) == 2 {
		if dataSize < condition.DataSizeRange[0] || dataSize > condition.DataSizeRange[1] {
			return false
		}
	}

	// 检查维度范围
	if len(condition.DimensionRange) == 2 {
		if dimension < condition.DimensionRange[0] || dimension > condition.DimensionRange[1] {
			return false
		}
	}

	// 检查质量范围
	if len(condition.QualityRange) == 2 {
		if quality < condition.QualityRange[0] || quality > condition.QualityRange[1] {
			return false
		}
	}

	// 检查延迟范围
	if len(condition.LatencyRange) == 2 {
		if latency < condition.LatencyRange[0] || latency > condition.LatencyRange[1] {
			return false
		}
	}

	return true
}

// getDefaultIndexForCategory 根据分类获取默认索引
func (dcm *DistributedConfigManager) getDefaultIndexForCategory(sizeCategory, qualityCategory, latencyCategory string) string {
	// 根据不同的分类组合选择最适合的索引
	switch {
	case sizeCategory == "small" && qualityCategory == "high":
		return "hnsw"
	case sizeCategory == "large" && latencyCategory == "low":
		return "lsh"
	case sizeCategory == "ultra_large":
		return "ivf"
	case qualityCategory == "acceptable" && latencyCategory == "low":
		return "lsh"
	default:
		return "hnsw"
	}
}
