package vector

import (
	"time"
)

// MonitoringConfig 监控配置
type MonitoringConfig struct {
	Metrics     MetricsConfig     `json:"metrics" yaml:"metrics"`
	Alerting    AlertingConfig    `json:"alerting" yaml:"alerting"`
	AutoScaling AutoScalingConfig `json:"auto_scaling" yaml:"auto_scaling"`
	Logging     LoggingConfig     `json:"logging" yaml:"logging"`
	Tracing     TracingConfig     `json:"tracing" yaml:"tracing"`
	Profiling   ProfilingConfig   `json:"profiling" yaml:"profiling"`
	Dashboard   DashboardConfig   `json:"dashboard" yaml:"dashboard"`
	HealthCheck HealthCheckConfig `json:"health_check" yaml:"health_check"`
}

// MetricsConfig 指标配置
type MetricsConfig struct {
	Enable             bool                     `json:"enable" yaml:"enable"`
	Interval           int                      `json:"interval"`
	CollectionInterval time.Duration            `json:"collection_interval" yaml:"collection_interval"`
	RetentionPeriod    time.Duration            `json:"retention_period" yaml:"retention_period"`
	Exporter           MetricsExporterConfig    `json:"exporter" yaml:"exporter"`
	Performance        PerformanceMetricsConfig `json:"performance" yaml:"performance"`
	Business           BusinessMetricsConfig    `json:"business" yaml:"business"`
	System             SystemMetricsConfig      `json:"system" yaml:"system"`
	Custom             CustomMetricsConfig      `json:"custom" yaml:"custom"`
}

// MetricsExporterConfig 指标导出配置
type MetricsExporterConfig struct {
	Prometheus    PrometheusConfig    `json:"prometheus" yaml:"prometheus"`
	InfluxDB      InfluxDBConfig      `json:"influxdb" yaml:"influxdb"`
	Elasticsearch ElasticsearchConfig `json:"elasticsearch" yaml:"elasticsearch"`
	CloudWatch    CloudWatchConfig    `json:"cloudwatch" yaml:"cloudwatch"`
	Datadog       DatadogConfig       `json:"datadog" yaml:"datadog"`
}

// PrometheusConfig Prometheus配置
type PrometheusConfig struct {
	Enable         bool              `json:"enable" yaml:"enable"`
	Endpoint       string            `json:"endpoint" yaml:"endpoint"`
	Port           int               `json:"port" yaml:"port"`
	Path           string            `json:"path" yaml:"path"`
	PushGateway    string            `json:"push_gateway" yaml:"push_gateway"`
	Labels         map[string]string `json:"labels" yaml:"labels"`
	ScrapeInterval time.Duration     `json:"scrape_interval" yaml:"scrape_interval"`
	Timeout        time.Duration     `json:"timeout" yaml:"timeout"`
}

// InfluxDBConfig InfluxDB配置
type InfluxDBConfig struct {
	Enable          bool          `json:"enable" yaml:"enable"`
	URL             string        `json:"url" yaml:"url"`
	Database        string        `json:"database" yaml:"database"`
	Username        string        `json:"username" yaml:"username"`
	Password        string        `json:"password" yaml:"password"`
	RetentionPolicy string        `json:"retention_policy" yaml:"retention_policy"`
	BatchSize       int           `json:"batch_size" yaml:"batch_size"`
	FlushInterval   time.Duration `json:"flush_interval" yaml:"flush_interval"`
}

// ElasticsearchConfig Elasticsearch配置
type ElasticsearchConfig struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	URLs              []string      `json:"urls" yaml:"urls"`
	Index             string        `json:"index" yaml:"index"`
	Username          string        `json:"username" yaml:"username"`
	Password          string        `json:"password" yaml:"password"`
	BatchSize         int           `json:"batch_size" yaml:"batch_size"`
	FlushInterval     time.Duration `json:"flush_interval" yaml:"flush_interval"`
	CompressionEnable bool          `json:"compression_enable" yaml:"compression_enable"`
}

// CloudWatchConfig CloudWatch配置
type CloudWatchConfig struct {
	Enable          bool          `json:"enable" yaml:"enable"`
	Region          string        `json:"region" yaml:"region"`
	Namespace       string        `json:"namespace" yaml:"namespace"`
	AccessKeyID     string        `json:"access_key_id" yaml:"access_key_id"`
	SecretAccessKey string        `json:"secret_access_key" yaml:"secret_access_key"`
	BatchSize       int           `json:"batch_size" yaml:"batch_size"`
	FlushInterval   time.Duration `json:"flush_interval" yaml:"flush_interval"`
}

// DatadogConfig Datadog配置
type DatadogConfig struct {
	Enable        bool              `json:"enable" yaml:"enable"`
	APIKey        string            `json:"api_key" yaml:"api_key"`
	Site          string            `json:"site" yaml:"site"`
	Tags          map[string]string `json:"tags" yaml:"tags"`
	BatchSize     int               `json:"batch_size" yaml:"batch_size"`
	FlushInterval time.Duration     `json:"flush_interval" yaml:"flush_interval"`
}

// PerformanceMetricsConfig 性能指标配置
type PerformanceMetricsConfig struct {
	SearchLatency    LatencyMetricsConfig    `json:"search_latency" yaml:"search_latency"`
	IndexBuildTime   DurationMetricsConfig   `json:"index_build_time" yaml:"index_build_time"`
	Throughput       ThroughputMetricsConfig `json:"throughput" yaml:"throughput"`
	ResourceUsage    ResourceMetricsConfig   `json:"resource_usage" yaml:"resource_usage"`
	CachePerformance CacheMetricsConfig      `json:"cache_performance" yaml:"cache_performance"`
}

// LatencyMetricsConfig 延迟指标配置
type LatencyMetricsConfig struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	Percentiles       []float64     `json:"percentiles" yaml:"percentiles"`
	Buckets           []float64     `json:"buckets" yaml:"buckets"`
	SamplingRate      float64       `json:"sampling_rate" yaml:"sampling_rate"`
	AggregationWindow time.Duration `json:"aggregation_window" yaml:"aggregation_window"`
}

// DurationMetricsConfig 持续时间指标配置
type DurationMetricsConfig struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	Buckets           []float64     `json:"buckets" yaml:"buckets"`
	AggregationWindow time.Duration `json:"aggregation_window" yaml:"aggregation_window"`
}

// ThroughputMetricsConfig 吞吐量指标配置
type ThroughputMetricsConfig struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	QPS               bool          `json:"qps" yaml:"qps"`
	RPS               bool          `json:"rps" yaml:"rps"`
	Bandwidth         bool          `json:"bandwidth" yaml:"bandwidth"`
	AggregationWindow time.Duration `json:"aggregation_window" yaml:"aggregation_window"`
}

// ResourceMetricsConfig 资源指标配置
type ResourceMetricsConfig struct {
	CPU     CPUMetricsConfig     `json:"cpu" yaml:"cpu"`
	Memory  MemoryMetricsConfig  `json:"memory" yaml:"memory"`
	Disk    DiskMetricsConfig    `json:"disk" yaml:"disk"`
	Network NetworkMetricsConfig `json:"network" yaml:"network"`
	GPU     GPUMetricsConfig     `json:"gpu" yaml:"gpu"`
}

// CPUMetricsConfig CPU指标配置
type CPUMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	Utilization        bool          `json:"utilization" yaml:"utilization"`
	LoadAverage        bool          `json:"load_average" yaml:"load_average"`
	ContextSwitches    bool          `json:"context_switches" yaml:"context_switches"`
	Interrupts         bool          `json:"interrupts" yaml:"interrupts"`
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
}

// MemoryMetricsConfig 内存指标配置
type MemoryMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	Usage              bool          `json:"usage" yaml:"usage"`
	Available          bool          `json:"available" yaml:"available"`
	Swap               bool          `json:"swap" yaml:"swap"`
	PageFaults         bool          `json:"page_faults" yaml:"page_faults"`
	GCStats            bool          `json:"gc_stats" yaml:"gc_stats"`
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
}

// DiskMetricsConfig 磁盘指标配置
type DiskMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	Usage              bool          `json:"usage" yaml:"usage"`
	IOPS               bool          `json:"iops" yaml:"iops"`
	Throughput         bool          `json:"throughput" yaml:"throughput"`
	Latency            bool          `json:"latency" yaml:"latency"`
	QueueDepth         bool          `json:"queue_depth" yaml:"queue_depth"`
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
}

// NetworkMetricsConfig 网络指标配置
type NetworkMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	Bandwidth          bool          `json:"bandwidth" yaml:"bandwidth"`
	Packets            bool          `json:"packets" yaml:"packets"`
	Errors             bool          `json:"errors" yaml:"errors"`
	Drops              bool          `json:"drops" yaml:"drops"`
	Connections        bool          `json:"connections" yaml:"connections"`
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
}

// GPUMetricsConfig GPU指标配置
type GPUMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	Utilization        bool          `json:"utilization" yaml:"utilization"`
	MemoryUsage        bool          `json:"memory_usage" yaml:"memory_usage"`
	Temperature        bool          `json:"temperature" yaml:"temperature"`
	PowerUsage         bool          `json:"power_usage" yaml:"power_usage"`
	FanSpeed           bool          `json:"fan_speed" yaml:"fan_speed"`
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
}

// BusinessMetricsConfig 业务指标配置
type BusinessMetricsConfig struct {
	Enable           bool                      `json:"enable" yaml:"enable"`
	SearchAccuracy   AccuracyMetricsConfig     `json:"search_accuracy" yaml:"search_accuracy"`
	UserSatisfaction SatisfactionMetricsConfig `json:"user_satisfaction" yaml:"user_satisfaction"`
	DataQuality      QualityMetricsConfig      `json:"data_quality" yaml:"data_quality"`
	CostMetrics      CostMetricsConfig         `json:"cost_metrics" yaml:"cost_metrics"`
}

// AccuracyMetricsConfig 准确性指标配置
type AccuracyMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	Recall             bool          `json:"recall" yaml:"recall"`
	Precision          bool          `json:"precision" yaml:"precision"`
	F1Score            bool          `json:"f1_score" yaml:"f1_score"`
	NDCG               bool          `json:"ndcg" yaml:"ndcg"`
	MRR                bool          `json:"mrr" yaml:"mrr"`
	EvaluationInterval time.Duration `json:"evaluation_interval" yaml:"evaluation_interval"`
}

// SatisfactionMetricsConfig 满意度指标配置
type SatisfactionMetricsConfig struct {
	Enable         bool          `json:"enable" yaml:"enable"`
	ClickThrough   bool          `json:"click_through" yaml:"click_through"`
	DwellTime      bool          `json:"dwell_time" yaml:"dwell_time"`
	BounceRate     bool          `json:"bounce_rate" yaml:"bounce_rate"`
	ConversionRate bool          `json:"conversion_rate" yaml:"conversion_rate"`
	UserFeedback   bool          `json:"user_feedback" yaml:"user_feedback"`
	TrackingWindow time.Duration `json:"tracking_window" yaml:"tracking_window"`
}

// QualityMetricsConfig 质量指标配置
type QualityMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	DataFreshness      bool          `json:"data_freshness" yaml:"data_freshness"`
	DataCompleteness   bool          `json:"data_completeness" yaml:"data_completeness"`
	DataConsistency    bool          `json:"data_consistency" yaml:"data_consistency"`
	IndexHealth        bool          `json:"index_health" yaml:"index_health"`
	ValidationInterval time.Duration `json:"validation_interval" yaml:"validation_interval"`
}

// CostMetricsConfig 成本指标配置
type CostMetricsConfig struct {
	Enable              bool          `json:"enable" yaml:"enable"`
	ComputeCost         bool          `json:"compute_cost" yaml:"compute_cost"`
	StorageCost         bool          `json:"storage_cost" yaml:"storage_cost"`
	NetworkCost         bool          `json:"network_cost" yaml:"network_cost"`
	OperationalCost     bool          `json:"operational_cost" yaml:"operational_cost"`
	CostPerQuery        bool          `json:"cost_per_query" yaml:"cost_per_query"`
	CalculationInterval time.Duration `json:"calculation_interval" yaml:"calculation_interval"`
}

// SystemMetricsConfig 系统指标配置
type SystemMetricsConfig struct {
	Enable          bool                   `json:"enable" yaml:"enable"`
	Goroutines      GoroutineMetricsConfig `json:"goroutines" yaml:"goroutines"`
	GC              GCMetricsConfig        `json:"gc" yaml:"gc"`
	FileDescriptors FDMetricsConfig        `json:"file_descriptors" yaml:"file_descriptors"`
	ProcessMetrics  ProcessMetricsConfig   `json:"process_metrics" yaml:"process_metrics"`
}

// GoroutineMetricsConfig Goroutine指标配置
type GoroutineMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	Count              bool          `json:"count" yaml:"count"`
	StackTrace         bool          `json:"stack_trace" yaml:"stack_trace"`
	LeakDetection      bool          `json:"leak_detection" yaml:"leak_detection"`
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
}

// GCMetricsConfig GC指标配置
type GCMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	PauseTime          bool          `json:"pause_time" yaml:"pause_time"`
	Frequency          bool          `json:"frequency" yaml:"frequency"`
	HeapSize           bool          `json:"heap_size" yaml:"heap_size"`
	Allocations        bool          `json:"allocations" yaml:"allocations"`
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
}

// FDMetricsConfig 文件描述符指标配置
type FDMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	OpenCount          bool          `json:"open_count" yaml:"open_count"`
	MaxCount           bool          `json:"max_count" yaml:"max_count"`
	UsageRatio         bool          `json:"usage_ratio" yaml:"usage_ratio"`
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
}

// ProcessMetricsConfig 进程指标配置
type ProcessMetricsConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	CPUTime            bool          `json:"cpu_time" yaml:"cpu_time"`
	MemoryUsage        bool          `json:"memory_usage" yaml:"memory_usage"`
	ThreadCount        bool          `json:"thread_count" yaml:"thread_count"`
	StartTime          bool          `json:"start_time" yaml:"start_time"`
	CollectionInterval time.Duration `json:"collection_interval" yaml:"collection_interval"`
}

// CustomMetricsConfig 自定义指标配置
type CustomMetricsConfig struct {
	Enable  bool                 `json:"enable" yaml:"enable"`
	Metrics []CustomMetricConfig `json:"metrics" yaml:"metrics"`
	Plugins []MetricPluginConfig `json:"plugins" yaml:"plugins"`
}

// CustomMetricConfig 自定义指标配置
type CustomMetricConfig struct {
	Name               string              `json:"name" yaml:"name"`
	Type               string              `json:"type" yaml:"type"` // "counter", "gauge", "histogram", "summary"
	Description        string              `json:"description" yaml:"description"`
	Labels             []string            `json:"labels" yaml:"labels"`
	Buckets            []float64           `json:"buckets" yaml:"buckets"`
	Objectives         map[float64]float64 `json:"objectives" yaml:"objectives"`
	CollectionInterval time.Duration       `json:"collection_interval" yaml:"collection_interval"`
}

// MetricPluginConfig 指标插件配置
type MetricPluginConfig struct {
	Name   string                 `json:"name" yaml:"name"`
	Path   string                 `json:"path" yaml:"path"`
	Config map[string]interface{} `json:"config" yaml:"config"`
	Enable bool                   `json:"enable" yaml:"enable"`
}

// AlertingConfig 告警配置
type AlertingConfig struct {
	Enable       bool               `json:"enable" yaml:"enable"`
	Rules        []AlertRule        `json:"rules" yaml:"rules"`
	Notification NotificationConfig `json:"notification" yaml:"notification"`
	Escalation   EscalationConfig   `json:"escalation" yaml:"escalation"`
	Suppression  SuppressionConfig  `json:"suppression" yaml:"suppression"`
	Inhibition   InhibitionConfig   `json:"inhibition" yaml:"inhibition"`
}

// AlertRule 告警规则
type AlertRule struct {
	Expression    string            `json:"expression" yaml:"expression"`
	Name          string            `json:"name" yaml:"name"`
	Description   string            `json:"description" yaml:"description"`
	Metric        string            `json:"metric" yaml:"metric"`
	Condition     AlertCondition    `json:"condition" yaml:"condition"`
	Duration      time.Duration     `json:"duration" yaml:"duration"`
	Severity      string            `json:"severity" yaml:"severity"` // "critical", "warning", "info"
	Labels        map[string]string `json:"labels" yaml:"labels"`
	Annotations   map[string]string `json:"annotations" yaml:"annotations"`
	Enabled       bool              `json:"enabled" yaml:"enabled"`
	Runbook       string            `json:"runbook" yaml:"runbook"`
	lastTriggered time.Time         // 内部字段，不导出
}

// AlertCondition 告警条件
type AlertCondition struct {
	Operator    string            `json:"operator" yaml:"operator"` // ">", "<", ">=", "<=", "==", "!="
	Threshold   float64           `json:"threshold" yaml:"threshold"`
	Aggregation string            `json:"aggregation" yaml:"aggregation"` // "avg", "sum", "min", "max", "count"
	TimeWindow  time.Duration     `json:"time_window" yaml:"time_window"`
	GroupBy     []string          `json:"group_by" yaml:"group_by"`
	Filters     map[string]string `json:"filters" yaml:"filters"`
}

// NotificationConfig 通知配置
type NotificationConfig struct {
	Channels     []NotificationChannel `json:"channels" yaml:"channels"`
	Templates    NotificationTemplates `json:"templates" yaml:"templates"`
	RateLimiting RateLimitingConfig    `json:"rate_limiting" yaml:"rate_limiting"`
	Grouping     GroupingConfig        `json:"grouping" yaml:"grouping"`
}

// NotificationChannel 通知渠道
type NotificationChannel struct {
	Name           string                 `json:"name" yaml:"name"`
	Type           string                 `json:"type" yaml:"type"` // "email", "slack", "webhook", "sms", "pagerduty"
	Config         map[string]interface{} `json:"config" yaml:"config"`
	Enabled        bool                   `json:"enabled" yaml:"enabled"`
	SeverityFilter []string               `json:"severity_filter" yaml:"severity_filter"`
	LabelMatchers  map[string]string      `json:"label_matchers" yaml:"label_matchers"`
}

// NotificationTemplates 通知模板
type NotificationTemplates struct {
	Subject        string `json:"subject" yaml:"subject"`
	Body           string `json:"body" yaml:"body"`
	SlackMessage   string `json:"slack_message" yaml:"slack_message"`
	WebhookPayload string `json:"webhook_payload" yaml:"webhook_payload"`
	SMSMessage     string `json:"sms_message" yaml:"sms_message"`
}

// RateLimitingConfig 限流配置

// GroupingConfig 分组配置
type GroupingConfig struct {
	Enable         bool          `json:"enable" yaml:"enable"`
	GroupBy        []string      `json:"group_by" yaml:"group_by"`
	GroupWait      time.Duration `json:"group_wait" yaml:"group_wait"`
	GroupInterval  time.Duration `json:"group_interval" yaml:"group_interval"`
	RepeatInterval time.Duration `json:"repeat_interval" yaml:"repeat_interval"`
}

// EscalationConfig 升级配置
type EscalationConfig struct {
	Enable          bool              `json:"enable" yaml:"enable"`
	Levels          []EscalationLevel `json:"levels" yaml:"levels"`
	AutoEscalation  bool              `json:"auto_escalation" yaml:"auto_escalation"`
	EscalationDelay time.Duration     `json:"escalation_delay" yaml:"escalation_delay"`
}

// EscalationLevel 升级级别
type EscalationLevel struct {
	Level      int           `json:"level" yaml:"level"`
	Name       string        `json:"name" yaml:"name"`
	Channels   []string      `json:"channels" yaml:"channels"`
	Delay      time.Duration `json:"delay" yaml:"delay"`
	Conditions []string      `json:"conditions" yaml:"conditions"`
	Actions    []string      `json:"actions" yaml:"actions"`
}

// SuppressionConfig 抑制配置
type SuppressionConfig struct {
	Enable          bool                  `json:"enable" yaml:"enable"`
	Rules           []SuppressionRule     `json:"rules" yaml:"rules"`
	MaintenanceMode MaintenanceModeConfig `json:"maintenance_mode" yaml:"maintenance_mode"`
}

// SuppressionRule 抑制规则
type SuppressionRule struct {
	Name      string            `json:"name" yaml:"name"`
	Matchers  map[string]string `json:"matchers" yaml:"matchers"`
	StartTime string            `json:"start_time" yaml:"start_time"`
	EndTime   string            `json:"end_time" yaml:"end_time"`
	Duration  time.Duration     `json:"duration" yaml:"duration"`
	Comment   string            `json:"comment" yaml:"comment"`
	CreatedBy string            `json:"created_by" yaml:"created_by"`
	Enabled   bool              `json:"enabled" yaml:"enabled"`
}

// MaintenanceModeConfig 维护模式配置
type MaintenanceModeConfig struct {
	Enable       bool          `json:"enable" yaml:"enable"`
	Schedule     string        `json:"schedule" yaml:"schedule"` // cron expression
	Duration     time.Duration `json:"duration" yaml:"duration"`
	NotifyBefore time.Duration `json:"notify_before" yaml:"notify_before"`
	AutoEnable   bool          `json:"auto_enable" yaml:"auto_enable"`
}

// InhibitionConfig 抑制配置
type InhibitionConfig struct {
	Enable bool             `json:"enable" yaml:"enable"`
	Rules  []InhibitionRule `json:"rules" yaml:"rules"`
}

// InhibitionRule 抑制规则
type InhibitionRule struct {
	SourceMatchers map[string]string `json:"source_matchers" yaml:"source_matchers"`
	TargetMatchers map[string]string `json:"target_matchers" yaml:"target_matchers"`
	Equal          []string          `json:"equal" yaml:"equal"`
}

// AutoScalingConfig 自动扩缩容配置
type AutoScalingConfig struct {
	Enable     bool                 `json:"enable" yaml:"enable"`
	Strategy   string               `json:"strategy" yaml:"strategy"` // "reactive", "predictive", "hybrid"
	Metrics    ScalingMetricsConfig `json:"metrics" yaml:"metrics"`
	Policies   []ScalingPolicy      `json:"policies" yaml:"policies"`
	Limits     ScalingLimitsConfig  `json:"limits" yaml:"limits"`
	Cooldown   CooldownConfig       `json:"cooldown" yaml:"cooldown"`
	Prediction PredictionConfig     `json:"prediction" yaml:"prediction"`
}

// ScalingMetricsConfig 扩缩容指标配置
type ScalingMetricsConfig struct {
	CPUUtilization    CPUScalingConfig      `json:"cpu_utilization" yaml:"cpu_utilization"`
	MemoryUtilization MemoryScalingConfig   `json:"memory_utilization" yaml:"memory_utilization"`
	QueryLatency      LatencyScalingConfig  `json:"query_latency" yaml:"query_latency"`
	QPS               QPSScalingConfig      `json:"qps" yaml:"qps"`
	QueueLength       QueueScalingConfig    `json:"queue_length" yaml:"queue_length"`
	CustomMetrics     []CustomScalingMetric `json:"custom_metrics" yaml:"custom_metrics"`
}

// CPUScalingConfig CPU扩缩容配置
type CPUScalingConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	ScaleUpThreshold   float64       `json:"scale_up_threshold" yaml:"scale_up_threshold"`
	ScaleDownThreshold float64       `json:"scale_down_threshold" yaml:"scale_down_threshold"`
	AggregationWindow  time.Duration `json:"aggregation_window" yaml:"aggregation_window"`
	Weight             float64       `json:"weight" yaml:"weight"`
}

// MemoryScalingConfig 内存扩缩容配置
type MemoryScalingConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	ScaleUpThreshold   float64       `json:"scale_up_threshold" yaml:"scale_up_threshold"`
	ScaleDownThreshold float64       `json:"scale_down_threshold" yaml:"scale_down_threshold"`
	AggregationWindow  time.Duration `json:"aggregation_window" yaml:"aggregation_window"`
	Weight             float64       `json:"weight" yaml:"weight"`
}

// LatencyScalingConfig 延迟扩缩容配置
type LatencyScalingConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	ScaleUpThreshold   time.Duration `json:"scale_up_threshold" yaml:"scale_up_threshold"`
	ScaleDownThreshold time.Duration `json:"scale_down_threshold" yaml:"scale_down_threshold"`
	Percentile         float64       `json:"percentile" yaml:"percentile"`
	AggregationWindow  time.Duration `json:"aggregation_window" yaml:"aggregation_window"`
	Weight             float64       `json:"weight" yaml:"weight"`
}

// QPSScalingConfig QPS扩缩容配置
type QPSScalingConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	ScaleUpThreshold   float64       `json:"scale_up_threshold" yaml:"scale_up_threshold"`
	ScaleDownThreshold float64       `json:"scale_down_threshold" yaml:"scale_down_threshold"`
	AggregationWindow  time.Duration `json:"aggregation_window" yaml:"aggregation_window"`
	Weight             float64       `json:"weight" yaml:"weight"`
}

// QueueScalingConfig 队列扩缩容配置
type QueueScalingConfig struct {
	Enable             bool          `json:"enable" yaml:"enable"`
	ScaleUpThreshold   float64       `json:"scale_up_threshold" yaml:"scale_up_threshold"`
	ScaleDownThreshold float64       `json:"scale_down_threshold" yaml:"scale_down_threshold"`
	AggregationWindow  time.Duration `json:"aggregation_window" yaml:"aggregation_window"`
	Weight             float64       `json:"weight" yaml:"weight"`
}

// CustomScalingMetric 自定义扩缩容指标
type CustomScalingMetric struct {
	Name               string        `json:"name" yaml:"name"`
	Query              string        `json:"query" yaml:"query"`
	ScaleUpThreshold   float64       `json:"scale_up_threshold" yaml:"scale_up_threshold"`
	ScaleDownThreshold float64       `json:"scale_down_threshold" yaml:"scale_down_threshold"`
	AggregationWindow  time.Duration `json:"aggregation_window" yaml:"aggregation_window"`
	Weight             float64       `json:"weight" yaml:"weight"`
}

// ScalingCondition 扩缩容条件
type ScalingCondition struct {
	Metric    string        `json:"metric" yaml:"metric"`
	Operator  string        `json:"operator" yaml:"operator"`
	Threshold float64       `json:"threshold" yaml:"threshold"`
	Duration  time.Duration `json:"duration" yaml:"duration"`
}

// ScalingLimitsConfig 扩缩容限制配置
type ScalingLimitsConfig struct {
	MinComputeNodes  int     `json:"min_compute_nodes" yaml:"min_compute_nodes"`
	MaxComputeNodes  int     `json:"max_compute_nodes" yaml:"max_compute_nodes"`
	MinStorageNodes  int     `json:"min_storage_nodes" yaml:"min_storage_nodes"`
	MaxStorageNodes  int     `json:"max_storage_nodes" yaml:"max_storage_nodes"`
	MinProxyNodes    int     `json:"min_proxy_nodes" yaml:"min_proxy_nodes"`
	MaxProxyNodes    int     `json:"max_proxy_nodes" yaml:"max_proxy_nodes"`
	MaxScaleUpRate   float64 `json:"max_scale_up_rate" yaml:"max_scale_up_rate"`
	MaxScaleDownRate float64 `json:"max_scale_down_rate" yaml:"max_scale_down_rate"`
}

// CooldownConfig 冷却配置
type CooldownConfig struct {
	ScaleUpCooldown     time.Duration `json:"scale_up_cooldown" yaml:"scale_up_cooldown"`
	ScaleDownCooldown   time.Duration `json:"scale_down_cooldown" yaml:"scale_down_cooldown"`
	StabilizationWindow time.Duration `json:"stabilization_window" yaml:"stabilization_window"`
}

// PredictionConfig 预测配置
type PredictionConfig struct {
	Enable            bool          `json:"enable" yaml:"enable"`
	Algorithm         string        `json:"algorithm" yaml:"algorithm"` // "linear", "arima", "lstm", "prophet"
	LookAheadWindow   time.Duration `json:"look_ahead_window" yaml:"look_ahead_window"`
	TrainingWindow    time.Duration `json:"training_window" yaml:"training_window"`
	UpdateInterval    time.Duration `json:"update_interval" yaml:"update_interval"`
	AccuracyThreshold float64       `json:"accuracy_threshold" yaml:"accuracy_threshold"`
	ModelPath         string        `json:"model_path" yaml:"model_path"`
}

// LoggingConfig 日志配置
type LoggingConfig struct {
	Enable     bool                    `json:"enable" yaml:"enable"`
	Level      string                  `json:"level" yaml:"level"`   // "DEBUG", "INFO", "WARN", "ERROR"
	Format     string                  `json:"format" yaml:"format"` // "json", "text"
	Output     LogOutputConfig         `json:"output" yaml:"output"`
	Rotation   LogRotationConfig       `json:"rotation" yaml:"rotation"`
	Sampling   LogSamplingConfig       `json:"sampling" yaml:"sampling"`
	Structured StructuredLoggingConfig `json:"structured" yaml:"structured"`
	Audit      AuditLoggingConfig      `json:"audit" yaml:"audit"`
}

// LogOutputConfig 日志输出配置
type LogOutputConfig struct {
	Console       bool                      `json:"console" yaml:"console"`
	File          FileOutputConfig          `json:"file" yaml:"file"`
	Syslog        SyslogOutputConfig        `json:"syslog" yaml:"syslog"`
	Elasticsearch ElasticsearchOutputConfig `json:"elasticsearch" yaml:"elasticsearch"`
	Kafka         KafkaOutputConfig         `json:"kafka" yaml:"kafka"`
}

// FileOutputConfig 文件输出配置
type FileOutputConfig struct {
	Enable     bool   `json:"enable" yaml:"enable"`
	Path       string `json:"path" yaml:"path"`
	MaxSize    int64  `json:"max_size" yaml:"max_size"` // bytes
	MaxAge     int    `json:"max_age" yaml:"max_age"`   // days
	MaxBackups int    `json:"max_backups" yaml:"max_backups"`
	Compress   bool   `json:"compress" yaml:"compress"`
}

// SyslogOutputConfig Syslog输出配置
type SyslogOutputConfig struct {
	Enable   bool   `json:"enable" yaml:"enable"`
	Network  string `json:"network" yaml:"network"` // "tcp", "udp", "unix"
	Address  string `json:"address" yaml:"address"`
	Facility string `json:"facility" yaml:"facility"`
	Tag      string `json:"tag" yaml:"tag"`
}

// ElasticsearchOutputConfig Elasticsearch输出配置
type ElasticsearchOutputConfig struct {
	Enable        bool          `json:"enable" yaml:"enable"`
	URLs          []string      `json:"urls" yaml:"urls"`
	Index         string        `json:"index" yaml:"index"`
	Username      string        `json:"username" yaml:"username"`
	Password      string        `json:"password" yaml:"password"`
	BatchSize     int           `json:"batch_size" yaml:"batch_size"`
	FlushInterval time.Duration `json:"flush_interval" yaml:"flush_interval"`
}

// KafkaOutputConfig Kafka输出配置
type KafkaOutputConfig struct {
	Enable        bool          `json:"enable" yaml:"enable"`
	Brokers       []string      `json:"brokers" yaml:"brokers"`
	Topic         string        `json:"topic" yaml:"topic"`
	Partition     int           `json:"partition" yaml:"partition"`
	BatchSize     int           `json:"batch_size" yaml:"batch_size"`
	FlushInterval time.Duration `json:"flush_interval" yaml:"flush_interval"`
}

// LogRotationConfig 日志轮转配置
type LogRotationConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	MaxSize          int64         `json:"max_size" yaml:"max_size"` // bytes
	MaxAge           time.Duration `json:"max_age" yaml:"max_age"`
	MaxBackups       int           `json:"max_backups" yaml:"max_backups"`
	Compress         bool          `json:"compress" yaml:"compress"`
	RotationSchedule string        `json:"rotation_schedule" yaml:"rotation_schedule"` // cron expression
}

// LogSamplingConfig 日志采样配置
type LogSamplingConfig struct {
	Enable     bool          `json:"enable" yaml:"enable"`
	Initial    int           `json:"initial" yaml:"initial"`
	Thereafter int           `json:"thereafter" yaml:"thereafter"`
	Tick       time.Duration `json:"tick" yaml:"tick"`
}

// StructuredLoggingConfig 结构化日志配置
type StructuredLoggingConfig struct {
	Enable          bool     `json:"enable" yaml:"enable"`
	Fields          []string `json:"fields" yaml:"fields"`
	TimestampFormat string   `json:"timestamp_format" yaml:"timestamp_format"`
	CallerInfo      bool     `json:"caller_info" yaml:"caller_info"`
	StackTrace      bool     `json:"stack_trace" yaml:"stack_trace"`
}

// AuditLoggingConfig 审计日志配置
type AuditLoggingConfig struct {
	Enable          bool          `json:"enable" yaml:"enable"`
	Events          []string      `json:"events" yaml:"events"`
	Path            string        `json:"path" yaml:"path"`
	Format          string        `json:"format" yaml:"format"`
	RetentionPeriod time.Duration `json:"retention_period" yaml:"retention_period"`
	Encryption      bool          `json:"encryption" yaml:"encryption"`
	IntegrityCheck  bool          `json:"integrity_check" yaml:"integrity_check"`
}

// TracingConfig 链路追踪配置
type TracingConfig struct {
	Enable           bool                `json:"enable" yaml:"enable"`
	Provider         string              `json:"provider" yaml:"provider"` // "jaeger", "zipkin", "opentelemetry"
	SamplingRate     float64             `json:"sampling_rate" yaml:"sampling_rate"`
	Jaeger           JaegerConfig        `json:"jaeger" yaml:"jaeger"`
	Zipkin           ZipkinConfig        `json:"zipkin" yaml:"zipkin"`
	OpenTelemetry    OpenTelemetryConfig `json:"opentelemetry" yaml:"opentelemetry"`
	CustomTags       map[string]string   `json:"custom_tags" yaml:"custom_tags"`
	OperationFilters []string            `json:"operation_filters" yaml:"operation_filters"`
}

// JaegerConfig Jaeger配置
type JaegerConfig struct {
	Endpoint          string `json:"endpoint" yaml:"endpoint"`
	ServiceName       string `json:"service_name" yaml:"service_name"`
	AgentHost         string `json:"agent_host" yaml:"agent_host"`
	AgentPort         int    `json:"agent_port" yaml:"agent_port"`
	CollectorEndpoint string `json:"collector_endpoint" yaml:"collector_endpoint"`
	Username          string `json:"username" yaml:"username"`
	Password          string `json:"password" yaml:"password"`
}

// ZipkinConfig Zipkin配置
type ZipkinConfig struct {
	Endpoint    string        `json:"endpoint" yaml:"endpoint"`
	ServiceName string        `json:"service_name" yaml:"service_name"`
	BatchSize   int           `json:"batch_size" yaml:"batch_size"`
	Timeout     time.Duration `json:"timeout" yaml:"timeout"`
}

// OpenTelemetryConfig OpenTelemetry配置
type OpenTelemetryConfig struct {
	Endpoint       string            `json:"endpoint" yaml:"endpoint"`
	ServiceName    string            `json:"service_name" yaml:"service_name"`
	ServiceVersion string            `json:"service_version" yaml:"service_version"`
	Headers        map[string]string `json:"headers" yaml:"headers"`
	BatchTimeout   time.Duration     `json:"batch_timeout" yaml:"batch_timeout"`
	ExportTimeout  time.Duration     `json:"export_timeout" yaml:"export_timeout"`
}

// ProfilingConfig 性能分析配置
type ProfilingConfig struct {
	Enable    bool                     `json:"enable" yaml:"enable"`
	CPU       CPUProfilingConfig       `json:"cpu" yaml:"cpu"`
	Memory    MemoryProfilingConfig    `json:"memory" yaml:"memory"`
	Goroutine GoroutineProfilingConfig `json:"goroutine" yaml:"goroutine"`
	Block     BlockProfilingConfig     `json:"block" yaml:"block"`
	Mutex     MutexProfilingConfig     `json:"mutex" yaml:"mutex"`
	Custom    CustomProfilingConfig    `json:"custom" yaml:"custom"`
}

// CPUProfilingConfig CPU性能分析配置
type CPUProfilingConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	SamplingRate     int           `json:"sampling_rate" yaml:"sampling_rate"` // Hz
	Duration         time.Duration `json:"duration" yaml:"duration"`
	OutputPath       string        `json:"output_path" yaml:"output_path"`
	AutoTrigger      bool          `json:"auto_trigger" yaml:"auto_trigger"`
	TriggerThreshold float64       `json:"trigger_threshold" yaml:"trigger_threshold"`
}

// MemoryProfilingConfig 内存性能分析配置
type MemoryProfilingConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	SamplingRate     int           `json:"sampling_rate" yaml:"sampling_rate"`
	Interval         time.Duration `json:"interval" yaml:"interval"`
	OutputPath       string        `json:"output_path" yaml:"output_path"`
	AutoTrigger      bool          `json:"auto_trigger" yaml:"auto_trigger"`
	TriggerThreshold float64       `json:"trigger_threshold" yaml:"trigger_threshold"`
	GCBeforeProfile  bool          `json:"gc_before_profile" yaml:"gc_before_profile"`
}

// GoroutineProfilingConfig Goroutine性能分析配置
type GoroutineProfilingConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	Interval         time.Duration `json:"interval" yaml:"interval"`
	OutputPath       string        `json:"output_path" yaml:"output_path"`
	AutoTrigger      bool          `json:"auto_trigger" yaml:"auto_trigger"`
	TriggerThreshold int           `json:"trigger_threshold" yaml:"trigger_threshold"`
	StackTrace       bool          `json:"stack_trace" yaml:"stack_trace"`
}

// BlockProfilingConfig 阻塞性能分析配置
type BlockProfilingConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	Rate             int           `json:"rate" yaml:"rate"`
	Interval         time.Duration `json:"interval" yaml:"interval"`
	OutputPath       string        `json:"output_path" yaml:"output_path"`
	AutoTrigger      bool          `json:"auto_trigger" yaml:"auto_trigger"`
	TriggerThreshold time.Duration `json:"trigger_threshold" yaml:"trigger_threshold"`
}

// MutexProfilingConfig 互斥锁性能分析配置
type MutexProfilingConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	Rate             int           `json:"rate" yaml:"rate"`
	Interval         time.Duration `json:"interval" yaml:"interval"`
	OutputPath       string        `json:"output_path" yaml:"output_path"`
	AutoTrigger      bool          `json:"auto_trigger" yaml:"auto_trigger"`
	TriggerThreshold time.Duration `json:"trigger_threshold" yaml:"trigger_threshold"`
}

// CustomProfilingConfig 自定义性能分析配置
type CustomProfilingConfig struct {
	Enable   bool                  `json:"enable" yaml:"enable"`
	Profiles []CustomProfileConfig `json:"profiles" yaml:"profiles"`
}

// CustomProfileConfig 自定义性能分析配置
type CustomProfileConfig struct {
	Name       string                 `json:"name" yaml:"name"`
	Type       string                 `json:"type" yaml:"type"`
	Config     map[string]interface{} `json:"config" yaml:"config"`
	Interval   time.Duration          `json:"interval" yaml:"interval"`
	OutputPath string                 `json:"output_path" yaml:"output_path"`
	Enabled    bool                   `json:"enabled" yaml:"enabled"`
}

// DashboardConfig 仪表板配置
type DashboardConfig struct {
	Enable          bool                   `json:"enable" yaml:"enable"`
	Port            int                    `json:"port" yaml:"port"`
	Host            string                 `json:"host" yaml:"host"`
	Path            string                 `json:"path" yaml:"path"`
	Authentication  DashboardAuthConfig    `json:"authentication" yaml:"authentication"`
	Charts          []DashboardChart       `json:"charts" yaml:"charts"`
	RefreshInterval time.Duration          `json:"refresh_interval" yaml:"refresh_interval"`
	Theme           string                 `json:"theme" yaml:"theme"`
	Customization   DashboardCustomization `json:"customization" yaml:"customization"`
}

// DashboardAuthConfig 仪表板认证配置
type DashboardAuthConfig struct {
	Enable         bool            `json:"enable" yaml:"enable"`
	Type           string          `json:"type" yaml:"type"` // "basic", "oauth", "jwt"
	Users          []DashboardUser `json:"users" yaml:"users"`
	OAuth          OAuthConfig     `json:"oauth" yaml:"oauth"`
	JWT            JWTConfig       `json:"jwt" yaml:"jwt"`
	SessionTimeout time.Duration   `json:"session_timeout" yaml:"session_timeout"`
}

// DashboardUser 仪表板用户
type DashboardUser struct {
	Username    string   `json:"username" yaml:"username"`
	Password    string   `json:"password" yaml:"password"`
	Roles       []string `json:"roles" yaml:"roles"`
	Permissions []string `json:"permissions" yaml:"permissions"`
}

// OAuthConfig OAuth配置
type OAuthConfig struct {
	Provider     string   `json:"provider" yaml:"provider"`
	ClientID     string   `json:"client_id" yaml:"client_id"`
	ClientSecret string   `json:"client_secret" yaml:"client_secret"`
	RedirectURL  string   `json:"redirect_url" yaml:"redirect_url"`
	Scopes       []string `json:"scopes" yaml:"scopes"`
	AuthURL      string   `json:"auth_url" yaml:"auth_url"`
	TokenURL     string   `json:"token_url" yaml:"token_url"`
}

// JWTConfig JWT配置
type JWTConfig struct {
	Secret            string        `json:"secret" yaml:"secret"`
	Issuer            string        `json:"issuer" yaml:"issuer"`
	Audience          string        `json:"audience" yaml:"audience"`
	Expiration        time.Duration `json:"expiration" yaml:"expiration"`
	RefreshEnable     bool          `json:"refresh_enable" yaml:"refresh_enable"`
	RefreshExpiration time.Duration `json:"refresh_expiration" yaml:"refresh_expiration"`
}

// DashboardChart 仪表板图表
type DashboardChart struct {
	ID              string                 `json:"id" yaml:"id"`
	Title           string                 `json:"title" yaml:"title"`
	Type            string                 `json:"type" yaml:"type"` // "line", "bar", "pie", "gauge", "table"
	Query           string                 `json:"query" yaml:"query"`
	DataSource      string                 `json:"data_source" yaml:"data_source"`
	RefreshInterval time.Duration          `json:"refresh_interval" yaml:"refresh_interval"`
	Position        ChartPosition          `json:"position" yaml:"position"`
	Size            ChartSize              `json:"size" yaml:"size"`
	Options         map[string]interface{} `json:"options" yaml:"options"`
	Enabled         bool                   `json:"enabled" yaml:"enabled"`
}

// ChartPosition 图表位置
type ChartPosition struct {
	X int `json:"x" yaml:"x"`
	Y int `json:"y" yaml:"y"`
}

// ChartSize 图表大小
type ChartSize struct {
	Width  int `json:"width" yaml:"width"`
	Height int `json:"height" yaml:"height"`
}

// DashboardCustomization 仪表板自定义
type DashboardCustomization struct {
	Logo      string            `json:"logo" yaml:"logo"`
	Title     string            `json:"title" yaml:"title"`
	Colors    map[string]string `json:"colors" yaml:"colors"`
	Fonts     map[string]string `json:"fonts" yaml:"fonts"`
	Layout    string            `json:"layout" yaml:"layout"`
	CustomCSS string            `json:"custom_css" yaml:"custom_css"`
	CustomJS  string            `json:"custom_js" yaml:"custom_js"`
}

// HealthCheckConfig 健康检查配置
type HealthCheckConfig struct {
	Enable       bool                  `json:"enable" yaml:"enable"`
	Interval     time.Duration         `json:"interval" yaml:"interval"`
	Timeout      time.Duration         `json:"timeout" yaml:"timeout"`
	Endpoints    []HealthCheckEndpoint `json:"endpoints" yaml:"endpoints"`
	Dependencies []DependencyCheck     `json:"dependencies" yaml:"dependencies"`
	Readiness    ReadinessCheckConfig  `json:"readiness" yaml:"readiness"`
	Liveness     LivenessCheckConfig   `json:"liveness" yaml:"liveness"`
	Startup      StartupCheckConfig    `json:"startup" yaml:"startup"`
}

// HealthCheckEndpoint 健康检查端点
type HealthCheckEndpoint struct {
	Name           string            `json:"name" yaml:"name"`
	Path           string            `json:"path" yaml:"path"`
	Method         string            `json:"method" yaml:"method"`
	ExpectedStatus int               `json:"expected_status" yaml:"expected_status"`
	Timeout        time.Duration     `json:"timeout" yaml:"timeout"`
	Headers        map[string]string `json:"headers" yaml:"headers"`
	Body           string            `json:"body" yaml:"body"`
	Enabled        bool              `json:"enabled" yaml:"enabled"`
}

// DependencyCheck 依赖检查
type DependencyCheck struct {
	Name       string        `json:"name" yaml:"name"`
	Type       string        `json:"type" yaml:"type"` // "database", "redis", "elasticsearch", "http"
	Address    string        `json:"address" yaml:"address"`
	Timeout    time.Duration `json:"timeout" yaml:"timeout"`
	RetryCount int           `json:"retry_count" yaml:"retry_count"`
	Critical   bool          `json:"critical" yaml:"critical"`
	Enabled    bool          `json:"enabled" yaml:"enabled"`
}

// ReadinessCheckConfig 就绪检查配置
type ReadinessCheckConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	Path             string        `json:"path" yaml:"path"`
	InitialDelay     time.Duration `json:"initial_delay" yaml:"initial_delay"`
	Period           time.Duration `json:"period" yaml:"period"`
	Timeout          time.Duration `json:"timeout" yaml:"timeout"`
	FailureThreshold int           `json:"failure_threshold" yaml:"failure_threshold"`
	SuccessThreshold int           `json:"success_threshold" yaml:"success_threshold"`
}

// LivenessCheckConfig 存活检查配置
type LivenessCheckConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	Path             string        `json:"path" yaml:"path"`
	InitialDelay     time.Duration `json:"initial_delay" yaml:"initial_delay"`
	Period           time.Duration `json:"period" yaml:"period"`
	Timeout          time.Duration `json:"timeout" yaml:"timeout"`
	FailureThreshold int           `json:"failure_threshold" yaml:"failure_threshold"`
}

// StartupCheckConfig 启动检查配置
type StartupCheckConfig struct {
	Enable           bool          `json:"enable" yaml:"enable"`
	Path             string        `json:"path" yaml:"path"`
	InitialDelay     time.Duration `json:"initial_delay" yaml:"initial_delay"`
	Period           time.Duration `json:"period" yaml:"period"`
	Timeout          time.Duration `json:"timeout" yaml:"timeout"`
	FailureThreshold int           `json:"failure_threshold" yaml:"failure_threshold"`
}

// GetDefaultMonitoringConfig 获取默认监控配置
func GetDefaultMonitoringConfig() *MonitoringConfig {
	return &MonitoringConfig{
		Metrics: MetricsConfig{
			Enable:             true,
			CollectionInterval: 10 * time.Second,
			RetentionPeriod:    24 * time.Hour,
			Exporter: MetricsExporterConfig{
				Prometheus: PrometheusConfig{
					Enable:         true,
					Port:           9090,
					Path:           "/metrics",
					ScrapeInterval: 15 * time.Second,
					Timeout:        10 * time.Second,
				},
			},
			Performance: PerformanceMetricsConfig{
				SearchLatency: LatencyMetricsConfig{
					Enable:            true,
					Percentiles:       []float64{0.5, 0.9, 0.95, 0.99},
					Buckets:           []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0},
					SamplingRate:      1.0,
					AggregationWindow: 1 * time.Minute,
				},
				Throughput: ThroughputMetricsConfig{
					Enable:            true,
					QPS:               true,
					RPS:               true,
					Bandwidth:         true,
					AggregationWindow: 1 * time.Minute,
				},
				ResourceUsage: ResourceMetricsConfig{
					CPU: CPUMetricsConfig{
						Enable:             true,
						Utilization:        true,
						LoadAverage:        true,
						CollectionInterval: 5 * time.Second,
					},
					Memory: MemoryMetricsConfig{
						Enable:             true,
						Usage:              true,
						Available:          true,
						GCStats:            true,
						CollectionInterval: 5 * time.Second,
					},
				},
			},
			Business: BusinessMetricsConfig{
				Enable: true,
				SearchAccuracy: AccuracyMetricsConfig{
					Enable:             true,
					Recall:             true,
					Precision:          true,
					F1Score:            true,
					EvaluationInterval: 1 * time.Hour,
				},
			},
		},
		Alerting: AlertingConfig{
			Enable: true,
			Rules: []AlertRule{
				{
					Name:        "HighCPUUsage",
					Description: "CPU使用率过高",
					Metric:      "cpu_utilization",
					Condition: AlertCondition{
						Operator:    ">",
						Threshold:   70.0,
						Aggregation: "avg",
						TimeWindow:  5 * time.Minute,
					},
					Duration: 2 * time.Minute,
					Severity: "warning",
					Enabled:  true,
				},
				{
					Name:        "HighQueryLatency",
					Description: "查询延迟过高",
					Metric:      "search_latency_p95",
					Condition: AlertCondition{
						Operator:    ">",
						Threshold:   200.0, // 200ms
						Aggregation: "avg",
						TimeWindow:  5 * time.Minute,
					},
					Duration: 1 * time.Minute,
					Severity: "critical",
					Enabled:  true,
				},
				{
					Name:        "HighMemoryUsage",
					Description: "内存使用率过高",
					Metric:      "memory_usage_percent",
					Condition: AlertCondition{
						Operator:    ">",
						Threshold:   80.0,
						Aggregation: "avg",
						TimeWindow:  5 * time.Minute,
					},
					Duration: 3 * time.Minute,
					Severity: "warning",
					Enabled:  true,
				},
			},
			Notification: NotificationConfig{
				Channels: []NotificationChannel{
					{
						Name:           "default-email",
						Type:           "email",
						Enabled:        true,
						SeverityFilter: []string{"critical", "warning"},
					},
				},
				RateLimiting: RateLimitingConfig{
					Enable: true,
					GlobalLimit: RateLimitRule{
						RequestsPerSecond: 10,
						BurstSize:         20,
						WindowSize:        1 * time.Hour,
						PenaltyDuration:   5 * time.Minute,
					},
					PerUserLimit: RateLimitRule{
						RequestsPerSecond: 5,
						BurstSize:         10,
						WindowSize:        1 * time.Hour,
						PenaltyDuration:   5 * time.Minute,
					},
					PerIPLimit: RateLimitRule{
						RequestsPerSecond: 3,
						BurstSize:         5,
						WindowSize:        1 * time.Hour,
						PenaltyDuration:   10 * time.Minute,
					},
				},
			},
		},
		AutoScaling: AutoScalingConfig{
			Enable:   true,
			Strategy: "reactive",
			Metrics: ScalingMetricsConfig{
				CPUUtilization: CPUScalingConfig{
					Enable:             true,
					ScaleUpThreshold:   70.0,
					ScaleDownThreshold: 30.0,
					AggregationWindow:  5 * time.Minute,
					Weight:             1.0,
				},
				MemoryUtilization: MemoryScalingConfig{
					Enable:             true,
					ScaleUpThreshold:   80.0,
					ScaleDownThreshold: 40.0,
					AggregationWindow:  5 * time.Minute,
					Weight:             1.0,
				},
				QueryLatency: LatencyScalingConfig{
					Enable:             true,
					ScaleUpThreshold:   200 * time.Millisecond,
					ScaleDownThreshold: 50 * time.Millisecond,
					Percentile:         0.95,
					AggregationWindow:  5 * time.Minute,
					Weight:             1.5,
				},
			},
			Policies: []ScalingPolicy{
				{
					Name:           "cpu-scale-up",
					MetricType:     CPUUtilization,
					Threshold:      70.0,
					Direction:      ScaleUp,
					CooldownPeriod: 5 * time.Minute,
					MinInstances:   1,
					MaxInstances:   10,
					ScalingFactor:  1.5,
					Enabled:        true,
				},
				{
					Name:           "memory-scale-up",
					MetricType:     MemoryUtilization,
					Threshold:      80.0,
					Direction:      ScaleUp,
					CooldownPeriod: 3 * time.Minute,
					MinInstances:   1,
					MaxInstances:   5,
					ScalingFactor:  1.3,
					Enabled:        true,
				},
			},
			Limits: ScalingLimitsConfig{
				MinComputeNodes:  1,
				MaxComputeNodes:  10,
				MinStorageNodes:  1,
				MaxStorageNodes:  5,
				MinProxyNodes:    1,
				MaxProxyNodes:    3,
				MaxScaleUpRate:   2.0,
				MaxScaleDownRate: 0.5,
			},
			Cooldown: CooldownConfig{
				ScaleUpCooldown:     5 * time.Minute,
				ScaleDownCooldown:   10 * time.Minute,
				StabilizationWindow: 3 * time.Minute,
			},
		},
		Logging: LoggingConfig{
			Enable: true,
			Level:  "INFO",
			Format: "json",
			Output: LogOutputConfig{
				Console: true,
				File: FileOutputConfig{
					Enable:     true,
					Path:       "./logs/vectorsphere.log",
					MaxSize:    100 * 1024 * 1024, // 100MB
					MaxAge:     7,                 // 7 days
					MaxBackups: 10,
					Compress:   true,
				},
			},
			Rotation: LogRotationConfig{
				Enable:     true,
				MaxSize:    100 * 1024 * 1024,  // 100MB
				MaxAge:     7 * 24 * time.Hour, // 7 days
				MaxBackups: 10,
				Compress:   true,
			},
			Structured: StructuredLoggingConfig{
				Enable:          true,
				Fields:          []string{"timestamp", "level", "message", "caller"},
				TimestampFormat: "2006-01-02T15:04:05.000Z",
				CallerInfo:      true,
				StackTrace:      false,
			},
		},
		Tracing: TracingConfig{
			Enable:       false,
			Provider:     "jaeger",
			SamplingRate: 0.1,
			Jaeger: JaegerConfig{
				ServiceName: "vectorsphere",
				AgentHost:   "localhost",
				AgentPort:   6831,
			},
		},
		Profiling: ProfilingConfig{
			Enable: false,
			CPU: CPUProfilingConfig{
				Enable:           false,
				SamplingRate:     100,
				Duration:         30 * time.Second,
				OutputPath:       "./profiles/cpu.prof",
				AutoTrigger:      false,
				TriggerThreshold: 80.0,
			},
			Memory: MemoryProfilingConfig{
				Enable:           false,
				Interval:         1 * time.Minute,
				OutputPath:       "./profiles/memory.prof",
				AutoTrigger:      false,
				TriggerThreshold: 80.0,
				GCBeforeProfile:  true,
			},
		},
		Dashboard: DashboardConfig{
			Enable:          true,
			Port:            8080,
			Host:            "0.0.0.0",
			Path:            "/dashboard",
			RefreshInterval: 30 * time.Second,
			Theme:           "dark",
			Authentication: DashboardAuthConfig{
				Enable:         false,
				Type:           "basic",
				SessionTimeout: 24 * time.Hour,
			},
		},
		HealthCheck: HealthCheckConfig{
			Enable:   true,
			Interval: 30 * time.Second,
			Timeout:  5 * time.Second,
			Endpoints: []HealthCheckEndpoint{
				{
					Name:           "api-health",
					Path:           "/health",
					Method:         "GET",
					ExpectedStatus: 200,
					Timeout:        3 * time.Second,
					Enabled:        true,
				},
			},
			Readiness: ReadinessCheckConfig{
				Enable:           true,
				Path:             "/ready",
				InitialDelay:     10 * time.Second,
				Period:           10 * time.Second,
				Timeout:          5 * time.Second,
				FailureThreshold: 3,
				SuccessThreshold: 1,
			},
			Liveness: LivenessCheckConfig{
				Enable:           true,
				Path:             "/live",
				InitialDelay:     30 * time.Second,
				Period:           30 * time.Second,
				Timeout:          5 * time.Second,
				FailureThreshold: 3,
			},
		},
	}
}
