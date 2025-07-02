package enhanced

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// SimpleConfig 简化的配置结构
type SimpleConfig struct {
	Port            int                        `json:"port" yaml:"port"`
	Host            string                     `json:"host" yaml:"host"`
	ServiceName     string                     `json:"service_name" yaml:"service_name"`
	LogLevel        string                     `json:"log_level" yaml:"log_level"`
	Gateway         SimpleGatewayConfig        `json:"gateway" yaml:"gateway"`
	LoadBalancer    SimpleLoadBalancerConfig   `json:"load_balancer" yaml:"load_balancer"`
	SecurityManager SimpleSecurityConfig       `json:"security_manager" yaml:"security_manager"`
	CircuitBreaker  SimpleCircuitBreakerConfig `json:"circuit_breaker" yaml:"circuit_breaker"`
	HealthCheck     SimpleHealthCheckConfig    `json:"health_check" yaml:"health_check"`
	Monitoring      MonitoringConfig           `json:"monitoring" yaml:"monitoring"`
	Features        FeatureConfig              `json:"features" yaml:"features"`
	Distributed     DistributedFeatureConfig   `json:"distributed" yaml:"distributed"`
}

// SimpleGatewayConfig 简化的网关配置
type SimpleGatewayConfig struct {
	Enabled        bool `json:"enabled"`
	Timeout        int  `json:"timeout"` // 秒
	MaxConnections int  `json:"max_connections"`
	RateLimit      int  `json:"rate_limit"` // 每秒请求数
}

// SimpleLoadBalancerConfig 简化的负载均衡器配置
type SimpleLoadBalancerConfig struct {
	Algorithm            string `json:"algorithm"`             // "round_robin", "weighted_round_robin", "least_connections", "ip_hash"
	HealthCheckInterval  int    `json:"health_check_interval"` // 秒
	MaxRetries           int    `json:"max_retries"`
	RetryDelay           int    `json:"retry_delay"` // 毫秒
	EnableStickySessions bool   `json:"enable_sticky_sessions"`
	SessionTimeout       int    `json:"session_timeout"` // 分钟
	EnableMetrics        bool   `json:"enable_metrics"`
	MaxConnections       int    `json:"max_connections"`
	ConnectionTimeout    int    `json:"connection_timeout"` // 秒
}

// SimpleSecurityConfig 简化的安全配置
type SimpleSecurityConfig struct {
	JWTSecret          string   `json:"jwt_secret"`
	TokenExpiration    int      `json:"token_expiration"` // 小时
	EnableRateLimit    bool     `json:"enable_rate_limit"`
	RateLimitPerSecond int      `json:"rate_limit_per_second"`
	EnableIPWhitelist  bool     `json:"enable_ip_whitelist"`
	IPWhitelist        []string `json:"ip_whitelist"`
	EnableEncryption   bool     `json:"enable_encryption"`
	EncryptionKey      string   `json:"encryption_key"`
	EnableAuditLog     bool     `json:"enable_audit_log"`
	AuditLogPath       string   `json:"audit_log_path"`
	MaxLoginAttempts   int      `json:"max_login_attempts"`
	LockoutDuration    int      `json:"lockout_duration"` // 分钟
}

// SimpleCircuitBreakerConfig 简化的熔断器配置
type SimpleCircuitBreakerConfig struct {
	FailureThreshold    int    `json:"failure_threshold"`
	RecoveryTimeout     int    `json:"recovery_timeout"` // 秒
	MaxRequests         int    `json:"max_requests"`
	Interval            int    `json:"interval"` // 秒
	Timeout             int    `json:"timeout"`  // 秒
	EnableMetrics       bool   `json:"enable_metrics"`
	EnableNotification  bool   `json:"enable_notification"`
	NotificationURL     string `json:"notification_url"`
	HalfOpenMaxRequests int    `json:"half_open_max_requests"`
}

// SimpleHealthCheckConfig 简化的健康检查配置
type SimpleHealthCheckConfig struct {
	Interval           int    `json:"interval"` // 秒
	Timeout            int    `json:"timeout"`  // 秒
	RetryCount         int    `json:"retry_count"`
	RetryDelay         int    `json:"retry_delay"` // 毫秒
	EnableDeepCheck    bool   `json:"enable_deep_check"`
	EnableMetrics      bool   `json:"enable_metrics"`
	EnableNotification bool   `json:"enable_notification"`
	UnhealthyThreshold int    `json:"unhealthy_threshold"`
	HealthyThreshold   int    `json:"healthy_threshold"`
	HealthCheckPath    string `json:"health_check_path"`
	ExpectedStatusCode int    `json:"expected_status_code"`
}

// MonitoringConfig 监控配置
type MonitoringConfig struct {
	EnableMetrics      bool   `json:"enable_metrics"`
	MetricsPort        int    `json:"metrics_port"`
	MetricsPath        string `json:"metrics_path"`
	EnableTracing      bool   `json:"enable_tracing"`
	TracingEndpoint    string `json:"tracing_endpoint"`
	EnableProfiling    bool   `json:"enable_profiling"`
	ProfilingPort      int    `json:"profiling_port"`
	CollectionInterval int    `json:"collection_interval"` // 秒
}

// FeatureConfig 功能开关配置
type FeatureConfig struct {
	EnableLoadBalancer    bool `json:"enable_load_balancer"`    // 是否启用负载均衡器
	EnableSecurityManager bool `json:"enable_security_manager"` // 是否启用安全管理器
	EnableCircuitBreaker  bool `json:"enable_circuit_breaker"`  // 是否启用熔断器
	EnableHealthChecker   bool `json:"enable_health_checker"`   // 是否启用健康检查器
	EnableAPIGateway      bool `json:"enable_api_gateway"`      // 是否启用API网关
	EnableMonitoring      bool `json:"enable_monitoring"`       // 是否启用监控
	EnableDistributed     bool `json:"enable_distributed"`      // 是否启用分布式功能
}

// DistributedFeatureConfig 分布式功能配置
type DistributedFeatureConfig struct {
	Enabled                  bool     `json:"enabled"`                    // 是否启用分布式功能
	NodeID                   string   `json:"node_id"`                    // 节点ID
	ServiceName              string   `json:"service_name"`               // 服务名称
	EtcdEndpoints            []string `json:"etcd_endpoints"`             // etcd端点列表
	EtcdDialTimeout          int      `json:"etcd_dial_timeout"`          // etcd连接超时时间(秒)
	ClusterNodes             []string `json:"cluster_nodes"`              // 集群节点列表
	EnableServiceDiscovery   bool     `json:"enable_service_discovery"`   // 是否启用服务发现
	EnableCommunication      bool     `json:"enable_communication"`       // 是否启用通信服务
	EnableDistributedManager bool     `json:"enable_distributed_manager"` // 是否启用分布式管理器
}

// SimpleConfigManager 简化配置管理器
type SimpleConfigManager struct {
	configPath    string
	config        *SimpleConfig
	lastModTime   time.Time
	mutex         sync.RWMutex
	watchers      []func(*SimpleConfig)
	watchersMutex sync.RWMutex
}

// NewSimpleConfigManager 创建简化配置管理器
func NewSimpleConfigManager(configPath string) (*SimpleConfigManager, error) {
	manager := &SimpleConfigManager{
		configPath: configPath,
		watchers:   make([]func(*SimpleConfig), 0),
	}

	// 加载配置
	if err := manager.loadConfig(); err != nil {
		return nil, err
	}

	return manager, nil
}

// GetConfig 获取配置
func (scm *SimpleConfigManager) GetConfig() *SimpleConfig {
	scm.mutex.RLock()
	defer scm.mutex.RUnlock()
	return scm.config
}

// ReloadConfig 重新加载配置
func (scm *SimpleConfigManager) ReloadConfig() (*SimpleConfig, error) {
	if err := scm.loadConfig(); err != nil {
		return nil, err
	}

	// 通知观察者
	scm.notifyWatchers()

	return scm.config, nil
}

// HasConfigChanged 检查配置是否已更改
func (scm *SimpleConfigManager) HasConfigChanged() bool {
	fileInfo, err := os.Stat(scm.configPath)
	if err != nil {
		return false
	}

	scm.mutex.RLock()
	lastModTime := scm.lastModTime
	scm.mutex.RUnlock()

	return fileInfo.ModTime().After(lastModTime)
}

// loadConfig 加载配置文件
func (scm *SimpleConfigManager) loadConfig() error {
	data, err := os.ReadFile(scm.configPath)
	if err != nil {
		return fmt.Errorf("读取配置文件失败: %v", err)
	}

	var config SimpleConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("解析配置文件失败: %v", err)
	}

	// 验证配置
	if err := scm.validateConfig(&config); err != nil {
		return fmt.Errorf("配置验证失败: %v", err)
	}

	scm.mutex.Lock()
	scm.config = &config

	// 更新文件修改时间
	fileInfo, err := os.Stat(scm.configPath)
	if err == nil {
		scm.lastModTime = fileInfo.ModTime()
	}
	scm.mutex.Unlock()

	return nil
}

// validateConfig 验证配置
func (scm *SimpleConfigManager) validateConfig(config *SimpleConfig) error {
	if config.ServiceName == "" {
		return fmt.Errorf("service_name不能为空")
	}

	if config.Port <= 0 || config.Port > 65535 {
		return fmt.Errorf("port必须在1-65535范围内")
	}

	if config.LoadBalancer.Algorithm == "" {
		return fmt.Errorf("load_balancer.algorithm不能为空")
	}

	validAlgorithms := []string{"round_robin", "weighted_round_robin", "least_connections", "ip_hash"}
	validAlgorithm := false
	for _, alg := range validAlgorithms {
		if config.LoadBalancer.Algorithm == alg {
			validAlgorithm = true
			break
		}
	}
	if !validAlgorithm {
		return fmt.Errorf("load_balancer.algorithm必须是: %v", validAlgorithms)
	}

	if config.SecurityManager.JWTSecret == "" {
		return fmt.Errorf("security_manager.jwt_secret不能为空")
	}

	if config.CircuitBreaker.FailureThreshold <= 0 {
		return fmt.Errorf("circuit_breaker.failure_threshold必须大于0")
	}

	if config.HealthCheck.Interval <= 0 {
		return fmt.Errorf("health_check.interval必须大于0")
	}

	return nil
}

// notifyWatchers 通知观察者
func (scm *SimpleConfigManager) notifyWatchers() {
	scm.watchersMutex.RLock()
	watchers := make([]func(*SimpleConfig), len(scm.watchers))
	copy(watchers, scm.watchers)
	scm.watchersMutex.RUnlock()

	for _, watcher := range watchers {
		go watcher(scm.config)
	}
}

// AddWatcher 添加配置变更观察者
func (scm *SimpleConfigManager) AddWatcher(watcher func(*SimpleConfig)) {
	scm.watchersMutex.Lock()
	scm.watchers = append(scm.watchers, watcher)
	scm.watchersMutex.Unlock()
}
