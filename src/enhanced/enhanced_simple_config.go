package enhanced

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"sync"
	"time"
)

// SimpleConfig 简化配置结构
type SimpleConfig struct {
	// 基础配置
	ServiceName string `json:"service_name"`
	Version     string `json:"version"`
	Environment string `json:"environment"`
	DataDir     string `json:"data_dir"`
	LogLevel    string `json:"log_level"`

	// 网关配置
	Gateway APIGatewayConfig `json:"gateway"`

	// 负载均衡器配置
	LoadBalancer SimpleLoadBalancerConfig `json:"load_balancer"`

	// 安全管理配置
	Security SimpleSecurityConfig `json:"security"`

	// 熔断器配置
	CircuitBreaker SimpleCircuitBreakerConfig `json:"circuit_breaker"`

	// 健康检查配置
	HealthCheck SimpleHealthCheckConfig `json:"health_check"`

	// 监控配置
	Monitoring MonitoringConfig `json:"monitoring"`
}

// SimpleLoadBalancerConfig 简化的负载均衡器配置
type SimpleLoadBalancerConfig struct {
	Algorithm             string `json:"algorithm"`               // "round_robin", "weighted_round_robin", "least_connections", "ip_hash"
	HealthCheckInterval   int    `json:"health_check_interval"`   // 秒
	MaxRetries            int    `json:"max_retries"`
	RetryDelay            int    `json:"retry_delay"`              // 毫秒
	EnableStickySessions  bool   `json:"enable_sticky_sessions"`
	SessionTimeout        int    `json:"session_timeout"`         // 分钟
	EnableMetrics         bool   `json:"enable_metrics"`
	MaxConnections        int    `json:"max_connections"`
	ConnectionTimeout     int    `json:"connection_timeout"`      // 秒
}

// SimpleSecurityConfig 简化的安全配置
type SimpleSecurityConfig struct {
	JWTSecret           string   `json:"jwt_secret"`
	TokenExpiration     int      `json:"token_expiration"`      // 小时
	EnableRateLimit     bool     `json:"enable_rate_limit"`
	RateLimitPerSecond  int      `json:"rate_limit_per_second"`
	EnableIPWhitelist   bool     `json:"enable_ip_whitelist"`
	IPWhitelist         []string `json:"ip_whitelist"`
	EnableEncryption    bool     `json:"enable_encryption"`
	EncryptionKey       string   `json:"encryption_key"`
	EnableAuditLog      bool     `json:"enable_audit_log"`
	AuditLogPath        string   `json:"audit_log_path"`
	MaxLoginAttempts    int      `json:"max_login_attempts"`
	LockoutDuration     int      `json:"lockout_duration"`       // 分钟
}

// SimpleCircuitBreakerConfig 简化的熔断器配置
type SimpleCircuitBreakerConfig struct {
	FailureThreshold    int  `json:"failure_threshold"`
	RecoveryTimeout     int  `json:"recovery_timeout"`      // 秒
	MaxRequests         int  `json:"max_requests"`
	Interval            int  `json:"interval"`              // 秒
	Timeout             int  `json:"timeout"`               // 秒
	EnableMetrics       bool `json:"enable_metrics"`
	EnableNotification  bool `json:"enable_notification"`
	NotificationURL     string `json:"notification_url"`
	HalfOpenMaxRequests int  `json:"half_open_max_requests"`
}

// SimpleHealthCheckConfig 简化的健康检查配置
type SimpleHealthCheckConfig struct {
	Interval            int    `json:"interval"`              // 秒
	Timeout             int    `json:"timeout"`               // 秒
	RetryCount          int    `json:"retry_count"`
	RetryDelay          int    `json:"retry_delay"`           // 毫秒
	EnableDeepCheck     bool   `json:"enable_deep_check"`
	EnableMetrics       bool   `json:"enable_metrics"`
	EnableNotification  bool   `json:"enable_notification"`
	UnhealthyThreshold  int    `json:"unhealthy_threshold"`
	HealthyThreshold    int    `json:"healthy_threshold"`
	HealthCheckPath     string `json:"health_check_path"`
	ExpectedStatusCode  int    `json:"expected_status_code"`
}

// MonitoringConfig 监控配置
type MonitoringConfig struct {
	EnableMetrics     bool   `json:"enable_metrics"`
	MetricsPort       int    `json:"metrics_port"`
	MetricsPath       string `json:"metrics_path"`
	EnableTracing     bool   `json:"enable_tracing"`
	TracingEndpoint   string `json:"tracing_endpoint"`
	EnableProfiling   bool   `json:"enable_profiling"`
	ProfilingPort     int    `json:"profiling_port"`
	CollectionInterval int   `json:"collection_interval"` // 秒
}

// SimpleConfigManager 简化配置管理器
type SimpleConfigManager struct {
	configPath   string
	config       *SimpleConfig
	lastModTime  time.Time
	mutex        sync.RWMutex
	watchers     []func(*SimpleConfig)
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
	data, err := ioutil.ReadFile(scm.configPath)
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

	if config.Gateway.Port <= 0 || config.Gateway.Port > 65535 {
		return fmt.Errorf("gateway.port必须在1-65535范围内")
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

	if config.Security.JWTSecret == "" {
		return fmt.Errorf("security.jwt_secret不能为空")
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