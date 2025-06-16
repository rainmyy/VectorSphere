package distributed

import (
	confType "VectorSphere/src/library/confType"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/server"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ConfigManager 配置管理器
type ConfigManager struct {
	configPath string
	config     *DistributedConfig
}

// NewConfigManager 创建配置管理器
func NewConfigManager(configPath string) *ConfigManager {
	return &ConfigManager{
		configPath: configPath,
	}
}

// LoadConfig 加载配置
func (cm *ConfigManager) LoadConfig() (*DistributedConfig, error) {
	logger.Info("Loading configuration from: %s", cm.configPath)

	var config DistributedConfig

	// 读取YAML配置文件
	err := confType.ReadYAML(cm.configPath, &config)
	if err != nil {
		return nil, fmt.Errorf("读取配置文件失败: %v", err)
	}

	// 验证配置
	if err := cm.validateConfig(&config); err != nil {
		return nil, fmt.Errorf("配置验证失败: %v", err)
	}

	// 应用环境变量覆盖
	cm.applyEnvOverrides(&config)

	// 设置默认值
	cm.setDefaults(&config)

	cm.config = &config
	logger.Info("Configuration loaded successfully")
	return &config, nil
}

// validateConfig 验证配置
func (cm *ConfigManager) validateConfig(config *DistributedConfig) error {
	if config.ServiceName == "" {
		return fmt.Errorf("serviceName不能为空")
	}

	if len(config.Etcd.Endpoints) == 0 {
		return fmt.Errorf("etcd endpoints不能为空")
	}

	if config.DefaultPort <= 0 || config.DefaultPort > 65535 {
		return fmt.Errorf("defaultPort必须在1-65535范围内")
	}

	if config.HttpPort <= 0 || config.HttpPort > 65535 {
		return fmt.Errorf("httpPort必须在1-65535范围内")
	}

	if config.TimeOut <= 0 {
		return fmt.Errorf("timeOut必须大于0")
	}

	if config.Heartbeat <= 0 {
		return fmt.Errorf("heartbeat必须大于0")
	}

	return nil
}

// applyEnvOverrides 应用环境变量覆盖
func (cm *ConfigManager) applyEnvOverrides(config *DistributedConfig) {
	// 服务名称
	if serviceName := os.Getenv("VECTOR_SPHERE_SERVICE_NAME"); serviceName != "" {
		config.ServiceName = serviceName
		logger.Info("Override serviceName from env: %s", serviceName)
	}

	// 节点类型
	if nodeType := os.Getenv("VECTOR_SPHERE_NODE_TYPE"); nodeType != "" {
		switch strings.ToLower(nodeType) {
		case "master":
			config.NodeType = MasterNode
		case "slave":
			config.NodeType = SlaveNode
		default:
			logger.Warning("Invalid node type in env: %s, using default", nodeType)
		}
	}

	// 端口配置
	if port := os.Getenv("VECTOR_SPHERE_PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {
			config.DefaultPort = p
			logger.Info("Override defaultPort from env: %d", p)
		}
	}

	if httpPort := os.Getenv("VECTOR_SPHERE_HTTP_PORT"); httpPort != "" {
		if p, err := strconv.Atoi(httpPort); err == nil {
			config.HttpPort = p
			logger.Info("Override httpPort from env: %d", p)
		}
	}

	// etcd端点
	if etcdEndpoints := os.Getenv("VECTOR_SPHERE_ETCD_ENDPOINTS"); etcdEndpoints != "" {
		endpoints := strings.Split(etcdEndpoints, ",")
		for i, ep := range endpoints {
			endpoints[i] = strings.TrimSpace(ep)
		}
		config.Etcd.Endpoints = endpoints
		logger.Info("Override etcd endpoints from env: %v", endpoints)
	}

	// 数据目录
	if dataDir := os.Getenv("VECTOR_SPHERE_DATA_DIR"); dataDir != "" {
		config.DataDir = dataDir
		logger.Info("Override dataDir from env: %s", dataDir)
	}

	// 超时配置
	if timeout := os.Getenv("VECTOR_SPHERE_TIMEOUT"); timeout != "" {
		if t, err := strconv.Atoi(timeout); err == nil {
			config.TimeOut = t
			logger.Info("Override timeout from env: %d", t)
		}
	}

	// 心跳间隔
	if heartbeat := os.Getenv("VECTOR_SPHERE_HEARTBEAT"); heartbeat != "" {
		if h, err := strconv.Atoi(heartbeat); err == nil {
			config.Heartbeat = h
			logger.Info("Override heartbeat from env: %d", h)
		}
	}

	// 调度器工作线程数
	if workerCount := os.Getenv("VECTOR_SPHERE_WORKER_COUNT"); workerCount != "" {
		if w, err := strconv.Atoi(workerCount); err == nil {
			config.SchedulerWorkerCount = w
			logger.Info("Override schedulerWorkerCount from env: %d", w)
		}
	}

	// 任务超时
	if taskTimeout := os.Getenv("VECTOR_SPHERE_TASK_TIMEOUT"); taskTimeout != "" {
		if t, err := strconv.Atoi(taskTimeout); err == nil {
			config.TaskTimeout = t
			logger.Info("Override taskTimeout from env: %d", t)
		}
	}

	// 健康检查间隔
	if healthInterval := os.Getenv("VECTOR_SPHERE_HEALTH_INTERVAL"); healthInterval != "" {
		if h, err := strconv.Atoi(healthInterval); err == nil {
			config.HealthCheckInterval = h
			logger.Info("Override healthCheckInterval from env: %d", h)
		}
	}
}

// setDefaults 设置默认值
func (cm *ConfigManager) setDefaults(config *DistributedConfig) {
	if config.TimeOut == 0 {
		config.TimeOut = 30
	}

	if config.Heartbeat == 0 {
		config.Heartbeat = 10
	}

	if config.SchedulerWorkerCount == 0 {
		config.SchedulerWorkerCount = 10
	}

	if config.TaskTimeout == 0 {
		config.TaskTimeout = 60
	}

	if config.HealthCheckInterval == 0 {
		config.HealthCheckInterval = 30
	}

	if config.DataDir == "" {
		config.DataDir = "./data"
	}

	// 确保数据目录存在
	if err := os.MkdirAll(config.DataDir, 0755); err != nil {
		logger.Warning("Failed to create data directory %s: %v", config.DataDir, err)
	}

	// 设置默认endpoints（如果为空）
	if config.Endpoints == nil {
		config.Endpoints = make(map[string]server.EndPoint)
	}

	logger.Info("Applied default configuration values")
}

// GetConfig 获取当前配置
func (cm *ConfigManager) GetConfig() *DistributedConfig {
	return cm.config
}

// ReloadConfig 重新加载配置
func (cm *ConfigManager) ReloadConfig() (*DistributedConfig, error) {
	logger.Info("Reloading configuration...")
	return cm.LoadConfig()
}

// SaveConfig 保存配置到文件
func (cm *ConfigManager) SaveConfig(config *DistributedConfig) error {
	return confType.WriteYAML(cm.configPath, config)
}

// GetDefaultConfigPath 获取默认配置文件路径
func GetDefaultConfigPath() string {
	// 优先使用环境变量
	if configPath := os.Getenv("VECTOR_SPHERE_CONFIG_PATH"); configPath != "" {
		return configPath
	}

	// 查找配置文件的可能位置
	possiblePaths := []string{
		"./config/service.yaml",
		"./conf/service.yaml",
		"./conf/idc/simple/service.yaml",
		"../conf/idc/simple/service.yaml",
		"/etc/vector_sphere/service.yaml",
	}

	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	// 返回默认路径
	return "./conf/idc/simple/service.yaml"
}

// CreateDefaultConfig 创建默认配置文件
func CreateDefaultConfig(configPath string) error {
	defaultConfig := &DistributedConfig{
		ServiceName:          "vector_sphere",
		NodeType:             SlaveNode,
		TimeOut:              30,
		DefaultPort:          8080,
		Heartbeat:            10,
		SchedulerWorkerCount: 10,
		HttpPort:             8090,
		TaskTimeout:          60,
		HealthCheckInterval:  30,
		DataDir:              "./data",
		Etcd: EtcdConfig{
			Endpoints: []string{"localhost:2379"},
		},
		Endpoints: make(map[string]server.EndPoint),
	}

	// 确保目录存在
	dir := filepath.Dir(configPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("创建配置目录失败: %v", err)
	}

	// 保存配置
	return confType.WriteYAML(configPath, defaultConfig)
}

// ValidateConfigFile 验证配置文件是否存在且有效
func ValidateConfigFile(configPath string) error {
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return fmt.Errorf("配置文件不存在: %s", configPath)
	}

	// 尝试读取配置
	var config DistributedConfig
	if err := confType.ReadYAML(configPath, &config); err != nil {
		return fmt.Errorf("配置文件格式错误: %v", err)
	}

	// 基本验证
	if config.ServiceName == "" {
		return fmt.Errorf("配置文件中serviceName不能为空")
	}

	if len(config.Etcd.Endpoints) == 0 {
		return fmt.Errorf("配置文件中etcd endpoints不能为空")
	}

	return nil
}

// GetConfigSummary 获取配置摘要信息
func (cm *ConfigManager) GetConfigSummary() map[string]interface{} {
	if cm.config == nil {
		return map[string]interface{}{"error": "配置未加载"}
	}

	return map[string]interface{}{
		"service_name":           cm.config.ServiceName,
		"node_type":              cm.config.NodeType,
		"default_port":           cm.config.DefaultPort,
		"http_port":              cm.config.HttpPort,
		"etcd_endpoints":         cm.config.Etcd.Endpoints,
		"data_dir":               cm.config.DataDir,
		"timeout":                cm.config.TimeOut,
		"heartbeat":              cm.config.Heartbeat,
		"scheduler_worker_count": cm.config.SchedulerWorkerCount,
		"task_timeout":           cm.config.TaskTimeout,
		"health_check_interval":  cm.config.HealthCheckInterval,
		"config_path":            cm.configPath,
	}
}

// UpdateConfig 更新配置的特定字段
func (cm *ConfigManager) UpdateConfig(updates map[string]interface{}) error {
	if cm.config == nil {
		return fmt.Errorf("配置未加载")
	}

	for key, value := range updates {
		switch key {
		case "service_name":
			if v, ok := value.(string); ok {
				cm.config.ServiceName = v
			}
		case "timeout":
			if v, ok := value.(int); ok {
				cm.config.TimeOut = v
			}
		case "heartbeat":
			if v, ok := value.(int); ok {
				cm.config.Heartbeat = v
			}
		case "scheduler_worker_count":
			if v, ok := value.(int); ok {
				cm.config.SchedulerWorkerCount = v
			}
		case "task_timeout":
			if v, ok := value.(int); ok {
				cm.config.TaskTimeout = v
			}
		case "health_check_interval":
			if v, ok := value.(int); ok {
				cm.config.HealthCheckInterval = v
			}
		case "data_dir":
			if v, ok := value.(string); ok {
				cm.config.DataDir = v
			}
		default:
			logger.Warning("Unknown config key: %s", key)
		}
	}

	// 验证更新后的配置
	if err := cm.validateConfig(cm.config); err != nil {
		return fmt.Errorf("配置更新后验证失败: %v", err)
	}

	logger.Info("Configuration updated successfully")
	return nil
}
