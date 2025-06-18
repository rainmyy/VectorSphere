package bootstrap

import (
	"VectorSphere/src/enhanced"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/server"
	"context"
	"fmt"
	"net"
	"strconv"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// AppConfig 应用配置
type AppConfig struct {
	// 基本配置
	ServiceName string   `json:"service_name" yaml:"service_name"`
	NodeID      string   `json:"node_id" yaml:"node_id"`
	NodeType    string   `json:"node_type" yaml:"node_type"` // "master" 或 "slave"
	Version     string   `json:"version" yaml:"version"`
	Region      string   `json:"region" yaml:"region"`
	Zone        string   `json:"zone" yaml:"zone"`
	Tags        []string `json:"tags" yaml:"tags"`

	// 网络配置
	Address  string `json:"address" yaml:"address"`
	Port     int    `json:"port" yaml:"port"`
	HttpPort int    `json:"http_port" yaml:"http_port"`

	// etcd配置
	EtcdEndpoints []string `json:"etcd_endpoints" yaml:"etcd_endpoints"`
	EtcdTimeout   int      `json:"etcd_timeout" yaml:"etcd_timeout"`

	// 服务注册配置
	ServiceTTL          int64         `json:"service_ttl" yaml:"service_ttl"`
	HeartbeatInterval   time.Duration `json:"heartbeat_interval" yaml:"heartbeat_interval"`
	HealthCheckInterval time.Duration `json:"health_check_interval" yaml:"health_check_interval"`
	HealthCheckEndpoint string        `json:"health_check_endpoint" yaml:"health_check_endpoint"`

	// 负载均衡配置
	LoadBalancerAlgorithm string        `json:"load_balancer_algorithm" yaml:"load_balancer_algorithm"`
	MaxRetries            int           `json:"max_retries" yaml:"max_retries"`
	RetryTimeout          time.Duration `json:"retry_timeout" yaml:"retry_timeout"`

	// 配置管理
	ConfigNamespace string `json:"config_namespace" yaml:"config_namespace"`
	ConfigEnv       string `json:"config_env" yaml:"config_env"`
}

// InitEtcdClient 初始化etcd客户端
func InitEtcdClient(config *AppConfig) (*clientv3.Client, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	if len(config.EtcdEndpoints) == 0 {
		return nil, fmt.Errorf("etcd endpoints cannot be empty")
	}

	logger.Info("Initializing etcd client with endpoints: %v", config.EtcdEndpoints)

	client, err := clientv3.New(clientv3.Config{
		Endpoints:   config.EtcdEndpoints,
		DialTimeout: time.Duration(config.EtcdTimeout) * time.Second,
	})

	if err != nil {
		return nil, fmt.Errorf("failed to create etcd client: %v", err)
	}

	// 测试连接
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.EtcdTimeout)*time.Second)
	defer cancel()

	_, err = client.Status(ctx, config.EtcdEndpoints[0])
	if err != nil {
		client.Close()
		return nil, fmt.Errorf("failed to connect to etcd: %v", err)
	}

	logger.Info("Etcd client initialized successfully")
	return client, nil
}

// InitServiceRegistry 初始化服务注册
func InitServiceRegistry(client *clientv3.Client, config *AppConfig) (*enhanced.EnhancedServiceRegistry, error) {
	logger.Info("Initializing enhanced service registry")

	// 创建服务注册配置
	registryConfig := &enhanced.ServiceRegistryConfig{
		BaseTTL:              config.ServiceTTL,
		MinTTL:               30,  // 最小30秒
		MaxTTL:               300, // 最大5分钟
		HeartbeatInterval:    config.HeartbeatInterval,
		HealthCheckInterval:  config.HealthCheckInterval,
		RetryAttempts:        3,
		RetryInterval:        time.Second * 5,
		EnableCache:          true,
		CacheTTL:             time.Minute * 5,
		EnableNotification:   true,
		NotificationChannels: []string{"service_change"},
	}

	// 创建服务注册实例
	registry := enhanced.NewEnhancedServiceRegistry(client, registryConfig)

	logger.Info("Enhanced service registry initialized successfully")
	return registry, nil
}

// RegisterService 注册服务
func RegisterService(ctx context.Context, registry *enhanced.EnhancedServiceRegistry, config *AppConfig) error {
	logger.Info("Registering service: %s, type: %s, address: %s:%d", config.ServiceName, config.NodeType, config.Address, config.Port)

	// 如果地址为空，尝试获取本地IP
	address := config.Address
	if address == "" {
		ip, err := getLocalIP()
		if err != nil {
			return fmt.Errorf("failed to get local IP: %v", err)
		}
		address = ip
	}
	timeNow := time.Now()
	// 创建服务元数据
	metadata := &enhanced.ServiceMetadata{
		ServiceName:   config.ServiceName,
		NodeID:        config.NodeID,
		Address:       address,
		Port:          config.Port,
		NodeType:      config.NodeType,
		Status:        "active",
		Version:       config.Version,
		Region:        config.Region,
		Zone:          config.Zone,
		Tags:          config.Tags,
		Capabilities:  []string{"search", "index", "config"},
		Load:          0.0,
		HealthScore:   1.0,
		LastHeartbeat: timeNow,
		StartTime:     timeNow,
		Metrics:       make(map[string]interface{}),
		CustomData: map[string]string{
			"health_check_endpoint": config.HealthCheckEndpoint,
			"http_port":             strconv.Itoa(config.HttpPort),
		},
	}

	// 注册服务
	err := registry.RegisterService(ctx, metadata)
	if err != nil {
		return fmt.Errorf("failed to register service: %v", err)
	}

	logger.Info("Service registered successfully")
	return nil
}

// InitLoadBalancer 初始化负载均衡器
func InitLoadBalancer(client *clientv3.Client, registry *enhanced.EnhancedServiceRegistry, config *AppConfig) (*enhanced.EnhancedLoadBalancer, error) {
	logger.Info("Initializing enhanced load balancer")

	// 确定负载均衡算法
	var algorithm enhanced.LoadBalancingAlgorithm
	switch config.LoadBalancerAlgorithm {
	case "round_robin":
		algorithm = server.RoundRobin
	case "weighted_round_robin":
		algorithm = server.WeightedRoundRobin
	case "least_connections":
		algorithm = server.LeastConnections
	case "weighted_least_connections":
		algorithm = enhanced.WeightedLeastConnections
	case "random":
		algorithm = server.Random
	case "weighted_random":
		algorithm = enhanced.WeightedRandom
	case "ip_hash":
		algorithm = enhanced.IPHash
	case "consistent_hash":
		algorithm = server.ConsistentHash
	case "least_response_time":
		algorithm = enhanced.LeastResponseTime
	case "resource_based":
		algorithm = enhanced.ResourceBased
	case "adaptive":
		algorithm = enhanced.Adaptive
	default:
		algorithm = server.RoundRobin
	}

	// 创建负载均衡器配置
	lbConfig := &enhanced.LoadBalancerConfig{
		Algorithm:           algorithm,
		HealthCheckEnabled:  true,
		HealthCheckInterval: config.HealthCheckInterval,
		HealthCheckTimeout:  time.Second * 5,
		FailureThreshold:    3,
		RecoveryThreshold:   2,
		MaxRetries:          config.MaxRetries,
		RetryTimeout:        config.RetryTimeout,
		SessionAffinity:     true,
		AffinityTimeout:     time.Minute * 30,
		SlowStart:           true,
		SlowStartDuration:   time.Minute * 2,
		CircuitBreaker:      true,
		MetricsEnabled:      true,
		BasePrefix:          "/vector_sphere/services",
	}

	// 创建负载均衡器实例
	lb := enhanced.NewEnhancedLoadBalancer(client, lbConfig)

	// 设置服务发现源
	err := lb.SetServiceDiscoverySource(registry)
	if err != nil {
		return nil, fmt.Errorf("failed to set service discovery source: %v", err)
	}

	logger.Info("Enhanced load balancer initialized successfully")
	return lb, nil
}

// InitConfigManager 初始化配置管理器
func InitConfigManager(client *clientv3.Client, config *AppConfig) (*enhanced.EnhancedConfigManager, error) {
	logger.Info("Initializing enhanced config manager")

	// 创建配置管理器实例
	configManager, err := enhanced.NewEnhancedConfigManager(client, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create enhanced config manager: %v", err)
	}

	// 初始化配置命名空间
	if config.ConfigNamespace != "" {
		err = configManager.InitNamespace(context.Background(), config.ConfigNamespace, config.ConfigEnv)
		if err != nil {
			logger.Warning("Failed to initialize config namespace: %v", err)
		}
	}

	logger.Info("Enhanced config manager initialized successfully")
	return configManager, nil
}

// LoadConfig 加载配置
func LoadConfig(configPath string) (*AppConfig, error) {
	// 这里应该实现从配置文件加载配置的逻辑
	// 为了示例，这里返回一个默认配置
	return &AppConfig{
		ServiceName:           "vector_sphere",
		NodeID:                "node1",
		NodeType:              "master",
		Version:               "1.0.0",
		Region:                "default",
		Zone:                  "default",
		Tags:                  []string{"production"},
		Port:                  8000,
		HttpPort:              8080,
		EtcdEndpoints:         []string{"localhost:2379"},
		EtcdTimeout:           50000000,
		ServiceTTL:            60,
		HeartbeatInterval:     time.Second * 30,
		HealthCheckInterval:   time.Second * 60,
		HealthCheckEndpoint:   "/health",
		LoadBalancerAlgorithm: "round_robin",
		MaxRetries:            3,
		RetryTimeout:          time.Second * 10,
		ConfigNamespace:       "vector_sphere",
		ConfigEnv:             "production",
	}, nil
}

// getLocalIP 获取本地IP地址
func getLocalIP() (string, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "", err
	}

	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				return ipnet.IP.String(), nil
			}
		}
	}

	return "", fmt.Errorf("no available IP address found")
}
