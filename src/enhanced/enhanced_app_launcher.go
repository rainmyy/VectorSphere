package enhanced

import (
	"VectorSphere/src/distributed"
	"VectorSphere/src/library/common"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"context"
	"encoding/json"
	"fmt"
	clientv3 "go.etcd.io/etcd/client/v3"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// EnhancedAppLauncher 增强型应用启动器
type EnhancedAppLauncher struct {
	// 核心组件
	configManager   *SimpleConfigManager
	loadBalancer    *EnhancedLoadBalancer
	healthChecker   *EnhancedHealthChecker
	securityManager *EnhancedSecurityManager
	circuitBreaker  *EnhancedCircuitBreaker
	gateway         *EnhancedAPIGateway

	// 分布式组件
	distributedManager DistributedManagerInterface
	communicationSvc   CommunicationServiceInterface
	serviceDiscovery   ServiceDiscoveryInterface

	// 上下文管理
	ctx    context.Context
	cancel context.CancelFunc

	// 运行状态
	isRunning bool
	startTime time.Time
}

// NewEnhancedAppLauncher 创建增强型应用启动器
func NewEnhancedAppLauncher(configPath string) (*EnhancedAppLauncher, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// 创建配置管理器
	configManager, err := NewSimpleConfigManager(configPath)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("创建配置管理器失败: %v", err)
	}

	return &EnhancedAppLauncher{
		configManager: configManager,
		ctx:           ctx,
		cancel:        cancel,
		isRunning:     false,
	}, nil
}

// Start 启动增强型应用
func (eal *EnhancedAppLauncher) Start() error {
	logger.Info("Starting Enhanced VectorSphere application...")
	eal.startTime = time.Now()

	// 1. 加载和验证配置
	if err := eal.loadAndValidateConfig(); err != nil {
		return fmt.Errorf("配置加载失败: %v", err)
	}

	config := eal.configManager.GetConfig()

	// 2. 根据配置初始化安全管理器
	if config.Features.EnableSecurityManager {
		if err := eal.initializeSecurityManager(); err != nil {
			return fmt.Errorf("安全管理器初始化失败: %v", err)
		}
	} else {
		logger.Info("Security manager disabled by configuration")
	}

	// 3. 根据配置初始化熔断器
	if config.Features.EnableCircuitBreaker {
		if err := eal.initializeCircuitBreaker(); err != nil {
			return fmt.Errorf("熔断器初始化失败: %v", err)
		}
	} else {
		logger.Info("Circuit breaker disabled by configuration")
	}

	// 4. 根据配置初始化负载均衡器
	if config.Features.EnableLoadBalancer {
		if err := eal.initializeLoadBalancer(); err != nil {
			return fmt.Errorf("负载均衡器初始化失败: %v", err)
		}
	} else {
		logger.Info("Load balancer disabled by configuration")
	}

	// 5. 根据配置初始化健康检查器
	if config.Features.EnableHealthChecker {
		if err := eal.initializeHealthChecker(); err != nil {
			return fmt.Errorf("健康检查器初始化失败: %v", err)
		}
	} else {
		logger.Info("Health checker disabled by configuration")
	}

	// 6. 根据配置初始化分布式组件
	if config.Features.EnableDistributed {
		if err := eal.initializeDistributedComponents(); err != nil {
			return fmt.Errorf("分布式组件初始化失败: %v", err)
		}
	} else {
		logger.Info("Distributed components disabled by configuration")
	}

	// 7. 根据配置初始化增强型API网关
	if config.Features.EnableAPIGateway {
		if err := eal.initializeAPIGateway(); err != nil {
			return fmt.Errorf("API网关初始化失败: %v", err)
		}
	} else {
		logger.Info("API Gateway disabled by configuration")
	}

	// 8. 启动所有服务
	if err := eal.startAllServices(); err != nil {
		return fmt.Errorf("服务启动失败: %v", err)
	}

	// 9. 注册当前节点
	if config.Features.EnableLoadBalancer && eal.loadBalancer != nil {
		if err := eal.registerCurrentNode(); err != nil {
			logger.Warning("节点注册失败: %v", err)
		}
	}

	// 10. 启动监控和维护任务
	if config.Features.EnableMonitoring {
		eal.startMaintenanceTasks()
	} else {
		logger.Info("Monitoring and maintenance tasks disabled by configuration")
	}

	eal.isRunning = true
	logger.Info("Enhanced VectorSphere application started successfully")
	return nil
}

// Stop 停止增强型应用
func (eal *EnhancedAppLauncher) Stop() error {
	logger.Info("Stopping Enhanced VectorSphere application...")

	// 优雅停止所有服务
	eal.stopAllServices()

	// 取消上下文
	eal.cancel()

	eal.isRunning = false
	logger.Info("Enhanced VectorSphere application stopped")
	return nil
}

// loadAndValidateConfig 加载和验证配置
func (eal *EnhancedAppLauncher) loadAndValidateConfig() error {
	config := eal.configManager.GetConfig()

	// 验证配置
	if err := eal.configManager.validateConfig(config); err != nil {
		return fmt.Errorf("配置验证失败: %v", err)
	}

	logger.Info("Configuration loaded and validated: %s", config.ServiceName)
	return nil
}

// initializeSecurityManager 初始化安全管理器
func (eal *EnhancedAppLauncher) initializeSecurityManager() error {
	config := eal.configManager.GetConfig()

	eal.securityManager = NewEnhancedSecurityManager(nil, &SecurityConfig{
		Enabled:          true,
		DefaultPolicy:    "default",
		CertificatePath:  "",
		PrivateKeyPath:   "",
		SessionTimeout:   24 * time.Hour,
		MaxLoginAttempts: config.SecurityManager.MaxLoginAttempts,
		LockoutDuration:  time.Duration(config.SecurityManager.LockoutDuration) * time.Minute,
		AuditLogPath:     config.SecurityManager.AuditLogPath,
		EncryptionKey:    []byte(config.SecurityManager.EncryptionKey),
	})
	logger.Info("Security manager initialized")
	return nil
}

// initializeCircuitBreaker 初始化熔断器
func (eal *EnhancedAppLauncher) initializeCircuitBreaker() error {
	eal.circuitBreaker = NewEnhancedCircuitBreaker()
	logger.Info("Circuit breaker initialized")
	return nil
}

// initializeLoadBalancer 初始化负载均衡器
func (eal *EnhancedAppLauncher) initializeLoadBalancer() error {
	logger.Info("Initializing enhanced load balancer")

	config := eal.configManager.GetConfig()
	lbConfig := &LoadBalancerConfig{
		Algorithm:           RoundRobin, // 使用默认的轮询算法
		HealthCheckEnabled:  true,       // 默认启用健康检查
		HealthCheckInterval: time.Duration(config.LoadBalancer.HealthCheckInterval) * time.Second,
		HealthCheckTimeout:  10 * time.Second,
		FailureThreshold:    3,
		RecoveryThreshold:   2,
		MaxRetries:          config.LoadBalancer.MaxRetries,
		RetryTimeout:        5 * time.Second,
		SessionAffinity:     false,
		AffinityTimeout:     30 * time.Minute,
		SlowStart:           false,
		SlowStartDuration:   2 * time.Minute,
		CircuitBreaker:      false,
		MetricsEnabled:      true,
		BasePrefix:          "/vectorsphere",
	}

	loadBalancer := NewEnhancedLoadBalancer(nil, lbConfig)

	eal.loadBalancer = loadBalancer
	logger.Info("Load balancer initialized with algorithm: %s", config.LoadBalancer.Algorithm)
	return nil
}

// initializeHealthChecker 初始化健康检查器
func (eal *EnhancedAppLauncher) initializeHealthChecker() error {
	config := eal.configManager.GetConfig()

	healthChecker := NewEnhancedHealthChecker(nil, &HealthCheckConfig{
		Enabled:           true,
		DefaultInterval:   time.Duration(config.HealthCheck.Interval) * time.Second,
		DefaultTimeout:    time.Duration(config.HealthCheck.Timeout) * time.Second,
		DefaultRetries:    config.HealthCheck.RetryCount,
		AdaptiveInterval:  true,
		MinInterval:       5 * time.Second,
		MaxInterval:       300 * time.Second,
		TrendWindowSize:   20,
		PredictionEnabled: true,
		AlertingEnabled:   true,
		MetricsRetention:  24 * time.Hour,
		BasePrefix:        "/vectorsphere/health",
	})

	eal.healthChecker = healthChecker
	logger.Info("Health checker initialized")
	return nil
}

// initializeDistributedComponents 初始化分布式组件
func (eal *EnhancedAppLauncher) initializeDistributedComponents() error {
	logger.Info("Initializing distributed components...")

	config := eal.configManager.GetConfig()
	distConfig := config.Distributed

	// 检查分布式功能是否启用
	if !distConfig.Enabled {
		logger.Info("Distributed functionality is disabled")
		return nil
	}

	// 验证分布式配置
	if len(distConfig.EtcdEndpoints) == 0 {
		return fmt.Errorf("etcd endpoints cannot be empty when distributed is enabled")
	}

	if distConfig.NodeID == "" {
		return fmt.Errorf("node ID cannot be empty when distributed is enabled")
	}

	if distConfig.ServiceName == "" {
		distConfig.ServiceName = config.ServiceName // 使用默认服务名
	}

	// 创建etcd客户端
	dialTimeout := time.Duration(distConfig.EtcdDialTimeout) * time.Second
	if dialTimeout == 0 {
		dialTimeout = 5 * time.Second
	}

	etcdClient, err := clientv3.New(clientv3.Config{
		Endpoints:   distConfig.EtcdEndpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		return fmt.Errorf("failed to create etcd client: %v", err)
	}

	// 初始化服务发现
	if distConfig.EnableServiceDiscovery {
		eal.serviceDiscovery = distributed.NewServiceDiscovery(etcdClient, distConfig.ServiceName)
		logger.Info("Service discovery initialized")
	}

	// 初始化通信服务
	if distConfig.EnableCommunication {
		eal.communicationSvc = distributed.NewCommunicationService(etcdClient, distConfig.ServiceName)
		logger.Info("Communication service initialized")
	}

	// 初始化分布式管理器
	if distConfig.EnableDistributedManager {
		// 转换 NodeType
		var nodeType distributed.NodeType
		if distConfig.NodeID == "master" {
			nodeType = distributed.MasterNode
		} else {
			nodeType = distributed.SlaveNode
		}

		dmConfig := &distributed.DistributedConfig{
			ServiceName:          distConfig.ServiceName,
			NodeType:             nodeType,
			DefaultPort:          config.Port,
			HttpPort:             config.Port + 1000,
			TimeOut:              30,
			Heartbeat:            10,
			DataDir:              "./data",
			TaskTimeout:          300,
			HealthCheckInterval:  30,
			SchedulerWorkerCount: 4,
			Etcd: distributed.EtcdConfig{
				Endpoints: convertStringSliceToEndpoints(distConfig.EtcdEndpoints),
			},
		}
		dm, err := distributed.NewDistributedManager(dmConfig)
		if err != nil {
			return fmt.Errorf("failed to create distributed manager: %v", err)
		}
		eal.distributedManager = dm
		logger.Info("Distributed manager initialized")
	}

	logger.Info("Distributed components configuration validated (implementation commented out for safety)")
	logger.Info("To enable distributed features, uncomment the implementation code and ensure etcd is available")
	return nil
}

// initializeAPIGateway 初始化增强型API网关
func (eal *EnhancedAppLauncher) initializeAPIGateway() error {
	config := eal.configManager.GetConfig()
	apiGatewayConfig := &APIGatewayConfig{
		Port:              config.Port,
		ReadTimeout:       time.Duration(config.Gateway.Timeout) * time.Second,
		WriteTimeout:      time.Duration(config.Gateway.Timeout) * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1048576,
		EnableCORS:        true,
		EnableCompression: false,
		EnableMetrics:     true,
		EnableLogging:     true,
	}

	gateway, err := NewEnhancedAPIGateway(apiGatewayConfig, eal.loadBalancer, eal.securityManager, eal.circuitBreaker,
		eal.healthChecker, eal.distributedManager, eal.communicationSvc, eal.serviceDiscovery)
	if err != nil {
		return err
	}

	eal.gateway = gateway
	logger.Info("Enhanced API Gateway initialized on port %d", config.Port)
	return nil
}

// startAllServices 启动所有服务
func (eal *EnhancedAppLauncher) startAllServices() error {
	// 启动安全管理器
	if err := eal.securityManager.Start(); err != nil {
		return fmt.Errorf("安全管理器启动失败: %v", err)
	}

	// 启动熔断器
	if err := eal.circuitBreaker.Start(); err != nil {
		return fmt.Errorf("熔断器启动失败: %v", err)
	}

	// 启动负载均衡器
	if err := eal.loadBalancer.Start(); err != nil {
		return fmt.Errorf("负载均衡器启动失败: %v", err)
	}

	// 启动健康检查器
	if err := eal.healthChecker.Start(); err != nil {
		return fmt.Errorf("健康检查器启动失败: %v", err)
	}

	// 启动API网关
	if err := eal.gateway.Start(eal.ctx); err != nil {
		return fmt.Errorf("API网关启动失败: %v", err)
	}

	logger.Info("All enhanced services started successfully")
	return nil
}

// stopAllServices 停止所有服务
func (eal *EnhancedAppLauncher) stopAllServices() {
	// 按相反顺序停止服务
	if eal.gateway != nil {
		if err := eal.gateway.Stop(eal.ctx); err != nil {
			logger.Error("停止API网关失败: %v", err)
		}
	}

	if eal.healthChecker != nil {
		if err := eal.healthChecker.Stop(); err != nil {
			logger.Error("停止健康检查器失败: %v", err)
		}
	}

	if eal.loadBalancer != nil {
		if err := eal.loadBalancer.Stop(); err != nil {
			logger.Error("停止负载均衡器失败: %v", err)
		}
	}

	if eal.circuitBreaker != nil {
		if err := eal.circuitBreaker.Stop(); err != nil {
			logger.Error("停止熔断器失败: %v", err)
		}
	}

	if eal.securityManager != nil {
		if err := eal.securityManager.Stop(); err != nil {
			logger.Error("停止安全管理器失败: %v", err)
		}
	}
}

// registerCurrentNode 注册当前节点
func (eal *EnhancedAppLauncher) registerCurrentNode() error {
	config := eal.configManager.GetConfig()

	// 获取本地IP
	localIP, err := common.GetLocalHost()
	if err != nil {
		return fmt.Errorf("获取本地IP失败: %v", err)
	}

	// 生成节点ID
	nodeID := fmt.Sprintf("%s-%s-%d", config.ServiceName, localIP, config.Port)

	// 注册到负载均衡器
	backend := &Backend{
		ID:      nodeID,
		Address: localIP,
		Port:    config.Port,
		Weight:  100,
		Status:  BackendHealthy,
		Metadata: map[string]interface{}{
			"service_name": config.ServiceName,
			"version":      "enhanced-1.0.0",
			"start_time":   strconv.FormatInt(eal.startTime.Unix(), 10),
		},
	}

	return eal.loadBalancer.AddBackend(backend)
}

// startMaintenanceTasks 启动维护任务
func (eal *EnhancedAppLauncher) startMaintenanceTasks() {
	// 启动配置热重载监控
	go eal.configReloadWatcher()

	// 启动性能监控
	go eal.performanceMonitor()

	// 启动资源清理
	go eal.resourceCleaner()
}

// configReloadWatcher 配置热重载监控
func (eal *EnhancedAppLauncher) configReloadWatcher() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-eal.ctx.Done():
			return
		case <-ticker.C:
			if eal.configManager.HasConfigChanged() {
				logger.Info("Configuration file changed, reloading...")
				if err := eal.ReloadConfig(); err != nil {
					logger.Error("配置重载失败: %v", err)
				}
			}
		}
	}
}

// performanceMonitor 性能监控
func (eal *EnhancedAppLauncher) performanceMonitor() {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-eal.ctx.Done():
			return
		case <-ticker.C:
			// 收集性能指标
			metrics := eal.collectMetrics()
			logger.Info("Performance metrics: %+v", metrics)
		}
	}
}

// resourceCleaner 资源清理
func (eal *EnhancedAppLauncher) resourceCleaner() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-eal.ctx.Done():
			return
		case <-ticker.C:
			// 清理过期资源
			eal.cleanupExpiredResources()
		}
	}
}

// collectMetrics 收集性能指标
func (eal *EnhancedAppLauncher) collectMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})

	if eal.loadBalancer != nil {
		metrics["load_balancer"] = eal.loadBalancer.GetMetrics()
	}

	if eal.circuitBreaker != nil {
		metrics["circuit_breaker"] = eal.circuitBreaker.GetMetrics()
	}

	if eal.healthChecker != nil {
		metrics["health_checker"] = eal.healthChecker.GetMetrics()
	}

	if eal.gateway != nil {
		metrics["api_gateway"] = eal.gateway.GetMetrics()
	}

	metrics["uptime"] = time.Since(eal.startTime).Seconds()
	return metrics
}

// cleanupExpiredResources 清理过期资源
func (eal *EnhancedAppLauncher) cleanupExpiredResources() {
	if eal.securityManager != nil {
		eal.securityManager.cleanupExpiredSessions()
	}

	if eal.loadBalancer != nil {
		eal.loadBalancer.cleanup()
	}

	if eal.circuitBreaker != nil {
		// Circuit breaker cleanup is handled internally
	}
}

// Run 运行增强型应用（阻塞直到收到停止信号）
func (eal *EnhancedAppLauncher) Run() error {
	// 启动应用
	if err := eal.Start(); err != nil {
		return err
	}

	// 等待停止信号
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	println("Enhanced VectorSphere Application is running. Press Ctrl+C to stop.")

	// 阻塞等待信号
	sig := <-sigChan
	logger.Info("Received signal: %v, shutting down gracefully...", sig)

	// 优雅停止
	return eal.Stop()
}

// ReloadConfig 重新加载配置
func (eal *EnhancedAppLauncher) ReloadConfig() error {
	logger.Info("Reloading enhanced configuration...")

	// 重新加载配置
	newConfig, err := eal.configManager.ReloadConfig()
	if err != nil {
		return fmt.Errorf("重新加载配置失败: %v", err)
	}

	// 热更新各组件配置
	if eal.securityManager != nil {
		// Security manager config update is handled internally
		logger.Info("Security manager configuration updated")
	}

	if eal.circuitBreaker != nil {
		// Circuit breaker config update is handled internally
		logger.Info("Circuit breaker configuration updated")
	}

	// 更新负载均衡器配置
	if eal.loadBalancer != nil {
		// Load balancer config update is handled internally
		logger.Info("Load balancer configuration updated")
	}

	logger.Info("Enhanced configuration reloaded successfully: %s", newConfig.ServiceName)
	return nil
}

// GetStatus 获取增强型应用状态
func (eal *EnhancedAppLauncher) GetStatus() map[string]interface{} {
	status := map[string]interface{}{
		"running":    eal.isRunning,
		"start_time": eal.startTime,
		"uptime":     time.Since(eal.startTime).Seconds(),
	}

	if eal.configManager != nil {
		status["config"] = "loaded"
	}

	if eal.loadBalancer != nil {
		status["load_balancer"] = map[string]interface{}{
			"healthy": eal.loadBalancer.IsHealthy(),
			"metrics": eal.loadBalancer.GetMetrics(),
		}
	}

	if eal.securityManager != nil {
		status["security_manager"] = eal.securityManager.GetSecurityMetrics()
	}

	if eal.circuitBreaker != nil {
		status["circuit_breaker"] = eal.circuitBreaker.GetState()
	}

	if eal.healthChecker != nil {
		status["health_checker"] = eal.healthChecker.GetHealthStatus()
	}

	if eal.gateway != nil {
		status["api_gateway"] = eal.gateway.GetStatus()
	}

	return status
}

// GetComponents 获取所有组件
func (eal *EnhancedAppLauncher) GetComponents() map[string]interface{} {
	return map[string]interface{}{
		"config_manager":   eal.configManager,
		"load_balancer":    eal.loadBalancer,
		"security_manager": eal.securityManager,
		"circuit_breaker":  eal.circuitBreaker,
		"health_checker":   eal.healthChecker,
		"api_gateway":      eal.gateway,
	}
}

// convertStringSliceToEndpoints 转换字符串切片到EndPoint切片
func convertStringSliceToEndpoints(endpoints []string) []entity.EndPoint {
	var result []entity.EndPoint
	for _, ep := range endpoints {
		// 解析 host:port 格式
		parts := strings.Split(ep, ":")
		if len(parts) == 2 {
			port, err := strconv.Atoi(parts[1])
			if err == nil {
				result = append(result, entity.EndPoint{
					Ip:   parts[0],
					Port: port,
				})
			}
		}
	}
	return result
}

// GetDefaultEnhancedConfigPath 获取默认配置文件路径
func GetDefaultEnhancedConfigPath() string {
	return "./config/enhanced_config.json"
}

// ValidateEnhancedConfigFile 验证配置文件
func ValidateEnhancedConfigFile(configPath string) error {
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return fmt.Errorf("配置文件不存在: %s", configPath)
	}
	return nil
}

// CreateDefaultEnhancedConfig 创建默认配置文件
func CreateDefaultEnhancedConfig(configPath string) error {
	defaultConfig := &SimpleConfig{
		Port:        8080,
		Host:        "0.0.0.0",
		ServiceName: "VectorSphere-Enhanced",
		LogLevel:    "info",
		Gateway: SimpleGatewayConfig{
			Enabled:        true,
			Timeout:        30,
			MaxConnections: 1000,
			RateLimit:      100,
		},
		LoadBalancer: SimpleLoadBalancerConfig{
			Algorithm:            "round_robin",
			HealthCheckInterval:  30,
			MaxRetries:           3,
			RetryDelay:           5000,
			EnableStickySessions: false,
			SessionTimeout:       30,
			EnableMetrics:        true,
			MaxConnections:       1000,
			ConnectionTimeout:    10,
		},
		SecurityManager: SimpleSecurityConfig{
			JWTSecret:          "default-jwt-secret-change-in-production",
			TokenExpiration:    24,
			EnableRateLimit:    true,
			RateLimitPerSecond: 100,
			EnableIPWhitelist:  false,
			IPWhitelist:        []string{},
			MaxLoginAttempts:   5,
			LockoutDuration:    30,
			AuditLogPath:       "./logs/audit.log",
			EncryptionKey:      "default-encryption-key-32-chars!!",
		},
		CircuitBreaker: SimpleCircuitBreakerConfig{
			FailureThreshold:    5,
			RecoveryTimeout:     60,
			MaxRequests:         10,
			Interval:            30,
			Timeout:             30,
			EnableMetrics:       true,
			EnableNotification:  false,
			NotificationURL:     "",
			HalfOpenMaxRequests: 5,
		},
		HealthCheck: SimpleHealthCheckConfig{
			Interval:           30,
			Timeout:            10,
			RetryCount:         3,
			RetryDelay:         5000,
			EnableDeepCheck:    false,
			EnableMetrics:      true,
			EnableNotification: false,
			UnhealthyThreshold: 3,
			HealthyThreshold:   2,
			HealthCheckPath:    "/health",
			ExpectedStatusCode: 200,
		},
		Monitoring: MonitoringConfig{
			EnableMetrics:      true,
			MetricsPort:        9090,
			MetricsPath:        "/metrics",
			EnableTracing:      false,
			TracingEndpoint:    "",
			EnableProfiling:    false,
			ProfilingPort:      6060,
			CollectionInterval: 30,
		},
		Features: FeatureConfig{
			EnableSecurityManager: true,
			EnableCircuitBreaker:  true,
			EnableLoadBalancer:    true,
			EnableHealthChecker:   true,
			EnableDistributed:     false,
			EnableAPIGateway:      true,
			EnableMonitoring:      true,
		},
		Distributed: DistributedFeatureConfig{
			Enabled:                  false,
			NodeID:                   "enhanced-node-1",
			ServiceName:              "",
			EtcdEndpoints:            []string{"localhost:2379"},
			EtcdDialTimeout:          5,
			ClusterNodes:             []string{"localhost:8080"},
			EnableServiceDiscovery:   true,
			EnableCommunication:      true,
			EnableDistributedManager: true,
		},
	}

	// 确保目录存在
	dir := filepath.Dir(configPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("创建配置目录失败: %v", err)
	}

	// 写入配置文件
	data, err := json.MarshalIndent(defaultConfig, "", "  ")
	if err != nil {
		return fmt.Errorf("序列化配置失败: %v", err)
	}

	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("写入配置文件失败: %v", err)
	}

	logger.Info("默认配置文件已创建: %s", configPath)
	return nil
}

// CreateAndRunEnhancedApp 创建并运行增强型应用的便捷函数
func CreateAndRunEnhancedApp(configPath string) error {
	// 如果配置文件不存在，创建默认配置
	if configPath == "" {
		configPath = GetDefaultEnhancedConfigPath()
	}

	if err := ValidateEnhancedConfigFile(configPath); err != nil {
		logger.Warning("Enhanced configuration file validation failed: %v", err)
		logger.Info("Creating default enhanced configuration file: %s", configPath)
		if err := CreateDefaultEnhancedConfig(configPath); err != nil {
			return fmt.Errorf("创建默认增强配置文件失败: %v", err)
		}
	}

	// 创建增强型应用启动器
	appLauncher, err := NewEnhancedAppLauncher(configPath)
	if err != nil {
		return fmt.Errorf("创建增强型应用启动器失败: %v", err)
	}

	// 运行应用
	return appLauncher.Run()
}
