package distributed

import (
	"VectorSphere/src/library/common"
	"VectorSphere/src/library/log"
	"context"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
)

// AppLauncher 应用启动器
type AppLauncher struct {
	configManager      *ConfigManager
	distributedManager *DistributedManager
	communicationSvc   *CommunicationService
	serviceDiscovery   *ServiceDiscovery
	apiGateway         *APIGateway

	ctx    context.Context
	cancel context.CancelFunc
}

// NewAppLauncher 创建应用启动器
func NewAppLauncher(configPath string) (*AppLauncher, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// 创建配置管理器
	configManager := NewConfigManager(configPath)

	return &AppLauncher{
		configManager: configManager,
		ctx:           ctx,
		cancel:        cancel,
	}, nil
}

// Start 启动应用
func (al *AppLauncher) Start() error {
	log.Info("Starting VectorSphere distributed application...")

	// 1. 加载配置
	config, err := al.configManager.LoadConfig()
	if err != nil {
		return fmt.Errorf("加载配置失败: %v", err)
	}
	log.Info("Configuration loaded: %s", config.ServiceName)

	// 2. 创建分布式管理器
	al.distributedManager, err = NewDistributedManager(config)
	if err != nil {
		return fmt.Errorf("创建分布式管理器失败: %v", err)
	}

	// 3. 启动分布式管理器
	if err := al.distributedManager.Start(); err != nil {
		return fmt.Errorf("启动分布式管理器失败: %v", err)
	}

	// 4. 创建服务发现组件
	al.serviceDiscovery = NewServiceDiscovery(al.distributedManager.GetEtcdClient(), config.ServiceName)

	// 5. 注册当前节点
	if err := al.registerCurrentNode(); err != nil {
		log.Warning("注册当前节点失败: %v", err)
	}

	// 6. 创建通信服务
	al.communicationSvc = NewCommunicationService(time.Duration(config.TaskTimeout) * time.Second)

	// 7. 启动API网关（仅master节点）
	if err := al.startAPIGateway(); err != nil {
		return fmt.Errorf("启动API网关失败: %v", err)
	}

	// 8. 启动服务监听
	al.startServiceWatchers()

	// 9. 启动健康检查
	go al.startHealthCheck()

	log.Info("VectorSphere application started successfully")
	return nil
}

// Stop 停止应用
func (al *AppLauncher) Stop() error {
	log.Info("Stopping VectorSphere application...")

	// 停止API网关
	if al.apiGateway != nil {
		if err := al.apiGateway.Stop(al.ctx); err != nil {
			log.Error("停止API网关失败: %v", err)
		}
	}

	// 停止通信服务
	if al.communicationSvc != nil {
		al.communicationSvc.Close()
	}

	// 停止服务发现
	if al.serviceDiscovery != nil {
		al.serviceDiscovery.Stop()
	}

	// 停止分布式管理器
	if al.distributedManager != nil {
		if err := al.distributedManager.Stop(); err != nil {
			log.Error("停止分布式管理器失败: %v", err)
		}
	}

	// 取消上下文
	al.cancel()

	log.Info("VectorSphere application stopped")
	return nil
}

// registerCurrentNode 注册当前节点
func (al *AppLauncher) registerCurrentNode() error {
	config := al.configManager.GetConfig()

	// 获取本地IP
	localIP, err := common.GetLocalHost()
	if err != nil {
		return fmt.Errorf("获取本地IP失败: %v", err)
	}

	// 生成节点ID
	nodeID := fmt.Sprintf("%s-%s-%d", config.ServiceName, localIP, config.DefaultPort)

	// 确定节点类型
	nodeType := "slave"
	if config.NodeType == MasterNode {
		nodeType = "master"
	}

	// 创建服务信息
	serviceInfo := CreateServiceInfo(
		config.ServiceName,
		nodeID,
		localIP,
		config.DefaultPort,
		nodeType,
	)

	// 添加元数据
	serviceInfo.Metadata["http_port"] = strconv.Itoa(config.HttpPort)
	serviceInfo.Metadata["data_dir"] = config.DataDir
	serviceInfo.Metadata["worker_count"] = strconv.Itoa(config.SchedulerWorkerCount)

	// 注册服务
	return al.serviceDiscovery.RegisterService(al.ctx, serviceInfo, int64(config.Heartbeat*3))
}

// startAPIGateway 启动API网关
func (al *AppLauncher) startAPIGateway() error {
	config := al.configManager.GetConfig()

	// 创建API网关
	al.apiGateway = NewAPIGateway(
		al.distributedManager,
		al.communicationSvc,
		al.serviceDiscovery,
		config.HttpPort,
	)

	// 启动API网关
	return al.apiGateway.Start(al.ctx)
}

// startServiceWatchers 启动服务监听器
func (al *AppLauncher) startServiceWatchers() {
	// 监听master变化
	al.serviceDiscovery.WatchMaster(al.ctx, func(masterInfo *ServiceInfo) {
		if masterInfo != nil {
			log.Info("Master changed: %s:%d", masterInfo.Address, masterInfo.Port)
		} else {
			log.Info("Master removed")
		}
	})

	// 监听slave变化
	al.serviceDiscovery.WatchSlaves(al.ctx, func(slaves map[string]*ServiceInfo) {
		log.Info("Slaves updated, count: %d", len(slaves))
		for nodeID, info := range slaves {
			log.Info("Slave: %s -> %s:%d (status: %s)", nodeID, info.Address, info.Port, info.Status)
		}
	})
}

// startHealthCheck 启动健康检查
func (al *AppLauncher) startHealthCheck() {
	config := al.configManager.GetConfig()
	ticker := time.NewTicker(time.Duration(config.HealthCheckInterval) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-al.ctx.Done():
			return
		case <-ticker.C:
			al.performHealthCheck()
		}
	}
}

// performHealthCheck 执行健康检查
func (al *AppLauncher) performHealthCheck() {
	// 检查自身状态
	if al.distributedManager.IsMaster() {
		// Master节点检查所有slave的健康状态
		slaveAddrs := al.serviceDiscovery.GetSlaveAddresses()
		if len(slaveAddrs) > 0 {
			healthStatus := al.communicationSvc.HealthCheckSlaves(al.ctx, slaveAddrs)
			healthyCount := 0
			for addr, healthy := range healthStatus {
				if healthy {
					healthyCount++
				} else {
					log.Warning("Slave %s is unhealthy", addr)
				}
			}
			log.Info("Health check completed: %d/%d slaves healthy", healthyCount, len(slaveAddrs))
		}
	} else {
		// Slave节点更新自己的状态
		if err := al.updateNodeStatus("active"); err != nil {
			log.Warning("Failed to update node status: %v", err)
		}
	}
}

// updateNodeStatus 更新节点状态
func (al *AppLauncher) updateNodeStatus(status string) error {
	config := al.configManager.GetConfig()
	localIP, err := common.GetLocalHost()
	if err != nil {
		return err
	}

	nodeID := fmt.Sprintf("%s-%s-%d", config.ServiceName, localIP, config.DefaultPort)
	nodeType := "slave"
	if al.distributedManager.IsMaster() {
		nodeType = "master"
	}

	return al.serviceDiscovery.UpdateServiceStatus(al.ctx, nodeType, nodeID, status)
}

// Run 运行应用（阻塞直到收到停止信号）
func (al *AppLauncher) Run() error {
	// 启动应用
	if err := al.Start(); err != nil {
		return err
	}

	// 等待停止信号
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	log.Info("Application is running. Press Ctrl+C to stop.")

	// 阻塞等待信号
	sig := <-sigChan
	log.Info("Received signal: %v, shutting down...", sig)

	// 优雅停止
	return al.Stop()
}

// GetStatus 获取应用状态
func (al *AppLauncher) GetStatus() map[string]interface{} {
	status := map[string]interface{}{
		"running": al.distributedManager != nil && al.distributedManager.isRunning,
	}

	if al.configManager != nil {
		status["config"] = al.configManager.GetConfigSummary()
	}

	if al.distributedManager != nil {
		status["is_master"] = al.distributedManager.IsMaster()
	}

	if al.serviceDiscovery != nil {
		status["master"] = al.serviceDiscovery.GetMaster()
		status["slaves"] = al.serviceDiscovery.GetSlaves()
	}

	if al.communicationSvc != nil {
		status["active_connections"] = al.communicationSvc.GetActiveConnections()
	}

	return status
}

// GetDistributedManager 获取分布式管理器
func (al *AppLauncher) GetDistributedManager() *DistributedManager {
	return al.distributedManager
}

// GetServiceDiscovery 获取服务发现组件
func (al *AppLauncher) GetServiceDiscovery() *ServiceDiscovery {
	return al.serviceDiscovery
}

// GetCommunicationService 获取通信服务
func (al *AppLauncher) GetCommunicationService() *CommunicationService {
	return al.communicationSvc
}

// GetAPIGateway 获取API网关
func (al *AppLauncher) GetAPIGateway() *APIGateway {
	return al.apiGateway
}

// GetConfigManager 获取配置管理器
func (al *AppLauncher) GetConfigManager() *ConfigManager {
	return al.configManager
}

// ReloadConfig 重新加载配置
func (al *AppLauncher) ReloadConfig() error {
	log.Info("Reloading configuration...")

	// 重新加载配置
	newConfig, err := al.configManager.ReloadConfig()
	if err != nil {
		return fmt.Errorf("重新加载配置失败: %v", err)
	}

	// 这里可以添加配置热更新的逻辑
	// 例如：更新限流配置、认证配置等
	if al.apiGateway != nil {
		al.apiGateway.SetRateLimit(100) // 示例：更新限流配置
	}

	log.Info("Configuration reloaded successfully: %s", newConfig.ServiceName)
	return nil
}

// CreateAndRunApp 创建并运行应用的便捷函数
func CreateAndRunApp(configPath string) error {
	// 如果配置文件不存在，创建默认配置
	if configPath == "" {
		configPath = GetDefaultConfigPath()
	}

	if err := ValidateConfigFile(configPath); err != nil {
		log.Warning("Configuration file validation failed: %v", err)
		log.Info("Creating default configuration file: %s", configPath)
		if err := CreateDefaultConfig(configPath); err != nil {
			return fmt.Errorf("创建默认配置文件失败: %v", err)
		}
	}

	// 创建应用启动器
	appLauncher, err := NewAppLauncher(configPath)
	if err != nil {
		return fmt.Errorf("创建应用启动器失败: %v", err)
	}

	// 运行应用
	return appLauncher.Run()
}