package distributed

import (
	"VectorSphere/src/library/common"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/library/pool"
	"VectorSphere/src/scheduler"
	"VectorSphere/src/server"
	"context"
	"fmt"
	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"
	"strconv"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	_ "google.golang.org/grpc"
)

// NodeType 节点类型
type NodeType int

const (
	MasterNode NodeType = iota
	SlaveNode
)

// DistributedConfig 分布式配置
type DistributedConfig struct {
	ServiceName          string     `yaml:"serviceName"`
	NodeType             NodeType   `yaml:"nodeType"`
	TimeOut              int        `yaml:"timeOut"`
	DefaultPort          int        `yaml:"defaultPort"`
	Heartbeat            int        `yaml:"heartbeat"`
	Etcd                 EtcdConfig `yaml:"etcd"`
	SchedulerWorkerCount int        `yaml:"schedulerWorkerCount"`
	HttpPort             int        `yaml:"httpPort"`
	TaskTimeout          int        `yaml:"taskTimeout"`
	HealthCheckInterval  int        `yaml:"healthCheckInterval"`
	DataDir              string     `yaml:"dataDir"`
	//LoadBalancerConfig    *enhanced.EnhancedLoadBalancer
	//EnhancedConfigManager *enhanced.EnhancedConfigManager
}

type EtcdConfig struct {
	Endpoints []entity.EndPoint `yaml:"endpoints"`
}

// DistributedManager 分布式管理器
type DistributedManager struct {
	config *DistributedConfig
	ctx    context.Context
	cancel context.CancelFunc
	mutex  sync.RWMutex

	// etcd相关
	etcdClient *clientv3.Client
	session    *concurrency.Session
	election   *concurrency.Election
	leaseID    clientv3.LeaseID

	// 服务相关
	localhost     string
	taskPool      *pool.Pool
	taskScheduler *scheduler.TaskPoolManager
	services      map[string]interface{} // 服务名称 -> 服务实例

	// 节点服务
	masterService    *server.MasterService
	slaveService     *server.SlaveService
	serviceDiscovery *ServiceDiscovery
	communicationSvc *CommunicationService
	apiGateway       *APIGateway

	// 状态管理
	isRunning bool
	stopCh    chan struct{}
}

// NewDistributedManager 创建分布式管理器
func NewDistributedManager(config *DistributedConfig) (*DistributedManager, error) {
	ctx, cancel := context.WithCancel(context.Background())
	// 获取本地IP
	localIP, err := common.GetLocalHost()

	if err != nil {
		cancel()
		return nil, fmt.Errorf("获取本地IP失败: %v", err)
	}

	localhost := localIP + ":" + strconv.Itoa(config.DefaultPort)

	// 创建任务调度器
	taskScheduler := scheduler.NewTaskPoolManager()

	dm := &DistributedManager{
		config:        config,
		ctx:           ctx,
		cancel:        cancel,
		localhost:     localhost,
		taskScheduler: taskScheduler,
		services:      make(map[string]interface{}),
		stopCh:        make(chan struct{}),
	}

	return dm, nil
}

// Start 启动分布式管理器
func (dm *DistributedManager) Start() error {
	// 初始化etcd客户端
	if err := dm.initEtcdClient(); err != nil {
		return fmt.Errorf("初始化etcd客户端失败: %v", err)
	}

	// 初始化leader选举
	if err := dm.initLeaderElection(); err != nil {
		return fmt.Errorf("初始化leader选举失败: %v", err)
	}

	// 启动基础服务
	if err := dm.startBaseServices(); err != nil {
		return fmt.Errorf("启动基础服务失败: %v", err)
	}

	// 优化的启动逻辑：检查etcd中是否已有master节点
	if dm.config.NodeType == MasterNode {
		// 配置为master节点，直接启动master服务和API Gateway
		logger.Info("Node configured as master, starting master services...")
		if err := dm.startMasterServices(); err != nil {
			return fmt.Errorf("启动master服务失败: %v", err)
		}
	} else {
		// 配置为slave或auto，检查etcd中是否已有master节点
		logger.Info("Node configured as slave/auto, checking for existing master...")
		if err := dm.startWithMasterCheck(); err != nil {
			return fmt.Errorf("启动服务失败: %v", err)
		}
	}

	dm.isRunning = true
	return nil
}

// Stop 停止分布式管理器
func (dm *DistributedManager) Stop() error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	if !dm.isRunning {
		return nil
	}

	logger.Info("Stopping distributed manager...")

	// 停止服务
	close(dm.stopCh)

	// 停止master服务
	if dm.masterService != nil {
		if err := dm.masterService.Stop(dm.ctx); err != nil {
			logger.Error("停止master服务失败: %v", err)
		}
	}

	// 停止slave服务
	if dm.slaveService != nil {
		if err := dm.slaveService.Stop(dm.ctx); err != nil {
			logger.Error("停止slave服务失败: %v", err)
		}
	}

	// 关闭etcd相关资源
	if dm.session != nil {
		dm.session.Close()
	}
	if dm.etcdClient != nil {
		dm.etcdClient.Close()
	}

	// 取消上下文
	dm.cancel()

	dm.isRunning = false
	logger.Info("Distributed manager stopped")
	return nil
}

// initEtcdClient 初始化etcd客户端
func (dm *DistributedManager) initEtcdClient() error {
	logger.Info("Initializing etcd client...")
	var endPoints []string
	for _, endPoint := range dm.config.Etcd.Endpoints {
		endPoints = append(endPoints, endPoint.Ip+":"+strconv.Itoa(endPoint.Port))
	}
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   endPoints,
		DialTimeout: time.Duration(dm.config.TimeOut) * time.Second,
		// 添加自动重连选项
		DialOptions: []grpc.DialOption{
			grpc.WithBackoffMaxDelay(5 * time.Second),
			grpc.WithKeepaliveParams(keepalive.ClientParameters{
				Time:                10 * time.Second,
				Timeout:             5 * time.Second,
				PermitWithoutStream: true,
			}),
		},
	})
	if err != nil {
		return fmt.Errorf("创建etcd客户端失败: %v", err)
	}

	dm.etcdClient = client
	logger.Info("Etcd client initialized successfully")
	return nil
}

// initLeaderElection 初始化leader选举
func (dm *DistributedManager) initLeaderElection() error {
	logger.Info("Initializing leader election...")

	// 创建session
	session, err := concurrency.NewSession(dm.etcdClient, concurrency.WithTTL(dm.config.Heartbeat))
	if err != nil {
		return fmt.Errorf("创建etcd session失败: %v", err)
	}
	dm.session = session
	dm.leaseID = session.Lease()

	// 创建election
	electionKey := fmt.Sprintf("/vector_sphere/election/%s", dm.config.ServiceName)
	dm.election = concurrency.NewElection(session, electionKey)

	logger.Info("Leader election initialized successfully")
	return nil
}

// startWithMasterCheck 检查master节点并决定启动逻辑
func (dm *DistributedManager) startWithMasterCheck() error {
	// 创建服务发现组件来检查master节点
	serviceDiscovery := NewServiceDiscovery(dm.etcdClient, dm.config.ServiceName)
	
	// 检查etcd中是否已有master节点
	masterInfo, err := serviceDiscovery.DiscoverMaster(dm.ctx)
	if err != nil {
		// 没有找到master节点，尝试竞选master
		logger.Info("No existing master found, attempting to become master...")
		return dm.attemptMasterElection()
	}
	
	// 已有master节点，启动为slave
	logger.Info("Found existing master: %s:%d, starting as slave...", masterInfo.Address, masterInfo.Port)
	return dm.startAsSlaveNode()
}

// attemptMasterElection 尝试竞选master节点
func (dm *DistributedManager) attemptMasterElection() error {
	// 尝试立即竞选master
	ctx, cancel := context.WithTimeout(dm.ctx, 5*time.Second)
	defer cancel()
	
	if err := dm.election.Campaign(ctx, dm.localhost); err != nil {
		// 竞选失败，说明其他节点已经成为master，启动为slave
		logger.Info("Master election failed, starting as slave: %v", err)
		return dm.startAsSlaveNode()
	}
	
	// 竞选成功，启动为master
	logger.Info("Successfully elected as master: %s", dm.localhost)
	return dm.startAsMasterNode()
}

// startAsMasterNode 启动为master节点
func (dm *DistributedManager) startAsMasterNode() error {
	dm.setMaster(true)
	logger.Info("Starting as master node...")
	
	// 启动master服务
	if err := dm.startMasterServices(); err != nil {
		return fmt.Errorf("启动master服务失败: %v", err)
	}
	
	// 注册master节点信息到etcd
	if err := dm.registerMasterService(); err != nil {
		logger.Error("注册master服务失败: %v", err)
	}
	
	// 启动leader选举监听（处理失去leadership的情况）
	go dm.monitorLeadership()
	
	return nil
}

// startAsSlaveNode 启动为slave节点
func (dm *DistributedManager) startAsSlaveNode() error {
	dm.setMaster(false)
	logger.Info("Starting as slave node...")
	
	// 启动slave服务
	if err := dm.startSlaveServices(); err != nil {
		return fmt.Errorf("启动slave服务失败: %v", err)
	}
	
	// 注册slave节点信息到etcd
	if err := dm.registerSlaveService(); err != nil {
		logger.Error("注册slave服务失败: %v", err)
	}
	
	// 启动leader选举监听（等待成为master的机会）
	go dm.runLeaderElection()
	
	return nil
}

// monitorLeadership 监听leadership状态
func (dm *DistributedManager) monitorLeadership() {
	logger.Info("Starting leadership monitoring...")
	
	for {
		select {
		case <-dm.session.Done():
			logger.Info("Lost leadership due to session expiry")
			dm.onLoseLeader()
			// 重新启动选举
			go dm.runLeaderElection()
			return
		case <-dm.stopCh:
			logger.Info("Leadership monitoring stopped")
			return
		}
	}
}

// registerSlaveService 注册slave服务到etcd
func (dm *DistributedManager) registerSlaveService() error {
	// 获取当前节点的IP地址
	localIP, err := common.GetLocalHost()
	if err != nil {
		return fmt.Errorf("获取本地IP失败: %v", err)
	}
	
	// 创建slave服务信息
	slaveInfo := &ServiceInfo{
		ServiceName: dm.config.ServiceName,
		NodeID:      dm.localhost, // 使用localhost作为节点ID
		Address:     localIP,
		Port:        dm.config.DefaultPort,
		NodeType:    "slave",
		Status:      "active",
		Metadata: map[string]string{
			"grpc_address": dm.localhost,
			"version":      "1.0.0",
		},
		LastSeen: time.Now(),
		Version:  "1.0.0",
	}
	
	// 创建服务发现组件并注册slave服务
	serviceDiscovery := NewServiceDiscovery(dm.etcdClient, dm.config.ServiceName)
	if err := serviceDiscovery.RegisterService(dm.ctx, slaveInfo, int64(dm.config.Heartbeat*3)); err != nil {
		return fmt.Errorf("注册slave服务失败: %v", err)
	}
	
	logger.Info("Slave service registered successfully: %s", localIP)
	return nil
}

// unregisterSlaveService 注销slave服务信息
func (dm *DistributedManager) unregisterSlaveService() error {
	// 创建服务发现组件并注销slave服务
	serviceDiscovery := NewServiceDiscovery(dm.etcdClient, dm.config.ServiceName)
	if err := serviceDiscovery.UnregisterService(dm.ctx, "slave", dm.localhost); err != nil {
		return fmt.Errorf("注销slave服务失败: %v", err)
	}
	
	logger.Info("Slave service unregistered successfully")
	return nil
}

// runLeaderElection 运行leader选举
func (dm *DistributedManager) runLeaderElection() {
	logger.Info("Starting leader election...")

	for {
		select {
		case <-dm.stopCh:
			logger.Info("Leader election stopped")
			return
		default:
			// 尝试成为leader
			if err := dm.election.Campaign(dm.ctx, dm.localhost); err != nil {
				logger.Error("Leader election campaign failed: %v", err)
				time.Sleep(time.Second)
				continue
			}

			logger.Info("Became leader: %s", dm.localhost)
			dm.onBecomeLeader()

			// 启动leadership监听
			dm.monitorLeadership()
			return
		}
	}
}

// onBecomeLeader 成为leader时的处理
func (dm *DistributedManager) onBecomeLeader() {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	dm.setMaster(true)
	logger.Info("Node became master through election")

	// 停止slave服务（如果正在运行）
	if dm.slaveService != nil {
		logger.Info("Stopping slave service...")
		if err := dm.slaveService.Stop(dm.ctx); err != nil {
			logger.Error("停止slave服务失败: %v", err)
		}
		dm.slaveService = nil
		// 等待一段时间确保slave服务完全停止
		time.Sleep(2 * time.Second)
		logger.Info("Slave service stopped successfully")
	}

	// 注销slave服务信息
	if err := dm.unregisterSlaveService(); err != nil {
		logger.Error("注销slave服务失败: %v", err)
	}

	// 启动master服务
	if dm.masterService == nil {
		logger.Info("Starting master service after election...")
		if err := dm.startMasterService(dm.communicationSvc); err != nil {
			logger.Error("启动master服务失败: %v", err)
			return
		}
	}

	// 启动API Gateway（监听HTTP请求）
	if dm.apiGateway == nil {
		logger.Info("Starting API Gateway after becoming master...")
		if err := dm.startAPIGateway(); err != nil {
			logger.Error("启动API Gateway失败: %v", err)
			return
		}
	}

	// 注册master节点信息到etcd，包括API Gateway地址
	if err := dm.registerMasterService(); err != nil {
		logger.Error("注册master服务到etcd失败: %v", err)
	}
}

func (dm *DistributedManager) setMaster(isMaster bool) {
	if isMaster {
		dm.config.NodeType = MasterNode
		return
	}

	dm.config.NodeType = SlaveNode
}

// onLoseLeader 失去leader时的处理
func (dm *DistributedManager) onLoseLeader() {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	dm.setMaster(false)
	logger.Info("Node lost master status, switching to slave mode")

	// 从etcd注销master服务信息
	if err := dm.unregisterMasterService(); err != nil {
		logger.Error("注销master服务失败: %v", err)
	}

	// 停止API Gateway
	if dm.apiGateway != nil {
		logger.Info("Stopping API Gateway...")
		if err := dm.apiGateway.Stop(dm.ctx); err != nil {
			logger.Error("停止API Gateway失败: %v", err)
		}
		dm.apiGateway = nil
	}

	// 停止master服务
	if dm.masterService != nil {
		logger.Info("Stopping master service...")
		if err := dm.masterService.Stop(dm.ctx); err != nil {
			logger.Error("停止master服务失败: %v", err)
		}
		dm.masterService = nil
	}

	// 启动slave服务（监听gRPC请求）
	if dm.slaveService == nil {
		logger.Info("Starting slave service after losing master status...")
		if err := dm.startSlaveService(dm.communicationSvc); err != nil {
			logger.Error("启动slave服务失败: %v", err)
		}
	}
}

// startBaseServices 启动基础服务
func (dm *DistributedManager) startBaseServices() error {
	// 创建服务发现
	dm.serviceDiscovery = NewServiceDiscovery(dm.etcdClient, dm.config.ServiceName)

	// 创建通信服务
	dm.communicationSvc = NewCommunicationService(dm.etcdClient, dm.config.ServiceName)

	// 创建分布式文件服务
	distributedFileService := NewDistributedFileService(dm, dm.serviceDiscovery, dm.communicationSvc, dm.config.DataDir)
	if err := distributedFileService.Start(); err != nil {
		return fmt.Errorf("启动分布式文件服务失败: %v", err)
	}
	// 注册分布式文件服务
	dm.RegisterService("distributed_file_service", distributedFileService)

	return nil
}

// startMasterServices 启动master服务和API Gateway
func (dm *DistributedManager) startMasterServices() error {
	// 启动master服务
	if err := dm.startMasterService(dm.communicationSvc); err != nil {
		return fmt.Errorf("启动master服务失败: %v", err)
	}

	// 启动API Gateway
	if err := dm.startAPIGateway(); err != nil {
		return fmt.Errorf("启动API Gateway失败: %v", err)
	}

	return nil
}

// startSlaveServices 启动slave服务
func (dm *DistributedManager) startSlaveServices() error {
	return dm.startSlaveService(dm.communicationSvc)
}

// startAPIGateway 启动API Gateway
func (dm *DistributedManager) startAPIGateway() error {
	logger.Info("Starting API Gateway...")

	// 获取当前节点的IP地址
	localIP, err := common.GetLocalHost()
	if err != nil {
		return fmt.Errorf("获取本地IP失败: %v", err)
	}

	// 使用当前节点的IP和HttpPort构建API Gateway地址
	apiGatewayPort := dm.config.HttpPort
	apiGatewayAddress := localIP + ":" + strconv.Itoa(apiGatewayPort)

	logger.Info("API Gateway will listen on: %s", apiGatewayAddress)

	// 创建API Gateway
	dm.apiGateway = NewAPIGateway(
		dm,
		dm.communicationSvc,
		dm.serviceDiscovery,
		apiGatewayPort,
	)

	// 启动API Gateway
	if err := dm.apiGateway.Start(dm.ctx); err != nil {
		return fmt.Errorf("启动API Gateway失败: %v", err)
	}

	logger.Info("API Gateway started successfully on %s", apiGatewayAddress)
	return nil
}

// startMasterService 启动master服务
func (dm *DistributedManager) startMasterService(communicationSvc *CommunicationService) error {
	logger.Info("Starting master service...")

	// 转换endpoints
	var endpoints []entity.EndPoint
	for _, ep := range dm.config.Etcd.Endpoints {
		endpoints = append(endpoints, ep)
	}

	masterService, err := server.NewMasterService(
		dm.ctx,
		endpoints,
		dm.config.ServiceName,
		dm.localhost,
		dm.config.HttpPort,
		time.Duration(dm.config.TaskTimeout)*time.Second,
		time.Duration(dm.config.HealthCheckInterval)*time.Second,
	)
	if err != nil {
		return fmt.Errorf("创建master服务失败: %v", err)
	}

	// 设置通信服务
	masterService.SetCommunicationService(communicationSvc)

	if err := masterService.Start(dm.ctx); err != nil {
		return fmt.Errorf("启动master服务失败: %v", err)
	}

	dm.masterService = masterService
	logger.Info("Master service started successfully")
	return nil
}

// startSlaveService 启动slave服务
func (dm *DistributedManager) startSlaveService(communicationSvc *CommunicationService) error {
	logger.Info("Starting slave service...")

	// 转换endpoints
	var endpoints []entity.EndPoint
	for _, ep := range dm.config.Etcd.Endpoints {
		endpoints = append(endpoints, ep)
	}

	slaveService, err := server.NewSlaveService(
		dm.ctx,
		endpoints,
		dm.config.ServiceName,
		dm.config.DefaultPort,
	)
	if err != nil {
		return fmt.Errorf("创建slave服务失败: %v", err)
	}

	// 设置通信服务
	slaveService.SetCommunicationService(communicationSvc)

	if err := slaveService.Start(dm.ctx); err != nil {
		return fmt.Errorf("启动slave服务失败: %v", err)
	}

	dm.slaveService = slaveService
	logger.Info("Slave service started successfully")
	return nil
}

// IsMaster 检查是否为master节点
func (dm *DistributedManager) IsMaster() bool {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()

	return dm.config.NodeType == MasterNode
}

// GetMasterService 获取master服务实例
func (dm *DistributedManager) GetMasterService() *server.MasterService {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()
	return dm.masterService
}

// GetSlaveService 获取slave服务实例
func (dm *DistributedManager) GetSlaveService() *server.SlaveService {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()
	return dm.slaveService
}

// GetEtcdClient 获取etcd客户端
func (dm *DistributedManager) GetEtcdClient() *clientv3.Client {
	return dm.etcdClient
}

// GetService 获取服务实例
func (dm *DistributedManager) GetService(serviceName string) interface{} {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()
	return dm.services[serviceName]
}

// RegisterService 注册服务
func (dm *DistributedManager) RegisterService(name string, service interface{}) {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()
	dm.services[name] = service
}

// GetServiceDiscovery 获取服务发现实例
func (dm *DistributedManager) GetServiceDiscovery() *ServiceDiscovery {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()
	return dm.serviceDiscovery
}

// GetCommunicationService 获取通信服务实例
func (dm *DistributedManager) GetCommunicationService() *CommunicationService {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()
	return dm.communicationSvc
}

// GetConfig 获取配置
func (dm *DistributedManager) GetConfig() *DistributedConfig {
	return dm.config
}

// registerMasterService 注册master服务信息到etcd
func (dm *DistributedManager) registerMasterService() error {
	// 获取当前节点的IP地址
	localIP, err := common.GetLocalHost()
	if err != nil {
		return fmt.Errorf("获取本地IP失败: %v", err)
	}

	// 构建API Gateway地址
	apiGatewayAddress := localIP + ":" + strconv.Itoa(dm.config.HttpPort)

	// 创建master服务信息
	masterInfo := &ServiceInfo{
		ServiceName: dm.config.ServiceName,
		NodeID:      dm.localhost, // 使用localhost作为节点ID
		Address:     localIP,
		Port:        dm.config.HttpPort,
		NodeType:    "master",
		Status:      "active",
		Metadata: map[string]string{
			"api_gateway_address": apiGatewayAddress,
			"grpc_address":        dm.localhost,
			"version":             "1.0.0",
		},
		LastSeen: time.Now(),
		Version:  "1.0.0",
	}

	// 注册master服务到etcd
	if err := dm.serviceDiscovery.RegisterService(dm.ctx, masterInfo, int64(dm.config.Heartbeat)); err != nil {
		return fmt.Errorf("注册master服务失败: %v", err)
	}

	logger.Info("Master service registered successfully with API Gateway address: %s", apiGatewayAddress)
	return nil
}

// unregisterMasterService 从etcd注销master服务信息
func (dm *DistributedManager) unregisterMasterService() error {
	if err := dm.serviceDiscovery.UnregisterService(dm.ctx, "master", dm.localhost); err != nil {
		return fmt.Errorf("注销master服务失败: %v", err)
	}

	logger.Info("Master service unregistered successfully")
	return nil
}
