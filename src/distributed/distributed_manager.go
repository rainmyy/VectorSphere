package distributed

import (
	"VectorSphere/src/enhanced"
	"VectorSphere/src/library/common"
	"VectorSphere/src/library/log"
	"VectorSphere/src/library/pool"
	"VectorSphere/src/scheduler"
	"VectorSphere/src/server"
	"context"
	"fmt"
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
	ServiceName           string                     `yaml:"serviceName"`
	NodeType              NodeType                   `yaml:"nodeType"`
	TimeOut               int                        `yaml:"timeOut"`
	DefaultPort           int                        `yaml:"defaultPort"`
	Heartbeat             int                        `yaml:"heartbeat"`
	Etcd                  EtcdConfig                 `yaml:"etcd"`
	SchedulerWorkerCount  int                        `yaml:"schedulerWorkerCount"`
	HttpPort              int                        `yaml:"httpPort"`
	TaskTimeout           int                        `yaml:"taskTimeout"`
	HealthCheckInterval   int                        `yaml:"healthCheckInterval"`
	Endpoints             map[string]server.EndPoint `yaml:"endpoints"`
	DataDir               string                     `yaml:"dataDir"`
	LoadBalancerConfig    *enhanced.EnhancedLoadBalancer
	EnhancedConfigManager *enhanced.EnhancedConfigManager
}

type EtcdConfig struct {
	Endpoints []string `yaml:"endpoints"`
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

	// 节点服务
	masterService *server.MasterService
	slaveService  *server.SlaveService

	// 状态管理
	isMaster  bool
	isRunning bool
	stopCh    chan struct{}
}

// NewDistributedManager 创建分布式管理器
func NewDistributedManager(config *DistributedConfig) (*DistributedManager, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 获取本地IP
	localIP, err := common.GetLocalHost()
	if err != nil {
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
		stopCh:        make(chan struct{}),
	}

	return dm, nil
}

// Start 启动分布式管理器
func (dm *DistributedManager) Start() error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	if dm.isRunning {
		return fmt.Errorf("distributed manager is already running")
	}

	log.Info("Starting distributed manager...")

	// 1. 初始化etcd连接
	if err := dm.initEtcdClient(); err != nil {
		return fmt.Errorf("初始化etcd客户端失败: %v", err)
	}

	// 2. 创建session和election
	if err := dm.initLeaderElection(); err != nil {
		return fmt.Errorf("初始化leader选举失败: %v", err)
	}

	// 3. 启动leader选举
	go dm.runLeaderElection()

	// 4. 根据配置启动对应的服务
	if err := dm.startServices(); err != nil {
		return fmt.Errorf("启动服务失败: %v", err)
	}

	dm.isRunning = true
	log.Info("Distributed manager started successfully")
	return nil
}

// Stop 停止分布式管理器
func (dm *DistributedManager) Stop() error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	if !dm.isRunning {
		return nil
	}

	log.Info("Stopping distributed manager...")

	// 停止服务
	close(dm.stopCh)

	// 停止master服务
	if dm.masterService != nil {
		if err := dm.masterService.Stop(dm.ctx); err != nil {
			log.Error("停止master服务失败: %v", err)
		}
	}

	// 停止slave服务
	if dm.slaveService != nil {
		if err := dm.slaveService.Stop(dm.ctx); err != nil {
			log.Error("停止slave服务失败: %v", err)
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
	log.Info("Distributed manager stopped")
	return nil
}

// initEtcdClient 初始化etcd客户端
func (dm *DistributedManager) initEtcdClient() error {
	log.Info("Initializing etcd client...")

	client, err := clientv3.New(clientv3.Config{
		Endpoints:   dm.config.Etcd.Endpoints,
		DialTimeout: time.Duration(dm.config.TimeOut) * time.Second,
	})
	if err != nil {
		return fmt.Errorf("创建etcd客户端失败: %v", err)
	}

	dm.etcdClient = client
	log.Info("Etcd client initialized successfully")
	return nil
}

// initLeaderElection 初始化leader选举
func (dm *DistributedManager) initLeaderElection() error {
	log.Info("Initializing leader election...")

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

	log.Info("Leader election initialized successfully")
	return nil
}

// runLeaderElection 运行leader选举
func (dm *DistributedManager) runLeaderElection() {
	log.Info("Starting leader election...")

	for {
		select {
		case <-dm.stopCh:
			log.Info("Leader election stopped")
			return
		default:
			// 尝试成为leader
			if err := dm.election.Campaign(dm.ctx, dm.localhost); err != nil {
				log.Error("Leader election campaign failed: %v", err)
				time.Sleep(time.Second)
				continue
			}

			log.Info("Became leader: %s", dm.localhost)
			dm.onBecomeLeader()

			// 等待失去leadership
			select {
			case <-dm.session.Done():
				log.Info("Lost leadership due to session expiry")
				dm.onLoseLeader()
			case <-dm.stopCh:
				log.Info("Leader election stopped")
				return
			}
		}
	}
}

// onBecomeLeader 成为leader时的处理
func (dm *DistributedManager) onBecomeLeader() {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	dm.isMaster = true
	log.Info("Node became master")

	// 启动master服务
	if dm.masterService == nil {
		if err := dm.startMasterService(); err != nil {
			log.Error("启动master服务失败: %v", err)
		}
	}

	// 停止slave服务（如果正在运行）
	if dm.slaveService != nil {
		if err := dm.slaveService.Stop(dm.ctx); err != nil {
			log.Error("停止slave服务失败: %v", err)
		}
		dm.slaveService = nil
	}
}

// onLoseLeader 失去leader时的处理
func (dm *DistributedManager) onLoseLeader() {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	dm.isMaster = false
	log.Info("Node lost master status")

	// 停止master服务
	if dm.masterService != nil {
		if err := dm.masterService.Stop(dm.ctx); err != nil {
			log.Error("停止master服务失败: %v", err)
		}
		dm.masterService = nil
	}

	// 启动slave服务
	if dm.slaveService == nil {
		if err := dm.startSlaveService(); err != nil {
			log.Error("启动slave服务失败: %v", err)
		}
	}
}

// startServices 启动服务
func (dm *DistributedManager) startServices() error {
	// 根据配置的节点类型启动对应服务
	switch dm.config.NodeType {
	case MasterNode:
		return dm.startMasterService()
	case SlaveNode:
		return dm.startSlaveService()
	default:
		// 默认启动slave服务，等待选举结果
		return dm.startSlaveService()
	}
}

// startMasterService 启动master服务
func (dm *DistributedManager) startMasterService() error {
	log.Info("Starting master service...")

	// 转换endpoints
	var endpoints []server.EndPoint
	for _, ep := range dm.config.Endpoints {
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

	if err := masterService.Start(dm.ctx); err != nil {
		return fmt.Errorf("启动master服务失败: %v", err)
	}

	dm.masterService = masterService
	log.Info("Master service started successfully")
	return nil
}

// startSlaveService 启动slave服务
func (dm *DistributedManager) startSlaveService() error {
	log.Info("Starting slave service...")

	// 转换endpoints
	var endpoints []server.EndPoint
	for _, ep := range dm.config.Endpoints {
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

	if err := slaveService.Start(dm.ctx); err != nil {
		return fmt.Errorf("启动slave服务失败: %v", err)
	}

	dm.slaveService = slaveService
	log.Info("Slave service started successfully")
	return nil
}

// IsMaster 检查是否为master节点
func (dm *DistributedManager) IsMaster() bool {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()
	return dm.isMaster
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
