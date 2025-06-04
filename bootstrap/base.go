package bootstrap

import (
	"context"
	"encoding/json"
	"fmt"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"net/http"
	"seetaSearch/library/common"
	"seetaSearch/library/log"
	"seetaSearch/messages"
	"seetaSearch/scheduler"
	"seetaSearch/server"
	"sync"
	"time"

	"seetaSearch/library/conf"
	PoolLib "seetaSearch/library/pool"
)

/**
*app执行入口
 */
type AppServer struct {
	mutex        sync.WaitGroup
	Ctx          context.Context
	Cancel       context.CancelFunc
	funcRegister map[string]func()
	server       *server.IndexServer
	sentinel     *server.Sentinel
	etcdCli      *clientv3.Client
	grpcConn     *grpc.ClientConn
	masterAddr   string

	taskPool *PoolLib.Pool  // 使用 library/pool 中的 Pool
	config   *ServiceConfig // 保存解析后的配置

	// 服务实例，根据需要添加
	masterService *server.MasterService
	slaveServices []*server.SlaveService
}

const (
	RPCSERVICE = iota
	TCPSERVICE
	READSERVICE
	WRITESERVICE
)

// EtcdReadyTask 确保 etcd 客户端连接成功的任务
type EtcdReadyTask struct {
	appServer *AppServer
	connected bool
	mutex     sync.Mutex
}

// EtcdConfig ReadServiceConf 读取服务配置 (修改以适应新的配置结构)
type EtcdConfig struct {
	Endpoints []string `yaml:"endpoints"`
}

// ServiceConfig 结构体映射整个 YAML 文件
type ServiceConfig struct {
	ServiceName          string                     `yaml:"serviceName"`
	TimeOut              int                        `yaml:"timeOut"`     // 通用超时
	DefaultPort          int                        `yaml:"defaultPort"` // 服务默认端口
	Heartbeat            int                        `yaml:"heartbeat"`   // 心跳间隔
	Etcd                 EtcdConfig                 `yaml:"etcd"`        // Etcd 配置
	Master               *MasterConfig              `yaml:"master,omitempty"`
	Slaves               []SlaveConfig              `yaml:"slaves,omitempty"`
	SchedulerWorkerCount int                        `yaml:"schedulerWorkerCount"` // TaskPoolManager 的工作线程数
	HttpPort             int                        `yaml:"httpPort"`             // Master HTTP 服务的端口
	TaskTimeout          int                        `yaml:"taskTimeout"`          // Master 任务超时（秒）
	HealthCheckInterval  int                        `yaml:"healthCheckInterval"`  // Master 健康检查间隔（秒）
	Endpoints            map[string]server.EndPoint `yaml:"endpoints"`
}

type MasterConfig struct {
	Enabled bool `yaml:"enabled"`
	// 可以添加 Master 特有的配置
}

type SlaveConfig struct {
	Enabled bool   `yaml:"enabled"`
	Port    int    `yaml:"port"` // Slave 服务的特定端口，如果为0则使用 DefaultPort
	DataDir string `yaml:"dataDir"`
	// 可以添加 Slave 特有的配置
}

func NewEtcdReadyTask(app *AppServer) *EtcdReadyTask {
	return &EtcdReadyTask{appServer: app}
}

func (app *AppServer) ReadServiceConf() (error, *ServiceConfig) {
	var cfg ServiceConfig
	//rootPath, err := util.GetProjectRoot()
	//if err != nil {
	//	return err, nil
	//}

	//err = conf.ReadYAML(path.Join(rootPath, "conf", "idc", "simple", "service.yaml"), &cfg)
	err := conf.ReadYAML("D:\\code\\seetaSearch\\conf\\idc\\simple\\service.yaml", &cfg)

	if err != nil {
		return err, nil
	}
	app.config = &cfg
	return nil, &cfg
}

func (t *EtcdReadyTask) ToPoolTask() *PoolLib.Queue {
	return PoolLib.QueryInit(t.GetName(), t.Execute, context.Background())
}

func (t *EtcdReadyTask) Execute(ctx context.Context) error {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	if t.connected {
		log.Info("Etcd already connected.")
		return nil
	}

	log.Info("Attempting to connect to Etcd...")
	cfg := t.appServer.config
	if cfg == nil || len(cfg.Etcd.Endpoints) == 0 {
		return fmt.Errorf("etcd endpoints not configured")
	}

	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   cfg.Etcd.Endpoints,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		log.Error("Failed to connect to Etcd: %v", err)
		return err
	}
	t.appServer.etcdCli = cli
	t.connected = true
	log.Info("Successfully connected to Etcd.")
	return nil
}

func (t *EtcdReadyTask) GetName() string {
	return "etcd_ready"
}

func (t *EtcdReadyTask) GetDependencies() []string {
	return nil
}
func (t *EtcdReadyTask) GetRetries() int {
	return 3 // 重试3次连接etcd
}

// MasterServiceDiscoveryReadyTask 标记 Master 服务已启动并可被发现的任务
type MasterServiceDiscoveryReadyTask struct {
	appServer *AppServer
}

func NewMasterServiceDiscoveryReadyTask(app *AppServer) *MasterServiceDiscoveryReadyTask {
	return &MasterServiceDiscoveryReadyTask{appServer: app}
}
func (t *MasterServiceDiscoveryReadyTask) ToPoolTask() *PoolLib.Queue {
	return PoolLib.QueryInit(t.GetName(), t.Execute, context.Background())
}

func (t *MasterServiceDiscoveryReadyTask) Execute(ctx context.Context) error {
	// 这个任务实际上不做任何操作，它的存在是为了作为依赖项
	// MasterService 启动后，其他服务可以依赖此任务来确保 Master 已就绪
	// 实际的发现逻辑在 SlaveService 的 watchMaster 中
	log.Info("MasterService is considered ready for discovery.")
	// 可以在这里添加一个检查，例如查询 etcd 中 master 的注册信息
	// 但为了简化，我们假设 MasterService.Start() 成功即代表可发现
	// 检查 MasterService 是否真的成为了 Master
	if t.appServer.masterService != nil {
		// 等待一段时间让选举完成
		timeout := time.After(15 * time.Second) // 等待最多15秒
		checkInterval := time.NewTicker(1 * time.Second)
		defer checkInterval.Stop()

		for {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-timeout:
				log.Warning("Timeout waiting for master election to complete for discovery readiness.")
				return fmt.Errorf("master election did not complete in time for discovery readiness")
			case <-checkInterval.C:
				if t.appServer.masterService.IsMaster() { // 假设 MasterService 有 IsMaster() 方法
					log.Info("MasterService confirmed as master and ready for discovery.")
					return nil
				}
			}
		}
	} else {
		return fmt.Errorf("master service instance is nil, cannot confirm discovery readiness")
	}
}

func (t *MasterServiceDiscoveryReadyTask) GetName() string {
	return "master_service_discovery_ready"
}

func (t *MasterServiceDiscoveryReadyTask) GetDependencies() []string {
	// 依赖 MasterService 任务本身成功启动
	return []string{"master_service"}
}
func (t *MasterServiceDiscoveryReadyTask) GetRetries() int {
	return 0
}

// NewAppServer 创建 AppServer 实例
func NewAppServer() *AppServer {
	ctx, cancel := context.WithCancel(context.Background())
	// 初始化 PoolLib.Pool，使用链式调用设置配置
	p := PoolLib.NewPool().Init(10, 10).WithPreAllocWorkers(false).WithBlock(false)
	return &AppServer{
		Ctx:      ctx,
		Cancel:   cancel,
		taskPool: p,
	}
}
func (app *AppServer) RegisterToEtcd(serviceName, addr string) error {
	key := fmt.Sprintf("/seetasearch/services/%s/%s", serviceName, addr)
	_, err := app.etcdCli.Put(context.Background(), key, addr)
	return err
}
func (app *AppServer) DiscoverMaster(serviceName string) (string, error) {
	resp, err := app.etcdCli.Get(context.Background(), fmt.Sprintf("/seetasearch/services/%s/", serviceName), clientv3.WithPrefix())
	if err != nil {
		return "", err
	}
	for _, kv := range resp.Kvs {
		return string(kv.Value), nil // 取第一个主服务地址
	}
	return "", fmt.Errorf("no master found")
}
func (app *AppServer) ConnectToMaster(masterAddr string) error {
	conn, err := grpc.Dial(masterAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	app.grpcConn = conn
	return nil
}

func (app *AppServer) RegisterService() {
	err, config := app.ReadServiceConf()
	if err != nil {
		log.Error("read service conf failed, err:%v", err)
		return
	}
	var masterEndpoint []server.EndPoint
	var sentinelEndpoint []server.EndPoint

	for name, endpoint := range config.Endpoints {
		port := endpoint.Port
		endpoint.Name = name
		if port == 0 {
			endpoint.Port = config.DefaultPort
		}
		if endpoint.IsMaster {
			masterEndpoint = append(masterEndpoint, endpoint)
		} else {
			sentinelEndpoint = append(sentinelEndpoint, endpoint)
		}

	}
	if len(masterEndpoint) > 0 {
		// master节点注册
		s := new(server.IndexServer)
		masterServiceName := config.ServiceName
		err := s.RegisterService(masterEndpoint, config.DefaultPort, masterServiceName)
		if err != nil {
			log.Error("Master注册失败:", err)
			return
		}
		log.Info("Master节点 %s 注册成功\n", masterServiceName)
		app.server = s
	}

	if len(sentinelEndpoint) > 0 {
		sentinel := server.NewSentinel(sentinelEndpoint, int64(config.Heartbeat), 100, config.ServiceName)
		err := sentinel.RegisterSentinel(int64(config.Heartbeat))
		if err != nil {
			log.Error("Sentinel注册失败:", err)
			return
		}

		app.sentinel = sentinel
	}

	//for name, ep := range config.Endpoints {
	//	port := ep.Port
	//	if port == 0 {
	//		port = config.DefaultPort
	//	}
	//	ep.Ip = ep.Ip + ":" + strconv.Itoa(port)
	//	println(ep.Ip)
	//	endpoints := []string{ep.Ip}
	//	if ep.IsMaster {
	//		// master节点注册
	//		s := new(server.IndexServer)
	//		masterServiceName := config.ServiceName
	//		err := s.RegisterService(endpoints, port, masterServiceName)
	//		if err != nil {
	//			log.Error("Master注册失败:", err)
	//			continue
	//		}
	//		log.Info("Master节点 %s 注册成功: %s:%d\n", name, ep.Ip, port)
	//	} else {
	//		// sentinel节点注册
	//		sentinel := server.NewSentinel(endpoints, int64(config.Heartbeat), 100, config.ServiceName)
	//		err := sentinel.RegisterSentinel(int64(config.Heartbeat))
	//		if err != nil {
	//			log.Error("Sentinel注册失败:", err)
	//			continue
	//		}
	//		log.Info("Sentinel节点 %s 注册成功: %s:%d\n", name, ep.Ip, port)
	//	}
	//}
}

func (app *AppServer) DiscoverService() {
	err, config := app.ReadServiceConf()
	if err != nil {
		log.Error("read service conf failed, err:%v", err)
		return
	}

	if config == nil || config.Endpoints == nil {
		log.Error("endpoints is nil")
		return
	}
	var sentinelEndpoint []server.EndPoint

	for name, endpoint := range config.Endpoints {
		port := endpoint.Port
		endpoint.Name = name
		if port == 0 {
			port = config.DefaultPort
		}
		if !endpoint.IsMaster {
			sentinelEndpoint = append(sentinelEndpoint, endpoint)
		}

	}

	if len(sentinelEndpoint) > 0 {
		if app.sentinel == nil {
			sentinel := server.NewSentinel(sentinelEndpoint, int64(config.Heartbeat), 100, config.ServiceName)
			app.sentinel = sentinel
		}

		endpoints := app.sentinel.Hub.GetServiceEndpoints(config.ServiceName)
		log.Info("Sentinel节点 %s 发现的master节点: %+v\n", config.ServiceName, endpoints)
	}
	//for name, ep := range config.Endpoints {
	//	if !ep.IsMaster {
	//		sentinel := server.NewSentinel([]string{ep.Ip}, int64(config.Heartbeat), 100, config.ServiceName)
	//		endpoints := sentinel.Hub.GetServiceEndpoints(config.ServiceName)
	//		log.Info("Sentinel节点 %s 发现的master节点: %+v\n", name, endpoints)
	//	}
	//}
}

// Setup /
func (app *AppServer) Setup() error {
	err, cfg := app.ReadServiceConf()
	if err != nil {
		return err
	}

	// 0. 创建 scheduler.TaskPoolManager 实例 (如果需要被服务共享)
	// 注意：这里的 scheduler.TaskPoolManager 是您项目中的任务调度器，
	// 而 app.taskPool 是 PoolLib.Pool，用于执行这些服务的启动任务。
	// 如果 MasterService 和 SlaveService 内部的 taskExecutor 需要共享一个实例，在这里创建。
	// 否则，它们可以在各自的 New 函数中创建自己的 manager。
	sharedTaskManager := scheduler.NewTaskPoolManager() // 假设默认 worker 数量
	if cfg.SchedulerWorkerCount > 0 {
		sharedTaskManager = scheduler.NewTaskPoolManager()
	}

	// 1. 添加 Etcd 连接任务
	etcdTask := NewEtcdReadyTask(app).ToPoolTask()
	if err := app.taskPool.Submit(etcdTask); err != nil {
		return fmt.Errorf("failed to submit etcd_ready task: %w", err)
	}
	var endpoints []server.EndPoint
	for _, endpoint := range cfg.Endpoints {
		endpoints = append(endpoints, endpoint)
	}
	// 2. 初始化并添加 Master 服务任务 (如果配置了 Master)
	if cfg.Master != nil && cfg.Master.Enabled {
		masterHost, _ := common.GetLocalHost() // 假设 Master 运行在当前节点
		// 如果 Master 有特定端口配置，则使用，否则使用 DefaultPort
		masterBindAddress := fmt.Sprintf("%s:%d", masterHost, cfg.DefaultPort) // Master 通常监听一个已知端口

		mService, err := server.NewMasterService(
			app.Ctx, // 传递应用上下文
			endpoints,
			cfg.ServiceName+"_master",
			masterBindAddress,
			cfg.HttpPort,                                       // 从配置中读取 HTTP 端口
			time.Duration(cfg.TaskTimeout)*time.Second,         // 从配置中读取任务超时
			time.Duration(cfg.HealthCheckInterval)*time.Second, // 从配置中读取健康检查间隔
		)
		if err != nil {
			return fmt.Errorf("failed to create master service: %w", err)
		}
		mService.SetTaskPoolManager(sharedTaskManager) // 设置共享的 manager
		app.masterService = mService                   // 保存实例
		queue := mService.ToPoolTask()
		if err := app.taskPool.Submit(queue); err != nil {
			return fmt.Errorf("failed to submit master_service task: %w", err)
		}

		// 添加 Master 服务发现就绪任务
		masterDiscoveryTask := NewMasterServiceDiscoveryReadyTask(app).ToPoolTask()
		if err := app.taskPool.Submit(masterDiscoveryTask); err != nil {
			return fmt.Errorf("failed to submit master_service_discovery_ready task: %w", err)
		}
	} else {
		log.Info("Master service is not enabled in the configuration.")
	}

	// 3. 初始化并添加 Slave 服务任务 (如果配置了 Slaves)
	app.slaveServices = []*server.SlaveService{}
	for i, slaveCfg := range cfg.Slaves {
		if slaveCfg.Enabled {
			slavePort := slaveCfg.Port
			if slavePort == 0 {
				slavePort = cfg.DefaultPort // 如果 slave 未指定端口，则使用默认端口或进行递增分配
				// 注意：如果多个 slave 在同一主机上使用 DefaultPort，会导致端口冲突
				// 更好的做法是为每个 slave 分配唯一端口或从配置中读取
				slavePort += i // 简单递增避免冲突，生产环境应有更好策略
			}

			sService, err := server.NewSlaveService(
				app.Ctx, // 传递应用上下文
				endpoints,
				cfg.ServiceName+"_slave",
				slavePort,
			)
			if err != nil {
				log.Error("Failed to create slave service for port %d: %v", slavePort, err)
				continue // 跳过这个 slave，继续其他的
			}
			sService.SetTaskPoolManager(sharedTaskManager) // 设置共享的 manager
			// 初始化 SlaveService 的 Index 等组件
			// 参数应从 slaveCfg 或全局配置中获取
			if err := sService.Init(100000, 0, slaveCfg.DataDir, nil, nil, nil); err != nil {
				log.Error("Failed to initialize slave service components for port %d: %v", slavePort, err)
				continue
			}

			app.slaveServices = append(app.slaveServices, sService)
			if err := app.taskPool.Submit(sService.ToPoolTask()); err != nil {
				log.Error("Failed to submit slave_service task for port %d: %v", slavePort, err)
				// 根据策略决定是否返回错误并停止启动
			}
		} else {
			log.Info("Slave service at index %d (port %d) is not enabled.", i, slaveCfg.Port)
		}
	}

	return nil
}

func (app *AppServer) Register() {
	//注册注册方法
	app.funcRegister["register_etcd"] = app.RegisterService
	//注册发现方法
	app.funcRegister["discover_etcd"] = app.DiscoverService
	app.funcRegister["listen_api"] = app.ListenAPI
}

func (app *AppServer) ListenAPI() {
	err, config := app.ReadServiceConf()
	if err != nil {
		log.Error("read service conf failed, err:%v", err)
		return
	}

	var serviceName string
	var serviceAddr string
	for name, ep := range config.Endpoints {
		if ep.IsMaster {
			port := ep.Port
			if port == 0 {
				port = config.DefaultPort
			}
			address := fmt.Sprintf(":%d", port)
			serviceName = name
			serviceAddr = address
		}
	}

	log.Info("Master节点 %s 开始监听 API 请求: %s\n", serviceName, serviceAddr)
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(rw http.ResponseWriter, req *http.Request) {
		rw.Write([]byte("hello world"))
	})

	mux.HandleFunc("/addDoc", func(w http.ResponseWriter, r *http.Request) {
		doc := &messages.Document{}
		if err := json.NewDecoder(r.Body).Decode(doc); err != nil {
			http.Error(w, "Invalid request payload", http.StatusBadRequest)
			return
		}
		s := app.sentinel
		affected, err := s.AddDoc(doc)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		log.Info("添加文档成功，影响文档数量: %d", affected)
		w.WriteHeader(http.StatusOK)
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte("ok"))
	})

	mux.HandleFunc("/count", func(w http.ResponseWriter, r *http.Request) {
		s := app.sentinel
		total := s.Count()
		log.Info("更新文档成功,total:%d", total)
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/deleteDoc", func(w http.ResponseWriter, r *http.Request) {
		docID := r.URL.Query().Get("id")
		if docID == "" {
			http.Error(w, "Missing document ID", http.StatusBadRequest)
			return
		}
		s := app.sentinel
		doc := &server.DocId{Id: docID}
		count := s.DelDoc(doc)
		if count == 0 {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		log.Info("删除文档成功，文档ID: %s", docID)
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		query := &messages.TermQuery{}
		if err := json.NewDecoder(r.Body).Decode(query); err != nil {
			http.Error(w, "Invalid request payload", http.StatusBadRequest)
			return
		}
		onFlag := uint64(0)
		offFlag := uint64(0)
		var orFlags []uint64
		s := app.sentinel
		err, result := s.Search(query, onFlag, offFlag, orFlags)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := json.NewEncoder(w).Encode(result); err != nil {
			log.Error("encode json failed, err:%v\n", err)
		}
	})

	if err := http.ListenAndServe(serviceAddr, mux); err != nil {
		log.Info("Master节点 %s 监听 API 请求失败: %v\n", serviceName, err)
	}
}

// Start 启动应用，开始执行任务池中的任务
func (app *AppServer) Start() {
	log.Info("Starting application server and task pool...")
	app.taskPool.Run()

	// 等待所有任务完成或应用关闭信号
	// PoolLib.Pool 的 Run() 是阻塞的，直到所有任务完成或池关闭
	// 如果 Run() 是非阻塞的，你可能需要一个等待机制
	log.Info("All tasks in the pool have been processed or pool is shutting down.")
}

// Stop 停止应用，释放资源
func (app *AppServer) Stop() {
	log.Info("Stopping application server...")
	app.Cancel() // 发送关闭信号给所有使用 app.Ctx 的组件

	// 关闭任务池
	if app.taskPool != nil {
		app.taskPool.Release() // PoolLib.Pool 的关闭方法
		log.Info("Task pool released.")
	}

	// 显式停止 MasterService (如果存在且 RunnableService.Stop 未被任务池自动调用)
	if app.masterService != nil {
		if err := app.masterService.Stop(context.Background()); err != nil {
			log.Warning("Error stopping master service: %v", err)
		}
	}

	// 显式停止 SlaveServices
	for _, slave := range app.slaveServices {
		if err := slave.Stop(context.Background()); err != nil {
			log.Warning("Error stopping slave service %s: %v", slave.GetName(), err)
		}
	}

	// 关闭 etcd 客户端
	if app.etcdCli != nil {
		if err := app.etcdCli.Close(); err != nil {
			log.Warning("Error closing etcd client: %v", err)
		}
		log.Info("Etcd client closed.")
	}

	app.mutex.Wait() // 等待所有可能的后台 goroutine 完成
	log.Info("Application server stopped.")
}
func GenInstance() *AppServer {
	app := &AppServer{funcRegister: make(map[string]func())}
	app.Register()
	return app
}

//func main() {
//	role := flag.String("role", "master", "Role: master or slave")
//	port := flag.Int("port", 8080, "HTTP port")
//	lb := flag.String("lb", "roundrobin", "Load balancer: roundrobin/weighted/leastconn")
//	token := flag.String("token", "your_token", "API token")
//	gray := flag.String("gray", "", "Gray tag for routing")
//	flag.Parse()
//	// ...根据参数初始化服务
//	// 例如：
//	if *role == "master" {
//		os.Setenv("MASTER_WEBHOOK", "http://your-webhook-url")
//		// ... 启动 MasterService，传递端口、token、lb 策略等
//	} else {
//		os.Setenv("SLAVE_ENV", *gray)
//		os.Setenv("SLAVE_MASTER_WEBHOOK", "http://your-webhook-url")
//		// ... 启动 SlaveService，传递端口、token、lb 策略等
//	}
//}
