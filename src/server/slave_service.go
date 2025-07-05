package server

import (
	"VectorSphere/src/backup"
	"VectorSphere/src/index"
	"VectorSphere/src/library/entity"
	PoolLib "VectorSphere/src/library/pool"
	"VectorSphere/src/library/tree"
	"VectorSphere/src/proto/messages"
	"VectorSphere/src/proto/serverProto"
	scheduler2 "VectorSphere/src/scheduler"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
	"net"
	"time"

	"go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"VectorSphere/src/library/common"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/search"
	"net/http"
	"os"
	"strconv"
	"sync"
)

var _ serverProto.IndexServiceServer = (*SlaveService)(nil)

// SlaveService 从服务结构体
type SlaveService struct {
	// etcd 相关
	client  *clientv3.Client
	leaseID clientv3.LeaseID

	// 服务相关
	Index       *index.Index
	serviceName string
	localhost   string
	stopCh      chan struct{} // 修改 stop 为 stopCh

	// 主节点相关
	masterEndpoint string
	masterConn     *grpc.ClientConn
	masterMutex    sync.RWMutex

	// 任务执行相关
	taskExecutor *scheduler2.TaskPoolManager

	// 多表搜索服务
	multiTableService *search.MultiTableSearchService

	// 通信服务
	communicationService interface{}

	// gRPC 服务器
	grpcServer *grpc.Server

	appCtx        context.Context   // 用于传递应用的全局上下文
	etcdEndpoints []entity.EndPoint // etcd 端点
	port          int               // 服务端口
}

func (s *SlaveService) CreateTable(ctx context.Context, request *serverProto.CreateTableRequest) (*serverProto.ResCount, error) {
	logger.Info("Creating table: %s", request.TableName)

	// 这里应该实现创建表的逻辑
	// 例如：在多表搜索服务中创建新表
	if s.multiTableService != nil {
		// 实现表创建逻辑
		logger.Info("Table %s created successfully", request.TableName)
	} else {
		logger.Warning("MultiTableService not initialized")
	}

	return &serverProto.ResCount{Count: 1}, nil
}

func (s *SlaveService) DeleteTable(ctx context.Context, request *serverProto.TableRequest) (*serverProto.ResCount, error) {
	logger.Info("Deleting table: %s", request.TableName)

	// 这里应该实现删除表的逻辑
	if s.multiTableService != nil {
		// 实现表删除逻辑
		logger.Info("Table %s deleted successfully", request.TableName)
	} else {
		logger.Warning("MultiTableService not initialized")
	}

	return &serverProto.ResCount{Count: 1}, nil
}

func (s *SlaveService) AddDocumentToTable(ctx context.Context, request *serverProto.AddDocumentRequest) (*serverProto.ResCount, error) {
	println(1111)
	return &serverProto.ResCount{Count: 1}, nil
}

func (s *SlaveService) DeleteDocumentFromTable(ctx context.Context, request *serverProto.DeleteDocumentRequest) (*serverProto.ResCount, error) {
	logger.Info("Deleting document %s from table: %s", request.DocId, request.TableName)

	// 这里应该实现从表中删除文档的逻辑
	if s.multiTableService != nil {
		// 实现文档删除逻辑
		logger.Info("Document %s deleted from table %s successfully", request.DocId, request.TableName)
	} else {
		logger.Warning("MultiTableService not initialized")
	}

	return &serverProto.ResCount{Count: 1}, nil
}

func (s *SlaveService) SearchTable(ctx context.Context, request *serverProto.SearchRequest) (*serverProto.SearchResult, error) {
	logger.Info("Searching in table: %s", request.TableName)

	// 这里应该实现表搜索的逻辑
	if s.multiTableService != nil {
		// 实现搜索逻辑
		// 返回搜索结果
		logger.Info("Search completed in table %s", request.TableName)
		return &serverProto.SearchResult{
			DocIds: []string{}, // 返回空结果作为示例
		}, nil
	} else {
		logger.Warning("MultiTableService not initialized")
		return &serverProto.SearchResult{
			DocIds: []string{},
		}, nil
	}
}

// NewSlaveService 创建从服务实例
func NewSlaveService(appCtx context.Context, endpoints []entity.EndPoint, serviceName string, port int) (*SlaveService, error) {
	// 获取本地 IP
	localIp, err := common.GetLocalHost()
	if err != nil {
		return nil, fmt.Errorf("获取本地 IP 失败: %v", err)
	}
	localhost := localIp + ":" + strconv.Itoa(port)

	// 创建 etcd 客户端
	var endpointUrls []string
	for _, ep := range endpoints {
		endpointUrls = append(endpointUrls, ep.Ip)
	}

	client, err := clientv3.New(clientv3.Config{
		Endpoints:   endpointUrls,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("创建 etcd 客户端失败: %v", err)
	}

	// 创建任务执行器
	taskExecutor := scheduler2.NewTaskPoolManager() // 设置合适的工作线程数

	return &SlaveService{
		client:        client,
		serviceName:   serviceName,
		localhost:     localhost,
		stopCh:        make(chan struct{}),
		appCtx:        appCtx,
		taskExecutor:  taskExecutor,
		etcdEndpoints: endpoints, // 保存 etcdEndpoints
		port:          port,      // 保存 port
	}, nil
}

// SetCommunicationService 设置通信服务
func (s *SlaveService) SetCommunicationService(communicationService interface{}) {
	s.communicationService = communicationService
}

// Init 初始化索引服务
func (s *SlaveService) Init(
	docNumEstimate int,
	dbType int,
	DataDir string,
	txMgr *tree.TransactionManager,
	lockMgr *tree.LockManager,
	wal *tree.WALManager,
) error {
	// 初始化索引
	s.Index = &index.Index{}
	err := s.Index.NewIndexServer(docNumEstimate, dbType, "", DataDir)
	if err != nil {
		return err
	}

	// 初始化多表搜索服务
	s.multiTableService = search.NewMultiTableSearchService(txMgr, lockMgr, wal)

	return nil
}

// Start 启动 SlaveService
func (s *SlaveService) Start(ctx context.Context) error {
	logger.Info("Starting SlaveService on %s...", s.localhost)
	var etcdEndpoints []string
	for _, v := range s.etcdEndpoints {
		etcdEndpoints = append(etcdEndpoints, v.Ip+":"+strconv.Itoa(v.Port))
	}
	// 如果 etcd client 未初始化，尝试重新初始化
	if s.client == nil {
		logger.Info("etcd client not initialized, attempting to connect...")
		// 异步连接etcd，避免阻塞启动流程
		go func() {
			logger.Info("Starting etcd connection...")
			client, err := clientv3.New(clientv3.Config{
				Endpoints:   etcdEndpoints,
				DialTimeout: 5 * time.Second,
			})
			if err != nil {
				logger.Warning("Failed to connect to etcd during SlaveService start: %v. Proceeding without etcd.", err)
				return
			}
			s.client = client
			logger.Info("Successfully connected to etcd.")

			// 连接成功后注册服务
			if err := s.registerService(); err != nil {
				logger.Warning("Failed to register slave service with etcd: %v.", err)
			} else {
				logger.Info("Slave service registered with etcd successfully.")
			}

			// 开始监听主节点
			go s.watchMaster(context.Background())
		}()
		logger.Info("etcd connection started in background.")
	}

	// etcd注册和监听已移到异步处理中
	logger.Info("etcd registration and master watching will be handled asynchronously.")

	// 初始化索引等（如果尚未初始化）
	if s.Index == nil {
		logger.Info("Starting slave service component initialization...")
		// 异步初始化索引，避免阻塞启动流程
		go func() {
			// 这里的参数需要从配置中获取或传递
			// 例如: docNumEstimate, dbType, DataDir 等
			// 此处仅为示例，您需要根据实际情况调整
			logger.Info("Initializing slave service components...")
			if err := s.Init(100000, 0, "./data/slave_"+s.localhost, nil, nil, nil); err != nil {
				logger.Error("Failed to initialize slave service components: %v", err)
				// 这里可以设置一个标志位表示初始化失败
				return
			}
			logger.Info("Slave service components initialized successfully.")
		}()
		logger.Info("Slave service component initialization started in background.")
	}

	// 启动gRPC服务器
	if err := s.startGRPCServer(); err != nil {
		logger.Error("Failed to start gRPC server: %v", err)
		return err
	}

	logger.Info("SlaveService %s started.", s.localhost)

	// 监听停止信号
	go func() {
		select {
		case <-s.stopCh:
			logger.Info("SlaveService %s received stop signal.", s.localhost)
		case <-ctx.Done():
			logger.Info("SlaveService %s context cancelled.", s.localhost)
			err := s.Stop(context.Background())
			if err != nil {
				return
			}
		case <-s.appCtx.Done():
			logger.Info("Application context cancelled, stopping SlaveService %s.", s.localhost)
			err := s.Stop(context.Background())
			if err != nil {
				return
			}
		}
	}()

	return nil
}

// registerService 注册服务
func (s *SlaveService) registerService() error {
	resp, err := s.client.Grant(context.Background(), 10)
	if err != nil {
		return fmt.Errorf("创建租约失败: %v", err)
	}
	s.leaseID = resp.ID
	// 注册服务时带权重和标签
	meta := map[string]interface{}{
		"ip":     s.localhost,
		"weight": 1,
		"tags":   map[string]string{"env": os.Getenv("SLAVE_ENV")},
	}
	val, _ := json.Marshal(meta)
	key := fmt.Sprintf("%s/%s/%s", backup.ServiceRootPath, s.serviceName, s.localhost)
	_, err = s.client.Put(context.Background(), key, string(val), clientv3.WithLease(s.leaseID))
	if err != nil {
		return fmt.Errorf("注册服务失败: %v", err)
	}

	ch, err := s.client.KeepAlive(context.Background(), s.leaseID)
	if err != nil {
		return fmt.Errorf("保持租约失败: %v", err)
	}
	go func() {
		for {
			select {
			case <-s.stopCh:
				return
			default:
				<-ch
			}
		}
	}()
	return nil
}

// GetName 返回服务名称
func (s *SlaveService) GetName() string {
	return "slave_service_" + s.localhost // 保证每个 slave 实例的任务名唯一
}

// GetDependencies 返回服务依赖的任务名称列表
func (s *SlaveService) GetDependencies() []string {
	// 依赖 etcd (如果 etcdClient 初始化成功) 和 master 服务发现
	deps := []string{"master_service_discovery_ready"}
	if s.client != nil { // 仅当 etcd client 初始化时才添加 etcd_ready 依赖
		deps = append(deps, "etcd_ready")
	}
	return deps
}

// connectMaster 连接主节点
func (s *SlaveService) connectMaster() {
	// 关闭旧连接
	if s.masterConn != nil {
		err := s.masterConn.Close()
		if err != nil {
			return
		}
		s.masterConn = nil
	}

	// 创建新连接
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, s.masterEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		logger.Error("连接主节点失败: %v", err)
		return
	}

	s.masterConn = conn
	logger.Info("连接主节点成功: %s", s.masterEndpoint)
}

// Stop 停止 SlaveService
func (s *SlaveService) Stop(ctx context.Context) error {
	logger.Info("Stopping SlaveService %s...", s.localhost)
	close(s.stopCh)

	// 停止gRPC服务器
	if s.grpcServer != nil {
		logger.Info("Stopping gRPC server...")
		s.grpcServer.GracefulStop()
		s.grpcServer = nil
	}

	// 从 etcd 取消注册
	if s.client != nil && s.leaseID != 0 {
		if _, err := s.client.Revoke(context.Background(), s.leaseID); err != nil {
			logger.Warning("Failed to revoke etcd lease for slave %s: %v", s.localhost, err)
		}
	}

	// 关闭到主节点的连接
	s.masterMutex.Lock()
	if s.masterConn != nil {
		err := s.masterConn.Close()
		if err != nil {
			return err
		}
		s.masterConn = nil
	}
	s.masterMutex.Unlock()

	// 关闭 etcd 客户端
	if s.client != nil {
		if err := s.client.Close(); err != nil {
			logger.Warning("Failed to close etcd client for slave %s: %v", s.localhost, err)
		}
	}

	// 关闭索引等资源
	if s.Index != nil {
		logger.Info("Closing Index for SlaveService %s...", s.localhost)
		err := s.Index.Close()
		if err != nil {
			logger.Warning("Failed to close Index for SlaveService %s: %v", s.localhost, err)
			return err
		}
		logger.Info("Index closed successfully for SlaveService %s", s.localhost)
	} else {
		logger.Info("Index is nil, skipping close for SlaveService %s", s.localhost)
	}

	logger.Info("SlaveService %s stopped.", s.localhost)
	return nil
}

// GetTaskSpec 返回任务规格
func (s *SlaveService) GetTaskSpec() *scheduler2.TaskSpec {
	return &scheduler2.TaskSpec{
		RetryCount: 5, // 示例：重试5次
		// Timeout:    60 * time.Second, // 示例：超时60秒
	}
}

// ToPoolTask 将 SlaveService 转换为 PoolLib.Task
func (s *SlaveService) ToPoolTask() *PoolLib.Queue {
	startWrapper := func() error {
		logger.Info("ServiceTask for SlaveService: %s is about to run Start()", s.GetName())
		// 注意：s.Start 需要一个 context.Context。
		// 在 bootstrap/base.go 中，任务池执行任务时会传递上下文。
		// 但 PoolLib.Queue 的执行不直接接受上下文参数给 ExcelFunc。
		// 因此，我们在这里使用 s.appCtx，或者需要一种机制将任务池的上下文传递给 Start。
		// 假设 s.Start 内部可以处理 s.appCtx，或者 bootstrap 层面在调用此 Task 的 Execute 时传递正确的上下文。
		// 为了与 PoolLib.Queue 兼容，包装器不接受参数。
		err := s.Start(s.appCtx)
		if err != nil {
			logger.Error("ServiceTask for SlaveService: %s failed to Start: %v", s.GetName(), err)
		}
		return err
	}

	return PoolLib.QueryInit(s.GetName(), startWrapper)
}

// SetTaskPoolManager 设置任务池管理器
func (s *SlaveService) SetTaskPoolManager(manager *scheduler2.TaskPoolManager) {
	s.taskExecutor = manager
}

// watchMaster 监听主节点变化
func (s *SlaveService) watchMaster(ctx context.Context) {
	if s.client == nil {
		logger.Warning("Cannot watch master: etcd client is nil.")
		return
	}
	masterKeyPrefix := "/VectorSphere/master_election/"
	logger.Info("Slave %s watching for master changes at prefix: %s", s.localhost, masterKeyPrefix)

	// 初始获取一次 Master 地址
	s.updateMasterConnectionFromEtcd(ctx, masterKeyPrefix)

	rch := s.client.Watch(ctx, masterKeyPrefix, clientv3.WithPrefix())
	for {
		select {
		case <-ctx.Done():
			logger.Info("Watch master process for slave %s cancelled by context.", s.localhost)
			return
		case <-s.stopCh:
			logger.Info("Watch master process for slave %s stopped by service stop signal.", s.localhost)
			return
		case wresp := <-rch:
			if wresp.Canceled {
				logger.Warning("Watch for master on slave %s was canceled: %v", s.localhost, wresp.Err())
				// 尝试重新 Watch
				time.Sleep(5 * time.Second)
				if s.client != nil {
					rch = s.client.Watch(ctx, masterKeyPrefix, clientv3.WithPrefix())
				}
				continue
			}
			for _, ev := range wresp.Events {
				logger.Trace("Slave %s received etcd event: Type: %s, Key: %s, Value: %s", s.localhost, ev.Type, string(ev.Kv.Key), string(ev.Kv.Value))
				// 当有 master 相关的事件时，重新获取并连接 master
				s.updateMasterConnectionFromEtcd(ctx, masterKeyPrefix)
			}
		}
	}
}

func (s *SlaveService) updateMasterConnectionFromEtcd(ctx context.Context, masterKeyPrefix string) {
	if s.client == nil {
		return
	}
	resp, err := s.client.Get(ctx, masterKeyPrefix, clientv3.WithPrefix(), clientv3.WithSort(clientv3.SortByKey, clientv3.SortAscend), clientv3.WithLimit(1))
	if err != nil {
		logger.Warning("Slave %s failed to get master from etcd: %v", s.localhost, err)
		return
	}

	var newMasterEndpoint string
	if len(resp.Kvs) > 0 {
		newMasterEndpoint = string(resp.Kvs[0].Value)
		logger.Info("Slave %s discovered master endpoint from etcd: %s", s.localhost, newMasterEndpoint)
	} else {
		logger.Warning("Slave %s: No master found in etcd with prefix %s", s.localhost, masterKeyPrefix)
		s.masterMutex.Lock()
		if s.masterConn != nil {
			err := s.masterConn.Close()
			if err != nil {
				return
			}
			s.masterConn = nil
			logger.Info("Slave %s: Disconnected from previous master as no master is currently elected.", s.localhost)
		}
		s.masterEndpoint = ""
		s.masterMutex.Unlock()
		return
	}

	s.masterMutex.Lock()
	defer s.masterMutex.Unlock()

	if newMasterEndpoint != s.masterEndpoint || s.masterConn == nil {
		if s.masterConn != nil {
			err := s.masterConn.Close()
			if err != nil {
				return
			}
			logger.Info("Slave %s: Disconnecting from old master %s", s.localhost, s.masterEndpoint)
		}
		logger.Info("Slave %s: Attempting to connect to new master %s", s.localhost, newMasterEndpoint)
		conn, err := grpc.DialContext(ctx, newMasterEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
		if err != nil {
			logger.Error("Slave %s: Failed to connect to master %s: %v", s.localhost, newMasterEndpoint, err)
			s.masterConn = nil
			s.masterEndpoint = "" // 连接失败，清除 endpoint
			return
		}
		logger.Info("Slave %s: Successfully connected to master %s", s.localhost, newMasterEndpoint)
		s.masterConn = conn
		s.masterEndpoint = newMasterEndpoint
		// --- 主节点变更通知 ---
		go func() {
			webhookUrl := os.Getenv("SLAVE_MASTER_WEBHOOK")
			if webhookUrl == "" {
				return
			}
			body := map[string]string{"msg": "Slave connected to new master: " + newMasterEndpoint, "time": time.Now().Format(time.RFC3339)}
			jsonBody, _ := json.Marshal(body)
			http.Post(webhookUrl, "application/json", bytes.NewReader(jsonBody))
		}()
	} else {
		logger.Trace("Slave %s: Master endpoint %s unchanged.", s.localhost, s.masterEndpoint)
	}
}

// 实现 IndexServiceServer 接口

func (s *SlaveService) DelDoc(ctx context.Context, docId *serverProto.DocId) (*serverProto.ResCount, error) {
	// 调用索引服务删除文档
	return &serverProto.ResCount{
		Count: int32(s.Index.DelDoc(docId.Id)),
	}, nil
}

func (s *SlaveService) AddDoc(ctx context.Context, doc *messages.Document) (*serverProto.ResCount, error) {
	// 调用索引服务添加文档
	n, err := s.Index.AddDoc(*doc)
	return &serverProto.ResCount{
		Count: int32(n),
	}, err
}

func (s *SlaveService) Search(ctx context.Context, request *serverProto.Request) (*serverProto.Result, error) {
	// 调用索引服务搜索
	result, err := s.Index.Search(request.Query, request.OnFlag, request.OffFlag, request.OrFlags)
	if err != nil {
		return nil, err
	}

	return &serverProto.Result{
		Results: result,
	}, nil
}

// HealthCheck 实现 IndexServiceServer 接口的 HealthCheck 方法
func (s *SlaveService) HealthCheck(ctx context.Context, req *serverProto.HealthCheckRequest) (*serverProto.HealthCheckResponse, error) {
	// 1. 检查核心依赖服务是否可用
	if s.client == nil {
		logger.Warning("HealthCheck: etcd client is nil")
		return &serverProto.HealthCheckResponse{
			Status: serverProto.HealthCheckResponse_NOT_SERVING,
			Load:   0, // 或者一个表示错误/不可用的特定负值
		}, nil
	}
	if s.Index == nil {
		logger.Warning("HealthCheck: Index service is nil")
		return &serverProto.HealthCheckResponse{
			Status: serverProto.HealthCheckResponse_NOT_SERVING,
			Load:   0,
		}, nil
	}
	if s.taskExecutor == nil {
		logger.Warning("HealthCheck: TaskExecutor is nil")
		return &serverProto.HealthCheckResponse{
			Status: serverProto.HealthCheckResponse_NOT_SERVING,
			Load:   0,
		}, nil
	}

	// 2. 计算当前负载
	var currentLoad float32
	var cpuUsage float32
	var memUsage float32
	var taskQueueLength int32

	// 获取 CPU 使用率 (需要 github.com/shirou/gopsutil/cpu)
	cpuPercentages, err := cpu.Percent(time.Second, false) // false 表示获取总体 CPU 使用率
	if err == nil && len(cpuPercentages) > 0 {
		cpuUsage = float32(cpuPercentages[0])
	} else {
		logger.Warning("HealthCheck: Failed to get CPU usage: %v", err)
		// 可以选择在此处返回 NOT_SERVING 或 UNKNOWN，或者继续计算其他指标
	}

	// 获取内存使用率
	vmStat, err := mem.VirtualMemory()
	if err == nil {
		memUsage = float32(vmStat.UsedPercent)
	} else {
		logger.Warning("HealthCheck: Failed to get memory usage: %v", err)
	}

	// 获取任务队列长度/当前运行任务数
	taskQueueLength = s.taskExecutor.GetRunningTaskCount() // 使用您实现的 GetRunningTaskCount

	// 获取当前处理的请求数 (这是一个示例，您需要根据您的服务逻辑来实现)
	// currentRequests := s.getCurrentActiveRequests() // 假设有这样一个方法

	// 综合计算负载：这是一个简单的加权平均示例，您可以根据实际情况调整权重和计算方式
	// 权重可以根据哪个指标对您的服务健康更重要来设定
	// 例如：CPU 40%, 内存 30%, 任务队列 30%
	// 注意：这里的负载值是一个 0-100 的百分比值，您可以根据需要调整其范围和含义
	cpuWeight := float32(0.4)
	memWeight := float32(0.3)
	taskWeight := float32(0.3)

	// 将任务队列长度归一化到一个可比较的范围，例如 0-100
	// 假设任务队列长度超过 200 就认为是满负荷
	maxExpectedTasks := float32(200)
	normalizedTaskLoad := (float32(taskQueueLength) / maxExpectedTasks) * 100
	if normalizedTaskLoad > 100 {
		normalizedTaskLoad = 100
	}
	if normalizedTaskLoad < 0 {
		normalizedTaskLoad = 0
	}

	currentLoad = (cpuUsage * cpuWeight) + (memUsage * memWeight) + (normalizedTaskLoad * taskWeight)
	if currentLoad > 100 {
		currentLoad = 100
	}
	if currentLoad < 0 {
		currentLoad = 0
	}

	logger.Trace("HealthCheck: CPU: %.2f%%, Mem: %.2f%%, Tasks: %d (Normalized: %.2f%%), Calculated Load: %.2f%%", cpuUsage, memUsage, taskQueueLength, normalizedTaskLoad, currentLoad)

	// 3. 判断健康状态
	// 示例：如果综合负载超过某个阈值，则认为服务过载，状态为 UNKNOWN 或自定义状态
	loadThreshold := float32(85.0) // 假设综合负载超过 85% 就认为过载

	if currentLoad > loadThreshold {
		logger.Warning("HealthCheck: Load (%.2f%%) exceeds threshold (%.2f%%)", currentLoad, loadThreshold)
		return &serverProto.HealthCheckResponse{
			Status: serverProto.HealthCheckResponse_UNKNOWN, // 或者一个表示高负载的状态
			Load:   float32(currentLoad),                    // protobuf 的 load 是 int32，进行转换
		}, nil
	}

	// 如果所有检查通过，并且负载在可接受范围内，则返回 SERVING 状态
	return &serverProto.HealthCheckResponse{
		Status: serverProto.HealthCheckResponse_SERVING,
		Load:   float32(currentLoad),
	}, nil
}

func (s *SlaveService) Count(ctx context.Context, request *serverProto.CountRequest) (*serverProto.ResCount, error) {
	// 调用索引服务计数
	return &serverProto.ResCount{
		Count: int32(s.Index.Total()),
	}, nil
}

func (s *SlaveService) ExecuteTask(ctx context.Context, request *serverProto.TaskRequest) (*serverProto.TaskResponse, error) {
	// 检查请求是否为空
	if request == nil {
		return &serverProto.TaskResponse{
			Success:      false,
			ErrorMessage: "请求为空",
		}, nil
	}

	// 解析任务数据
	var taskData map[string]interface{}
	err := json.Unmarshal(request.TaskData, &taskData)
	if err != nil {
		return &serverProto.TaskResponse{
			TaskId:       request.TaskId,
			Success:      false,
			ErrorMessage: fmt.Sprintf("解析任务数据失败: %v", err),
		}, nil
	}

	// 检查多表服务是否初始化
	if s.multiTableService == nil {
		return &serverProto.TaskResponse{
			TaskId:       request.TaskId,
			Success:      false,
			ErrorMessage: "多表搜索服务未初始化",
		}, nil
	}

	// 根据任务类型执行不同的操作
	var resultData []byte
	var success bool
	var errorMessage string

	switch request.TaskType {
	case "rebuild_index":
		// 重建索引任务
		tableName, ok := taskData["table_name"].(string)
		if !ok {
			errorMessage = "缺少表名参数"
			break
		}

		// 获取表实例
		table, err := s.multiTableService.GetTable(tableName)
		if err != nil {
			errorMessage = fmt.Sprintf("获取表失败: %v", err)
			break
		}

		// 检查表实例是否为空
		if table == nil {
			errorMessage = fmt.Sprintf("表 %s 不存在", tableName)
			break
		}

		// 重建索引
		if table.VectorDB != nil {
			// 使用新实现的 RebuildIndex 方法
			err = table.VectorDB.RebuildIndex()
			if err != nil {
				errorMessage = fmt.Sprintf("重建索引失败: %v", err)
				break
			}

			// 如果表有倒排索引，也尝试优化它
			if table.InvertedIndex != nil {
				optErr := table.InvertedIndex.Optimize()
				if optErr != nil {
					logger.Warning("优化倒排索引失败: %v", optErr)
					// 不中断流程，只记录警告
				}
			}
		} else {
			// 如果没有向量数据库，但有倒排索引，尝试只优化倒排索引
			if table.InvertedIndex != nil {
				optErr := table.InvertedIndex.Optimize()
				if optErr != nil {
					errorMessage = fmt.Sprintf("优化倒排索引失败: %v", optErr)
					break
				}
				success = true
				resultData, _ = json.Marshal(map[string]interface{}{
					"message": fmt.Sprintf("表 %s 倒排索引优化成功", tableName),
				})
				break
			} else {
				errorMessage = "表没有向量数据库和倒排索引实例"
				break
			}
		}

		success = true
		resultData, _ = json.Marshal(map[string]interface{}{
			"message": fmt.Sprintf("表 %s 索引重建成功", tableName),
		})

	case "optimize_index":
		// 优化索引任务
		tableName, ok := taskData["table_name"].(string)
		if !ok {
			errorMessage = "缺少表名参数"
			break
		}

		// 获取表实例
		table, err := s.multiTableService.GetTable(tableName)
		if err != nil {
			errorMessage = fmt.Sprintf("获取表失败: %v", err)
			break
		}

		// 检查表实例是否为空
		if table == nil {
			errorMessage = fmt.Sprintf("表 %s 不存在", tableName)
			break
		}

		// 优化索引
		if table.InvertedIndex != nil {
			err = table.InvertedIndex.Optimize()
			if err != nil {
				errorMessage = fmt.Sprintf("优化索引失败: %v", err)
				break
			}
		} else {
			errorMessage = "表没有倒排索引实例"
			break
		}

		success = true
		resultData, _ = json.Marshal(map[string]interface{}{
			"message": fmt.Sprintf("表 %s 索引优化成功", tableName),
		})

	case "backup_data":
		// 备份数据任务
		tableName, ok := taskData["table_name"].(string)
		if !ok {
			errorMessage = "缺少表名参数"
			break
		}

		backupPath, ok := taskData["backup_path"].(string)
		if !ok {
			errorMessage = "缺少备份路径参数"
			break
		}

		// 获取表实例
		table, err := s.multiTableService.GetTable(tableName)
		if err != nil {
			errorMessage = fmt.Sprintf("获取表失败: %v", err)
			break
		}

		// 检查表实例是否为空
		if table == nil {
			errorMessage = fmt.Sprintf("表 %s 不存在", tableName)
			break
		}

		// 备份数据
		if table.VectorDB != nil {
			// 设置备份路径
			originalPath := table.VectorDB.GetFilePath()
			table.VectorDB.SetFilePath(backupPath)

			// 执行备份
			err = table.VectorDB.SaveToFile(table.VectorDB.GetBackupPath())

			// 恢复原始路径
			table.VectorDB.SetFilePath(originalPath)

			if err != nil {
				errorMessage = fmt.Sprintf("备份数据失败: %v", err)
				break
			}
		} else {
			errorMessage = "表没有向量数据库实例"
			break
		}

		success = true
		resultData, _ = json.Marshal(map[string]interface{}{
			"message": fmt.Sprintf("表 %s 数据备份成功，路径: %s", tableName, backupPath),
		})

	default:
		errorMessage = fmt.Sprintf("不支持的任务类型: %s", request.TaskType)
	}

	// 创建任务响应
	response := &serverProto.TaskResponse{
		TaskId:       request.TaskId,
		Success:      success,
		ResultData:   resultData,
		ErrorMessage: errorMessage,
	}

	// 异步上报任务结果给主服务
	go s.reportTaskResult(response)

	return response, nil
}

// reportTaskResult 上报任务结果给主服务
func (s *SlaveService) reportTaskResult(response *serverProto.TaskResponse) {
	// 检查主节点连接
	s.masterMutex.RLock()
	masterConn := s.masterConn
	s.masterMutex.RUnlock()

	if masterConn == nil {
		logger.Error("上报任务结果失败: 没有主节点连接")
		return
	}

	// 创建客户端
	client := serverProto.NewIndexServiceClient(masterConn)

	// 创建上下文，设置超时
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 上报任务结果
	_, err := client.ReportTaskResult(ctx, response)
	if err != nil {
		logger.Error("上报任务结果失败: %v", err)
	}
}

// ReportTaskResult 接收任务结果上报（主服务调用）
func (s *SlaveService) ReportTaskResult(ctx context.Context, response *serverProto.TaskResponse) (*serverProto.ResCount, error) {
	return nil, fmt.Errorf("从服务不支持接收任务结果上报")
}

// startGRPCServer 启动gRPC服务器
func (s *SlaveService) startGRPCServer() error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		return fmt.Errorf("failed to listen on port %d: %v", s.port, err)
	}

	s.grpcServer = grpc.NewServer()
	serverProto.RegisterIndexServiceServer(s.grpcServer, s)

	logger.Info("Starting gRPC server on port %d", s.port)
	go func() {
		if err := s.grpcServer.Serve(lis); err != nil {
			logger.Error("gRPC server failed to serve: %v", err)
		}
	}()

	return nil
}
