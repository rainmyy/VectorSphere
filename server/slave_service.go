package server

import (
	"context"
	"encoding/json"
	"fmt"
	PoolLib "seetaSearch/library/pool"
	"seetaSearch/library/tree"
	"time"

	"go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"seetaSearch/index"
	"seetaSearch/library/common"
	"seetaSearch/library/log"
	"seetaSearch/messages"
	"seetaSearch/scheduler"
	"seetaSearch/search"
	"strconv"
	"sync"
)

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
	taskExecutor *scheduler.TaskPoolManager

	// 多表搜索服务
	multiTableService *search.MultiTableSearchService

	appCtx        context.Context // 用于传递应用的全局上下文
	etcdEndpoints []EndPoint      // etcd 端点
	port          int             // 服务端口
}

func (s *SlaveService) CreateTable(ctx context.Context, request *CreateTableRequest) (*ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (s *SlaveService) DeleteTable(ctx context.Context, request *TableRequest) (*ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (s *SlaveService) AddDocumentToTable(ctx context.Context, request *AddDocumentRequest) (*ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (s *SlaveService) DeleteDocumentFromTable(ctx context.Context, request *DeleteDocumentRequest) (*ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (s *SlaveService) SearchTable(ctx context.Context, request *SearchRequest) (*SearchResult, error) {
	//TODO implement me
	panic("implement me")
}

// NewSlaveService 创建从服务实例
func NewSlaveService(appCtx context.Context, endpoints []EndPoint, serviceName string, port int) (*SlaveService, error) {
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
	taskExecutor := scheduler.NewTaskPoolManager() // 设置合适的工作线程数

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
	log.Info("Starting SlaveService on %s...", s.localhost)
	var etcdEndpoints []string
	for _, v := range s.etcdEndpoints {
		etcdEndpoints = append(etcdEndpoints, v.Ip+":"+strconv.Itoa(v.Port))
	}
	// 如果 etcd client 未初始化，尝试重新初始化
	if s.client == nil {
		log.Info("etcd client not initialized, attempting to connect...")
		client, err := clientv3.New(clientv3.Config{
			Endpoints:   etcdEndpoints,
			DialTimeout: 5 * time.Second,
		})
		if err != nil {
			log.Warning("Failed to connect to etcd during SlaveService start: %v. Proceeding without etcd.", err)
		} else {
			s.client = client
			log.Info("Successfully connected to etcd.")
		}
	}

	// 注册服务 (如果 etcd client 存在)
	if s.client != nil {
		if err := s.registerService(); err != nil {
			// 非致命错误，服务可以尝试无etcd运行或依赖master拉取
			log.Warning("Failed to register slave service with etcd: %v. Continuing without registration.", err)
		} else {
			log.Info("Slave service registered with etcd successfully.")
		}
	} else {
		log.Info("Slave service running without etcd registration.")
	}

	// 监听主节点 (如果 etcd client 存在)
	if s.client != nil {
		go s.watchMaster(ctx) // 使用传递的上下文
	} else {
		log.Info("Cannot watch master as etcd client is not available.")
		// 这里可以考虑实现一个备用机制来发现 master，例如通过配置或广播
	}

	// 初始化索引等（如果尚未初始化）
	if s.Index == nil {
		// 这里的参数需要从配置中获取或传递
		// 例如: docNumEstimate, dbType, DataDir 等
		// 此处仅为示例，您需要根据实际情况调整
		if err := s.Init(100000, 0, "./data/slave_"+s.localhost, nil, nil, nil); err != nil {
			log.Error("Failed to initialize slave service components: %v", err)
			return err // 初始化失败是致命的
		}
		log.Info("Slave service components initialized.")
	}

	log.Info("SlaveService %s started.", s.localhost)

	// 监听停止信号
	go func() {
		select {
		case <-s.stopCh:
			log.Info("SlaveService %s received stop signal.", s.localhost)
		case <-ctx.Done():
			log.Info("SlaveService %s context cancelled.", s.localhost)
			s.Stop(context.Background())
		case <-s.appCtx.Done():
			log.Info("Application context cancelled, stopping SlaveService %s.", s.localhost)
			s.Stop(context.Background())
		}
	}()

	return nil
}

// registerService 注册服务
func (s *SlaveService) registerService() error {
	// 创建租约
	resp, err := s.client.Grant(context.Background(), 10)
	if err != nil {
		return fmt.Errorf("创建租约失败: %v", err)
	}
	s.leaseID = resp.ID

	// 注册服务
	key := fmt.Sprintf("%s/%s/%s", ServiceRootPath, s.serviceName, s.localhost)
	_, err = s.client.Put(context.Background(), key, s.localhost, clientv3.WithLease(s.leaseID))
	if err != nil {
		return fmt.Errorf("注册服务失败: %v", err)
	}

	// 保持租约
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
				// 租约保持成功
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
		s.masterConn.Close()
		s.masterConn = nil
	}

	// 创建新连接
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, s.masterEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		log.Error("连接主节点失败: %v", err)
		return
	}

	s.masterConn = conn
	log.Info("连接主节点成功: %s", s.masterEndpoint)
}

// Stop 停止 SlaveService
func (s *SlaveService) Stop(ctx context.Context) error {
	log.Info("Stopping SlaveService %s...", s.localhost)
	close(s.stopCh)

	// 从 etcd 取消注册
	if s.client != nil && s.leaseID != 0 {
		if _, err := s.client.Revoke(context.Background(), s.leaseID); err != nil {
			log.Warning("Failed to revoke etcd lease for slave %s: %v", s.localhost, err)
		}
	}

	// 关闭到主节点的连接
	s.masterMutex.Lock()
	if s.masterConn != nil {
		s.masterConn.Close()
		s.masterConn = nil
	}
	s.masterMutex.Unlock()

	// 关闭 etcd 客户端
	if s.client != nil {
		if err := s.client.Close(); err != nil {
			log.Warning("Failed to close etcd client for slave %s: %v", s.localhost, err)
		}
	}

	// 关闭索引等资源
	if s.Index != nil {
		s.Index.Close()
	}

	log.Info("SlaveService %s stopped.", s.localhost)
	return nil
}

// GetTaskSpec 返回任务规格
func (s *SlaveService) GetTaskSpec() *scheduler.TaskSpec {
	return &scheduler.TaskSpec{
		RetryCount: 5, // 示例：重试5次
		// Timeout:    60 * time.Second, // 示例：超时60秒
	}
}

// ToPoolTask 将 SlaveService 转换为 PoolLib.Task
func (s *SlaveService) ToPoolTask() *PoolLib.Queue {
	startWrapper := func() error {
		log.Info("ServiceTask for SlaveService: %s is about to run Start()", s.GetName())
		// 注意：s.Start 需要一个 context.Context。
		// 在 bootstrap/base.go 中，任务池执行任务时会传递上下文。
		// 但 PoolLib.Queue 的执行不直接接受上下文参数给 ExcelFunc。
		// 因此，我们在这里使用 s.appCtx，或者需要一种机制将任务池的上下文传递给 Start。
		// 假设 s.Start 内部可以处理 s.appCtx，或者 bootstrap 层面在调用此 Task 的 Execute 时传递正确的上下文。
		// 为了与 PoolLib.Queue 兼容，包装器不接受参数。
		err := s.Start(s.appCtx)
		if err != nil {
			log.Error("ServiceTask for SlaveService: %s failed to Start: %v", s.GetName(), err)
		}
		return err
	}

	return PoolLib.QueryInit(s.GetName(), startWrapper)
}

// SetTaskPoolManager 设置任务池管理器
func (s *SlaveService) SetTaskPoolManager(manager *scheduler.TaskPoolManager) {
	s.taskExecutor = manager
}

// watchMaster 监听主节点变化
func (s *SlaveService) watchMaster(ctx context.Context) {
	if s.client == nil {
		log.Warning("Cannot watch master: etcd client is nil.")
		return
	}
	masterKeyPrefix := "/seetasearch/master_election/"
	log.Info("Slave %s watching for master changes at prefix: %s", s.localhost, masterKeyPrefix)

	// 初始获取一次 Master 地址
	s.updateMasterConnectionFromEtcd(ctx, masterKeyPrefix)

	rch := s.client.Watch(ctx, masterKeyPrefix, clientv3.WithPrefix())
	for {
		select {
		case <-ctx.Done():
			log.Info("Watch master process for slave %s cancelled by context.", s.localhost)
			return
		case <-s.stopCh:
			log.Info("Watch master process for slave %s stopped by service stop signal.", s.localhost)
			return
		case wresp := <-rch:
			if wresp.Canceled {
				log.Warning("Watch for master on slave %s was canceled: %v", s.localhost, wresp.Err())
				// 尝试重新 Watch
				time.Sleep(5 * time.Second)
				if s.client != nil {
					rch = s.client.Watch(ctx, masterKeyPrefix, clientv3.WithPrefix())
				}
				continue
			}
			for _, ev := range wresp.Events {
				log.Trace("Slave %s received etcd event: Type: %s, Key: %s, Value: %s", s.localhost, ev.Type, string(ev.Kv.Key), string(ev.Kv.Value))
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
		log.Warning("Slave %s failed to get master from etcd: %v", s.localhost, err)
		return
	}

	var newMasterEndpoint string
	if len(resp.Kvs) > 0 {
		newMasterEndpoint = string(resp.Kvs[0].Value)
		log.Info("Slave %s discovered master endpoint from etcd: %s", s.localhost, newMasterEndpoint)
	} else {
		log.Warning("Slave %s: No master found in etcd with prefix %s", s.localhost, masterKeyPrefix)
		s.masterMutex.Lock()
		if s.masterConn != nil {
			s.masterConn.Close()
			s.masterConn = nil
			log.Info("Slave %s: Disconnected from previous master as no master is currently elected.", s.localhost)
		}
		s.masterEndpoint = ""
		s.masterMutex.Unlock()
		return
	}

	s.masterMutex.Lock()
	defer s.masterMutex.Unlock()

	if newMasterEndpoint != s.masterEndpoint || s.masterConn == nil {
		if s.masterConn != nil {
			s.masterConn.Close()
			log.Info("Slave %s: Disconnecting from old master %s", s.localhost, s.masterEndpoint)
		}
		log.Info("Slave %s: Attempting to connect to new master %s", s.localhost, newMasterEndpoint)
		conn, err := grpc.DialContext(ctx, newMasterEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
		if err != nil {
			log.Error("Slave %s: Failed to connect to master %s: %v", s.localhost, newMasterEndpoint, err)
			s.masterConn = nil
			s.masterEndpoint = "" // 连接失败，清除 endpoint
			return
		}
		log.Info("Slave %s: Successfully connected to master %s", s.localhost, newMasterEndpoint)
		s.masterConn = conn
		s.masterEndpoint = newMasterEndpoint
	} else {
		log.Trace("Slave %s: Master endpoint %s unchanged.", s.localhost, s.masterEndpoint)
	}
}

// 实现 IndexServiceServer 接口

func (s *SlaveService) DelDoc(ctx context.Context, docId *DocId) (*ResCount, error) {
	// 调用索引服务删除文档
	return &ResCount{
		Count: int32(s.Index.DelDoc(docId.Id)),
	}, nil
}

func (s *SlaveService) AddDoc(ctx context.Context, doc *messages.Document) (*ResCount, error) {
	// 调用索引服务添加文档
	n, err := s.Index.AddDoc(*doc)
	return &ResCount{
		Count: int32(n),
	}, err
}

func (s *SlaveService) Search(ctx context.Context, request *Request) (*Result, error) {
	// 调用索引服务搜索
	result, err := s.Index.Search(request.Query, request.OnFlag, request.OffFlag, request.OrFlags)
	if err != nil {
		return nil, err
	}

	return &Result{
		Results: result,
	}, nil
}

func (s *SlaveService) Count(ctx context.Context, request *CountRequest) (*ResCount, error) {
	// 调用索引服务计数
	return &ResCount{
		Count: int32(s.Index.Total()),
	}, nil
}

func (s *SlaveService) ExecuteTask(ctx context.Context, request *TaskRequest) (*TaskResponse, error) {
	// 检查请求是否为空
	if request == nil {
		return &TaskResponse{
			Success:      false,
			ErrorMessage: "请求为空",
		}, nil
	}

	// 解析任务数据
	var taskData map[string]interface{}
	err := json.Unmarshal(request.TaskData, &taskData)
	if err != nil {
		return &TaskResponse{
			TaskId:       request.TaskId,
			Success:      false,
			ErrorMessage: fmt.Sprintf("解析任务数据失败: %v", err),
		}, nil
	}

	// 检查多表服务是否初始化
	if s.multiTableService == nil {
		return &TaskResponse{
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
					log.Warning("优化倒排索引失败: %v", optErr)
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
	response := &TaskResponse{
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
func (s *SlaveService) reportTaskResult(response *TaskResponse) {
	// 检查主节点连接
	s.masterMutex.RLock()
	masterConn := s.masterConn
	s.masterMutex.RUnlock()

	if masterConn == nil {
		log.Error("上报任务结果失败: 没有主节点连接")
		return
	}

	// 创建客户端
	client := NewIndexServiceClient(masterConn)

	// 创建上下文，设置超时
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 上报任务结果
	_, err := client.ReportTaskResult(ctx, response)
	if err != nil {
		log.Error("上报任务结果失败: %v", err)
	}
}

// ReportTaskResult 接收任务结果上报（主服务调用）
func (s *SlaveService) ReportTaskResult(ctx context.Context, response *TaskResponse) (*ResCount, error) {
	return nil, fmt.Errorf("从服务不支持接收任务结果上报")
}
