package server

import (
	"context"
	"encoding/json"
	"fmt"
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
	stop        bool

	// 主节点相关
	masterEndpoint string
	masterConn     *grpc.ClientConn
	masterMutex    sync.RWMutex

	// 任务执行相关
	taskExecutor *scheduler.TaskPoolManager

	// 多表搜索服务
	multiTableService *search.MultiTableSearchService
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
func NewSlaveService(endpoints []EndPoint, serviceName string, port int) (*SlaveService, error) {
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
		client:       client,
		serviceName:  serviceName,
		localhost:    localhost,
		taskExecutor: taskExecutor,
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

// Start 启动从服务
func (s *SlaveService) Start() error {
	// 注册服务
	err := s.registerService()
	if err != nil {
		return err
	}

	// 监听主节点
	go s.watchMaster()

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
		for !s.stop {
			<-ch
			// 租约保持成功
		}
	}()

	return nil
}

// watchMaster 监听主节点
func (s *SlaveService) watchMaster() {
	for !s.stop {
		// 获取主节点
		resp, err := s.client.Get(context.Background(), "/election/"+s.serviceName, clientv3.WithPrefix())
		if err != nil {
			log.Error("获取主节点失败: %v", err)
			time.Sleep(3 * time.Second)
			continue
		}

		if len(resp.Kvs) == 0 {
			log.Warning("没有主节点")
			time.Sleep(3 * time.Second)
			continue
		}

		// 获取主节点地址
		masterEndpoint := string(resp.Kvs[0].Value)

		s.masterMutex.Lock()
		if masterEndpoint != s.masterEndpoint {
			// 主节点变更
			log.Info("主节点变更为: %s", masterEndpoint)
			s.masterEndpoint = masterEndpoint

			// 连接主节点
			s.connectMaster()
		}
		s.masterMutex.Unlock()

		// 监听主节点变化
		watchChan := s.client.Watch(context.Background(), "/election/"+s.serviceName, clientv3.WithPrefix())
		for resp := range watchChan {
			for _, ev := range resp.Events {
				if ev.Type == clientv3.EventTypePut {
					// 主节点变更
					masterEndpoint := string(ev.Kv.Value)

					s.masterMutex.Lock()
					if masterEndpoint != s.masterEndpoint {
						log.Info("主节点变更为: %s", masterEndpoint)
						s.masterEndpoint = masterEndpoint

						// 连接主节点
						s.connectMaster()
					}
					s.masterMutex.Unlock()
				}
			}
		}
	}
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

// Stop 停止服务
func (s *SlaveService) Stop() {
	s.stop = true

	// 撤销租约
	_, err := s.client.Revoke(context.Background(), s.leaseID)
	if err != nil {
		log.Error("撤销租约失败: %v", err)
	}

	// 关闭主节点连接
	s.masterMutex.Lock()
	if s.masterConn != nil {
		s.masterConn.Close()
		s.masterConn = nil
	}
	s.masterMutex.Unlock()

	// 关闭客户端
	if s.client != nil {
		s.client.Close()
	}

	// 停止任务执行器
	if s.taskExecutor != nil {
		s.taskExecutor.Stop()
	}

	// 关闭索引
	if s.Index != nil {
		s.Index.Close()
	}

	// 关闭多表搜索服务
	if s.multiTableService != nil {
		// 遍历所有表并关闭
		tables, err := s.multiTableService.ListTables()
		if err == nil {
			for _, tableName := range tables {
				s.multiTableService.CloseTable(tableName)
			}
		}
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
			err = table.VectorDB.SaveToFile()

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
