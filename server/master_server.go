package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"seetaSearch/library/pool"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"

	"seetaSearch/library/log"
	"seetaSearch/messages"
	"seetaSearch/scheduler"
)

// TaskResult 任务结果结构体
type TaskResult struct {
	TaskID       string
	Success      bool
	ResultData   []byte
	ErrorMessage string
	SlaveIP      string
	Timestamp    time.Time
}

// TaskInfo 任务信息结构体
type TaskInfo struct {
	TaskID      string
	Name        string
	Params      map[string]interface{}
	StartTime   time.Time
	Timeout     time.Duration
	Slaves      []string
	Results     map[string]*TaskResponse
	ResultsLock sync.RWMutex
}

// MasterService 主服务结构体
type MasterService struct {
	// etcd 相关
	client      *clientv3.Client
	session     *concurrency.Session
	election    *concurrency.Election
	leaseID     clientv3.LeaseID
	isMaster    bool
	masterMutex sync.RWMutex

	// 服务相关
	serviceName string
	localhost   string
	stopCh      chan struct{} // 修改 stop 为 stopCh
	connPool    sync.Map      // 连接池

	// 任务调度相关
	taskScheduler *scheduler.TaskPoolManager
	taskResults   sync.Map // 存储任务结果
	taskTimeout   time.Duration

	// 健康检查相关
	healthCheckInterval time.Duration
	slaveStatus         sync.Map // 存储从节点状态
	slaveLoad           sync.Map // 存储从节点负载

	// 任务管理相关
	tasks      map[string]*TaskInfo
	tasksMutex sync.RWMutex

	// HTTP 服务器相关
	httpServer     *http.Server
	httpServerPort int

	appCtx context.Context // 用于传递应用的全局上下文
}

func (m *MasterService) HealthCheck(ctx context.Context, request *HealthCheckRequest) (*HealthCheckResponse, error) {
	//TODO implement me
	panic("implement me")
}

// startHTTPServer 启动 HTTP 服务器
func (m *MasterService) startHTTPServer() error {
	// 检查是否是主节点
	m.masterMutex.RLock()
	isMaster := m.isMaster
	m.masterMutex.RUnlock()

	if !isMaster {
		return errors.New("当前节点不是主节点")
	}
	if m.httpServer != nil {
		log.Info("HTTP server already running.")
		return nil
	}
	// 创建 HTTP 服务器
	mux := http.NewServeMux()

	// 注册路由
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("SeetaSearch Master Node"))
	})

	// 添加文档接口
	mux.HandleFunc("/api/addDoc", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 解析请求体
		doc := &messages.Document{}
		if err := json.NewDecoder(r.Body).Decode(doc); err != nil {
			http.Error(w, "Invalid request payload: "+err.Error(), http.StatusBadRequest)
			return
		}

		// 调用 gRPC 方法
		resp, err := m.AddDoc(r.Context(), doc)
		if err != nil {
			http.Error(w, "Failed to add document: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// 返回结果
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"count":   resp.Count,
		})
	})

	// 删除文档接口
	mux.HandleFunc("/api/delDoc", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete && r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 获取文档 ID
		var docID string
		if r.Method == http.MethodDelete {
			docID = r.URL.Query().Get("id")
		} else {
			// 解析请求体
			var reqBody struct {
				ID string `json:"id"`
			}
			if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
				http.Error(w, "Invalid request payload: "+err.Error(), http.StatusBadRequest)
				return
			}
			docID = reqBody.ID
		}

		if docID == "" {
			http.Error(w, "Missing document ID", http.StatusBadRequest)
			return
		}

		// 调用 gRPC 方法
		resp, err := m.DelDoc(r.Context(), &DocId{Id: docID})
		if err != nil {
			http.Error(w, "Failed to delete document: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// 返回结果
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"count":   resp.Count,
		})
	})

	// 搜索接口
	mux.HandleFunc("/api/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 解析请求体
		var searchReq struct {
			Query   messages.TermQuery `json:"query"`
			OnFlag  uint64             `json:"onFlag"`
			OffFlag uint64             `json:"offFlag"`
			OrFlags []uint64           `json:"orFlags"`
		}
		if err := json.NewDecoder(r.Body).Decode(&searchReq); err != nil {
			http.Error(w, "Invalid request payload: "+err.Error(), http.StatusBadRequest)
			return
		}

		// 调用 gRPC 方法
		resp, err := m.Search(r.Context(), &Request{
			Query:   &searchReq.Query,
			OnFlag:  searchReq.OnFlag,
			OffFlag: searchReq.OffFlag,
			OrFlags: searchReq.OrFlags,
		})
		if err != nil {
			http.Error(w, "Failed to search: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// 返回结果
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"results": resp.Results,
		})
	})

	// 计数接口
	mux.HandleFunc("/api/count", func(w http.ResponseWriter, r *http.Request) {
		// 调用 gRPC 方法
		resp, err := m.Count(r.Context(), &CountRequest{})
		if err != nil {
			http.Error(w, "Failed to count: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// 返回结果
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"count":   resp.Count,
		})
	})

	// 创建表接口
	mux.HandleFunc("/api/createTable", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 解析请求体
		var req CreateTableRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request payload: "+err.Error(), http.StatusBadRequest)
			return
		}

		// 调用 gRPC 方法
		resp, err := m.CreateTable(r.Context(), &req)
		if err != nil {
			http.Error(w, "Failed to create table: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// 返回结果
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"count":   resp.Count,
		})
	})

	// 删除表接口
	mux.HandleFunc("/api/deleteTable", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete && r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 获取表名
		var tableName string
		if r.Method == http.MethodDelete {
			tableName = r.URL.Query().Get("name")
		} else {
			// 解析请求体
			var reqBody struct {
				Name string `json:"name"`
			}
			if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
				http.Error(w, "Invalid request payload: "+err.Error(), http.StatusBadRequest)
				return
			}
			tableName = reqBody.Name
		}

		if tableName == "" {
			http.Error(w, "Missing table name", http.StatusBadRequest)
			return
		}

		// 调用 gRPC 方法
		resp, err := m.DeleteTable(r.Context(), &TableRequest{TableName: tableName})
		if err != nil {
			http.Error(w, "Failed to delete table: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// 返回结果
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"count":   resp.Count,
		})
	})

	// 添加文档到表接口
	mux.HandleFunc("/api/addDocToTable", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 解析请求体
		var req AddDocumentRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request payload: "+err.Error(), http.StatusBadRequest)
			return
		}

		// 调用 gRPC 方法
		resp, err := m.AddDocumentToTable(r.Context(), &req)
		if err != nil {
			http.Error(w, "Failed to add document to table: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// 返回结果
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"count":   resp.Count,
		})
	})

	// 从表中删除文档接口
	mux.HandleFunc("/api/delDocFromTable", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete && r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 解析请求体
		var req DeleteDocumentRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request payload: "+err.Error(), http.StatusBadRequest)
			return
		}

		// 调用 gRPC 方法
		resp, err := m.DeleteDocumentFromTable(r.Context(), &req)
		if err != nil {
			http.Error(w, "Failed to delete document from table: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// 返回结果
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"count":   resp.Count,
		})
	})

	// 表搜索接口
	mux.HandleFunc("/api/searchTable", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 解析请求体
		var req SearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request payload: "+err.Error(), http.StatusBadRequest)
			return
		}

		// 调用 gRPC 方法
		resp, err := m.SearchTable(r.Context(), &req)
		if err != nil {
			http.Error(w, "Failed to search table: "+err.Error(), http.StatusInternalServerError)
			return
		}

		// 返回结果
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"results": resp.DocIds,
		})
	})

	// 启动 HTTP 服务器
	addr := fmt.Sprintf(":%d", m.httpServerPort)
	m.httpServer = &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	log.Info("HTTP 服务器启动，监听端口: %d", m.httpServerPort)
	go func() {
		if err := m.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("HTTP 服务器启动失败: %v", err)
		}
	}()

	return nil
}
func (m *MasterService) CreateTable(ctx context.Context, request *CreateTableRequest) (*ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (m *MasterService) DeleteTable(ctx context.Context, request *TableRequest) (*ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (m *MasterService) AddDocumentToTable(ctx context.Context, request *AddDocumentRequest) (*ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (m *MasterService) DeleteDocumentFromTable(ctx context.Context, request *DeleteDocumentRequest) (*ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (m *MasterService) SearchTable(ctx context.Context, request *SearchRequest) (*SearchResult, error) {
	//TODO implement me
	panic("implement me")
}

// NewMasterService 创建主服务实例
func NewMasterService(appCtx context.Context, endpoints []EndPoint, serviceName string, localhost string, httpPort int, taskTimeout time.Duration, healthCheckInterval time.Duration) (*MasterService, error) {
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

	// 创建会话
	session, err := concurrency.NewSession(client, concurrency.WithTTL(10))
	if err != nil {
		client.Close()
		return nil, fmt.Errorf("创建 etcd 会话失败: %v", err)
	}

	// 创建选举
	election := concurrency.NewElection(session, "/election/"+serviceName)

	// 创建任务调度器
	taskScheduler := scheduler.NewTaskPoolManager() // 设置合适的工作线程数

	return &MasterService{
		client:              client,
		session:             session,
		election:            election,
		serviceName:         serviceName,
		localhost:           localhost,
		taskScheduler:       taskScheduler,
		healthCheckInterval: healthCheckInterval,
		tasks:               make(map[string]*TaskInfo),
		taskTimeout:         taskTimeout,
		httpServerPort:      httpPort,
		appCtx:              appCtx,
	}, nil
}

// GetName 返回服务名称
func (m *MasterService) GetName() string {
	return "master_service"
}

// GetDependencies 返回服务依赖的任务名称列表
func (m *MasterService) GetDependencies() []string {
	return []string{"etcd_ready"} // 依赖 etcd 连接成功
}

// ScheduleTask 调度任务
func (m *MasterService) ScheduleTask(task scheduler.ScheduledTask) error {
	// 检查是否是主节点
	m.masterMutex.RLock()
	isMaster := m.isMaster
	m.masterMutex.RUnlock()

	if !isMaster {
		return errors.New("当前节点不是主节点")
	}

	// 获取可用的从节点
	slaves := m.getAvailableSlaves()
	if len(slaves) == 0 {
		return errors.New("没有可用的从节点")
	}

	// 创建任务信息
	taskID := fmt.Sprintf("%s-%d", task.GetName(), time.Now().UnixNano())
	taskInfo := &TaskInfo{
		TaskID:    taskID,
		Name:      task.GetName(),
		Params:    task.Params(),
		StartTime: time.Now(),
		Timeout:   task.Timeout(),
		Slaves:    make([]string, 0, len(slaves)),
		Results:   make(map[string]*TaskResponse), // 修改这里，使用TaskResponse而不是TaskResult
	}

	// 将任务分配给从节点
	for _, slave := range slaves {
		// 创建子任务
		subtask := task.Clone()
		subtask.SetID(fmt.Sprintf("%s-%s", taskID, slave.Ip))
		subtask.SetTarget(slave.Ip)

		// 提交子任务
		m.taskScheduler.Submit(subtask)

		// 记录任务分配信息
		taskInfo.Slaves = append(taskInfo.Slaves, slave.Ip)
	}

	// 保存任务信息
	m.tasksMutex.Lock()
	m.tasks[taskID] = taskInfo
	m.tasksMutex.Unlock()

	// 启动任务超时检查
	if taskInfo.Timeout > 0 {
		go m.checkTaskTimeout(taskID, taskInfo.Timeout)
	}

	return nil
}

// Start 启动 MasterService
func (m *MasterService) Start(ctx context.Context) error {
	log.Info("Starting MasterService...")

	// 尝试成为主节点
	go m.runElection(ctx) // 使用传递的上下文

	// 等待成为主节点或上下文取消
	select {
	case <-m.appCtx.Done(): // 使用 appCtx 检查应用是否关闭
		log.Info("MasterService startup cancelled as application is shutting down.")
		return m.appCtx.Err()
	case <-time.After(30 * time.Second): // 等待一段时间成为 master
		m.masterMutex.RLock()
		isMaster := m.isMaster
		m.masterMutex.RUnlock()
		if !isMaster {
			log.Warning("Failed to become master within timeout, proceeding with limited functionality or retrying based on election logic.")
			// 根据实际需求，这里可以返回错误，或者允许服务以非 master 模式启动（如果支持）
			// return errors.New("failed to become master within timeout")
		}
	}

	m.masterMutex.RLock()
	becameMaster := m.isMaster
	m.masterMutex.RUnlock()

	if becameMaster {
		log.Info("Successfully became master. Initializing master functionalities.")
		// 启动 HTTP 服务器 (仅当是主节点时)
		go func() {
			if err := m.startHTTPServer(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Error("Failed to start HTTP server: %v", err)
			}
		}()

		// 启动健康检查
		go m.startHealthChecks(ctx) // 使用传递的上下文

		// 注册服务到 etcd (如果需要，并且由选举逻辑处理)
		err := m.registerService()
		if err != nil {
			return err
		}
	} else {
		log.Info("Node is not master. Master functionalities (HTTP server, health checks) will not be started by this instance.")
	}

	log.Info("MasterService started.")

	// 监听停止信号
	go func() {
		select {
		case <-m.stopCh:
			log.Info("MasterService received stop signal.")
			// 执行清理操作
		case <-ctx.Done(): // 监听任务池传递的上下文
			log.Info("MasterService context cancelled.")
			m.Stop(context.Background()) // 调用 Stop 进行清理
		case <-m.appCtx.Done(): // 监听应用全局上下文
			log.Info("Application context cancelled, stopping MasterService.")
			m.Stop(context.Background()) // 调用 Stop 进行清理
		}
	}()

	return nil
}

// performHealthChecks 执行健康检查
func (m *MasterService) performHealthChecks() {
	// 获取从节点列表
	slaves := m.getSlavesFromEtcd() // 假设有一个方法从 etcd 获取从节点列表

	for _, slaveIP := range slaves {
		go func(ip string) {
			conn, err := m.getOrCreateGRPCConn(ip) // 获取或创建 gRPC 连接
			if err != nil {
				log.Warning("Failed to connect to slave %s for health check: %v", ip, err)
				m.slaveStatus.Store(ip, "unhealthy")
				return
			}

			client := NewIndexServiceClient(conn) // 假设 SlaveService 定义了 gRPC 服务
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			healthStatus, err := client.HealthCheck(ctx, &HealthCheckRequest{}) // 假设 HealthCheckRequest 是空的
			if err != nil {
				log.Warning("Health check failed for slave %s: %v", ip, err)
				m.slaveStatus.Store(ip, "unhealthy")
				// 考虑关闭不健康的连接
				conn.Close()
				m.connPool.Delete(ip)
				return
			}

			if healthStatus.Status == HealthCheckResponse_SERVING { // 假设 HealthCheckResponse 有 Status 字段
				m.slaveStatus.Store(ip, "healthy")
				// 更新从节点负载信息，假设 HealthCheckResponse 包含 LoadInfo
				// m.slaveLoad.Store(ip, healthStatus.LoadInfo)
			} else {
				m.slaveStatus.Store(ip, "unhealthy")
			}
			log.Trace("Health check for slave %s: %s", ip, healthStatus.Status)
		}(slaveIP)
	}
}

// getSlavesFromEtcd 从 etcd 获取从节点列表 (示例实现)
func (m *MasterService) getSlavesFromEtcd() []string {
	// 在实际应用中，这里会从 etcd 查询已注册的从节点
	// 为了示例，我们返回一个硬编码的列表或者从 m.connPool 的键中提取
	var slaves []string
	m.connPool.Range(func(key, value interface{}) bool {
		slaves = append(slaves, key.(string))
		return true
	})
	if len(slaves) == 0 {
		// 如果 connPool 为空，可以尝试从 etcd 获取初始列表
		// 例如: m.discoverSlaves()
	}
	return slaves
}

// getOrCreateGRPCConn 获取或创建到从节点的 gRPC 连接
func (m *MasterService) getOrCreateGRPCConn(slaveIP string) (*grpc.ClientConn, error) {
	if conn, ok := m.connPool.Load(slaveIP); ok {
		clientConn := conn.(*grpc.ClientConn)
		// 检查连接状态
		if clientConn.GetState() == connectivity.Ready || clientConn.GetState() == connectivity.Idle {
			return clientConn, nil
		}
		// 如果连接不是 Ready 或 Idle，则关闭并重新创建
		clientConn.Close()
		m.connPool.Delete(slaveIP)
	}

	// 创建新连接
	conn, err := grpc.Dial(slaveIP, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		return nil, err
	}
	m.connPool.Store(slaveIP, conn)
	return conn, nil
}

// startHealthChecks 启动健康检查
func (m *MasterService) startHealthChecks(ctx context.Context) {
	// ... (健康检查逻辑保持不变或根据需要调整，确保使用 ctx 进行取消)
	ticker := time.NewTicker(m.healthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.masterMutex.RLock()
			isMasterNode := m.isMaster
			m.masterMutex.RUnlock()
			if isMasterNode {
				log.Trace("Running health checks for slave nodes...") // 可以取消注释以进行调试
				m.performHealthChecks()
			}
		case <-ctx.Done():
			log.Info("Health check process cancelled by context.")
			return
		case <-m.stopCh:
			log.Info("Health check process stopped by service stop signal.")
			return
		}
	}
}

// GetTaskSpec 返回任务规格
func (m *MasterService) GetTaskSpec() *scheduler.TaskSpec {
	return &scheduler.TaskSpec{
		RetryCount: 3,                // 示例：重试3次
		Timeout:    30 * time.Second, // 示例：超时30秒
	}
}

func (m *MasterService) ToPoolTask() *pool.Queue {
	startWrapper := func() error {
		log.Info("ServiceTask for MasterService: %s is about to run Start()", m.GetName())
		// 与 SlaveService 类似，Start 需要 context.Context。
		// 我们使用 m.appCtx。
		err := m.Start(m.appCtx)
		if err != nil {
			log.Error("ServiceTask for MasterService: %s failed to Start: %v", m.GetName(), err)
		}
		return err
	}

	return pool.QueryInit(m.GetName(), startWrapper)
}

// SetTaskPoolManager 设置任务池管理器
func (m *MasterService) SetTaskPoolManager(manager *scheduler.TaskPoolManager) {
	m.taskScheduler = manager
}

// runElection 尝试成为主节点
func (m *MasterService) runElection(ctx context.Context) {
	// ... (选举逻辑保持不变或根据需要调整，确保使用 ctx 进行取消)
	// 在选举成功后，设置 m.isMaster = true，并可以启动主节点特定的服务
	// 例如: go m.startHTTPServer()
	//       go m.startHealthChecks()
	// 在失去 master 身份或服务停止时，设置 m.isMaster = false 并停止相关服务
	log.Info("Attempting to become master...")

	// 创建一个新的会话
	sess, err := concurrency.NewSession(m.client, concurrency.WithTTL(15)) // TTL 设置为15秒
	if err != nil {
		log.Error("Failed to create etcd session: %v", err)
		return
	}
	m.session = sess
	m.leaseID = sess.Lease()

	e := concurrency.NewElection(sess, "/seetasearch/master_election")
	m.election = e

	// 循环尝试获取领导权
	for {
		select {
		case <-ctx.Done(): // 检查外部上下文是否已取消
			log.Info("Election process cancelled by context.")
			m.resignLeadership()
			return
		case <-m.stopCh: // 检查服务是否已停止
			log.Info("Election process stopped by service stop signal.")
			m.resignLeadership()
			return
		default:
			if err := e.Campaign(ctx, m.localhost); err != nil {
				log.Warning("Error during election campaign: %v. Retrying in 5 seconds...", err)
				time.Sleep(5 * time.Second)
				continue
			}

			// 成为主节点
			log.Info("Successfully elected as master: %s", m.localhost)
			m.masterMutex.Lock()
			m.isMaster = true
			m.masterMutex.Unlock()

			// 主节点逻辑，例如启动特定服务
			// go m.startHTTPServer() // 确保这个方法是幂等的或者在之前没有启动
			// go m.startHealthChecks(ctx)

			// 监听会话是否结束，如果结束则重新选举
			select {
			case <-ctx.Done():
				log.Info("Master leadership cancelled by context.")
				m.resignLeadership()
				return
			case <-m.stopCh:
				log.Info("Master leadership stopped by service stop signal.")
				m.resignLeadership()
				return
			case <-sess.Done():
				log.Warning("Etcd session lost. Resigning leadership and attempting re-election.")
				m.masterMutex.Lock()
				m.isMaster = false
				m.masterMutex.Unlock()
				// 停止主节点特定的服务
				// if m.httpServer != nil { m.httpServer.Close(); m.httpServer = nil }
				// 重新进入选举循环
			}
		}
	}
}

func (m *MasterService) resignLeadership() {
	m.masterMutex.Lock()
	defer m.masterMutex.Unlock()
	if m.isMaster {
		log.Info("Resigning master leadership.")
		if m.election != nil {
			if err := m.election.Resign(context.Background()); err != nil {
				log.Warning("Error resigning leadership: %v", err)
			}
		}
		m.isMaster = false
		// 停止主节点特定的服务
		// if m.httpServer != nil { m.httpServer.Close(); m.httpServer = nil }
	}
	if m.session != nil {
		m.session.Close() // 关闭会话
		m.session = nil
	}
}

// campaignMaster 竞选主节点
func (m *MasterService) campaignMaster() {
	for {
		select {
		case <-m.stopCh: // 检查服务是否已停止
			log.Info("campaignMaster: Stop signal received, exiting campaign loop.")
			return
		default:
			// 尝试成为主节点
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			err := m.election.Campaign(ctx, m.localhost)
			cancel()

			if err != nil {
				log.Error("竞选主节点失败: %v", err)
				// 在重试前检查停止信号
				select {
				case <-time.After(3 * time.Second):
				case <-m.stopCh:
					log.Info("campaignMaster: Stop signal received during retry wait, exiting.")
					return
				}
				continue
			}

			// 成为主节点
			m.masterMutex.Lock()
			m.isMaster = true
			m.masterMutex.Unlock()

			log.Info("成为主节点: %s", m.localhost)

			// 监听主节点变化
			observeCtx, observeCancel := context.WithCancel(context.Background())
			go func() { // 在 goroutine 中监听，避免阻塞 campaignMaster 的主循环
				select {
				case <-m.stopCh:
					observeCancel() // 如果服务停止，也取消 observe
				case <-observeCtx.Done(): // 由外部取消，例如 resignLeadership
				}
			}()

			ch := m.election.Observe(observeCtx)
			keepLeadership := true
			for resp := range ch {
				if len(resp.Kvs) == 0 || string(resp.Kvs[0].Value) != m.localhost {
					log.Info("主节点变更为其他节点或键被删除，当前节点 %s 不再是主节点", m.localhost)
					m.masterMutex.Lock()
					m.isMaster = false
					m.masterMutex.Unlock()
					keepLeadership = false
					observeCancel() // 停止观察
					break
				}
			}
			observeCancel() // 确保 observeCancel 在循环结束后被调用

			if !keepLeadership {
				log.Info("不再是主节点，重新竞选")
				// 不需要 continue，循环会自动进行下一次迭代
			} else {
				// 如果是因为 observeCtx 被取消 (例如 m.stopCh 关闭) 而退出 observe 循环，则这里也应该退出 campaignMaster
				select {
				case <-m.stopCh:
					log.Info("campaignMaster: Stop signal detected after leadership observation, exiting.")
					return
				default:
					// 如果通道未关闭，但领导权仍然保持（例如，etcd连接问题导致Observe通道关闭），则重新竞选
					log.Info("Leadership observation ended, but still master or stop signal not received. Re-evaluating leadership.")
				}
			}
		}
	}
}

// registerService 注册服务
func (m *MasterService) registerService() error {
	// 创建租约
	resp, err := m.client.Grant(context.Background(), 10)
	if err != nil {
		return fmt.Errorf("创建租约失败: %v", err)
	}
	m.leaseID = resp.ID

	// 注册服务
	key := fmt.Sprintf("%s/%s/%s", ServiceRootPath, m.serviceName, m.localhost)
	_, err = m.client.Put(context.Background(), key, m.localhost, clientv3.WithLease(m.leaseID))
	if err != nil {
		return fmt.Errorf("注册服务失败: %v", err)
	}

	// 保持租约
	ch, err := m.client.KeepAlive(context.Background(), m.leaseID)
	if err != nil {
		return fmt.Errorf("保持租约失败: %v", err)
	}

	// 监听停止信号
	go func() {
		select {
		case <-m.stopCh:
			<-ch
		case <-m.appCtx.Done(): // 监听应用全局上下文
			log.Info("Application context cancelled, stopping MasterService.")
			m.Stop(context.Background()) // 调用 Stop 进行清理
		}
	}()
	return nil
}
func (m *MasterService) IsMaster() bool {
	m.masterMutex.RLock()
	defer m.masterMutex.RUnlock()
	return m.isMaster
}

// healthCheck 健康检查
func (m *MasterService) healthCheck() {
	for {
		select {
		case <-m.stopCh:
			log.Info("healthCheck: Stop signal received, exiting health check loop.")
			return
		default:
			// 检查是否是主节点
			m.masterMutex.RLock()
			isMaster := m.isMaster
			m.masterMutex.RUnlock()

			if isMaster {
				// 获取所有从节点
				slaves := m.getSlaveEndpoints()

				// 检查每个从节点的健康状态
				for _, slave := range slaves {
					go m.checkSlaveHealth(slave)
				}
			}

			time.Sleep(m.healthCheckInterval)
		}
	}
}

// getSlaveEndpoints 获取所有从节点
func (m *MasterService) getSlaveEndpoints() []EndPoint {
	prefix := fmt.Sprintf("%s/%s/", ServiceRootPath, m.serviceName)
	resp, err := m.client.Get(context.Background(), prefix, clientv3.WithPrefix())
	if err != nil {
		log.Error("获取从节点失败: %v", err)
		return nil
	}

	var endpoints []EndPoint
	for _, kv := range resp.Kvs {
		// 排除自己
		if string(kv.Value) != m.localhost {
			endpoints = append(endpoints, EndPoint{Ip: string(kv.Value)})
		}
	}

	return endpoints
}

// checkSlaveHealth 检查从节点健康状态
func (m *MasterService) checkSlaveHealth(slave EndPoint) {
	// 获取连接
	conn := m.getGrpcConn(slave)
	if conn == nil {
		// 连接失败，标记为不可用
		m.slaveStatus.Store(slave.Ip, false)
		log.Error("从节点不可用: %s", slave.Ip)
		return
	}

	// 创建客户端
	client := NewIndexServiceClient(conn)

	// 发送健康检查请求（使用 Count 方法作为健康检查）
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// 记录请求开始时间，用于计算响应时间
	startTime := time.Now()

	resp, err := client.Count(ctx, &CountRequest{})
	if err != nil {
		// 健康检查失败，标记为不可用
		m.slaveStatus.Store(slave.Ip, false)
		log.Error("从节点健康检查失败: %s, %v", slave.Ip, err)
		return
	}

	// 计算响应时间
	responseTime := time.Since(startTime)

	// 健康检查成功，标记为可用
	m.slaveStatus.Store(slave.Ip, true)

	// 更新从节点负载信息
	m.slaveLoad.Store(slave.Ip, struct {
		DocCount     int32
		ResponseTime time.Duration
		LastCheck    time.Time
	}{
		DocCount:     resp.Count,
		ResponseTime: responseTime,
		LastCheck:    time.Now(),
	})
}

// getGrpcConn 获取 gRPC 连接
func (m *MasterService) getGrpcConn(point EndPoint) *grpc.ClientConn {
	v, ok := m.connPool.Load(point.Ip)
	if ok {
		conn := v.(*grpc.ClientConn)
		state := conn.GetState()
		if state != connectivity.TransientFailure && state != connectivity.Shutdown {
			return conn
		}
		conn.Close()
		m.connPool.Delete(point.Ip)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	grpcConn, err := grpc.DialContext(ctx, point.Ip, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		log.Error("连接从节点失败: %s, %v", point.Ip, err)
		return nil
	}

	m.connPool.Store(point.Ip, grpcConn)
	return grpcConn
}

// getBestSlave 根据负载均衡策略选择最佳从节点
func (m *MasterService) getBestSlave() (EndPoint, error) {
	availableSlaves := m.getAvailableSlaves()
	if len(availableSlaves) == 0 {
		return EndPoint{}, errors.New("没有可用的从节点")
	}

	// 实现加权轮询负载均衡
	type slaveScore struct {
		endpoint EndPoint
		score    float64
	}

	var scores []slaveScore
	for _, slave := range availableSlaves {
		// 获取从节点负载信息
		v, ok := m.slaveLoad.Load(slave.Ip)
		if !ok {
			// 如果没有负载信息，给一个默认分数
			scores = append(scores, slaveScore{endpoint: slave, score: 1.0})
			continue
		}

		loadInfo := v.(struct {
			DocCount     int32
			ResponseTime time.Duration
			LastCheck    time.Time
		})

		// 计算分数（响应时间越短，文档数越少，分数越高）
		responseScore := 1.0 / (float64(loadInfo.ResponseTime.Milliseconds()) + 1.0)
		docCountScore := 1.0 / (float64(loadInfo.DocCount) + 1.0)

		// 综合分数
		score := responseScore*0.7 + docCountScore*0.3
		scores = append(scores, slaveScore{endpoint: slave, score: score})
	}

	// 按分数排序
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// 返回得分最高的从节点
	return scores[0].endpoint, nil
}

// cleanupTaskResults 定期清理过期的任务结果
func (m *MasterService) cleanupTaskResults() {
	cleanupInterval := 5 * time.Minute
	for {
		select {
		case <-m.stopCh:
			return
		default:
			time.Sleep(cleanupInterval)
			// 清理超过 1 小时的任务结果
			expireTime := time.Now().Add(-1 * time.Hour)
			m.taskResults.Range(func(key, value interface{}) bool {
				result := value.(TaskResult)
				if result.Timestamp.Before(expireTime) {
					m.taskResults.Delete(key)
				}
				return true
			})
		}
	}
}

// Stop 停止 MasterService
func (m *MasterService) Stop(ctx context.Context) error {
	log.Info("Stopping MasterService...")
	close(m.stopCh) // 发送停止信号

	// 关闭 HTTP 服务器
	if m.httpServer != nil {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := m.httpServer.Shutdown(shutdownCtx); err != nil {
			log.Warning("HTTP server shutdown error: %v", err)
		}
	}

	// 辞去主节点
	if m.election != nil {
		m.masterMutex.RLock()
		isMaster := m.isMaster
		m.masterMutex.RUnlock()
		if isMaster {
			if err := m.election.Resign(context.Background()); err != nil {
				log.Warning("Failed to resign master leadership: %v", err)
			}
		}
	}

	// 关闭 etcd 会话和客户端
	if m.session != nil {
		if err := m.session.Close(); err != nil {
			log.Warning("Failed to close etcd session: %v", err)
		}
	}
	if m.client != nil {
		if err := m.client.Close(); err != nil {
			log.Warning("Failed to close etcd client: %v", err)
		}
	}

	// 关闭连接池中的所有连接
	m.connPool.Range(func(key, value interface{}) bool {
		if conn, ok := value.(*grpc.ClientConn); ok {
			conn.Close()
		}
		return true
	})

	log.Info("MasterService stopped.")
	return nil
}

// 实现 IndexServiceServer 接口

func (m *MasterService) DelDoc(ctx context.Context, docId *DocId) (*ResCount, error) {
	// 检查是否是主节点
	m.masterMutex.RLock()
	isMaster := m.isMaster
	m.masterMutex.RUnlock()

	if !isMaster {
		return nil, errors.New("当前节点不是主节点")
	}

	// 获取可用的从节点
	slaves := m.getAvailableSlaves()
	if len(slaves) == 0 {
		return nil, errors.New("没有可用的从节点")
	}

	// 并发删除文档
	var totalCount int32
	var wg sync.WaitGroup
	wg.Add(len(slaves))

	for _, slave := range slaves {
		go func(endpoint EndPoint) {
			defer wg.Done()

			// 获取连接
			conn := m.getGrpcConn(endpoint)
			if conn == nil {
				return
			}

			// 创建客户端
			client := NewIndexServiceClient(conn)

			// 发送删除请求
			resp, err := client.DelDoc(ctx, docId)
			if err != nil {
				log.Error("从节点删除文档失败: %s, %v", endpoint.Ip, err)
				return
			}

			// 累加删除数量
			atomic.AddInt32(&totalCount, resp.Count)
		}(slave)
	}

	// 等待所有请求完成
	wg.Wait()

	return &ResCount{Count: totalCount}, nil
}

func (m *MasterService) AddDoc(ctx context.Context, doc *messages.Document) (*ResCount, error) {
	// 检查是否是主节点
	m.masterMutex.RLock()
	isMaster := m.isMaster
	m.masterMutex.RUnlock()

	if !isMaster {
		return nil, errors.New("当前节点不是主节点")
	}

	// 获取最佳从节点
	slave, err := m.getBestSlave()
	if err != nil {
		return nil, err
	}

	// 获取连接
	conn := m.getGrpcConn(slave)
	if conn == nil {
		return nil, fmt.Errorf("无法连接到从节点: %s", slave.Ip)
	}

	// 创建客户端
	client := NewIndexServiceClient(conn)

	// 发送添加请求
	return client.AddDoc(ctx, doc)
}

func (m *MasterService) Search(ctx context.Context, request *Request) (*Result, error) {
	// 检查是否是主节点
	m.masterMutex.RLock()
	isMaster := m.isMaster
	m.masterMutex.RUnlock()

	if !isMaster {
		return nil, errors.New("当前节点不是主节点")
	}

	// 获取可用的从节点
	slaves := m.getAvailableSlaves()
	if len(slaves) == 0 {
		return nil, errors.New("没有可用的从节点")
	}

	// 并发搜索
	resultChan := make(chan *messages.Document, 1000)
	var wg sync.WaitGroup
	wg.Add(len(slaves))

	for _, slave := range slaves {
		go func(endpoint EndPoint) {
			defer wg.Done()

			// 获取连接
			conn := m.getGrpcConn(endpoint)
			if conn == nil {
				return
			}

			// 创建客户端
			client := NewIndexServiceClient(conn)

			// 发送搜索请求
			resp, err := client.Search(ctx, request)
			if err != nil {
				log.Error("从节点搜索失败: %s, %v", endpoint.Ip, err)
				return
			}

			// 收集结果
			for _, doc := range resp.Results {
				resultChan <- doc
			}
		}(slave)
	}

	// 等待所有请求完成
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 收集结果
	var results []*messages.Document
	for doc := range resultChan {
		results = append(results, doc)
	}

	// 对结果进行去重
	deduped := m.deduplicateResults(results)

	return &Result{Results: deduped}, nil
}

// deduplicateResults 对搜索结果进行去重
func (m *MasterService) deduplicateResults(docs []*messages.Document) []*messages.Document {
	deduped := make([]*messages.Document, 0, len(docs))
	idMap := make(map[string]bool)

	for _, doc := range docs {
		if _, exists := idMap[doc.Id]; !exists {
			deduped = append(deduped, doc)
			idMap[doc.Id] = true
		}
	}

	return deduped
}

func (m *MasterService) Count(ctx context.Context, request *CountRequest) (*ResCount, error) {
	// 检查是否是主节点
	m.masterMutex.RLock()
	isMaster := m.isMaster
	m.masterMutex.RUnlock()

	if !isMaster {
		return nil, errors.New("当前节点不是主节点")
	}

	// 获取可用的从节点
	slaves := m.getAvailableSlaves()
	if len(slaves) == 0 {
		return nil, errors.New("没有可用的从节点")
	}

	// 并发计数
	var totalCount int32
	var wg sync.WaitGroup
	wg.Add(len(slaves))

	for _, slave := range slaves {
		go func(endpoint EndPoint) {
			defer wg.Done()

			// 获取连接
			conn := m.getGrpcConn(endpoint)
			if conn == nil {
				return
			}

			// 创建客户端
			client := NewIndexServiceClient(conn)

			// 发送计数请求
			resp, err := client.Count(ctx, request)
			if err != nil {
				log.Error("从节点计数失败: %s, %v", endpoint.Ip, err)
				return
			}

			// 累加计数
			atomic.AddInt32(&totalCount, resp.Count)
		}(slave)
	}

	// 等待所有请求完成
	wg.Wait()

	return &ResCount{Count: totalCount}, nil
}

// ExecuteTask 执行任务（从服务调用）
func (m *MasterService) ExecuteTask(ctx context.Context, request *TaskRequest) (*TaskResponse, error) {
	return nil, errors.New("主服务不支持执行任务")
}

// ReportTaskResult 接收从节点的任务执行结果
func (m *MasterService) ReportTaskResult(ctx context.Context, result *TaskResponse) (*ResCount, error) {
	// 检查是否是主节点
	m.masterMutex.RLock()
	isMaster := m.isMaster
	m.masterMutex.RUnlock()

	if !isMaster {
		return nil, errors.New("当前节点不是主节点")
	}

	// 提取任务ID
	parts := strings.Split(result.TaskId, "-")
	if len(parts) < 2 {
		return nil, fmt.Errorf("无效的任务ID格式: %s", result.TaskId)
	}
	taskID := strings.Join(parts[:len(parts)-1], "-")

	// 查找任务信息
	m.tasksMutex.RLock()
	taskInfo, exists := m.tasks[taskID]
	m.tasksMutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("任务不存在: %s", taskID)
	}

	// 保存任务结果
	taskInfo.ResultsLock.Lock()
	taskInfo.Results[result.SlaveId] = result
	taskInfo.ResultsLock.Unlock()

	// 检查任务是否完成
	m.checkTaskCompletion(taskID)

	return &ResCount{Count: 1}, nil
}

// checkTaskTimeout 检查任务是否超时
func (m *MasterService) checkTaskTimeout(taskID string, timeout time.Duration) {
	time.Sleep(timeout)

	m.tasksMutex.RLock()
	taskInfo, exists := m.tasks[taskID]
	m.tasksMutex.RUnlock()

	if !exists {
		return
	}

	// 检查是否所有从节点都已报告结果
	taskInfo.ResultsLock.RLock()
	pendingSlaves := make([]string, 0)
	for _, slaveID := range taskInfo.Slaves {
		if _, reported := taskInfo.Results[slaveID]; !reported {
			pendingSlaves = append(pendingSlaves, slaveID)
		}
	}
	taskInfo.ResultsLock.RUnlock()

	// 对于未报告结果的从节点，标记为超时
	for _, slaveID := range pendingSlaves {
		taskInfo.ResultsLock.Lock()
		taskInfo.Results[slaveID] = &TaskResponse{
			TaskId:       taskID + "-" + slaveID,
			SlaveId:      slaveID,
			Success:      false,
			ErrorMessage: "任务执行超时",
			EndTime:      time.Now().Unix(),
		}
		taskInfo.ResultsLock.Unlock()
	}

	// 检查任务是否完成
	m.checkTaskCompletion(taskID)
}

// processTaskResults 处理任务结果
func (m *MasterService) processTaskResults(taskID string) {
	m.tasksMutex.RLock()
	taskInfo, exists := m.tasks[taskID]
	m.tasksMutex.RUnlock()

	if !exists {
		return
	}

	// 汇总任务结果
	taskInfo.ResultsLock.RLock()
	successCount := 0
	failureCount := 0
	for _, result := range taskInfo.Results {
		if result.Success {
			successCount++
		} else {
			failureCount++
		}
	}
	taskInfo.ResultsLock.RUnlock()

	// 记录任务执行情况
	log.Info("任务 %s 执行完成: 成功 %d, 失败 %d", taskID, successCount, failureCount)

	// 清理任务信息
	m.tasksMutex.Lock()
	delete(m.tasks, taskID)
	m.tasksMutex.Unlock()
}

// 改进的负载均衡策略
func (m *MasterService) getAvailableSlaves() []EndPoint {
	slaves := m.getSlaveEndpoints()
	var availableSlaves []EndPoint

	// 统计每个从节点的连接数
	connCounts := make(map[string]int)
	m.connPool.Range(func(key, value interface{}) bool {
		slaveIP := key.(string)
		connCounts[slaveIP]++
		return true
	})

	for _, slave := range slaves {
		v, ok := m.slaveStatus.Load(slave.Ip)
		if ok && v.(bool) {
			availableSlaves = append(availableSlaves, slave)
		}
	}

	// 根据连接数排序，优先选择连接数少的从节点
	sort.Slice(availableSlaves, func(i, j int) bool {
		return connCounts[availableSlaves[i].Ip] < connCounts[availableSlaves[j].Ip]
	})

	return availableSlaves
}

// checkTaskCompletion 检查任务是否完成
func (m *MasterService) checkTaskCompletion(taskID string) {
	m.tasksMutex.RLock()
	taskInfo, exists := m.tasks[taskID]
	m.tasksMutex.RUnlock()

	if !exists {
		return
	}

	// 检查是否所有从节点都已报告结果
	taskInfo.ResultsLock.RLock()
	allReported := true
	for _, slaveID := range taskInfo.Slaves {
		if _, reported := taskInfo.Results[slaveID]; !reported {
			allReported = false
			break
		}
	}
	taskInfo.ResultsLock.RUnlock()

	if allReported {
		// 所有从节点都已报告结果，处理任务完成逻辑
		go m.processTaskResults(taskID)
	}
}

// GetTaskResult 获取任务结果
func (m *MasterService) GetTaskResult(taskID string) (*TaskResult, bool) {
	value, ok := m.taskResults.Load(taskID)
	if !ok {
		return nil, false
	}

	result := value.(TaskResult)
	return &result, true
}
