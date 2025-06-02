package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
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
	stop        bool
	connPool    sync.Map // 连接池

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
func NewMasterService(endpoints []EndPoint, serviceName string, localhost string, httpPort int) (*MasterService, error) {
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
		healthCheckInterval: 5 * time.Second,
		tasks:               make(map[string]*TaskInfo),
		httpServerPort:      httpPort,
	}, nil
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

// Start 启动主服务
func (m *MasterService) Start() error {
	// 尝试成为主节点
	go m.campaignMaster()

	// 注册服务
	err := m.registerService()
	if err != nil {
		return err
	}

	// 启动健康检查
	go m.healthCheck()

	// 启动任务结果清理
	go m.cleanupTaskResults()
	// 启动 HTTP 服务器（只有主节点才会启动）
	go func() {
		for !m.stop {
			// 检查是否是主节点
			m.masterMutex.RLock()
			isMaster := m.isMaster
			m.masterMutex.RUnlock()

			if isMaster {
				// 如果是主节点且 HTTP 服务器未启动，则启动 HTTP 服务器
				if m.httpServer == nil {
					err := m.startHTTPServer()
					if err != nil {
						log.Error("启动 HTTP 服务器失败: %v", err)
					}
				}
			} else {
				// 如果不是主节点但 HTTP 服务器已启动，则关闭 HTTP 服务器
				if m.httpServer != nil {
					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
					m.httpServer.Shutdown(ctx)
					cancel()
					m.httpServer = nil
				}
			}

			// 每隔一段时间检查一次
			time.Sleep(5 * time.Second)
		}
	}()
	return nil
}

// campaignMaster 竞选主节点
func (m *MasterService) campaignMaster() {
	for !m.stop {
		// 尝试成为主节点
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		err := m.election.Campaign(ctx, m.localhost)
		cancel()

		if err != nil {
			log.Error("竞选主节点失败: %v", err)
			time.Sleep(3 * time.Second)
			continue
		}

		// 成为主节点
		m.masterMutex.Lock()
		m.isMaster = true
		m.masterMutex.Unlock()

		log.Info("成为主节点: %s", m.localhost)

		// 监听主节点变化
		ch := m.election.Observe(context.Background())
		for resp := range ch {
			if string(resp.Kvs[0].Value) != m.localhost {
				log.Info("主节点变更为: %s", string(resp.Kvs[0].Value))
				m.masterMutex.Lock()
				m.isMaster = false
				m.masterMutex.Unlock()
				break
			}
		}

		// 如果不再是主节点，重新竞选
		log.Info("不再是主节点，重新竞选")
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

	go func() {
		for !m.stop {
			<-ch
			// 租约保持成功
		}
	}()

	return nil
}

// healthCheck 健康检查
func (m *MasterService) healthCheck() {
	for !m.stop {
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
	for !m.stop {
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

// Stop 停止服务
func (m *MasterService) Stop() {
	m.stop = true
	// 关闭 HTTP 服务器
	if m.httpServer != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		m.httpServer.Shutdown(ctx)
		cancel()
	}
	// 撤销租约
	_, err := m.client.Revoke(context.Background(), m.leaseID)
	if err != nil {
		log.Error("撤销租约失败: %v", err)
	}

	// 关闭会话
	m.session.Close()

	// 关闭客户端
	m.client.Close()

	// 关闭连接池
	m.connPool.Range(func(key, value interface{}) bool {
		conn := value.(*grpc.ClientConn)
		conn.Close()
		return true
	})

	// 停止任务调度器
	m.taskScheduler.Stop()
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
