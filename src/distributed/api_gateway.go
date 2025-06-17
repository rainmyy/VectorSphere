package distributed

import (
	"VectorSphere/src/library/logger"
	"VectorSphere/src/messages"
	"VectorSphere/src/server"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	_ "strconv"
	"sync"
	"time"
)

// APIGateway HTTP API网关
type APIGateway struct {
	distributedManager *DistributedManager
	communicationSvc   *CommunicationService
	serviceDiscovery   *ServiceDiscovery
	httpServer         *http.Server
	port               int
	mutex              sync.RWMutex

	// 限流器
	rateLimiter chan struct{}

	// 中间件配置
	authEnabled bool
	authToken   string
}

// NewAPIGateway 创建API网关
func NewAPIGateway(dm *DistributedManager, commSvc *CommunicationService, sd *ServiceDiscovery, port int) *APIGateway {
	return &APIGateway{
		distributedManager: dm,
		communicationSvc:   commSvc,
		serviceDiscovery:   sd,
		port:               port,
		rateLimiter:        make(chan struct{}, 100), // 100 QPS
		authEnabled:        false,
		authToken:          "your_token",
	}
}

// Start 启动API网关
func (gw *APIGateway) Start(ctx context.Context) error {
	logger.Info("Starting API Gateway on port %d...", gw.port)

	mux := http.NewServeMux()

	// 注册路由
	gw.registerRoutes(mux)

	gw.httpServer = &http.Server{
		Addr:         fmt.Sprintf(":%d", gw.port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		if err := gw.httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("API Gateway server error: %v", err)
		}
	}()

	logger.Info("API Gateway started successfully")
	return nil
}

// Stop 停止API网关
func (gw *APIGateway) Stop(ctx context.Context) error {
	if gw.httpServer != nil {
		ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		return gw.httpServer.Shutdown(ctx)
	}
	return nil
}

func (gw *APIGateway) registerFunc(requestMethod string, handler func(http.ResponseWriter, *http.Request)) http.Handler {
	fn := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != requestMethod {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		handler(w, r)
	}

	httpHandle := gw.rateLimitMiddleware(http.HandlerFunc(fn))
	return gw.authMiddleware(httpHandle)
}

// registerRoutes 注册路由
func (gw *APIGateway) registerRoutes(mux *http.ServeMux) {
	// 健康检查
	mux.Handle("/health", gw.rateLimitMiddleware(http.HandlerFunc(gw.handleHealth)))

	// 表管理接口
	mux.Handle("/api/createTable", gw.registerFunc(http.MethodPost, gw.handleCreateTable))
	mux.Handle("/api/deleteTable", gw.registerFunc(http.MethodDelete, gw.handleDeleteTable))

	// 文档管理接口
	mux.Handle("/api/addDoc", gw.registerFunc(http.MethodPost, gw.handleAddDocument))
	mux.Handle("/api/delDoc", gw.registerFunc(http.MethodDelete, gw.handleDeleteDocument))

	// 搜索接口
	mux.Handle("/api/search", gw.registerFunc(http.MethodGet, gw.handleSearch))
	mux.Handle("/api/searchTable", gw.registerFunc(http.MethodGet, gw.handleSearchTable))

	// 统计接口
	mux.Handle("/api/count", gw.registerFunc(http.MethodGet, gw.handleCount))

	// 集群管理接口
	mux.Handle("/api/cluster/status", gw.registerFunc(http.MethodGet, gw.handleClusterStatus))
	mux.Handle("/api/cluster/slaves", gw.registerFunc(http.MethodGet, gw.handleSlaveList))

	// 系统管理接口
	mux.Handle("/api/system/info", gw.registerFunc(http.MethodGet, gw.handleSystemInfo))
}

// 中间件
func (gw *APIGateway) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if gw.authEnabled {
			token := r.Header.Get("Authorization")
			if token != "Bearer "+gw.authToken {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

func (gw *APIGateway) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case gw.rateLimiter <- struct{}{}:
			defer func() { <-gw.rateLimiter }()
			next.ServeHTTP(w, r)
		default:
			http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
		}
	})
}

// API处理函数
func (gw *APIGateway) handleHealth(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":    "ok",
		"timestamp": time.Now().Unix(),
		"is_master": gw.distributedManager.IsMaster(),
	}

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		logger.Error("encode handleHealth response: %v", err)
	}
}

func (gw *APIGateway) handleCreateTable(w http.ResponseWriter, r *http.Request) {
	// 检查是否为master
	if !gw.distributedManager.IsMaster() {
		http.Error(w, "Only master can create tables", http.StatusForbidden)
		return
	}

	var req struct {
		TableName          string `json:"table_name"`
		VectorDBPath       string `json:"vector_db_path"`
		NumClusters        int32  `json:"num_clusters"`
		InvertedIndexOrder int32  `json:"inverted_index_order"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// 创建protobuf请求
	createReq := &server.CreateTableRequest{
		TableName:          req.TableName,
		VectorDbPath:       req.VectorDBPath,
		NumClusters:        req.NumClusters,
		InvertedIndexOrder: req.InvertedIndexOrder,
	}

	// 获取所有slave地址
	slaveAddrs := gw.serviceDiscovery.GetHealthySlaveAddresses()
	if len(slaveAddrs) == 0 {
		http.Error(w, "No healthy slaves available", http.StatusServiceUnavailable)
		return
	}

	// 向所有slave发送创建表请求
	results := gw.communicationSvc.CreateTableOnSlaves(r.Context(), slaveAddrs, createReq)

	// 统计结果
	successCount := 0
	var errs []string
	for addr, err := range results {
		if err == nil {
			successCount++
		} else {
			errs = append(errs, fmt.Sprintf("%s: %v", addr, err))
		}
	}

	response := map[string]interface{}{
		"success_count": successCount,
		"total_slaves":  len(slaveAddrs),
		"errs":          errs,
	}

	w.Header().Set("Content-Type", "application/json")
	if successCount == len(slaveAddrs) {
		w.WriteHeader(http.StatusOK)
	} else {
		w.WriteHeader(http.StatusPartialContent)
	}
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		logger.Error("encode handleCreateTable response: %v", err)
	}
}

func (gw *APIGateway) handleDeleteTable(w http.ResponseWriter, r *http.Request) {
	if !gw.distributedManager.IsMaster() {
		http.Error(w, "Only master can delete tables", http.StatusForbidden)
		return
	}

	tableName := r.URL.Query().Get("table_name")
	if tableName == "" {
		http.Error(w, "table_name parameter required", http.StatusBadRequest)
		return
	}

	deleteReq := &server.TableRequest{
		TableName: tableName,
	}

	slaveAddrs := gw.serviceDiscovery.GetHealthySlaveAddresses()
	if len(slaveAddrs) == 0 {
		http.Error(w, "No healthy slaves available", http.StatusServiceUnavailable)
		return
	}

	results := gw.communicationSvc.DeleteTableOnSlaves(r.Context(), slaveAddrs, deleteReq)

	successCount := 0
	var errs []string
	for addr, err := range results {
		if err == nil {
			successCount++
		} else {
			errs = append(errs, fmt.Sprintf("%s: %v", addr, err))
		}
	}

	response := map[string]interface{}{
		"success_count": successCount,
		"total_slaves":  len(slaveAddrs),
		"errs":          errs,
	}

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		logger.Error("encode handleDeleteTable response: %v", err)
	}
}

func (gw *APIGateway) handleAddDocument(w http.ResponseWriter, r *http.Request) {
	if !gw.distributedManager.IsMaster() {
		http.Error(w, "Only master can add documents", http.StatusForbidden)
		return
	}

	var req struct {
		TableName      string             `json:"table_name"`
		Document       *messages.Document `json:"document"`
		VectorizedType int32              `json:"vectorized_type"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	addReq := &server.AddDocumentRequest{
		TableName:      req.TableName,
		Document:       req.Document,
		VectorizedType: req.VectorizedType,
	}

	slaveAddrs := gw.serviceDiscovery.GetHealthySlaveAddresses()
	if len(slaveAddrs) == 0 {
		http.Error(w, "No healthy slaves available", http.StatusServiceUnavailable)
		return
	}

	results := gw.communicationSvc.AddDocumentToSlaves(r.Context(), slaveAddrs, addReq)

	successCount := 0
	var errs []string
	for addr, err := range results {
		if err == nil {
			successCount++
		} else {
			errs = append(errs, fmt.Sprintf("%s: %v", addr, err))
		}
	}

	response := map[string]interface{}{
		"success_count": successCount,
		"total_slaves":  len(slaveAddrs),
		"errs":          errs,
	}

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		logger.Error("encode handleAddDocument response: %v", err)
	}
}

func (gw *APIGateway) handleDeleteDocument(w http.ResponseWriter, r *http.Request) {
	if !gw.distributedManager.IsMaster() {
		http.Error(w, "Only master can delete documents", http.StatusForbidden)
		return
	}

	var req struct {
		TableName string              `json:"table_name"`
		DocID     string              `json:"doc_id"`
		Keywords  []*messages.KeyWord `json:"keywords"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	delReq := &server.DeleteDocumentRequest{
		TableName: req.TableName,
		DocId:     req.DocID,
		Keywords:  req.Keywords,
	}

	slaveAddrs := gw.serviceDiscovery.GetHealthySlaveAddresses()
	if len(slaveAddrs) == 0 {
		http.Error(w, "No healthy slaves available", http.StatusServiceUnavailable)
		return
	}

	results := gw.communicationSvc.DeleteDocumentFromSlaves(r.Context(), slaveAddrs, delReq)

	successCount := 0
	var errs []string
	for addr, err := range results {
		if err == nil {
			successCount++
		} else {
			errs = append(errs, fmt.Sprintf("%s: %v", addr, err))
		}
	}

	response := map[string]interface{}{
		"success_count": successCount,
		"total_slaves":  len(slaveAddrs),
		"errs":          errs,
	}

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		logger.Error("encode handleDeleteDocument response: %v", err)
	}
}

func (gw *APIGateway) handleSearchTable(w http.ResponseWriter, r *http.Request) {

	var req struct {
		TableName      string              `json:"table_name"`
		Query          *messages.TermQuery `json:"query"`
		VectorizedType int32               `json:"vectorized_type"`
		K              int32               `json:"k"`
		Probe          int32               `json:"probe"`
		OnFlag         uint64              `json:"on_flag"`
		OffFlag        uint64              `json:"off_flag"`
		OrFlags        []uint64            `json:"or_flags"`
		UseAnn         bool                `json:"use_ann"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	searchReq := &server.SearchRequest{
		TableName:      req.TableName,
		Query:          req.Query,
		VectorizedType: req.VectorizedType,
		K:              req.K,
		Probe:          req.Probe,
		OnFlag:         req.OnFlag,
		OffFlag:        req.OffFlag,
		OrFlags:        req.OrFlags,
		UseAnn:         req.UseAnn,
	}

	slaveAddrs := gw.serviceDiscovery.GetHealthySlaveAddresses()
	if len(slaveAddrs) == 0 {
		http.Error(w, "No healthy slaves available", http.StatusServiceUnavailable)
		return
	}

	// 并行搜索所有slave
	results := gw.communicationSvc.SearchOnSlaves(r.Context(), slaveAddrs, searchReq)

	// 合并搜索结果
	allDocIDs := make([]string, 0)
	for addr, result := range results {
		if result != nil {
			allDocIDs = append(allDocIDs, result.DocIds...)
			logger.Info("Search result from %s: %d docs", addr, len(result.DocIds))
		}
	}

	response := map[string]interface{}{
		"doc_ids":     allDocIDs,
		"total_count": len(allDocIDs),
		"slave_count": len(results),
	}

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		logger.Error("encode handleSearchTable response: %v", err)
	}
}

func (gw *APIGateway) handleSearch(w http.ResponseWriter, r *http.Request) {
	// 简化的搜索接口，使用默认表
	gw.handleSearchTable(w, r)
}

func (gw *APIGateway) handleCount(w http.ResponseWriter, r *http.Request) {
	// 统计所有slave的文档数量
	slaveAddrs := gw.serviceDiscovery.GetHealthySlaveAddresses()
	totalCount := 0

	//for _, addr := range slaveAddrs {
	//	// 这里需要实现获取slave文档数量的逻辑
	//	// 暂时返回模拟数据
	//	totalCount += 1000 // 模拟数据
	//}

	response := map[string]interface{}{
		"total_count": totalCount,
		"slave_count": len(slaveAddrs),
	}

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		logger.Error("encode handleCount response: %v", err)
	}
}

func (gw *APIGateway) handleClusterStatus(w http.ResponseWriter, r *http.Request) {
	master := gw.serviceDiscovery.GetMaster()
	slaves := gw.serviceDiscovery.GetSlaves()

	// 检查slave健康状态
	slaveAddrs := gw.serviceDiscovery.GetSlaveAddresses()
	healthStatus := gw.communicationSvc.HealthCheckSlaves(r.Context(), slaveAddrs)

	response := map[string]interface{}{
		"master":        master,
		"slaves":        slaves,
		"health_status": healthStatus,
		"is_master":     gw.distributedManager.IsMaster(),
	}

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		logger.Error("encode handleClusterStatus response: %v", err)
	}
}

func (gw *APIGateway) handleSlaveList(w http.ResponseWriter, r *http.Request) {
	slaves := gw.serviceDiscovery.GetSlaves()
	slaveAddrs := gw.serviceDiscovery.GetSlaveAddresses()
	healthyAddrs := gw.serviceDiscovery.GetHealthySlaveAddresses()

	response := map[string]interface{}{
		"slaves":            slaves,
		"addresses":         slaveAddrs,
		"healthy_addresses": healthyAddrs,
		"total_count":       len(slaves),
		"healthy_count":     len(healthyAddrs),
	}

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		logger.Error("encode handleSlaveList response: %v", err)
	}
}

func (gw *APIGateway) handleSystemInfo(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"service_name": gw.distributedManager.config.ServiceName,
		"is_master":    gw.distributedManager.IsMaster(),
		"port":         gw.port,
		"version":      "1.0.0",
		"uptime":       time.Now().Unix(), // 简化的运行时间
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// SetAuthConfig 设置认证配置
func (gw *APIGateway) SetAuthConfig(enabled bool, token string) {
	gw.mutex.Lock()
	defer gw.mutex.Unlock()
	gw.authEnabled = enabled
	gw.authToken = token
}

// SetRateLimit 设置限流配置
func (gw *APIGateway) SetRateLimit(qps int) {
	gw.mutex.Lock()
	defer gw.mutex.Unlock()
	gw.rateLimiter = make(chan struct{}, qps)
}
