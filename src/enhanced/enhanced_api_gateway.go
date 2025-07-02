package enhanced

import (
	"VectorSphere/src/library/logger"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"strings"
	"sync"
	"time"
)

// APIGatewayConfig API网关配置
type APIGatewayConfig struct {
	Port              int           `json:"port"`
	ReadTimeout       time.Duration `json:"read_timeout"`
	WriteTimeout      time.Duration `json:"write_timeout"`
	IdleTimeout       time.Duration `json:"idle_timeout"`
	MaxHeaderBytes    int           `json:"max_header_bytes"`
	EnableCORS        bool          `json:"enable_cors"`
	EnableCompression bool          `json:"enable_compression"`
	EnableMetrics     bool          `json:"enable_metrics"`
	EnableLogging     bool          `json:"enable_logging"`
}

// EnhancedAPIGateway 增强型API网关
type EnhancedAPIGateway struct {
	// 配置
	config *APIGatewayConfig

	// 核心组件
	loadBalancer    *EnhancedLoadBalancer
	securityManager *EnhancedSecurityManager
	circuitBreaker  *EnhancedCircuitBreaker
	healthChecker   *EnhancedHealthChecker

	// HTTP服务器
	httpServer *http.Server
	mux        *http.ServeMux

	// 中间件
	middlewares []Middleware

	// 指标和监控
	metrics *GatewayMetrics
	mutex   sync.RWMutex

	// 运行状态
	isRunning bool
	startTime time.Time
}

// Middleware 中间件接口
type Middleware func(http.Handler) http.Handler

// GatewayMetrics 网关指标
type GatewayMetrics struct {
	TotalRequests      int64            `json:"total_requests"`
	SuccessfulRequests int64            `json:"successful_requests"`
	FailedRequests     int64            `json:"failed_requests"`
	AverageLatency     time.Duration    `json:"average_latency"`
	ActiveConnections  int32            `json:"active_connections"`
	RequestsByPath     map[string]int64 `json:"requests_by_path"`
	RequestsByMethod   map[string]int64 `json:"requests_by_method"`
	ErrorsByCode       map[int]int64    `json:"errors_by_code"`
	mutex              sync.RWMutex
}

// NewEnhancedAPIGateway 创建增强型API网关
func NewEnhancedAPIGateway(
	config *APIGatewayConfig,
	loadBalancer *EnhancedLoadBalancer,
	securityManager *EnhancedSecurityManager,
	circuitBreaker *EnhancedCircuitBreaker,
	healthChecker *EnhancedHealthChecker,
) (*EnhancedAPIGateway, error) {
	if config == nil {
		return nil, fmt.Errorf("API网关配置不能为空")
	}

	gateway := &EnhancedAPIGateway{
		config:          config,
		loadBalancer:    loadBalancer,
		securityManager: securityManager,
		circuitBreaker:  circuitBreaker,
		healthChecker:   healthChecker,
		mux:             http.NewServeMux(),
		middlewares:     make([]Middleware, 0),
		metrics:         NewGatewayMetrics(),
		isRunning:       false,
	}

	// 注册默认中间件
	gateway.registerDefaultMiddlewares()

	// 注册路由
	gateway.registerRoutes()

	return gateway, nil
}

// NewGatewayMetrics 创建网关指标
func NewGatewayMetrics() *GatewayMetrics {
	return &GatewayMetrics{
		RequestsByPath:   make(map[string]int64),
		RequestsByMethod: make(map[string]int64),
		ErrorsByCode:     make(map[int]int64),
	}
}

// Start 启动增强型API网关
func (gw *EnhancedAPIGateway) Start(ctx context.Context) error {
	logger.Info("Starting Enhanced API Gateway on port %d...", gw.config.Port)
	gw.startTime = time.Now()

	// 创建HTTP服务器
	gw.httpServer = &http.Server{
		Addr:           fmt.Sprintf(":%d", gw.config.Port),
		Handler:        gw.buildHandler(),
		ReadTimeout:    gw.config.ReadTimeout,
		WriteTimeout:   gw.config.WriteTimeout,
		IdleTimeout:    gw.config.IdleTimeout,
		MaxHeaderBytes: gw.config.MaxHeaderBytes,
	}

	// 启动服务器
	go func() {
		if err := gw.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("Enhanced API Gateway server error: %v", err)
		}
	}()

	gw.isRunning = true
	logger.Info("Enhanced API Gateway started successfully")
	return nil
}

// Stop 停止增强型API网关
func (gw *EnhancedAPIGateway) Stop(ctx context.Context) error {
	logger.Info("Stopping Enhanced API Gateway...")

	if gw.httpServer != nil {
		ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		if err := gw.httpServer.Shutdown(ctx); err != nil {
			return err
		}
	}

	gw.isRunning = false
	logger.Info("Enhanced API Gateway stopped")
	return nil
}

// registerDefaultMiddlewares 注册默认中间件
func (gw *EnhancedAPIGateway) registerDefaultMiddlewares() {
	// 1. 日志中间件
	if gw.config.EnableLogging {
		gw.AddMiddleware(gw.loggingMiddleware)
	}

	// 2. 指标中间件
	if gw.config.EnableMetrics {
		gw.AddMiddleware(gw.metricsMiddleware)
	}

	// 3. CORS中间件
	if gw.config.EnableCORS {
		gw.AddMiddleware(gw.corsMiddleware)
	}

	// 4. 压缩中间件
	if gw.config.EnableCompression {
		gw.AddMiddleware(gw.compressionMiddleware)
	}

	// 5. 安全中间件
	if gw.securityManager != nil {
		gw.AddMiddleware(gw.securityMiddleware)
	}

	// 6. 熔断器中间件
	if gw.circuitBreaker != nil {
		gw.AddMiddleware(gw.circuitBreakerMiddleware)
	}

	// 7. 负载均衡中间件
	if gw.loadBalancer != nil {
		gw.AddMiddleware(gw.loadBalancerMiddleware)
	}
}

// registerRoutes 注册路由
func (gw *EnhancedAPIGateway) registerRoutes() {
	// 健康检查
	gw.mux.HandleFunc("/health", gw.handleHealth)
	gw.mux.HandleFunc("/health/deep", gw.handleDeepHealth)

	// 指标接口
	gw.mux.HandleFunc("/metrics", gw.handleMetrics)
	gw.mux.HandleFunc("/metrics/detailed", gw.handleDetailedMetrics)

	// 管理接口
	gw.mux.HandleFunc("/admin/status", gw.handleAdminStatus)
	gw.mux.HandleFunc("/admin/config", gw.handleAdminConfig)
	gw.mux.HandleFunc("/admin/servers", gw.handleAdminServers)

	// 向量数据库API
	gw.mux.HandleFunc("/api/v1/tables", gw.handleTables)
	gw.mux.HandleFunc("/api/v1/documents", gw.handleDocuments)
	gw.mux.HandleFunc("/api/v1/search", gw.handleSearch)
	gw.mux.HandleFunc("/api/v1/count", gw.handleCount)

	// 集群管理API
	gw.mux.HandleFunc("/api/v1/cluster/status", gw.handleClusterStatus)
	gw.mux.HandleFunc("/api/v1/cluster/nodes", gw.handleClusterNodes)

	// 安全管理API
	gw.mux.HandleFunc("/api/v1/auth/login", gw.handleLogin)
	gw.mux.HandleFunc("/api/v1/auth/refresh", gw.handleRefreshToken)
	gw.mux.HandleFunc("/api/v1/auth/logout", gw.handleLogout)

	// 通用代理路由（必须放在最后）
	gw.mux.HandleFunc("/", gw.handleProxy)
}

// AddMiddleware 添加中间件
func (gw *EnhancedAPIGateway) AddMiddleware(middleware Middleware) {
	gw.mutex.Lock()
	defer gw.mutex.Unlock()
	gw.middlewares = append(gw.middlewares, middleware)
}

// buildHandler 构建处理器链
func (gw *EnhancedAPIGateway) buildHandler() http.Handler {
	handler := http.Handler(gw.mux)

	// 反向应用中间件（最后添加的先执行）
	for i := len(gw.middlewares) - 1; i >= 0; i-- {
		handler = gw.middlewares[i](handler)
	}

	return handler
}

// 中间件实现

// loggingMiddleware 日志中间件
func (gw *EnhancedAPIGateway) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// 创建响应记录器
		rec := &responseRecorder{ResponseWriter: w, statusCode: 200}

		// 处理请求
		next.ServeHTTP(rec, r)

		// 记录日志
		duration := time.Since(start)
		logger.Info("%s %s %d %v %s", r.Method, r.URL.Path, rec.statusCode, duration, r.RemoteAddr)
	})
}

// metricsMiddleware 指标中间件
func (gw *EnhancedAPIGateway) metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// 创建响应记录器
		rec := &responseRecorder{ResponseWriter: w, statusCode: 200}

		// 处理请求
		next.ServeHTTP(rec, r)

		// 更新指标
		duration := time.Since(start)
		gw.updateMetrics(r.Method, r.URL.Path, rec.statusCode, duration)
	})
}

// corsMiddleware CORS中间件
func (gw *EnhancedAPIGateway) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// compressionMiddleware 压缩中间件
func (gw *EnhancedAPIGateway) compressionMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 简化的压缩实现，实际应用中可以使用gzip
		if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			w.Header().Set("Content-Encoding", "gzip")
		}
		next.ServeHTTP(w, r)
	})
}

// securityMiddleware 安全中间件
func (gw *EnhancedAPIGateway) securityMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 跳过公开路径
		if gw.isPublicPath(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}

		// 验证认证
		if err := gw.securityManager.ValidateRequest(r); err != nil {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		// 检查速率限制
		if err := gw.securityManager.CheckRateLimit(r.RemoteAddr); err != nil {
			http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// circuitBreakerMiddleware 熔断器中间件
func (gw *EnhancedAPIGateway) circuitBreakerMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 检查熔断器状态
		if !gw.circuitBreaker.AllowRequest() {
			http.Error(w, "Service Temporarily Unavailable", http.StatusServiceUnavailable)
			return
		}

		// 创建响应记录器
		rec := &responseRecorder{ResponseWriter: w, statusCode: 200}

		// 处理请求
		next.ServeHTTP(rec, r)

		// 记录结果
		if rec.statusCode >= 500 {
			gw.circuitBreaker.RecordFailure()
		} else {
			gw.circuitBreaker.RecordSuccess()
		}
	})
}

// loadBalancerMiddleware 负载均衡中间件
func (gw *EnhancedAPIGateway) loadBalancerMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 对于代理请求，选择后端服务器
		if strings.HasPrefix(r.URL.Path, "/api/") {
			server, err := gw.loadBalancer.SelectServer(r)
			if err != nil {
				http.Error(w, "No Available Servers", http.StatusServiceUnavailable)
				return
			}

			// 将服务器信息添加到请求上下文
			ctx := context.WithValue(r.Context(), "selected_server", server)
			r = r.WithContext(ctx)
		}

		next.ServeHTTP(w, r)
	})
}

// 路由处理函数

// handleHealth 健康检查
func (gw *EnhancedAPIGateway) handleHealth(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":    "ok",
		"timestamp": time.Now().Unix(),
		"uptime":    time.Since(gw.startTime).Seconds(),
		"version":   "enhanced-1.0.0",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleDeepHealth 深度健康检查
func (gw *EnhancedAPIGateway) handleDeepHealth(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":    "ok",
		"timestamp": time.Now().Unix(),
		"uptime":    time.Since(gw.startTime).Seconds(),
		"components": map[string]interface{}{
			"load_balancer":    gw.loadBalancer != nil && gw.loadBalancer.IsHealthy(),
			"security_manager": gw.securityManager != nil,
			"circuit_breaker":  gw.circuitBreaker != nil && gw.circuitBreaker.GetState() != "open",
			"health_checker":   gw.healthChecker != nil,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleMetrics 指标接口
func (gw *EnhancedAPIGateway) handleMetrics(w http.ResponseWriter, r *http.Request) {
	gw.metrics.mutex.RLock()
	defer gw.metrics.mutex.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(gw.metrics)
}

// handleDetailedMetrics 详细指标接口
func (gw *EnhancedAPIGateway) handleDetailedMetrics(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"gateway":         gw.GetMetrics(),
		"load_balancer":   gw.loadBalancer.GetMetrics(),
		"circuit_breaker": gw.circuitBreaker.GetMetrics(),
		"health_checker":  gw.healthChecker.GetMetrics(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleAdminStatus 管理状态接口
func (gw *EnhancedAPIGateway) handleAdminStatus(w http.ResponseWriter, r *http.Request) {
	response := gw.GetStatus()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleAdminConfig 管理配置接口
func (gw *EnhancedAPIGateway) handleAdminConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(gw.config)
}

// handleAdminServers 管理服务器接口
func (gw *EnhancedAPIGateway) handleAdminServers(w http.ResponseWriter, r *http.Request) {
	servers := gw.loadBalancer.GetServers()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(servers)
}

// handleTables 表管理接口
func (gw *EnhancedAPIGateway) handleTables(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		gw.handleCreateTable(w, r)
	case http.MethodDelete:
		gw.handleDeleteTable(w, r)
	case http.MethodGet:
		gw.handleListTables(w, r)
	default:
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	}
}

// handleDocuments 文档管理接口
func (gw *EnhancedAPIGateway) handleDocuments(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		gw.handleAddDocument(w, r)
	case http.MethodDelete:
		gw.handleDeleteDocument(w, r)
	case http.MethodGet:
		gw.handleGetDocument(w, r)
	default:
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	}
}

// handleSearch 搜索接口
func (gw *EnhancedAPIGateway) handleSearch(w http.ResponseWriter, r *http.Request) {
	// 代理到后端服务
	gw.proxyToBackend(w, r)
}

// handleCount 计数接口
func (gw *EnhancedAPIGateway) handleCount(w http.ResponseWriter, r *http.Request) {
	// 代理到后端服务
	gw.proxyToBackend(w, r)
}

// handleClusterStatus 集群状态接口
func (gw *EnhancedAPIGateway) handleClusterStatus(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"servers": gw.loadBalancer.GetServers(),
		"health":  gw.healthChecker.GetHealthStatus(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleClusterNodes 集群节点接口
func (gw *EnhancedAPIGateway) handleClusterNodes(w http.ResponseWriter, r *http.Request) {
	nodes := gw.loadBalancer.GetServers()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(nodes)
}

// handleLogin 登录接口
func (gw *EnhancedAPIGateway) handleLogin(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	session, err := gw.securityManager.Authenticate(req.Username, req.Password, r.RemoteAddr, r.UserAgent())
	if err != nil {
		http.Error(w, "Authentication Failed", http.StatusUnauthorized)
		return
	}

	token := session.ID

	response := map[string]interface{}{
		"token":      token,
		"expires_in": 3600,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleRefreshToken 刷新令牌接口
func (gw *EnhancedAPIGateway) handleRefreshToken(w http.ResponseWriter, r *http.Request) {
	token := r.Header.Get("Authorization")
	if token == "" {
		http.Error(w, "Missing Authorization Header", http.StatusBadRequest)
		return
	}

	// 简化的token刷新实现
	newToken := token + "_refreshed"

	response := map[string]interface{}{
		"token":      newToken,
		"expires_in": 3600,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleLogout 登出接口
func (gw *EnhancedAPIGateway) handleLogout(w http.ResponseWriter, r *http.Request) {
	token := r.Header.Get("Authorization")
	if token != "" {
		// 简化的token撤销实现
		logger.Info("Token revoked: %s", token)
	}

	response := map[string]interface{}{
		"message": "Logged out successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleProxy 通用代理处理
func (gw *EnhancedAPIGateway) handleProxy(w http.ResponseWriter, r *http.Request) {
	gw.proxyToBackend(w, r)
}

// 具体业务处理函数

// handleCreateTable 创建表
func (gw *EnhancedAPIGateway) handleCreateTable(w http.ResponseWriter, r *http.Request) {
	// 解析请求
	var req map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// 代理到后端服务
	gw.proxyToBackend(w, r)
}

// handleDeleteTable 删除表
func (gw *EnhancedAPIGateway) handleDeleteTable(w http.ResponseWriter, r *http.Request) {
	tableName := r.URL.Query().Get("table_name")
	if tableName == "" {
		http.Error(w, "table_name parameter required", http.StatusBadRequest)
		return
	}

	// 代理到后端服务
	gw.proxyToBackend(w, r)
}

// handleListTables 列出表
func (gw *EnhancedAPIGateway) handleListTables(w http.ResponseWriter, r *http.Request) {
	// 代理到后端服务
	gw.proxyToBackend(w, r)
}

// handleAddDocument 添加文档
func (gw *EnhancedAPIGateway) handleAddDocument(w http.ResponseWriter, r *http.Request) {
	// 代理到后端服务
	gw.proxyToBackend(w, r)
}

// handleDeleteDocument 删除文档
func (gw *EnhancedAPIGateway) handleDeleteDocument(w http.ResponseWriter, r *http.Request) {
	// 代理到后端服务
	gw.proxyToBackend(w, r)
}

// handleGetDocument 获取文档
func (gw *EnhancedAPIGateway) handleGetDocument(w http.ResponseWriter, r *http.Request) {
	// 代理到后端服务
	gw.proxyToBackend(w, r)
}

// proxyToBackend 代理到后端服务
func (gw *EnhancedAPIGateway) proxyToBackend(w http.ResponseWriter, r *http.Request) {
	// 从上下文获取选中的服务器
	backend, ok := r.Context().Value("selected_server").(*Backend)
	if !ok {
		http.Error(w, "No Server Selected", http.StatusInternalServerError)
		return
	}

	// 简化的代理实现
	// 实际应用中应该使用httputil.ReverseProxy
	response := map[string]interface{}{
		"message": fmt.Sprintf("Request proxied to %s:%d", backend.Address, backend.Port),
		"path":    r.URL.Path,
		"method":  r.Method,
		"server":  backend.ID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// 辅助函数

// isPublicPath 检查是否为公开路径
func (gw *EnhancedAPIGateway) isPublicPath(path string) bool {
	publicPaths := []string{
		"/health",
		"/metrics",
		"/api/v1/auth/login",
	}

	for _, publicPath := range publicPaths {
		if strings.HasPrefix(path, publicPath) {
			return true
		}
	}
	return false
}

// updateMetrics 更新指标
func (gw *EnhancedAPIGateway) updateMetrics(method, path string, statusCode int, duration time.Duration) {
	gw.metrics.mutex.Lock()
	defer gw.metrics.mutex.Unlock()

	gw.metrics.TotalRequests++
	if statusCode < 400 {
		gw.metrics.SuccessfulRequests++
	} else {
		gw.metrics.FailedRequests++
	}

	// 更新平均延迟（简化计算）
	gw.metrics.AverageLatency = (gw.metrics.AverageLatency + duration) / 2

	// 按路径统计
	gw.metrics.RequestsByPath[path]++

	// 按方法统计
	gw.metrics.RequestsByMethod[method]++

	// 按错误码统计
	if statusCode >= 400 {
		gw.metrics.ErrorsByCode[statusCode]++
	}
}

// GetMetrics 获取指标
func (gw *EnhancedAPIGateway) GetMetrics() map[string]interface{} {
	gw.metrics.mutex.RLock()
	defer gw.metrics.mutex.RUnlock()

	return map[string]interface{}{
		"total_requests":      gw.metrics.TotalRequests,
		"successful_requests": gw.metrics.SuccessfulRequests,
		"failed_requests":     gw.metrics.FailedRequests,
		"average_latency":     gw.metrics.AverageLatency.Milliseconds(),
		"active_connections":  gw.metrics.ActiveConnections,
		"requests_by_path":    gw.metrics.RequestsByPath,
		"requests_by_method":  gw.metrics.RequestsByMethod,
		"errors_by_code":      gw.metrics.ErrorsByCode,
	}
}

// GetStatus 获取状态
func (gw *EnhancedAPIGateway) GetStatus() map[string]interface{} {
	return map[string]interface{}{
		"running":    gw.isRunning,
		"start_time": gw.startTime,
		"uptime":     time.Since(gw.startTime).Seconds(),
		"port":       gw.config.Port,
		"config":     gw.config,
		"metrics":    gw.GetMetrics(),
	}
}

// responseRecorder 响应记录器
type responseRecorder struct {
	http.ResponseWriter
	statusCode int
}

func (rec *responseRecorder) WriteHeader(code int) {
	rec.statusCode = code
	rec.ResponseWriter.WriteHeader(code)
}

// UpdateConfig 更新配置
func (gw *EnhancedAPIGateway) UpdateConfig(config *APIGatewayConfig) {
	gw.mutex.Lock()
	defer gw.mutex.Unlock()
	gw.config = config
}
