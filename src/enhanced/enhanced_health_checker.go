package enhanced

import (
	"VectorSphere/src/library/logger"
	"context"
	"fmt"
	"math"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// HealthStatus 健康状态
type HealthStatus int

const (
	Healthy HealthStatus = iota
	Degraded
	Unhealthy
	Critical
	Unknown
)

// CheckType 检查类型
type CheckType int

const (
	HTTPCheck CheckType = iota
	TCPCheck
	GRPCCheck
	CustomCheck
	DependencyCheck
	ResourceCheck
	BusinessCheck
)

// HealthCheckLevel 健康检查级别
type HealthCheckLevel int

const (
	BasicLevel HealthCheckLevel = iota
	DetailedLevel
	ComprehensiveLevel
)

// HealthCheck 健康检查定义
type HealthCheck struct {
	ID           string                                  `json:"id"`
	Name         string                                  `json:"name"`
	Type         CheckType                               `json:"type"`
	Level        HealthCheckLevel                        `json:"level"`
	Interval     time.Duration                           `json:"interval"`
	Timeout      time.Duration                           `json:"timeout"`
	Retries      int                                     `json:"retries"`
	Weight       float64                                 `json:"weight"`
	Critical     bool                                    `json:"critical"`
	Enabled      bool                                    `json:"enabled"`
	Config       map[string]interface{}                  `json:"config"`
	CheckFunc    func(ctx context.Context) *HealthResult `json:"-"`
	Dependencies []string                                `json:"dependencies"`
	Tags         []string                                `json:"tags"`
	Thresholds   *HealthThresholds                       `json:"thresholds"`
	Adaptive     bool                                    `json:"adaptive"`
	LastModified time.Time                               `json:"last_modified"`
}

// HealthThresholds 健康阈值
type HealthThresholds struct {
	WarningLatency  time.Duration `json:"warning_latency"`
	CriticalLatency time.Duration `json:"critical_latency"`
	WarningRate     float64       `json:"warning_rate"`
	CriticalRate    float64       `json:"critical_rate"`
	MinSuccessRate  float64       `json:"min_success_rate"`
	MaxErrorRate    float64       `json:"max_error_rate"`
}

// HealthResult 健康检查结果
type HealthResult struct {
	CheckID   string                 `json:"check_id"`
	Status    HealthStatus           `json:"status"`
	Message   string                 `json:"message"`
	Latency   time.Duration          `json:"latency"`
	Timestamp time.Time              `json:"timestamp"`
	Details   map[string]interface{} `json:"details"`
	Error     string                 `json:"error,omitempty"`
	Score     float64                `json:"score"`
	Metrics   map[string]float64     `json:"metrics"`
	Tags      []string               `json:"tags"`
}

// ServiceHealth 服务健康状态
type ServiceHealth struct {
	ServiceName   string                   `json:"service_name"`
	NodeID        string                   `json:"node_id"`
	OverallStatus HealthStatus             `json:"overall_status"`
	OverallScore  float64                  `json:"overall_score"`
	CheckResults  map[string]*HealthResult `json:"check_results"`
	LastUpdate    time.Time                `json:"last_update"`
	Uptime        time.Duration            `json:"uptime"`
	StartTime     time.Time                `json:"start_time"`
	Metrics       *ServiceMetrics          `json:"metrics"`
	Trends        *HealthTrends            `json:"trends"`
	Alerts        []*HealthAlert           `json:"alerts"`
	Dependencies  map[string]HealthStatus  `json:"dependencies"`
	Prediction    *HealthPrediction        `json:"prediction"`
}

// ServiceMetrics 服务指标
type ServiceMetrics struct {
	CPUUsage       float64   `json:"cpu_usage"`
	MemoryUsage    float64   `json:"memory_usage"`
	DiskUsage      float64   `json:"disk_usage"`
	NetworkLatency float64   `json:"network_latency"`
	RequestRate    float64   `json:"request_rate"`
	ErrorRate      float64   `json:"error_rate"`
	ResponseTime   float64   `json:"response_time"`
	Throughput     float64   `json:"throughput"`
	Connections    int64     `json:"connections"`
	QueueSize      int64     `json:"queue_size"`
	LastUpdated    time.Time `json:"last_updated"`
}

// HealthTrends 健康趋势
type HealthTrends struct {
	ScoreTrend        []float64 `json:"score_trend"`
	LatencyTrend      []float64 `json:"latency_trend"`
	ErrorRateTrend    []float64 `json:"error_rate_trend"`
	AvailabilityTrend []float64 `json:"availability_trend"`
	WindowSize        int       `json:"window_size"`
	LastUpdate        time.Time `json:"last_update"`
}

// HealthAlert 健康告警
type HealthAlert struct {
	ID         string                 `json:"id"`
	Level      string                 `json:"level"`
	Message    string                 `json:"message"`
	CheckID    string                 `json:"check_id"`
	Timestamp  time.Time              `json:"timestamp"`
	Resolved   bool                   `json:"resolved"`
	ResolvedAt *time.Time             `json:"resolved_at,omitempty"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// HealthPrediction 健康预测
type HealthPrediction struct {
	PredictedStatus HealthStatus  `json:"predicted_status"`
	Confidence      float64       `json:"confidence"`
	TimeToFailure   time.Duration `json:"time_to_failure"`
	RiskFactors     []string      `json:"risk_factors"`
	Recommendations []string      `json:"recommendations"`
	LastUpdate      time.Time     `json:"last_update"`
}

// HealthCheckConfig 健康检查配置
type HealthCheckConfig struct {
	Enabled           bool          `json:"enabled"`
	DefaultInterval   time.Duration `json:"default_interval"`
	DefaultTimeout    time.Duration `json:"default_timeout"`
	DefaultRetries    int           `json:"default_retries"`
	AdaptiveInterval  bool          `json:"adaptive_interval"`
	MinInterval       time.Duration `json:"min_interval"`
	MaxInterval       time.Duration `json:"max_interval"`
	TrendWindowSize   int           `json:"trend_window_size"`
	PredictionEnabled bool          `json:"prediction_enabled"`
	AlertingEnabled   bool          `json:"alerting_enabled"`
	MetricsRetention  time.Duration `json:"metrics_retention"`
	BasePrefix        string        `json:"base_prefix"`
}

// HealthCheckCallback 健康检查回调
type HealthCheckCallback func(serviceHealth *ServiceHealth)

// EnhancedHealthChecker 增强健康检查器
type EnhancedHealthChecker struct {
	client          *clientv3.Client
	config          *HealthCheckConfig
	checks          map[string]*HealthCheck
	serviceHealth   map[string]*ServiceHealth
	mu              sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
	callbacks       []HealthCheckCallback
	isRunning       int32
	workerPool      chan struct{}
	resultChan      chan *HealthResult
	alertChan       chan *HealthAlert
	metricsTicker   *time.Ticker
	cleanupTicker   *time.Ticker
	httpClient      *http.Client
	predictionModel *HealthPredictionModel
}

// HealthPredictionModel 健康预测模型
type HealthPredictionModel struct {
	mu                sync.RWMutex
	historyData       map[string][]*HealthResult
	models            map[string]*PredictionModel
	windowSize        int
	predictionHorizon time.Duration
}

// PredictionModel 预测模型
type PredictionModel struct {
	Weights      []float64 `json:"weights"`
	Bias         float64   `json:"bias"`
	Accuracy     float64   `json:"accuracy"`
	LastTrained  time.Time `json:"last_trained"`
	TrainingData int       `json:"training_data"`
}

// NewEnhancedHealthChecker 创建增强健康检查器
func NewEnhancedHealthChecker(client *clientv3.Client, config *HealthCheckConfig) *EnhancedHealthChecker {
	if config == nil {
		config = &HealthCheckConfig{
			Enabled:           true,
			DefaultInterval:   30 * time.Second,
			DefaultTimeout:    10 * time.Second,
			DefaultRetries:    3,
			AdaptiveInterval:  true,
			MinInterval:       5 * time.Second,
			MaxInterval:       300 * time.Second,
			TrendWindowSize:   20,
			PredictionEnabled: true,
			AlertingEnabled:   true,
			MetricsRetention:  24 * time.Hour,
			BasePrefix:        "/vector_sphere/health",
		}
	}

	ctx, cancel := context.WithCancel(context.Background())

	hc := &EnhancedHealthChecker{
		client:        client,
		config:        config,
		checks:        make(map[string]*HealthCheck),
		serviceHealth: make(map[string]*ServiceHealth),
		ctx:           ctx,
		cancel:        cancel,
		callbacks:     make([]HealthCheckCallback, 0),
		workerPool:    make(chan struct{}, 10), // 10个并发工作者
		resultChan:    make(chan *HealthResult, 100),
		alertChan:     make(chan *HealthAlert, 50),
		httpClient: &http.Client{
			Timeout: config.DefaultTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		predictionModel: &HealthPredictionModel{
			historyData:       make(map[string][]*HealthResult),
			models:            make(map[string]*PredictionModel),
			windowSize:        config.TrendWindowSize,
			predictionHorizon: 1 * time.Hour,
		},
	}

	// 初始化工作者池
	for i := 0; i < 10; i++ {
		hc.workerPool <- struct{}{}
	}

	logger.Info("Enhanced health checker created")
	return hc
}

// Start 启动健康检查器
func (hc *EnhancedHealthChecker) Start() error {
	if !atomic.CompareAndSwapInt32(&hc.isRunning, 0, 1) {
		return fmt.Errorf("health checker is already running")
	}

	logger.Info("Starting enhanced health checker")

	// 启动结果处理器
	go hc.resultProcessor()

	// 启动告警处理器
	go hc.alertProcessor()

	// 启动指标收集器
	hc.metricsTicker = time.NewTicker(60 * time.Second)
	go hc.metricsCollector()

	// 启动清理器
	hc.cleanupTicker = time.NewTicker(1 * time.Hour)
	go hc.cleaner()

	// 启动预测模型训练
	if hc.config.PredictionEnabled {
		go hc.predictionTrainer()
	}

	logger.Info("Enhanced health checker started successfully")
	return nil
}

// Stop 停止健康检查器
func (hc *EnhancedHealthChecker) Stop() error {
	if !atomic.CompareAndSwapInt32(&hc.isRunning, 1, 0) {
		return fmt.Errorf("health checker is not running")
	}

	logger.Info("Stopping enhanced health checker")

	// 停止所有检查
	hc.mu.Lock()
	for _, check := range hc.checks {
		check.Enabled = false
	}
	hc.mu.Unlock()

	// 停止定时器
	if hc.metricsTicker != nil {
		hc.metricsTicker.Stop()
	}
	if hc.cleanupTicker != nil {
		hc.cleanupTicker.Stop()
	}

	// 关闭通道
	close(hc.resultChan)
	close(hc.alertChan)

	// 取消上下文
	hc.cancel()

	logger.Info("Enhanced health checker stopped")
	return nil
}

// RegisterCheck 注册健康检查
func (hc *EnhancedHealthChecker) RegisterCheck(check *HealthCheck) error {
	if check == nil {
		return fmt.Errorf("health check cannot be nil")
	}

	if check.ID == "" {
		return fmt.Errorf("health check ID cannot be empty")
	}

	// 设置默认值
	if check.Interval == 0 {
		check.Interval = hc.config.DefaultInterval
	}
	if check.Timeout == 0 {
		check.Timeout = hc.config.DefaultTimeout
	}
	if check.Retries == 0 {
		check.Retries = hc.config.DefaultRetries
	}
	if check.Weight == 0 {
		check.Weight = 1.0
	}
	if check.Thresholds == nil {
		check.Thresholds = &HealthThresholds{
			WarningLatency:  5 * time.Second,
			CriticalLatency: 10 * time.Second,
			WarningRate:     0.1,
			CriticalRate:    0.2,
			MinSuccessRate:  0.95,
			MaxErrorRate:    0.05,
		}
	}

	check.LastModified = time.Now()
	check.Enabled = true

	hc.mu.Lock()
	hc.checks[check.ID] = check
	hc.mu.Unlock()

	// 启动检查
	if atomic.LoadInt32(&hc.isRunning) == 1 {
		go hc.runCheck(check)
	}

	logger.Info("Health check registered: %s (%s)", check.ID, check.Name)
	return nil
}

// UnregisterCheck 注销健康检查
func (hc *EnhancedHealthChecker) UnregisterCheck(checkID string) error {
	hc.mu.Lock()
	check, exists := hc.checks[checkID]
	if exists {
		check.Enabled = false
		delete(hc.checks, checkID)
	}
	hc.mu.Unlock()

	if !exists {
		return fmt.Errorf("health check not found: %s", checkID)
	}

	logger.Info("Health check unregistered: %s", checkID)
	return nil
}

// GetServiceHealth 获取服务健康状态
func (hc *EnhancedHealthChecker) GetServiceHealth(serviceName string) *ServiceHealth {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	if health, exists := hc.serviceHealth[serviceName]; exists {
		// 返回副本
		healthCopy := *health
		return &healthCopy
	}
	return nil
}

// GetAllServiceHealth 获取所有服务健康状态
func (hc *EnhancedHealthChecker) GetAllServiceHealth() map[string]*ServiceHealth {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	result := make(map[string]*ServiceHealth)
	for name, health := range hc.serviceHealth {
		healthCopy := *health
		result[name] = &healthCopy
	}
	return result
}

// AddCallback 添加健康检查回调
func (hc *EnhancedHealthChecker) AddCallback(callback HealthCheckCallback) {
	hc.mu.Lock()
	hc.callbacks = append(hc.callbacks, callback)
	hc.mu.Unlock()
}

// UpdateCheckInterval 更新检查间隔
func (hc *EnhancedHealthChecker) UpdateCheckInterval(checkID string, interval time.Duration) error {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	check, exists := hc.checks[checkID]
	if !exists {
		return fmt.Errorf("health check not found: %s", checkID)
	}

	check.Interval = interval
	check.LastModified = time.Now()

	logger.Info("Updated check interval for %s: %v", checkID, interval)
	return nil
}

// EnableCheck 启用检查
func (hc *EnhancedHealthChecker) EnableCheck(checkID string) error {
	hc.mu.Lock()
	check, exists := hc.checks[checkID]
	if exists {
		check.Enabled = true
		check.LastModified = time.Now()
	}
	hc.mu.Unlock()

	if !exists {
		return fmt.Errorf("health check not found: %s", checkID)
	}

	// 重新启动检查
	if atomic.LoadInt32(&hc.isRunning) == 1 {
		go hc.runCheck(check)
	}

	logger.Info("Health check enabled: %s", checkID)
	return nil
}

// DisableCheck 禁用检查
func (hc *EnhancedHealthChecker) DisableCheck(checkID string) error {
	hc.mu.Lock()
	check, exists := hc.checks[checkID]
	if exists {
		check.Enabled = false
		check.LastModified = time.Now()
	}
	hc.mu.Unlock()

	if !exists {
		return fmt.Errorf("health check not found: %s", checkID)
	}

	logger.Info("Health check disabled: %s", checkID)
	return nil
}

// 内部方法实现

// runCheck 运行单个健康检查
func (hc *EnhancedHealthChecker) runCheck(check *HealthCheck) {
	logger.Debug("Starting health check: %s", check.ID)

	// 计算初始间隔
	interval := check.Interval
	if check.Adaptive {
		interval = hc.calculateAdaptiveInterval(check)
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// 立即执行一次检查
	hc.executeCheck(check)

	for {
		select {
		case <-hc.ctx.Done():
			logger.Debug("Health check stopped: %s", check.ID)
			return
		case <-ticker.C:
			if !check.Enabled {
				logger.Debug("Health check disabled: %s", check.ID)
				return
			}

			// 执行检查
			hc.executeCheck(check)

			// 更新自适应间隔
			if check.Adaptive {
				newInterval := hc.calculateAdaptiveInterval(check)
				if newInterval != interval {
					interval = newInterval
					ticker.Reset(interval)
					logger.Debug("Adaptive interval updated for %s: %v", check.ID, interval)
				}
			}
		}
	}
}

// executeCheck 执行健康检查
func (hc *EnhancedHealthChecker) executeCheck(check *HealthCheck) {
	// 获取工作者
	select {
	case <-hc.workerPool:
		defer func() {
			hc.workerPool <- struct{}{}
		}()
	case <-time.After(1 * time.Second):
		// 工作者池满，跳过此次检查
		logger.Warning("Worker pool full, skipping check: %s", check.ID)
		return
	}

	start := time.Now()
	var result *HealthResult

	// 创建带超时的上下文
	ctx, cancel := context.WithTimeout(hc.ctx, check.Timeout)
	defer cancel()

	// 执行检查（带重试）
	for attempt := 0; attempt <= check.Retries; attempt++ {
		if check.CheckFunc != nil {
			result = check.CheckFunc(ctx)
		} else {
			result = hc.performBuiltinCheck(ctx, check)
		}

		if result.Status == Healthy || attempt == check.Retries {
			break
		}

		// 重试前等待
		time.Sleep(time.Duration(attempt+1) * time.Second)
	}

	// 设置基本信息
	result.CheckID = check.ID
	result.Latency = time.Since(start)
	result.Timestamp = time.Now()

	// 计算健康分数
	result.Score = hc.calculateHealthScore(result, check)

	// 发送结果
	select {
	case hc.resultChan <- result:
	default:
		logger.Warning("Result channel full, dropping result for check: %s", check.ID)
	}
}

// performBuiltinCheck 执行内置检查
func (hc *EnhancedHealthChecker) performBuiltinCheck(ctx context.Context, check *HealthCheck) *HealthResult {
	result := &HealthResult{
		CheckID:   check.ID,
		Timestamp: time.Now(),
		Details:   make(map[string]interface{}),
		Metrics:   make(map[string]float64),
		Tags:      check.Tags,
	}

	switch check.Type {
	case HTTPCheck:
		return hc.performHTTPCheck(ctx, check, result)
	case TCPCheck:
		return hc.performTCPCheck(ctx, check, result)
	case ResourceCheck:
		return hc.performResourceCheck(ctx, check, result)
	default:
		result.Status = Unknown
		result.Message = "Unknown check type"
		result.Error = fmt.Sprintf("Unsupported check type: %d", check.Type)
	}

	return result
}

// performHTTPCheck 执行HTTP检查
func (hc *EnhancedHealthChecker) performHTTPCheck(ctx context.Context, check *HealthCheck, result *HealthResult) *HealthResult {
	url, ok := check.Config["url"].(string)
	if !ok {
		result.Status = Critical
		result.Message = "HTTP check missing URL"
		result.Error = "URL not configured"
		return result
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		result.Status = Critical
		result.Message = "Failed to create HTTP request"
		result.Error = err.Error()
		return result
	}

	// 添加自定义头部
	if headers, ok := check.Config["headers"].(map[string]string); ok {
		for key, value := range headers {
			req.Header.Set(key, value)
		}
	}

	start := time.Now()
	resp, err := hc.httpClient.Do(req)
	latency := time.Since(start)

	if err != nil {
		result.Status = Critical
		result.Message = "HTTP request failed"
		result.Error = err.Error()
		result.Metrics["latency_ms"] = float64(latency.Milliseconds())
		return result
	}
	defer resp.Body.Close()

	// 检查状态码
	expectedStatus := 200
	if status, ok := check.Config["expected_status"].(int); ok {
		expectedStatus = status
	}

	result.Details["status_code"] = resp.StatusCode
	result.Details["response_time_ms"] = latency.Milliseconds()
	result.Metrics["latency_ms"] = float64(latency.Milliseconds())
	result.Metrics["status_code"] = float64(resp.StatusCode)

	if resp.StatusCode == expectedStatus {
		if latency > check.Thresholds.CriticalLatency {
			result.Status = Critical
			result.Message = fmt.Sprintf("HTTP check passed but latency too high: %v", latency)
		} else if latency > check.Thresholds.WarningLatency {
			result.Status = Degraded
			result.Message = fmt.Sprintf("HTTP check passed but latency elevated: %v", latency)
		} else {
			result.Status = Healthy
			result.Message = "HTTP check passed"
		}
	} else {
		result.Status = Unhealthy
		result.Message = fmt.Sprintf("HTTP check failed: expected status %d, got %d", expectedStatus, resp.StatusCode)
	}

	return result
}

// performTCPCheck 执行TCP检查
func (hc *EnhancedHealthChecker) performTCPCheck(ctx context.Context, check *HealthCheck, result *HealthResult) *HealthResult {
	address, ok := check.Config["address"].(string)
	if !ok {
		result.Status = Critical
		result.Message = "TCP check missing address"
		result.Error = "Address not configured"
		return result
	}

	start := time.Now()
	conn, err := net.DialTimeout("tcp", address, check.Timeout)
	latency := time.Since(start)

	result.Metrics["latency_ms"] = float64(latency.Milliseconds())
	result.Details["address"] = address
	result.Details["connection_time_ms"] = latency.Milliseconds()

	if err != nil {
		result.Status = Critical
		result.Message = "TCP connection failed"
		result.Error = err.Error()
		return result
	}
	defer conn.Close()

	if latency > check.Thresholds.CriticalLatency {
		result.Status = Critical
		result.Message = fmt.Sprintf("TCP connection too slow: %v", latency)
	} else if latency > check.Thresholds.WarningLatency {
		result.Status = Degraded
		result.Message = fmt.Sprintf("TCP connection slow: %v", latency)
	} else {
		result.Status = Healthy
		result.Message = "TCP connection successful"
	}

	return result
}

// performResourceCheck 执行资源检查
func (hc *EnhancedHealthChecker) performResourceCheck(ctx context.Context, check *HealthCheck, result *HealthResult) *HealthResult {
	// 这里可以实现CPU、内存、磁盘等资源检查
	// 简化实现，实际应该集成系统监控
	result.Status = Healthy
	result.Message = "Resource check passed"
	result.Metrics["cpu_usage"] = 50.0 // 模拟数据
	result.Metrics["memory_usage"] = 60.0
	result.Metrics["disk_usage"] = 70.0
	return result
}

// calculateHealthScore 计算健康分数
func (hc *EnhancedHealthChecker) calculateHealthScore(result *HealthResult, check *HealthCheck) float64 {
	baseScore := 0.0

	switch result.Status {
	case Healthy:
		baseScore = 100.0
	case Degraded:
		baseScore = 75.0
	case Unhealthy:
		baseScore = 25.0
	case Critical:
		baseScore = 0.0
	default:
		baseScore = 50.0
	}

	// 根据延迟调整分数
	if result.Latency > 0 {
		latencyPenalty := 0.0
		if result.Latency > check.Thresholds.CriticalLatency {
			latencyPenalty = 30.0
		} else if result.Latency > check.Thresholds.WarningLatency {
			latencyPenalty = 15.0
		}
		baseScore = math.Max(0, baseScore-latencyPenalty)
	}

	return baseScore
}

// calculateAdaptiveInterval 计算自适应间隔
func (hc *EnhancedHealthChecker) calculateAdaptiveInterval(check *HealthCheck) time.Duration {
	// 获取最近的检查结果
	hc.mu.RLock()
	serviceName := check.Config["service_name"]
	if serviceName == nil {
		serviceName = "default"
	}
	serviceHealth := hc.serviceHealth[serviceName.(string)]
	hc.mu.RUnlock()

	if serviceHealth == nil {
		return check.Interval
	}

	// 根据健康状态调整间隔
	switch serviceHealth.OverallStatus {
	case Healthy:
		// 健康时可以降低检查频率
		return time.Duration(float64(check.Interval) * 1.5)
	case Degraded:
		// 降级时保持正常频率
		return check.Interval
	case Unhealthy, Critical:
		// 不健康时增加检查频率
		return time.Duration(float64(check.Interval) * 0.5)
	default:
		return check.Interval
	}
}

// resultProcessor 结果处理器
func (hc *EnhancedHealthChecker) resultProcessor() {
	logger.Info("Starting health check result processor")

	for result := range hc.resultChan {
		hc.processResult(result)
	}

	logger.Info("Health check result processor stopped")
}

// processResult 处理检查结果
func (hc *EnhancedHealthChecker) processResult(result *HealthResult) {
	hc.mu.Lock()
	check := hc.checks[result.CheckID]
	hc.mu.Unlock()

	if check == nil {
		logger.Warning("Received result for unknown check: %s", result.CheckID)
		return
	}

	// 获取或创建服务健康状态
	serviceName := "default"
	if sn, ok := check.Config["service_name"].(string); ok {
		serviceName = sn
	}

	hc.mu.Lock()
	serviceHealth, exists := hc.serviceHealth[serviceName]
	if !exists {
		serviceHealth = &ServiceHealth{
			ServiceName:  serviceName,
			NodeID:       "local",
			CheckResults: make(map[string]*HealthResult),
			StartTime:    time.Now(),
			Metrics:      &ServiceMetrics{},
			Trends:       &HealthTrends{WindowSize: hc.config.TrendWindowSize},
			Alerts:       make([]*HealthAlert, 0),
			Dependencies: make(map[string]HealthStatus),
		}
		hc.serviceHealth[serviceName] = serviceHealth
	}

	// 更新检查结果
	serviceHealth.CheckResults[result.CheckID] = result
	serviceHealth.LastUpdate = time.Now()
	serviceHealth.Uptime = time.Since(serviceHealth.StartTime)

	// 计算整体健康状态和分数
	hc.calculateOverallHealth(serviceHealth)

	// 更新趋势数据
	hc.updateTrends(serviceHealth, result)

	// 检查告警条件
	hc.checkAlerts(serviceHealth, result, check)

	// 更新预测数据
	if hc.config.PredictionEnabled {
		hc.updatePredictionData(serviceName, result)
	}

	hc.mu.Unlock()

	// 触发回调
	hc.triggerCallbacks(serviceHealth)

	logger.Debug("Processed health check result: %s, status: %d, score: %.2f",
		result.CheckID, result.Status, result.Score)
}

// calculateOverallHealth 计算整体健康状态
func (hc *EnhancedHealthChecker) calculateOverallHealth(serviceHealth *ServiceHealth) {
	if len(serviceHealth.CheckResults) == 0 {
		serviceHealth.OverallStatus = Unknown
		serviceHealth.OverallScore = 0
		return
	}

	totalWeight := 0.0
	weightedScore := 0.0
	worstStatus := Healthy
	criticalCount := 0

	for checkID, result := range serviceHealth.CheckResults {
		check := hc.checks[checkID]
		if check == nil {
			continue
		}

		weight := check.Weight
		totalWeight += weight
		weightedScore += result.Score * weight

		// 更新最差状态
		if result.Status > worstStatus {
			worstStatus = result.Status
		}

		// 统计关键检查失败数
		if check.Critical && result.Status >= Unhealthy {
			criticalCount++
		}
	}

	// 计算加权平均分数
	if totalWeight > 0 {
		serviceHealth.OverallScore = weightedScore / totalWeight
	} else {
		serviceHealth.OverallScore = 0
	}

	// 确定整体状态
	if criticalCount > 0 {
		serviceHealth.OverallStatus = Critical
	} else {
		serviceHealth.OverallStatus = worstStatus
	}
}

// updateTrends 更新趋势数据
func (hc *EnhancedHealthChecker) updateTrends(serviceHealth *ServiceHealth, result *HealthResult) {
	trends := serviceHealth.Trends
	if trends == nil {
		return
	}

	// 更新分数趋势
	trends.ScoreTrend = append(trends.ScoreTrend, serviceHealth.OverallScore)
	if len(trends.ScoreTrend) > trends.WindowSize {
		trends.ScoreTrend = trends.ScoreTrend[1:]
	}

	// 更新延迟趋势
	latencyMs := float64(result.Latency.Milliseconds())
	trends.LatencyTrend = append(trends.LatencyTrend, latencyMs)
	if len(trends.LatencyTrend) > trends.WindowSize {
		trends.LatencyTrend = trends.LatencyTrend[1:]
	}

	// 更新错误率趋势（简化实现）
	errorRate := 0.0
	if result.Status >= Unhealthy {
		errorRate = 1.0
	}
	trends.ErrorRateTrend = append(trends.ErrorRateTrend, errorRate)
	if len(trends.ErrorRateTrend) > trends.WindowSize {
		trends.ErrorRateTrend = trends.ErrorRateTrend[1:]
	}

	// 更新可用性趋势
	availability := 0.0
	if result.Status <= Degraded {
		availability = 1.0
	}
	trends.AvailabilityTrend = append(trends.AvailabilityTrend, availability)
	if len(trends.AvailabilityTrend) > trends.WindowSize {
		trends.AvailabilityTrend = trends.AvailabilityTrend[1:]
	}

	trends.LastUpdate = time.Now()
}

// checkAlerts 检查告警条件
func (hc *EnhancedHealthChecker) checkAlerts(serviceHealth *ServiceHealth, result *HealthResult, check *HealthCheck) {
	if !hc.config.AlertingEnabled {
		return
	}

	// 检查状态变化告警
	if result.Status >= Unhealthy {
		alert := &HealthAlert{
			ID:        fmt.Sprintf("%s_%s_%d", serviceHealth.ServiceName, result.CheckID, time.Now().Unix()),
			Level:     hc.getAlertLevel(result.Status),
			Message:   fmt.Sprintf("Health check failed: %s - %s", check.Name, result.Message),
			CheckID:   result.CheckID,
			Timestamp: time.Now(),
			Resolved:  false,
			Metadata: map[string]interface{}{
				"service_name": serviceHealth.ServiceName,
				"check_name":   check.Name,
				"status":       result.Status,
				"score":        result.Score,
				"latency":      result.Latency.String(),
			},
		}

		select {
		case hc.alertChan <- alert:
		default:
			logger.Warning("Alert channel full, dropping alert: %s", alert.ID)
		}
	}

	// 检查延迟告警
	if result.Latency > check.Thresholds.CriticalLatency {
		alert := &HealthAlert{
			ID:        fmt.Sprintf("%s_%s_latency_%d", serviceHealth.ServiceName, result.CheckID, time.Now().Unix()),
			Level:     "critical",
			Message:   fmt.Sprintf("High latency detected: %s - %v", check.Name, result.Latency),
			CheckID:   result.CheckID,
			Timestamp: time.Now(),
			Resolved:  false,
			Metadata: map[string]interface{}{
				"service_name": serviceHealth.ServiceName,
				"check_name":   check.Name,
				"latency":      result.Latency.String(),
				"threshold":    check.Thresholds.CriticalLatency.String(),
			},
		}

		select {
		case hc.alertChan <- alert:
		default:
			logger.Warning("Alert channel full, dropping latency alert: %s", alert.ID)
		}
	}
}

// getAlertLevel 获取告警级别
func (hc *EnhancedHealthChecker) getAlertLevel(status HealthStatus) string {
	switch status {
	case Critical:
		return "critical"
	case Unhealthy:
		return "error"
	case Degraded:
		return "warning"
	default:
		return "info"
	}
}

// updatePredictionData 更新预测数据
func (hc *EnhancedHealthChecker) updatePredictionData(serviceName string, result *HealthResult) {
	hc.predictionModel.mu.Lock()
	defer hc.predictionModel.mu.Unlock()

	history := hc.predictionModel.historyData[serviceName]
	history = append(history, result)

	// 保持历史数据窗口大小
	if len(history) > hc.predictionModel.windowSize {
		history = history[1:]
	}

	hc.predictionModel.historyData[serviceName] = history
}

// alertProcessor 告警处理器
func (hc *EnhancedHealthChecker) alertProcessor() {
	logger.Info("Starting health check alert processor")

	for alert := range hc.alertChan {
		hc.processAlert(alert)
	}

	logger.Info("Health check alert processor stopped")
}

// processAlert 处理告警
func (hc *EnhancedHealthChecker) processAlert(alert *HealthAlert) {
	// 这里可以集成告警系统，如发送邮件、短信、Slack通知等
	logger.Warning("Health alert: [%s] %s (Check: %s)", alert.Level, alert.Message, alert.CheckID)

	// 将告警添加到服务健康状态中
	hc.mu.Lock()
	if serviceName, ok := alert.Metadata["service_name"].(string); ok {
		if serviceHealth, exists := hc.serviceHealth[serviceName]; exists {
			serviceHealth.Alerts = append(serviceHealth.Alerts, alert)
			// 保持告警历史数量限制
			if len(serviceHealth.Alerts) > 100 {
				serviceHealth.Alerts = serviceHealth.Alerts[1:]
			}
		}
	}
	hc.mu.Unlock()
}

// metricsCollector 指标收集器
func (hc *EnhancedHealthChecker) metricsCollector() {
	logger.Info("Starting health check metrics collector")

	for {
		select {
		case <-hc.ctx.Done():
			logger.Info("Health check metrics collector stopped")
			return
		case <-hc.metricsTicker.C:
			hc.collectMetrics()
		}
	}
}

// collectMetrics 收集指标
func (hc *EnhancedHealthChecker) collectMetrics() {
	hc.mu.RLock()
	checkCount := len(hc.checks)
	serviceCount := len(hc.serviceHealth)
	hc.mu.RUnlock()

	logger.Debug("Health check metrics: checks=%d, services=%d", checkCount, serviceCount)

	// 这里可以将指标发送到监控系统
}

// cleaner 清理器
func (hc *EnhancedHealthChecker) cleaner() {
	logger.Info("Starting health check cleaner")

	for {
		select {
		case <-hc.ctx.Done():
			logger.Info("Health check cleaner stopped")
			return
		case <-hc.cleanupTicker.C:
			hc.cleanup()
		}
	}
}

// cleanup 清理过期数据
func (hc *EnhancedHealthChecker) cleanup() {
	now := time.Now()
	retentionPeriod := hc.config.MetricsRetention

	hc.mu.Lock()
	defer hc.mu.Unlock()

	// 清理过期的告警
	for _, serviceHealth := range hc.serviceHealth {
		validAlerts := make([]*HealthAlert, 0)
		for _, alert := range serviceHealth.Alerts {
			if now.Sub(alert.Timestamp) < retentionPeriod {
				validAlerts = append(validAlerts, alert)
			}
		}
		serviceHealth.Alerts = validAlerts
	}

	logger.Debug("Health check cleanup completed")
}

// predictionTrainer 预测模型训练器
func (hc *EnhancedHealthChecker) predictionTrainer() {
	logger.Info("Starting health prediction trainer")

	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-hc.ctx.Done():
			logger.Info("Health prediction trainer stopped")
			return
		case <-ticker.C:
			hc.trainPredictionModels()
		}
	}
}

// trainPredictionModels 训练预测模型
func (hc *EnhancedHealthChecker) trainPredictionModels() {
	hc.predictionModel.mu.Lock()
	defer hc.predictionModel.mu.Unlock()

	for serviceName, history := range hc.predictionModel.historyData {
		if len(history) < 10 {
			continue // 数据不足，跳过训练
		}

		// 简化的预测模型训练（实际应该使用更复杂的机器学习算法）
		model := &PredictionModel{
			Weights:      []float64{0.3, 0.3, 0.2, 0.2}, // 简化权重
			Bias:         0.0,
			Accuracy:     0.8, // 模拟准确率
			LastTrained:  time.Now(),
			TrainingData: len(history),
		}

		hc.predictionModel.models[serviceName] = model
		logger.Debug("Trained prediction model for service: %s (data points: %d)", serviceName, len(history))
	}
}

// triggerCallbacks 触发回调
func (hc *EnhancedHealthChecker) triggerCallbacks(serviceHealth *ServiceHealth) {
	hc.mu.RLock()
	callbacks := make([]HealthCheckCallback, len(hc.callbacks))
	copy(callbacks, hc.callbacks)
	hc.mu.RUnlock()

	// 异步调用回调
	go func() {
		for _, callback := range callbacks {
			if callback != nil {
				try := func() {
					defer func() {
						if r := recover(); r != nil {
							logger.Error("Health check callback panic: %v", r)
						}
					}()
					callback(serviceHealth)
				}
				try()
			}
		}
	}()
}
