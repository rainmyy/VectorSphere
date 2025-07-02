package enhanced

import (
	"VectorSphere/src/library/logger"
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// LoadBalancingAlgorithm 负载均衡算法
type LoadBalancingAlgorithm int

// ServiceDiscoverySource 定义了服务发现源的接口
// 负载均衡器可以通过该接口获取服务实例信息
type ServiceDiscoverySource interface {
	// DiscoverServices 根据服务名称和过滤条件发现服务
	DiscoverServices(ctx context.Context, serviceName string, filter *ServiceFilter) ([]*ServiceMetadata, error)
}

// 添加服务发现源字段到EnhancedLoadBalancer结构体
var _ ServiceDiscoverySource = (*EnhancedServiceRegistry)(nil) // 确保EnhancedServiceRegistry实现了ServiceDiscoverySource接口

const (
	RoundRobin LoadBalancingAlgorithm = iota
	WeightedRoundRobin
	LeastConnections
	WeightedLeastConnections
	Random
	WeightedRandom
	IPHash
	ConsistentHash
	LeastResponseTime
	ResourceBased
	Adaptive
)

// BackendStatus 后端状态
type BackendStatus int

const (
	BackendHealthy BackendStatus = iota
	BackendDegraded
	BackendUnhealthy
	BackendMaintenance
	BackendDraining
)

// Backend 后端服务器
type Backend struct {
	ID              string                 `json:"id"`
	Address         string                 `json:"address"`
	Port            int                    `json:"port"`
	Weight          int                    `json:"weight"`
	CurrentWeight   int                    `json:"current_weight"`
	EffectiveWeight int                    `json:"effective_weight"`
	Status          BackendStatus          `json:"status"`
	HealthScore     float64                `json:"health_score"`
	Connections     int64                  `json:"connections"`
	MaxConnections  int64                  `json:"max_connections"`
	ResponseTime    time.Duration          `json:"response_time"`
	CPUUsage        float64                `json:"cpu_usage"`
	MemoryUsage     float64                `json:"memory_usage"`
	Throughput      float64                `json:"throughput"`
	ErrorRate       float64                `json:"error_rate"`
	LastCheck       time.Time              `json:"last_check"`
	FailureCount    int64                  `json:"failure_count"`
	SuccessCount    int64                  `json:"success_count"`
	TotalRequests   int64                  `json:"total_requests"`
	Metadata        map[string]interface{} `json:"metadata"`
	Tags            []string               `json:"tags"`
	Region          string                 `json:"region"`
	Zone            string                 `json:"zone"`
	Version         string                 `json:"version"`
	Mu              sync.RWMutex           `json:"-"`
}

// LoadBalancerConfig 负载均衡器配置
type LoadBalancerConfig struct {
	Algorithm           LoadBalancingAlgorithm `json:"algorithm"`
	HealthCheckEnabled  bool                   `json:"health_check_enabled"`
	HealthCheckInterval time.Duration          `json:"health_check_interval"`
	HealthCheckTimeout  time.Duration          `json:"health_check_timeout"`
	FailureThreshold    int                    `json:"failure_threshold"`
	RecoveryThreshold   int                    `json:"recovery_threshold"`
	MaxRetries          int                    `json:"max_retries"`
	RetryTimeout        time.Duration          `json:"retry_timeout"`
	SessionAffinity     bool                   `json:"session_affinity"`
	AffinityTimeout     time.Duration          `json:"affinity_timeout"`
	SlowStart           bool                   `json:"slow_start"`
	SlowStartDuration   time.Duration          `json:"slow_start_duration"`
	CircuitBreaker      bool                   `json:"circuit_breaker"`
	MetricsEnabled      bool                   `json:"metrics_enabled"`
	BasePrefix          string                 `json:"base_prefix"`
}

// LoadBalancerMetrics 负载均衡器指标
type LoadBalancerMetrics struct {
	TotalRequests       int64         `json:"total_requests"`
	SuccessfulRequests  int64         `json:"successful_requests"`
	FailedRequests      int64         `json:"failed_requests"`
	AverageResponseTime time.Duration `json:"average_response_time"`
	Throughput          float64       `json:"throughput"`
	ActiveConnections   int64         `json:"active_connections"`
	BackendCount        int           `json:"backend_count"`
	HealthyBackends     int           `json:"healthy_backends"`
	UnhealthyBackends   int           `json:"unhealthy_backends"`
	LastUpdate          time.Time     `json:"last_update"`
}

// SessionInfo 会话信息
type SessionInfo struct {
	SessionID    string    `json:"session_id"`
	BackendID    string    `json:"backend_id"`
	CreatedAt    time.Time `json:"created_at"`
	LastAccess   time.Time `json:"last_access"`
	RequestCount int64     `json:"request_count"`
}

// RequestContext 请求上下文
type RequestContext struct {
	ClientIP  string                 `json:"client_ip"`
	SessionID string                 `json:"session_id"`
	Headers   map[string]string      `json:"headers"`
	Path      string                 `json:"path"`
	Method    string                 `json:"method"`
	Timestamp time.Time              `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// LoadBalancingResult 负载均衡结果
type LoadBalancingResult struct {
	Backend    *Backend      `json:"backend"`
	Algorithm  string        `json:"algorithm"`
	Latency    time.Duration `json:"latency"`
	Error      error         `json:"error,omitempty"`
	RetryCount int           `json:"retry_count"`
	Timestamp  time.Time     `json:"timestamp"`
}

// ConsistentHashRing 一致性哈希环
type ConsistentHashRing struct {
	ring         map[uint32]*Backend
	sortedKeys   []uint32
	virtualNodes int
	mu           sync.RWMutex
}

// EnhancedLoadBalancer 增强负载均衡器
type EnhancedLoadBalancer struct {
	client           *clientv3.Client
	config           *LoadBalancerConfig
	backends         map[string]*Backend
	healthyBackends  []*Backend
	mu               sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
	metrics          *LoadBalancerMetrics
	sessions         map[string]*SessionInfo
	sessionMu        sync.RWMutex
	consistentHash   *ConsistentHashRing
	roundRobinIndex  int64
	isRunning        int32
	healthChecker    *EnhancedHealthChecker
	circuitBreakers  map[string]*CircuitBreaker
	basePrefix       string
	metricsTicker    *time.Ticker
	cleanupTicker    *time.Ticker
	httpClient       *http.Client
	serviceDiscovery ServiceDiscoverySource // 服务发现源
}

// NewEnhancedLoadBalancer 创建增强负载均衡器
func NewEnhancedLoadBalancer(client *clientv3.Client, config *LoadBalancerConfig) *EnhancedLoadBalancer {
	if config == nil {
		config = &LoadBalancerConfig{
			Algorithm:           RoundRobin,
			HealthCheckEnabled:  true,
			HealthCheckInterval: 30 * time.Second,
			HealthCheckTimeout:  10 * time.Second,
			FailureThreshold:    3,
			RecoveryThreshold:   2,
			MaxRetries:          3,
			RetryTimeout:        5 * time.Second,
			SessionAffinity:     false,
			AffinityTimeout:     30 * time.Minute,
			SlowStart:           true,
			SlowStartDuration:   5 * time.Minute,
			CircuitBreaker:      true,
			MetricsEnabled:      true,
			BasePrefix:          "/vector_sphere/load_balancer",
		}
	}

	ctx, cancel := context.WithCancel(context.Background())

	lb := &EnhancedLoadBalancer{
		client:          client,
		config:          config,
		backends:        make(map[string]*Backend),
		healthyBackends: make([]*Backend, 0),
		ctx:             ctx,
		cancel:          cancel,
		metrics:         &LoadBalancerMetrics{},
		sessions:        make(map[string]*SessionInfo),
		consistentHash:  NewConsistentHashRing(100), // 100个虚拟节点
		circuitBreakers: make(map[string]*CircuitBreaker),
		basePrefix:      config.BasePrefix,
		httpClient: &http.Client{
			Timeout: config.HealthCheckTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}

	// 创建健康检查器
	if config.HealthCheckEnabled {
		healthConfig := &HealthCheckConfig{
			Enabled:         true,
			DefaultInterval: config.HealthCheckInterval,
			DefaultTimeout:  config.HealthCheckTimeout,
			BasePrefix:      config.BasePrefix + "/health",
		}
		lb.healthChecker = NewEnhancedHealthChecker(client, healthConfig)
	}

	logger.Info("Enhanced load balancer created with algorithm: %d", config.Algorithm)
	return lb
}

// Start 启动负载均衡器
func (lb *EnhancedLoadBalancer) Start() error {
	if !atomic.CompareAndSwapInt32(&lb.isRunning, 0, 1) {
		return fmt.Errorf("load balancer is already running")
	}

	logger.Info("Starting enhanced load balancer")

	// 启动健康检查器
	if lb.healthChecker != nil {
		if err := lb.healthChecker.Start(); err != nil {
			logger.Error("Failed to start health checker: %v", err)
			return err
		}
	}

	// 启动指标收集
	if lb.config.MetricsEnabled {
		lb.metricsTicker = time.NewTicker(60 * time.Second)
		go lb.metricsCollector()
	}

	// 启动清理器
	lb.cleanupTicker = time.NewTicker(10 * time.Minute)
	go lb.cleaner()

	// 启动后端监控
	go lb.backendMonitor()

	logger.Info("Enhanced load balancer started successfully")
	return nil
}

// Stop 停止负载均衡器
func (lb *EnhancedLoadBalancer) Stop() error {
	if !atomic.CompareAndSwapInt32(&lb.isRunning, 1, 0) {
		return fmt.Errorf("load balancer is not running")
	}

	logger.Info("Stopping enhanced load balancer")

	// 停止健康检查器
	if lb.healthChecker != nil {
		lb.healthChecker.Stop()
	}

	// 停止定时器
	if lb.metricsTicker != nil {
		lb.metricsTicker.Stop()
	}
	if lb.cleanupTicker != nil {
		lb.cleanupTicker.Stop()
	}

	// 取消上下文
	lb.cancel()

	logger.Info("Enhanced load balancer stopped")
	return nil
}

// AddBackend 添加后端服务器
func (lb *EnhancedLoadBalancer) AddBackend(backend *Backend) error {
	if backend == nil {
		return fmt.Errorf("backend cannot be nil")
	}

	if backend.ID == "" {
		return fmt.Errorf("backend ID cannot be empty")
	}

	// 设置默认值
	if backend.Weight == 0 {
		backend.Weight = 1
	}
	backend.CurrentWeight = 0
	backend.EffectiveWeight = backend.Weight
	backend.Status = BackendHealthy
	backend.HealthScore = 100.0
	backend.LastCheck = time.Now()
	if backend.MaxConnections == 0 {
		backend.MaxConnections = 1000
	}
	if backend.Metadata == nil {
		backend.Metadata = make(map[string]interface{})
	}

	lb.mu.Lock()
	lb.backends[backend.ID] = backend
	lb.updateHealthyBackends()
	lb.mu.Unlock()

	// 添加到一致性哈希环
	lb.consistentHash.AddBackend(backend)

	// 注册健康检查
	if lb.healthChecker != nil {
		lb.registerBackendHealthCheck(backend)
	}

	// 注册熔断器
	if lb.config.CircuitBreaker {
		lb.registerCircuitBreaker(backend)
	}

	logger.Info("Backend added: %s (%s:%d)", backend.ID, backend.Address, backend.Port)
	return nil
}

// RemoveBackend 移除后端服务器
func (lb *EnhancedLoadBalancer) RemoveBackend(backendID string) error {
	lb.mu.Lock()
	backend, exists := lb.backends[backendID]
	if exists {
		delete(lb.backends, backendID)
		lb.updateHealthyBackends()
	}
	lb.mu.Unlock()

	if !exists {
		return fmt.Errorf("backend not found: %s", backendID)
	}

	// 从一致性哈希环移除
	lb.consistentHash.RemoveBackend(backend)

	// 注销健康检查
	if lb.healthChecker != nil {
		err := lb.healthChecker.UnregisterCheck(fmt.Sprintf("backend_%s", backendID))
		if err != nil {
			logger.Error("unable to unregister health checker: %v", err)
		}
	}

	// 移除熔断器
	delete(lb.circuitBreakers, backendID)

	logger.Info("Backend removed: %s", backendID)
	return nil
}

// UpdateBackend 更新后端服务器
func (lb *EnhancedLoadBalancer) UpdateBackend(backend *Backend) error {
	if backend == nil || backend.ID == "" {
		return fmt.Errorf("invalid backend")
	}

	lb.mu.Lock()
	existingBackend, exists := lb.backends[backend.ID]
	if !exists {
		lb.mu.Unlock()
		return fmt.Errorf("backend not found: %s", backend.ID)
	}

	// 更新后端信息
	existingBackend.Address = backend.Address
	existingBackend.Port = backend.Port
	existingBackend.Weight = backend.Weight
	existingBackend.EffectiveWeight = backend.Weight
	existingBackend.MaxConnections = backend.MaxConnections
	existingBackend.Metadata = backend.Metadata
	existingBackend.Tags = backend.Tags
	existingBackend.Region = backend.Region
	existingBackend.Zone = backend.Zone
	existingBackend.Version = backend.Version

	lb.updateHealthyBackends()
	lb.mu.Unlock()

	// 更新一致性哈希环
	lb.consistentHash.UpdateBackend(existingBackend)

	logger.Info("Backend updated: %s", backend.ID)
	return nil
}

// SelectBackend 选择后端服务器
func (lb *EnhancedLoadBalancer) SelectBackend(ctx *RequestContext) *LoadBalancingResult {
	start := time.Now()
	result := &LoadBalancingResult{
		Timestamp: start,
		Algorithm: lb.getAlgorithmName(),
	}

	// 检查会话亲和性
	if lb.config.SessionAffinity && ctx.SessionID != "" {
		if backend := lb.getSessionBackend(ctx.SessionID); backend != nil {
			result.Backend = backend
			result.Latency = time.Since(start)
			return result
		}
	}

	// 获取健康的后端列表
	lb.mu.RLock()
	healthyBackends := make([]*Backend, len(lb.healthyBackends))
	copy(healthyBackends, lb.healthyBackends)
	lb.mu.RUnlock()

	if len(healthyBackends) == 0 {
		result.Error = fmt.Errorf("no healthy backends available")
		result.Latency = time.Since(start)
		return result
	}

	// 根据算法选择后端
	var backend *Backend
	switch lb.config.Algorithm {
	case RoundRobin:
		backend = lb.selectRoundRobin(healthyBackends)
	case WeightedRoundRobin:
		backend = lb.selectWeightedRoundRobin(healthyBackends)
	case LeastConnections:
		backend = lb.selectLeastConnections(healthyBackends)
	case WeightedLeastConnections:
		backend = lb.selectWeightedLeastConnections(healthyBackends)
	case Random:
		backend = lb.selectRandom(healthyBackends)
	case WeightedRandom:
		backend = lb.selectWeightedRandom(healthyBackends)
	case IPHash:
		backend = lb.selectIPHash(healthyBackends, ctx.ClientIP)
	case ConsistentHash:
		backend = lb.selectConsistentHash(ctx.ClientIP)
	case LeastResponseTime:
		backend = lb.selectLeastResponseTime(healthyBackends)
	case ResourceBased:
		backend = lb.selectResourceBased(healthyBackends)
	case Adaptive:
		backend = lb.selectAdaptive(healthyBackends, ctx)
	default:
		backend = lb.selectRoundRobin(healthyBackends)
	}

	if backend == nil {
		result.Error = fmt.Errorf("failed to select backend")
		result.Latency = time.Since(start)
		return result
	}

	// 检查熔断器
	if lb.config.CircuitBreaker {
		if cb, exists := lb.circuitBreakers[backend.ID]; exists {
			if !cb.AllowRequest() {
				// 熔断器打开，尝试选择其他后端
				logger.Warning("Circuit breaker open for backend %s, selecting alternative", backend.ID)
				backend = lb.selectAlternativeBackend(healthyBackends, backend.ID)
				if backend == nil {
					result.Error = fmt.Errorf("all backends circuit breaker open")
					result.Latency = time.Since(start)
					return result
				}
			}
		}
	}

	// 检查连接数限制
	if backend.Connections >= backend.MaxConnections {
		logger.Warning("Backend %s reached max connections, selecting alternative", backend.ID)
		backend = lb.selectAlternativeBackend(healthyBackends, backend.ID)
		if backend == nil {
			result.Error = fmt.Errorf("all backends reached max connections")
			result.Latency = time.Since(start)
			return result
		}
	}

	// 更新会话亲和性
	if lb.config.SessionAffinity && ctx.SessionID != "" {
		lb.updateSession(ctx.SessionID, backend.ID)
	}

	// 增加连接数
	atomic.AddInt64(&backend.Connections, 1)
	atomic.AddInt64(&backend.TotalRequests, 1)

	// 更新指标
	atomic.AddInt64(&lb.metrics.TotalRequests, 1)
	atomic.AddInt64(&lb.metrics.ActiveConnections, 1)

	result.Backend = backend
	result.Latency = time.Since(start)

	logger.Debug("Selected backend: %s (algorithm: %s, latency: %v)", backend.ID, result.Algorithm, result.Latency)
	return result
}

// ReleaseBackend 释放后端连接
func (lb *EnhancedLoadBalancer) ReleaseBackend(backendID string, success bool, responseTime time.Duration) {
	lb.mu.RLock()
	backend, exists := lb.backends[backendID]
	lb.mu.RUnlock()

	if !exists {
		logger.Warning("Attempted to release unknown backend: %s", backendID)
		return
	}

	// 减少连接数
	atomic.AddInt64(&backend.Connections, -1)
	atomic.AddInt64(&lb.metrics.ActiveConnections, -1)

	// 更新响应时间
	backend.Mu.Lock()
	if backend.ResponseTime == 0 {
		backend.ResponseTime = responseTime
	} else {
		backend.ResponseTime = (backend.ResponseTime + responseTime) / 2
	}
	backend.Mu.Unlock()

	// 更新成功/失败计数
	if success {
		atomic.AddInt64(&backend.SuccessCount, 1)
		atomic.AddInt64(&lb.metrics.SuccessfulRequests, 1)

		// 重置失败计数
		atomic.StoreInt64(&backend.FailureCount, 0)

		// 更新熔断器
		if cb, exists := lb.circuitBreakers[backendID]; exists {
			cb.recordSuccess()
		}
	} else {
		atomic.AddInt64(&backend.FailureCount, 1)
		atomic.AddInt64(&lb.metrics.FailedRequests, 1)

		// 更新熔断器
		if cb, exists := lb.circuitBreakers[backendID]; exists {
			cb.RecordFailure()
		}

		// 检查是否需要标记为不健康
		if atomic.LoadInt64(&backend.FailureCount) >= int64(lb.config.FailureThreshold) {
			lb.markBackendUnhealthy(backendID)
		}
	}

	logger.Debug("Released backend: %s (success: %t, response_time: %v)", backendID, success, responseTime)
}

// GetBackends 获取所有后端
func (lb *EnhancedLoadBalancer) GetBackends() map[string]*Backend {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	result := make(map[string]*Backend)
	for id, backend := range lb.backends {
		// 创建不包含锁的副本
		backendCopy := &Backend{
			ID:              backend.ID,
			Address:         backend.Address,
			Port:            backend.Port,
			Weight:          backend.Weight,
			CurrentWeight:   backend.CurrentWeight,
			EffectiveWeight: backend.EffectiveWeight,
			Status:          backend.Status,
			HealthScore:     backend.HealthScore,
			Connections:     backend.Connections,
			MaxConnections:  backend.MaxConnections,
			ResponseTime:    backend.ResponseTime,
			CPUUsage:        backend.CPUUsage,
			MemoryUsage:     backend.MemoryUsage,
			Throughput:      backend.Throughput,
			ErrorRate:       backend.ErrorRate,
			LastCheck:       backend.LastCheck,
			FailureCount:    backend.FailureCount,
			SuccessCount:    backend.SuccessCount,
			TotalRequests:   backend.TotalRequests,
			Metadata:        backend.Metadata,
			Tags:            backend.Tags,
			Region:          backend.Region,
			Zone:            backend.Zone,
			Version:         backend.Version,
		}
		result[id] = backendCopy
	}
	return result
}

// GetHealthyBackends 获取健康的后端
func (lb *EnhancedLoadBalancer) GetHealthyBackends() []*Backend {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	result := make([]*Backend, len(lb.healthyBackends))
	for i, backend := range lb.healthyBackends {
		// 创建不包含锁的副本
		backendCopy := &Backend{
			ID:              backend.ID,
			Address:         backend.Address,
			Port:            backend.Port,
			Weight:          backend.Weight,
			CurrentWeight:   backend.CurrentWeight,
			EffectiveWeight: backend.EffectiveWeight,
			Status:          backend.Status,
			HealthScore:     backend.HealthScore,
			Connections:     backend.Connections,
			MaxConnections:  backend.MaxConnections,
			ResponseTime:    backend.ResponseTime,
			CPUUsage:        backend.CPUUsage,
			MemoryUsage:     backend.MemoryUsage,
			Throughput:      backend.Throughput,
			ErrorRate:       backend.ErrorRate,
			LastCheck:       backend.LastCheck,
			FailureCount:    backend.FailureCount,
			SuccessCount:    backend.SuccessCount,
			TotalRequests:   backend.TotalRequests,
			Metadata:        backend.Metadata,
			Tags:            backend.Tags,
			Region:          backend.Region,
			Zone:            backend.Zone,
			Version:         backend.Version,
		}
		result[i] = backendCopy
	}
	return result
}

// GetMetrics 获取负载均衡器指标
func (lb *EnhancedLoadBalancer) GetMetrics() *LoadBalancerMetrics {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	metricsCopy := *lb.metrics
	metricsCopy.BackendCount = len(lb.backends)
	metricsCopy.HealthyBackends = len(lb.healthyBackends)
	metricsCopy.UnhealthyBackends = len(lb.backends) - len(lb.healthyBackends)
	metricsCopy.LastUpdate = time.Now()

	return &metricsCopy
}

// 负载均衡算法实现

// selectRoundRobin 轮询算法
func (lb *EnhancedLoadBalancer) selectRoundRobin(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	index := atomic.AddInt64(&lb.roundRobinIndex, 1) % int64(len(backends))
	return backends[index]
}

// selectWeightedRoundRobin 加权轮询算法
func (lb *EnhancedLoadBalancer) selectWeightedRoundRobin(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	totalWeight := 0
	var selected *Backend

	for _, backend := range backends {
		backend.Mu.Lock()
		backend.CurrentWeight += backend.EffectiveWeight
		totalWeight += backend.EffectiveWeight

		if selected == nil || backend.CurrentWeight > selected.CurrentWeight {
			selected = backend
		}
		backend.Mu.Unlock()
	}

	if selected != nil {
		selected.Mu.Lock()
		selected.CurrentWeight -= totalWeight
		selected.Mu.Unlock()
	}

	return selected
}

// selectLeastConnections 最少连接算法
func (lb *EnhancedLoadBalancer) selectLeastConnections(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	var selected *Backend
	minConnections := int64(math.MaxInt64)

	for _, backend := range backends {
		connections := atomic.LoadInt64(&backend.Connections)
		if connections < minConnections {
			minConnections = connections
			selected = backend
		}
	}

	return selected
}

// selectWeightedLeastConnections 加权最少连接算法
func (lb *EnhancedLoadBalancer) selectWeightedLeastConnections(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	var selected *Backend
	minRatio := float64(math.MaxFloat64)

	for _, backend := range backends {
		connections := atomic.LoadInt64(&backend.Connections)
		ratio := float64(connections) / float64(backend.Weight)
		if ratio < minRatio {
			minRatio = ratio
			selected = backend
		}
	}

	return selected
}

// selectRandom 随机算法
func (lb *EnhancedLoadBalancer) selectRandom(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	index := rand.Intn(len(backends))
	return backends[index]
}

// selectWeightedRandom 加权随机算法
func (lb *EnhancedLoadBalancer) selectWeightedRandom(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	totalWeight := 0
	for _, backend := range backends {
		totalWeight += backend.Weight
	}

	if totalWeight == 0 {
		return lb.selectRandom(backends)
	}

	randomWeight := rand.Intn(totalWeight)
	currentWeight := 0

	for _, backend := range backends {
		currentWeight += backend.Weight
		if randomWeight < currentWeight {
			return backend
		}
	}

	return backends[len(backends)-1]
}

// selectIPHash IP哈希算法
func (lb *EnhancedLoadBalancer) selectIPHash(backends []*Backend, clientIP string) *Backend {
	if len(backends) == 0 {
		return nil
	}

	hash := fnv.New32a()
	_, err := hash.Write([]byte(clientIP))
	if err != nil {
		logger.Error("hash write failed:%v", err)
	}
	index := hash.Sum32() % uint32(len(backends))

	return backends[index]
}

// selectConsistentHash 一致性哈希算法
func (lb *EnhancedLoadBalancer) selectConsistentHash(key string) *Backend {
	return lb.consistentHash.GetBackend(key)
}

// selectLeastResponseTime 最短响应时间算法
func (lb *EnhancedLoadBalancer) selectLeastResponseTime(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	var selected *Backend
	minResponseTime := time.Duration(math.MaxInt64)

	for _, backend := range backends {
		backend.Mu.RLock()
		responseTime := backend.ResponseTime
		backend.Mu.RUnlock()

		if responseTime < minResponseTime {
			minResponseTime = responseTime
			selected = backend
		}
	}

	return selected
}

// selectResourceBased 基于资源的算法
func (lb *EnhancedLoadBalancer) selectResourceBased(backends []*Backend) *Backend {
	if len(backends) == 0 {
		return nil
	}

	var selected *Backend
	bestScore := float64(-1)

	for _, backend := range backends {
		// 计算资源分数（CPU使用率越低越好，内存使用率越低越好）
		cpuScore := 100.0 - backend.CPUUsage
		memoryScore := 100.0 - backend.MemoryUsage
		connectionScore := 100.0 - (float64(backend.Connections)/float64(backend.MaxConnections))*100.0

		// 综合分数
		score := (cpuScore + memoryScore + connectionScore) / 3.0

		if score > bestScore {
			bestScore = score
			selected = backend
		}
	}

	return selected
}

// selectAdaptive 自适应算法
func (lb *EnhancedLoadBalancer) selectAdaptive(backends []*Backend, ctx *RequestContext) *Backend {
	if len(backends) == 0 {
		return nil
	}

	// 根据当前系统状态选择最合适的算法
	totalConnections := int64(0)
	for _, backend := range backends {
		totalConnections += atomic.LoadInt64(&backend.Connections)
	}

	avgConnections := float64(totalConnections) / float64(len(backends))

	// 如果平均连接数较低，使用轮询
	if avgConnections < 10 {
		return lb.selectRoundRobin(backends)
	}

	// 如果连接数不均衡，使用最少连接
	if lb.isConnectionImbalanced(backends) {
		return lb.selectLeastConnections(backends)
	}

	// 如果有会话ID，使用一致性哈希
	if ctx.SessionID != "" {
		return lb.selectConsistentHash(ctx.SessionID)
	}

	// 默认使用加权轮询
	return lb.selectWeightedRoundRobin(backends)
}

// selectAlternativeBackend 选择替代后端
func (lb *EnhancedLoadBalancer) selectAlternativeBackend(backends []*Backend, excludeID string) *Backend {
	alternatives := make([]*Backend, 0)
	for _, backend := range backends {
		if backend.ID != excludeID {
			// 检查熔断器状态
			if cb, exists := lb.circuitBreakers[backend.ID]; exists {
				if !cb.AllowRequest() {
					continue
				}
			}

			// 检查连接数限制
			if backend.Connections >= backend.MaxConnections {
				continue
			}

			alternatives = append(alternatives, backend)
		}
	}

	if len(alternatives) == 0 {
		return nil
	}

	// 使用最少连接算法选择替代后端
	return lb.selectLeastConnections(alternatives)
}

// 辅助方法

// updateHealthyBackends 更新健康后端列表
func (lb *EnhancedLoadBalancer) updateHealthyBackends() {
	healthyBackends := make([]*Backend, 0)
	for _, backend := range lb.backends {
		if backend.Status == BackendHealthy || backend.Status == BackendDegraded {
			healthyBackends = append(healthyBackends, backend)
		}
	}
	lb.healthyBackends = healthyBackends
}

// markBackendUnhealthy 标记后端为不健康
func (lb *EnhancedLoadBalancer) markBackendUnhealthy(backendID string) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	if backend, exists := lb.backends[backendID]; exists {
		backend.Status = BackendUnhealthy
		lb.updateHealthyBackends()
		logger.Warning("Backend marked as unhealthy: %s", backendID)
	}
}

// markBackendHealthy 标记后端为健康
func (lb *EnhancedLoadBalancer) markBackendHealthy(backendID string) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	if backend, exists := lb.backends[backendID]; exists {
		backend.Status = BackendHealthy
		atomic.StoreInt64(&backend.FailureCount, 0)
		lb.updateHealthyBackends()
		logger.Info("Backend marked as healthy: %s", backendID)
	}
}

// isConnectionImbalanced 检查连接是否不均衡
func (lb *EnhancedLoadBalancer) isConnectionImbalanced(backends []*Backend) bool {
	if len(backends) < 2 {
		return false
	}

	connections := make([]int64, len(backends))
	for i, backend := range backends {
		connections[i] = atomic.LoadInt64(&backend.Connections)
	}

	// 计算标准差
	var sum, mean, variance float64
	for _, conn := range connections {
		sum += float64(conn)
	}
	mean = sum / float64(len(connections))

	for _, conn := range connections {
		variance += math.Pow(float64(conn)-mean, 2)
	}
	variance /= float64(len(connections))
	stdDev := math.Sqrt(variance)

	// 如果标准差大于平均值的50%，认为不均衡
	return stdDev > mean*0.5
}

// getAlgorithmName 获取算法名称
func (lb *EnhancedLoadBalancer) getAlgorithmName() string {
	switch lb.config.Algorithm {
	case RoundRobin:
		return "round_robin"
	case WeightedRoundRobin:
		return "weighted_round_robin"
	case LeastConnections:
		return "least_connections"
	case WeightedLeastConnections:
		return "weighted_least_connections"
	case Random:
		return "random"
	case WeightedRandom:
		return "weighted_random"
	case IPHash:
		return "ip_hash"
	case ConsistentHash:
		return "consistent_hash"
	case LeastResponseTime:
		return "least_response_time"
	case ResourceBased:
		return "resource_based"
	case Adaptive:
		return "adaptive"
	default:
		return "unknown"
	}
}

// 会话管理

// getSessionBackend 获取会话对应的后端
func (lb *EnhancedLoadBalancer) getSessionBackend(sessionID string) *Backend {
	lb.sessionMu.RLock()
	session, exists := lb.sessions[sessionID]
	lb.sessionMu.RUnlock()

	if !exists {
		return nil
	}

	// 检查会话是否过期
	if time.Since(session.LastAccess) > lb.config.AffinityTimeout {
		lb.sessionMu.Lock()
		delete(lb.sessions, sessionID)
		lb.sessionMu.Unlock()
		return nil
	}

	// 检查后端是否仍然健康
	lb.mu.RLock()
	backend, exists := lb.backends[session.BackendID]
	lb.mu.RUnlock()

	if !exists || backend.Status != BackendHealthy {
		lb.sessionMu.Lock()
		delete(lb.sessions, sessionID)
		lb.sessionMu.Unlock()
		return nil
	}

	return backend
}

// updateSession 更新会话信息
func (lb *EnhancedLoadBalancer) updateSession(sessionID, backendID string) {
	lb.sessionMu.Lock()
	defer lb.sessionMu.Unlock()

	session, exists := lb.sessions[sessionID]
	if !exists {
		session = &SessionInfo{
			SessionID: sessionID,
			BackendID: backendID,
			CreatedAt: time.Now(),
		}
		lb.sessions[sessionID] = session
	}

	session.LastAccess = time.Now()
	session.RequestCount++
}

// 健康检查集成

// registerBackendHealthCheck 注册后端健康检查
func (lb *EnhancedLoadBalancer) registerBackendHealthCheck(backend *Backend) {
	if lb.healthChecker == nil {
		return
	}

	check := &HealthCheck{
		ID:       fmt.Sprintf("backend_%s", backend.ID),
		Name:     fmt.Sprintf("Backend %s Health Check", backend.ID),
		Type:     HTTPCheck,
		Level:    BasicLevel,
		Interval: lb.config.HealthCheckInterval,
		Timeout:  lb.config.HealthCheckTimeout,
		Retries:  3,
		Weight:   1.0,
		Critical: true,
		Enabled:  true,
		Config: map[string]interface{}{
			"url":             fmt.Sprintf("http://%s:%d/health", backend.Address, backend.Port),
			"expected_status": 200,
			"service_name":    fmt.Sprintf("backend_%s", backend.ID),
		},
		Tags:     []string{"backend", "load_balancer"},
		Adaptive: true,
	}

	// 添加健康检查回调
	lb.healthChecker.AddCallback(func(serviceHealth *ServiceHealth) {
		if serviceHealth.ServiceName == fmt.Sprintf("backend_%s", backend.ID) {
			lb.updateBackendHealth(backend.ID, serviceHealth)
		}
	})

	lb.healthChecker.RegisterCheck(check)
}

// updateBackendHealth 更新后端健康状态
func (lb *EnhancedLoadBalancer) updateBackendHealth(backendID string, serviceHealth *ServiceHealth) {
	lb.mu.Lock()
	backend, exists := lb.backends[backendID]
	if !exists {
		lb.mu.Unlock()
		return
	}

	// 更新健康分数
	backend.HealthScore = serviceHealth.OverallScore
	backend.LastCheck = serviceHealth.LastUpdate

	// 更新状态
	oldStatus := backend.Status
	switch serviceHealth.OverallStatus {
	case Healthy:
		backend.Status = BackendHealthy
	case Degraded:
		backend.Status = BackendDegraded
	case Unhealthy, Critical:
		backend.Status = BackendUnhealthy
	default:
		backend.Status = BackendUnhealthy
	}

	// 如果状态发生变化，更新健康后端列表
	if oldStatus != backend.Status {
		lb.updateHealthyBackends()
		logger.Info("Backend %s status changed from %d to %d", backendID, oldStatus, backend.Status)
	}

	lb.mu.Unlock()
}

// 熔断器集成

// registerCircuitBreaker 注册熔断器
func (lb *EnhancedLoadBalancer) registerCircuitBreaker(backend *Backend) {
	config := &CircuitBreakerConfig{
		FailureThreshold:  lb.config.FailureThreshold,
		SuccessThreshold:  lb.config.RecoveryThreshold,
		Timeout:           lb.config.RetryTimeout,
		ResetTimeout:      30 * time.Second,
		HalfOpenMaxCalls:  5,
		FailureRate:       0.5,
		MinRequestAmount:  10,
		SlidingWindowSize: 100,
	}

	cb := &CircuitBreaker{
		Name:            fmt.Sprintf("backend_%s", backend.ID),
		config:          config,
		state:           Closed,
		lastStateChange: time.Now(),
		metrics: &CircuitBreakerMetrics{
			CurrentState: "closed",
		},
	}

	lb.circuitBreakers[backend.ID] = cb
}

// 监控和指标收集

// backendMonitor 后端监控
func (lb *EnhancedLoadBalancer) backendMonitor() {
	logger.Info("Starting backend monitor")

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-lb.ctx.Done():
			logger.Info("Backend monitor stopped")
			return
		case <-ticker.C:
			lb.monitorBackends()
		}
	}
}

// monitorBackends 监控后端状态
func (lb *EnhancedLoadBalancer) monitorBackends() {
	lb.mu.RLock()
	backends := make([]*Backend, 0, len(lb.backends))
	for _, backend := range lb.backends {
		backends = append(backends, backend)
	}
	lb.mu.RUnlock()

	for _, backend := range backends {
		// 检查慢启动
		if lb.config.SlowStart {
			lb.updateSlowStartWeight(backend)
		}

		// 更新错误率
		lb.updateErrorRate(backend)

		// 检查恢复条件
		if backend.Status == BackendUnhealthy {
			lb.checkRecovery(backend)
		}
	}
}

// updateSlowStartWeight 更新慢启动权重
func (lb *EnhancedLoadBalancer) updateSlowStartWeight(backend *Backend) {
	if backend.Status != BackendHealthy {
		return
	}

	// 计算启动时间
	startupTime := time.Since(backend.LastCheck)
	if startupTime > lb.config.SlowStartDuration {
		// 慢启动完成，使用正常权重
		backend.EffectiveWeight = backend.Weight
		return
	}

	// 慢启动期间，逐渐增加权重
	progress := float64(startupTime) / float64(lb.config.SlowStartDuration)
	backend.EffectiveWeight = int(float64(backend.Weight) * progress)
	if backend.EffectiveWeight < 1 {
		backend.EffectiveWeight = 1
	}
}

// updateErrorRate 更新错误率
func (lb *EnhancedLoadBalancer) updateErrorRate(backend *Backend) {
	totalRequests := atomic.LoadInt64(&backend.TotalRequests)
	failureCount := atomic.LoadInt64(&backend.FailureCount)

	if totalRequests > 0 {
		backend.ErrorRate = float64(failureCount) / float64(totalRequests)
	} else {
		backend.ErrorRate = 0
	}
}

// checkRecovery 检查恢复条件
func (lb *EnhancedLoadBalancer) checkRecovery(backend *Backend) {
	successCount := atomic.LoadInt64(&backend.SuccessCount)
	if successCount >= int64(lb.config.RecoveryThreshold) {
		lb.markBackendHealthy(backend.ID)
	}
}

// metricsCollector 指标收集器
func (lb *EnhancedLoadBalancer) metricsCollector() {
	logger.Info("Starting load balancer metrics collector")

	for {
		select {
		case <-lb.ctx.Done():
			logger.Info("Load balancer metrics collector stopped")
			return
		case <-lb.metricsTicker.C:
			lb.collectMetrics()
		}
	}
}

// collectMetrics 收集指标
func (lb *EnhancedLoadBalancer) collectMetrics() {
	lb.mu.RLock()
	totalBackends := len(lb.backends)
	healthyBackends := len(lb.healthyBackends)
	lb.mu.RUnlock()

	lb.sessionMu.RLock()
	activeSessions := len(lb.sessions)
	lb.sessionMu.RUnlock()

	logger.Debug("Load balancer metrics: total_backends=%d, healthy_backends=%d, active_sessions=%d",
		totalBackends, healthyBackends, activeSessions)

	// 更新指标
	lb.metrics.BackendCount = totalBackends
	lb.metrics.HealthyBackends = healthyBackends
	lb.metrics.UnhealthyBackends = totalBackends - healthyBackends
	lb.metrics.LastUpdate = time.Now()
}

// cleaner 清理器
func (lb *EnhancedLoadBalancer) cleaner() {
	logger.Info("Starting load balancer cleaner")

	for {
		select {
		case <-lb.ctx.Done():
			logger.Info("Load balancer cleaner stopped")
			return
		case <-lb.cleanupTicker.C:
			lb.cleanup()
		}
	}
}

// cleanup 清理过期数据
func (lb *EnhancedLoadBalancer) cleanup() {
	now := time.Now()

	// 清理过期会话
	lb.sessionMu.Lock()
	for sessionID, session := range lb.sessions {
		if now.Sub(session.LastAccess) > lb.config.AffinityTimeout {
			delete(lb.sessions, sessionID)
		}
	}
	lb.sessionMu.Unlock()

	logger.Debug("Load balancer cleanup completed")
}

// 一致性哈希环实现

// NewConsistentHashRing 创建一致性哈希环
func NewConsistentHashRing(virtualNodes int) *ConsistentHashRing {
	return &ConsistentHashRing{
		ring:         make(map[uint32]*Backend),
		sortedKeys:   make([]uint32, 0),
		virtualNodes: virtualNodes,
	}
}

// AddBackend 添加后端到哈希环
func (chr *ConsistentHashRing) AddBackend(backend *Backend) {
	chr.mu.Lock()
	defer chr.mu.Unlock()

	for i := 0; i < chr.virtualNodes; i++ {
		key := chr.hash(fmt.Sprintf("%s:%d", backend.ID, i))
		chr.ring[key] = backend
		chr.sortedKeys = append(chr.sortedKeys, key)
	}

	sort.Slice(chr.sortedKeys, func(i, j int) bool {
		return chr.sortedKeys[i] < chr.sortedKeys[j]
	})
}

// RemoveBackend 从哈希环移除后端
func (chr *ConsistentHashRing) RemoveBackend(backend *Backend) {
	chr.mu.Lock()
	defer chr.mu.Unlock()

	for i := 0; i < chr.virtualNodes; i++ {
		key := chr.hash(fmt.Sprintf("%s:%d", backend.ID, i))
		delete(chr.ring, key)

		// 从排序键中移除
		for j, sortedKey := range chr.sortedKeys {
			if sortedKey == key {
				chr.sortedKeys = append(chr.sortedKeys[:j], chr.sortedKeys[j+1:]...)
				break
			}
		}
	}
}

// UpdateBackend 更新哈希环中的后端
func (chr *ConsistentHashRing) UpdateBackend(backend *Backend) {
	// 简单实现：先移除再添加
	chr.RemoveBackend(backend)
	chr.AddBackend(backend)
}

// GetBackend 根据键获取后端
func (chr *ConsistentHashRing) GetBackend(key string) *Backend {
	chr.mu.RLock()
	defer chr.mu.RUnlock()

	if len(chr.sortedKeys) == 0 {
		return nil
	}

	hash := chr.hash(key)

	// 在哈希环上查找第一个大于等于hash值的节点
	idx := sort.Search(len(chr.sortedKeys), func(i int) bool {
		return chr.sortedKeys[i] >= hash
	})

	// 如果没有找到，则选择第一个节点（环形结构）
	if idx == len(chr.sortedKeys) {
		idx = 0
	}

	return chr.ring[chr.sortedKeys[idx]]
}

// hash 哈希函数
func (chr *ConsistentHashRing) hash(key string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(key))
	return h.Sum32()
}

// SetServiceDiscoverySource 设置负载均衡器的服务发现源
func (lb *EnhancedLoadBalancer) SetServiceDiscoverySource(source ServiceDiscoverySource) error {
	// 基本验证
	if source == nil {
		return fmt.Errorf("service discovery source cannot be nil")
	}

	// 获取源类型信息，用于日志和验证
	sourceType := fmt.Sprintf("%T", source)
	logger.Info("Validating service discovery source: %s", sourceType)

	// 类型断言检查 - 检查是否为EnhancedServiceRegistry类型
	// 这是一个可选的检查，如果有其他实现ServiceDiscoverySource接口的类型，可以移除此检查
	if _, ok := source.(*EnhancedServiceRegistry); !ok {
		logger.Warning("Service discovery source is not an EnhancedServiceRegistry, but %s", sourceType)
		// 注意：这里只是警告，不返回错误，因为任何实现了ServiceDiscoverySource接口的类型都应该可以使用
	}

	// 验证服务发现源是否能够正常工作
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 尝试调用DiscoverServices方法，使用空过滤器测试基本功能
	testFilter := &ServiceFilter{
		Name:      "_test_filter_", // 用特殊名称标记测试过滤器
		MinHealth: 0.0,             // 不过滤健康分数
	}

	// 使用一个通用服务名称进行测试
	testServiceName := "_test_service_"
	services, err := source.DiscoverServices(ctx, testServiceName, testFilter)
	if err != nil {
		// 如果是测试服务不存在的错误，我们可以忽略
		// 但其他类型的错误可能表明服务发现源存在问题
		if !strings.Contains(err.Error(), "not found") &&
			!strings.Contains(err.Error(), "no services") &&
			!strings.Contains(err.Error(), "not exist") {
			return fmt.Errorf("service discovery source validation failed: %v", err)
		}
		logger.Info("Service discovery test returned expected 'not found' error for test service")
	} else {
		logger.Info("Service discovery test found %d services for test service", len(services))
	}

	// 清理之前的服务发现源（如果存在）
	if lb.serviceDiscovery != nil {
		logger.Info("Replacing existing service discovery source of type %T", lb.serviceDiscovery)
	}

	// 存储服务发现源
	lb.mu.Lock()
	lb.serviceDiscovery = source
	lb.mu.Unlock()

	// 记录设置成功
	logger.Info("Service discovery source set successfully: %s", sourceType)
	return nil
}

// UpdateBackendsFromService 从服务发现源更新后端服务器列表
func (lb *EnhancedLoadBalancer) UpdateBackendsFromService(ctx context.Context, serviceName string) error {
	if lb.serviceDiscovery == nil {
		return fmt.Errorf("service discovery source not set")
	}

	// 使用服务发现源获取服务实例
	filter := &ServiceFilter{
		MinHealth: 50.0, // 只获取健康分数大于50的服务
	}

	services, err := lb.serviceDiscovery.DiscoverServices(ctx, serviceName, filter)
	if err != nil {
		return fmt.Errorf("failed to discover services: %v", err)
	}

	// 将服务实例转换为后端服务器
	backends := make([]*Backend, 0, len(services))
	for _, service := range services {
		backend := &Backend{
			ID:          service.NodeID,
			Address:     service.Address,
			Port:        service.Port,
			Weight:      int(service.HealthScore), // 使用健康分数作为权重
			HealthScore: service.HealthScore,
			Status:      BackendHealthy,
			LastCheck:   time.Now(),
		}
		backends = append(backends, backend)
	}

	// 更新负载均衡器的后端服务器列表
	lb.mu.Lock()
	defer lb.mu.Unlock()

	// 移除不再存在的后端
	existingBackends := make(map[string]bool)
	for _, backend := range backends {
		existingBackends[backend.ID] = true

		// 更新或添加后端
		if existingBackend, exists := lb.backends[backend.ID]; exists {
			// 更新现有后端
			existingBackend.Address = backend.Address
			existingBackend.Port = backend.Port
			existingBackend.Weight = backend.Weight
			existingBackend.HealthScore = backend.HealthScore
		} else {
			// 添加新后端
			lb.backends[backend.ID] = backend
		}
	}

	// 移除不再存在的后端
	for id := range lb.backends {
		if !existingBackends[id] {
			delete(lb.backends, id)
		}
	}

	// 更新健康后端列表
	lb.updateHealthyBackends()

	return nil
}

// StartServiceDiscoverySync 启动服务发现同步
func (lb *EnhancedLoadBalancer) StartServiceDiscoverySync(serviceName string, interval time.Duration) {
	if lb.serviceDiscovery == nil {
		logger.Error("Cannot start service discovery sync: service discovery source not set")
		return
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-lb.ctx.Done():
				return
			case <-ticker.C:
				ctx, cancel := context.WithTimeout(lb.ctx, 5*time.Second)
				err := lb.UpdateBackendsFromService(ctx, serviceName)
				if err != nil {
					logger.Error("Failed to update backends from service discovery: %v", err)
				}
				cancel()
			}
		}
	}()

	logger.Info("Started service discovery sync for service %s with interval %v", serviceName, interval)
}

// SelectServer 选择后端服务器
func (lb *EnhancedLoadBalancer) SelectServer(r *http.Request) (*Backend, error) {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	if len(lb.healthyBackends) == 0 {
		return nil, fmt.Errorf("no healthy backends available")
	}

	// 根据负载均衡算法选择后端
	switch lb.config.Algorithm {
	case RoundRobin:
		return lb.selectRoundRobin(lb.healthyBackends), nil
	case WeightedRoundRobin:
		return lb.selectWeightedRoundRobin(lb.healthyBackends), nil
	case LeastConnections:
		return lb.selectLeastConnections(lb.healthyBackends), nil
	case ConsistentHash:
		key := r.URL.Path
		if r.Header.Get("X-Session-ID") != "" {
			key = r.Header.Get("X-Session-ID")
		}
		return lb.selectConsistentHash(key), nil
	case IPHash:
		clientIP := r.RemoteAddr
		if idx := strings.LastIndex(clientIP, ":"); idx != -1 {
			clientIP = clientIP[:idx]
		}
		return lb.selectIPHash(lb.healthyBackends, clientIP), nil
	default:
		return lb.selectRoundRobin(lb.healthyBackends), nil
	}
}

// IsHealthy 检查负载均衡器是否健康
func (lb *EnhancedLoadBalancer) IsHealthy() bool {
	lb.mu.RLock()
	defer lb.mu.RUnlock()
	return len(lb.healthyBackends) > 0
}

// GetServers 获取所有后端服务器
func (lb *EnhancedLoadBalancer) GetServers() []*Backend {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	servers := make([]*Backend, 0, len(lb.backends))
	for _, backend := range lb.backends {
		servers = append(servers, backend)
	}
	return servers
}
