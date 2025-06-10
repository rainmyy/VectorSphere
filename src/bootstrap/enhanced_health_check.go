package bootstrap

import (
	"VectorSphere/src/library/config"
	"VectorSphere/src/library/log"
	"VectorSphere/src/server"
	"context"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health/grpc_health_v1"
)

// HealthCheckManager 健康检查管理器
type HealthCheckManager struct {
	services      map[string]*ServiceHealthInfo
	mu            sync.RWMutex
	checkInterval time.Duration
	ctx           context.Context
	cancel        context.CancelFunc
	callbacks     []HealthChangeCallback

	config           *config.HealthCheckConfig
	failureThreshold int
	successThreshold int
	checkTimeout     time.Duration
	retryPolicy      *config.RetryPolicy
	metrics          *HealthMetrics
	alertManager     *AlertManager
}

// ServiceHealthInfo 服务健康信息
type ServiceHealthInfo struct {
	Endpoint     server.EndPoint
	Status       grpc_health_v1.HealthCheckResponse_ServingStatus
	LastCheck    time.Time
	FailureCount int
	Latency      time.Duration
	Load         float32
	Metrics      map[string]interface{}
}

// HealthMetrics 健康检查指标
type HealthMetrics struct {
	TotalChecks      int64
	SuccessfulChecks int64
	FailedChecks     int64
	AverageLatency   time.Duration
	LastCheckTime    time.Time
	mu               sync.RWMutex
}

// AlertManager 告警管理器
type AlertManager struct {
	webhookURL  string
	emailConfig *config.EmailConfig
	alertRules  []AlertRule
	mu          sync.RWMutex
}

// AlertRule 告警规则
type AlertRule struct {
	Name          string
	Condition     func(*ServiceHealthInfo) bool
	Action        func(*ServiceHealthInfo) error
	Cooldown      time.Duration
	LastTriggered time.Time
}

// HealthChangeCallback 健康状态变化回调
type HealthChangeCallback func(serviceName string, endpoint server.EndPoint, oldStatus, newStatus grpc_health_v1.HealthCheckResponse_ServingStatus)

// NewHealthCheckManager 创建健康检查管理器
func NewHealthCheckManager(checkInterval time.Duration) *HealthCheckManager {
	ctx, cancel := context.WithCancel(context.Background())
	return &HealthCheckManager{
		services:      make(map[string]*ServiceHealthInfo),
		checkInterval: checkInterval,
		ctx:           ctx,
		cancel:        cancel,
	}
}

// RegisterService 注册服务进行健康检查
func (hm *HealthCheckManager) RegisterService(serviceName string, endpoint server.EndPoint) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	key := fmt.Sprintf("%s:%s:%d", serviceName, endpoint.Ip, endpoint.Port)
	hm.services[key] = &ServiceHealthInfo{
		Endpoint:  endpoint,
		Status:    grpc_health_v1.HealthCheckResponse_UNKNOWN,
		LastCheck: time.Now(),
		Metrics:   make(map[string]interface{}),
	}
}

// StartHealthChecks 启动健康检查
func (hm *HealthCheckManager) StartHealthChecks() {
	ticker := time.NewTicker(hm.checkInterval)
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-hm.ctx.Done():
				return
			case <-ticker.C:
				hm.performHealthChecks()
			}
		}
	}()
}

// performHealthChecks 执行健康检查（增强版）
func (hm *HealthCheckManager) performHealthChecks() {
	hm.mu.RLock()
	services := make(map[string]*ServiceHealthInfo)
	for k, v := range hm.services {
		services[k] = v
	}
	hm.mu.RUnlock()

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 10) // 限制并发数

	for serviceName, serviceInfo := range services {
		wg.Add(1)
		go func(name string, info *ServiceHealthInfo) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			hm.checkServiceHealth(name, info)
		}(serviceName, serviceInfo)
	}

	wg.Wait()
	hm.updateMetrics()
	hm.checkAlerts()
}

// checkServiceHealth 检查单个服务健康状态
func (hm *HealthCheckManager) checkServiceHealth(serviceName string, info *ServiceHealthInfo) {
	startTime := time.Now()
	oldStatus := info.Status

	// 执行健康检查
	ctx, cancel := context.WithTimeout(hm.ctx, hm.checkTimeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx, fmt.Sprintf("%s:%d", info.Endpoint.Ip, info.Endpoint.Port),
		grpc.WithInsecure(),
		grpc.WithBlock(),
	)
	if err != nil {
		hm.handleHealthCheckFailure(serviceName, info, err)
		return
	}
	defer conn.Close()

	client := grpc_health_v1.NewHealthClient(conn)
	resp, err := client.Check(ctx, &grpc_health_v1.HealthCheckRequest{
		Service: serviceName,
	})

	latency := time.Since(startTime)
	info.Latency = latency
	info.LastCheck = time.Now()

	if err != nil {
		hm.handleHealthCheckFailure(serviceName, info, err)
		return
	}

	// 更新健康状态
	newStatus := resp.Status
	if newStatus == grpc_health_v1.HealthCheckResponse_SERVING {
		info.FailureCount = 0
		if info.Status != grpc_health_v1.HealthCheckResponse_SERVING {
			info.Status = grpc_health_v1.HealthCheckResponse_SERVING
		}
	} else {
		hm.handleHealthCheckFailure(serviceName, info, fmt.Errorf("service not serving: %v", newStatus))
		return
	}

	// 触发状态变化回调
	if oldStatus != newStatus {
		for _, callback := range hm.callbacks {
			go callback(serviceName, info.Endpoint, oldStatus, newStatus)
		}
	}
}

// handleHealthCheckFailure 处理健康检查失败
func (hm *HealthCheckManager) handleHealthCheckFailure(serviceName string, info *ServiceHealthInfo, err error) {
	info.FailureCount++

	if info.FailureCount >= hm.failureThreshold {
		oldStatus := info.Status
		info.Status = grpc_health_v1.HealthCheckResponse_NOT_SERVING

		// 触发状态变化回调
		if oldStatus != info.Status {
			for _, callback := range hm.callbacks {
				go callback(serviceName, info.Endpoint, oldStatus, info.Status)
			}
		}
	}

	log.Warning("Health check failed for service %s (%s:%d): %v (failure count: %d)",
		serviceName, info.Endpoint.Ip, info.Endpoint.Port, err, info.FailureCount)
}

// checkSingleService 检查单个服务
func (hm *HealthCheckManager) checkSingleService(key string, serviceInfo *ServiceHealthInfo) {
	startTime := time.Now()

	// 建立连接
	conn, err := grpc.DialContext(hm.ctx,
		fmt.Sprintf("%s:%d", serviceInfo.Endpoint.Ip, serviceInfo.Endpoint.Port),
		grpc.WithInsecure(),
		grpc.WithTimeout(5*time.Second),
	)
	if err != nil {
		hm.updateServiceStatus(key, serviceInfo, grpc_health_v1.HealthCheckResponse_NOT_SERVING, time.Since(startTime))
		return
	}
	defer conn.Close()

	// 执行健康检查
	client := grpc_health_v1.NewHealthClient(conn)
	ctx, cancel := context.WithTimeout(hm.ctx, 3*time.Second)
	defer cancel()

	resp, err := client.Check(ctx, &grpc_health_v1.HealthCheckRequest{})
	latency := time.Since(startTime)

	if err != nil {
		hm.updateServiceStatus(key, serviceInfo, grpc_health_v1.HealthCheckResponse_NOT_SERVING, latency)
		return
	}

	// 更新服务状态
	hm.updateServiceStatus(key, serviceInfo, resp.Status, latency)

	// 如果是自定义健康检查响应，更新负载信息
	if customResp, ok := interface{}(resp).(*server.HealthCheckResponse); ok {
		serviceInfo.Load = customResp.Load
	}
}

// updateServiceStatus 更新服务状态
func (hm *HealthCheckManager) updateServiceStatus(key string, serviceInfo *ServiceHealthInfo, newStatus grpc_health_v1.HealthCheckResponse_ServingStatus, latency time.Duration) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	oldStatus := serviceInfo.Status
	serviceInfo.Status = newStatus
	serviceInfo.LastCheck = time.Now()
	serviceInfo.Latency = latency

	if newStatus != grpc_health_v1.HealthCheckResponse_SERVING {
		serviceInfo.FailureCount++
	} else {
		serviceInfo.FailureCount = 0
	}

	// 触发状态变化回调
	if oldStatus != newStatus {
		for _, callback := range hm.callbacks {
			go callback(key, serviceInfo.Endpoint, oldStatus, newStatus)
		}
	}
}

// GetHealthyServices 获取健康的服务列表
func (hm *HealthCheckManager) GetHealthyServices(serviceName string) []server.EndPoint {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	var healthyEndpoints []server.EndPoint
	for _, serviceInfo := range hm.services {
		if serviceInfo.Status == grpc_health_v1.HealthCheckResponse_SERVING {
			healthyEndpoints = append(healthyEndpoints, serviceInfo.Endpoint)
		}
	}

	return healthyEndpoints
}

// GetServiceMetrics 获取服务指标
func (hm *HealthCheckManager) GetServiceMetrics(serviceName string) map[string]*ServiceHealthInfo {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	metrics := make(map[string]*ServiceHealthInfo)
	for key, serviceInfo := range hm.services {
		metrics[key] = &ServiceHealthInfo{
			Endpoint:     serviceInfo.Endpoint,
			Status:       serviceInfo.Status,
			LastCheck:    serviceInfo.LastCheck,
			FailureCount: serviceInfo.FailureCount,
			Latency:      serviceInfo.Latency,
			Load:         serviceInfo.Load,
			Metrics:      serviceInfo.Metrics,
		}
	}

	return metrics
}

// AddHealthChangeCallback 添加健康状态变化回调
func (hm *HealthCheckManager) AddHealthChangeCallback(callback HealthChangeCallback) {
	hm.callbacks = append(hm.callbacks, callback)
}

// Stop 停止健康检查
func (hm *HealthCheckManager) Stop() {
	hm.cancel()
}
