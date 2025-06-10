package healthCheck

import (
	"VectorSphere/src/email"
	"VectorSphere/src/library/config"
	"VectorSphere/src/library/log"
	"VectorSphere/src/server"
	"context"
	"encoding/json"
	"fmt"
	etcdv3 "go.etcd.io/etcd/client/v3"
	"math"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health/grpc_health_v1"
)

// HealthCheckManager 健康检查管理器
type HealthCheckManager struct {
	Services      map[string]*ServiceHealthInfo
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

	// 新增字段
	etcdClient   *etcdv3.Client // etcd 客户端
	leaseTimeout int64          // 租约超时时间（秒）
	heartbeatMap sync.Map       // 心跳记录映射
	healthScores sync.Map       // 健康评分缓存
	serviceTags  sync.Map       // 服务标签映射
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
	// 新增字段
	LeaseID       int64     // etcd 租约 ID
	LastHeartbeat time.Time // 最后一次心跳时间
	HealthScore   float64   // 健康评分（0-1）
	Tags          []string  // 服务标签，用于分组和筛选
	IsHealthy     bool
}

// HealthMetrics 健康检查指标
type HealthMetrics struct {
	TotalChecks      int64
	SuccessfulChecks int64
	FailedChecks     int64
	AverageLatency   time.Duration
	LastCheckTime    time.Time
	mu               sync.RWMutex

	TotalServices     int
	HealthyServices   int
	UnhealthyServices int
	AverageLatencyMs  int64 // 直接存储毫秒数
	SuccessRate       float64
}

// AlertManager 告警管理器
type AlertManager struct {
	webhookURL  string
	emailConfig *config.EmailConfig
	alertRules  []AlertRule
	mu          sync.RWMutex
	emailSender email.EmailSender
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
	emailSender := email.NewSMTPEmailSender(config.EmailConfig{})
	return &HealthCheckManager{
		Services:      make(map[string]*ServiceHealthInfo),
		checkInterval: checkInterval,
		ctx:           ctx,
		cancel:        cancel,
		metrics:       &HealthMetrics{},                        // 初始化 metrics
		alertManager:  &AlertManager{emailSender: emailSender}, // 初始化 alertManager，或者根据需要传入配置
	}
}

// 新增方法：集成心跳检查和主动健康检查的结果
func (hm *HealthCheckManager) integrateHealthStatus(serviceName string, info *ServiceHealthInfo) {
	// 计算综合健康评分
	var score float64 = 1.0

	// 1. 检查心跳状态
	if time.Since(info.LastHeartbeat) > time.Duration(hm.leaseTimeout)*time.Second {
		score *= 0.5 // 心跳超时，降低评分
	}

	// 2. 检查主动健康检查结果
	if info.Status != grpc_health_v1.HealthCheckResponse_SERVING {
		score *= 0.3 // 健康检查失败，显著降低评分
	}

	// 3. 考虑历史失败次数
	if info.FailureCount > 0 {
		score *= math.Pow(0.9, float64(info.FailureCount)) // 每次失败降低10%
	}

	// 4. 考虑延迟情况
	if info.Latency > hm.config.MaxLatency {
		score *= 0.8 // 高延迟，适度降低评分
	}

	// 更新健康评分
	info.HealthScore = score

	// 根据评分决定是否需要告警
	if score < 0.5 {
		hm.triggerAlert(serviceName, info)
	}
}

func (am *AlertManager) sendEmailAlert(serviceName string, info *ServiceHealthInfo, ruleName string) error {
	am.mu.RLock()
	defer am.mu.RUnlock()

	if am.emailConfig == nil || !am.emailConfig.Enabled {
		log.Info("Email notifications are disabled or not configured.")
		return nil // 如果邮件配置未启用或不存在，则不执行任何操作
	}

	subject := fmt.Sprintf("Alert: Service '%s' - Rule '%s' Triggered", serviceName, ruleName)
	body := fmt.Sprintf(
		"Service: %s\nEndpoint: %s:%d\nStatus: %s\nHealth Score: %.2f\nLast Check: %s\nFailure Count: %d\nLatency: %s\nTriggered Rule: %s\nDetails: %+v",
		serviceName,
		info.Endpoint.Ip,
		info.Endpoint.Port,
		info.Status.String(),
		info.HealthScore,
		info.LastCheck.Format(time.RFC3339),
		info.FailureCount,
		info.Latency.String(),
		ruleName,
		info.Metrics,
	)

	// 这里需要一个实际的邮件发送函数，例如使用 net/smtp 包
	// 以下是一个占位符，您需要替换为实际的邮件发送实现
	log.Info("Attempting to send email alert: Subject: %s, To: %s", subject, am.emailConfig.To)
	err := am.emailSender.SendEmail(am.emailConfig.To, subject, body)
	if err != nil {
		log.Error("Failed to send email alert for service %s, rule %s: %v", serviceName, ruleName, err)
		return err
	}

	log.Info("Email alert sent successfully for service %s, rule %s", serviceName, ruleName)
	return nil
}

func (hm *HealthCheckManager) triggerAlert(serviceName string, info *ServiceHealthInfo) {
	if hm.alertManager == nil {
		return
	}

	hm.alertManager.mu.RLock()
	rules := make([]AlertRule, len(hm.alertManager.alertRules)) // 创建副本以避免长时间持有锁
	copy(rules, hm.alertManager.alertRules)
	hm.alertManager.mu.RUnlock()

	for i := range rules { // 使用索引来修改原始切片中的 LastTriggered
		rule := &rules[i] // 获取规则的指针
		if rule.Condition(info) {
			hm.alertManager.mu.Lock() // 加锁以安全地更新 LastTriggered
			if time.Since(rule.LastTriggered) > rule.Cooldown {
				log.Warning("Alert triggered for service %s: %s", serviceName, rule.Name)
				// 优先使用规则中定义的 Action
				if rule.Action != nil {
					err := rule.Action(info)
					if err != nil {
						log.Error("Failed to execute alert action for %s: %v", serviceName, err)
					}
				} else {
					// 如果没有定义 Action，则尝试发送邮件告警
					err := hm.alertManager.sendEmailAlert(serviceName, info, rule.Name)
					if err != nil {
						log.Error("Failed to send email alert via AlertManager for service %s, rule %s: %v", serviceName, rule.Name, err)
					}
				}
				rule.LastTriggered = time.Now()
				// 更新原始 alertRules 中的 LastTriggered
				hm.alertManager.alertRules[i].LastTriggered = rule.LastTriggered
			}
			hm.alertManager.mu.Unlock()
		}
	}
}

// 新增方法：处理服务心跳
func (hm *HealthCheckManager) handleHeartbeat(serviceName string, endpoint server.EndPoint, leaseID int64) {
	key := fmt.Sprintf("%s:%s:%d", serviceName, endpoint.Ip, endpoint.Port)

	hm.mu.Lock()
	defer hm.mu.Unlock()

	if info, exists := hm.Services[key]; exists {
		info.LastHeartbeat = time.Now()
		info.LeaseID = leaseID

		// 更新 etcd 中的服务状态
		go hm.updateEtcdServiceStatus(serviceName, info)
	}
}

// 新增方法：更新 etcd 中的服务状态
func (hm *HealthCheckManager) updateEtcdServiceStatus(serviceName string, info *ServiceHealthInfo) {
	if hm.etcdClient == nil {
		return
	}

	// 构建服务状态信息
	statusInfo := map[string]interface{}{
		"status":       info.Status.String(),
		"health_score": info.HealthScore,
		"last_check":   info.LastCheck.Unix(),
		"latency_ms":   info.Latency.Milliseconds(),
		"load":         info.Load,
		"tags":         info.Tags,
	}

	// 序列化状态信息
	statusData, err := json.Marshal(statusInfo)
	if err != nil {
		log.Error("Failed to marshal service status: %v", err)
		return
	}

	// 更新 etcd 中的状态
	key := fmt.Sprintf("/services/%s/%s:%d/status", serviceName, info.Endpoint.Ip, info.Endpoint.Port)
	_, err = hm.etcdClient.Put(context.Background(), key, string(statusData), etcdv3.WithLease(etcdv3.LeaseID(info.LeaseID)))
	if err != nil {
		log.Error("Failed to update service status in etcd: %v", err)
	}
}

// RegisterService 注册服务进行健康检查
func (hm *HealthCheckManager) RegisterService(serviceName string, endpoint server.EndPoint) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	key := fmt.Sprintf("%s:%s:%d", serviceName, endpoint.Ip, endpoint.Port)
	hm.Services[key] = &ServiceHealthInfo{
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
	for k, v := range hm.Services {
		services[k] = v
	}
	hm.mu.RUnlock()

	var wg sync.WaitGroup
	// 限制并发数，例如10，可以根据实际情况调整
	concurrencyLimit := 10
	if hm.config != nil && hm.config.ConcurrencyLimit > 0 {
		concurrencyLimit = hm.config.ConcurrencyLimit
	}
	semaphore := make(chan struct{}, concurrencyLimit)

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
	// hm.checkAlerts() // 告警检查逻辑已移至 integrateHealthStatus -> triggerAlert
}

// checkServiceHealth 检查单个服务健康状态
func (hm *HealthCheckManager) checkServiceHealth(serviceName string, info *ServiceHealthInfo) {
	startTime := time.Now()
	oldStatus := info.Status

	// 创建带总超时的上下文，用于整个重试过程
	// 如果 hm.config 或 hm.config.CheckTimeout 未初始化，需要提供默认值
	var overallTimeout time.Duration
	if hm.config != nil && hm.config.Timeout > 0 {
		overallTimeout = hm.config.Timeout
	} else {
		overallTimeout = 30 * time.Second // 默认总超时
	}
	ctx, cancel := context.WithTimeout(hm.ctx, overallTimeout)
	defer cancel()

	// 定义重试策略配置
	// 这些值应该从 HealthCheckConfig 中获取
	retryConfig := &RetryPolicyConfig{
		InitialInterval: 1 * time.Second,
		MaxInterval:     5 * time.Second,
		MaxElapsedTime:  overallTimeout, // 重试的总时长不超过整体超时
		MaxRetries:      3,              // 默认重试次数
	}
	if hm.config != nil && hm.config.RetryPolicy != nil {
		retryConfig.InitialInterval = hm.config.RetryPolicy.InitialInterval
		retryConfig.MaxInterval = hm.config.RetryPolicy.MaxInterval
		retryConfig.MaxElapsedTime = hm.config.RetryPolicy.MaxElapsedTime
		retryConfig.MaxRetries = hm.config.RetryPolicy.MaxRetries
	}

	// 使用重试策略进行健康检查
	err := RetryableOperation(ctx, func() error {
		// 每次尝试都使用新的带有单次检查超时的上下文
		// checkTimeout 应该是单次尝试的超时
		var singleCheckTimeout time.Duration
		if hm.config != nil && hm.config.CheckTimeoutPerTry > 0 { // 假设配置中有 CheckTimeoutPerTry
			singleCheckTimeout = hm.config.CheckTimeoutPerTry
		} else {
			singleCheckTimeout = 5 * time.Second // 默认单次检查超时
		}
		attemptCtx, attemptCancel := context.WithTimeout(ctx, singleCheckTimeout)
		defer attemptCancel()
		return hm.performHealthCheck(attemptCtx, serviceName, info) // performHealthCheck 现在负责更新 info.Status
	}, retryConfig)

	hm.mu.Lock() // 加锁以安全地更新共享的 serviceInfo
	info.Latency = time.Since(startTime)
	info.LastCheck = time.Now()

	if err != nil {
		// 重试后仍然失败
		hm.handleHealthCheckFailure(serviceName, info, err) // handleHealthCheckFailure 内部会更新 Status 和 FailureCount
	} else {
		// 重试成功或首次尝试即成功
		// performHealthCheck 内部在成功时已将 Status 设置为 SERVING 并重置 FailureCount
		// 这里确保一下，如果 performHealthCheck 内部没有完全覆盖所有成功情况的状态更新
		if info.Status != grpc_health_v1.HealthCheckResponse_SERVING {
			info.Status = grpc_health_v1.HealthCheckResponse_SERVING
		}
		info.FailureCount = 0 // 再次确保成功后失败计数为0
	}
	currentStatusAfterCheck := info.Status // 保存当前检查后的状态，用于比较	hm.mu.Unlock()

	// 集成健康状态评估 (在锁外部进行，因为它可能调用 triggerAlert，其中有自己的锁)
	hm.integrateHealthStatus(serviceName, info)

	hm.mu.Lock() // 再次加锁以进行回调的条件判断
	// 触发状态变化回调 (比较的是检查开始前的状态和检查结束后的状态)
	if oldStatus != currentStatusAfterCheck { // 使用检查后的状态进行比较
		callbacks := make([]HealthChangeCallback, len(hm.callbacks))
		copy(callbacks, hm.callbacks) // 复制回调列表以在锁外执行
		hm.mu.Unlock()
		for _, callback := range callbacks {
			go callback(serviceName, info.Endpoint, oldStatus, currentStatusAfterCheck)
		}
	} else {
		hm.mu.Unlock()
	}
}

// performHealthCheck 执行单次 gRPC 健康检查的内部逻辑
func (hm *HealthCheckManager) performHealthCheck(ctx context.Context, serviceName string, info *ServiceHealthInfo) error {
	conn, err := grpc.DialContext(ctx, // 使用传入的带超时的上下文
		fmt.Sprintf("%s:%d", info.Endpoint.Ip, info.Endpoint.Port),
		grpc.WithInsecure(), // 假设是内部服务，如果需要TLS，请修改
		grpc.WithBlock(),    // 使用 Block 来确保 Dial 在超时前完成或失败
	)
	if err != nil {
		log.Debug("Health check DialContext failed for %s (%s:%d): %v", serviceName, info.Endpoint.Ip, info.Endpoint.Port, err)
		return fmt.Errorf("failed to connect to service %s: %w", serviceName, err)
	}
	defer conn.Close()

	client := grpc_health_v1.NewHealthClient(conn)
	// 健康检查请求的超时应该小于或等于 Dial 的超时
	// 这里使用 ctx，它已经包含了 checkTimeout
	healthCheckCtx, healthCheckCancel := context.WithTimeout(ctx, hm.checkTimeout) // 可以考虑为单次 check 设置更短的超时
	defer healthCheckCancel()

	resp, err := client.Check(healthCheckCtx, &grpc_health_v1.HealthCheckRequest{Service: serviceName})
	if err != nil {
		log.Debug("Health check RPC failed for %s (%s:%d): %v", serviceName, info.Endpoint.Ip, info.Endpoint.Port, err)
		// 根据 gRPC 错误码判断是否可重试，或者直接返回错误由 RetryableOperation 处理
		return fmt.Errorf("health check failed for service %s: %w", serviceName, err)
	}

	hm.mu.Lock()
	info.Status = resp.Status
	if resp.Status == grpc_health_v1.HealthCheckResponse_SERVING {
		info.FailureCount = 0 // 成功后重置失败计数
	} else {
		// 如果不是SERVING，也算作一次潜在的失败，但不立即增加FailureCount，由checkServiceHealth的逻辑处理
	}
	// 如果响应中包含自定义指标，可以在这里解析和存储
	// 例如: info.Load = customResp.Load (需要类型断言和检查)
	hm.mu.Unlock()

	return nil
}

// handleHealthCheckFailure 处理健康检查失败
func (hm *HealthCheckManager) handleHealthCheckFailure(serviceName string, info *ServiceHealthInfo, err error) {
	// 注意：此方法在调用时，外部应该已经获取了 hm.mu 的锁
	info.FailureCount++

	// failureThreshold 应该从配置中读取
	failureThreshold := 3 // 默认值
	if hm.config != nil && hm.config.FailureThreshold > 0 {
		failureThreshold = hm.config.FailureThreshold
	}

	if info.FailureCount >= failureThreshold {
		// oldStatus := info.Status // oldStatus 应该在 checkServiceHealth 开始时获取
		info.Status = grpc_health_v1.HealthCheckResponse_NOT_SERVING
		// 状态变化回调的触发移到 checkServiceHealth 的末尾，以确保在所有状态更新后进行
	}

	log.Warning("Health check failed for service %s (%s:%d) after retries: %v (failure count: %d, status: %s)",
		serviceName, info.Endpoint.Ip, info.Endpoint.Port, err, info.FailureCount, info.Status.String())
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
	for _, serviceInfo := range hm.Services {
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
	for key, serviceInfo := range hm.Services {
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

// updateMetrics 更新健康检查指标
func (hm *HealthCheckManager) updateMetrics() {
	hm.metrics.mu.Lock()
	defer hm.metrics.mu.Unlock()

	hm.metrics.TotalChecks = 0
	hm.metrics.SuccessfulChecks = 0
	hm.metrics.FailedChecks = 0
	var totalLatency time.Duration
	var checkedServices int64

	hm.mu.RLock()
	defer hm.mu.RUnlock()

	for _, info := range hm.Services {
		hm.metrics.TotalChecks++
		if info.Status == grpc_health_v1.HealthCheckResponse_SERVING {
			hm.metrics.SuccessfulChecks++
		} else {
			hm.metrics.FailedChecks++
		}
		totalLatency += info.Latency
		checkedServices++
	}

	if checkedServices > 0 {
		hm.metrics.AverageLatency = totalLatency / time.Duration(checkedServices)
	}
	hm.metrics.LastCheckTime = time.Now()
}

// checkAlerts 检查并触发告警
func (hm *HealthCheckManager) checkAlerts() {
	if hm.alertManager == nil {
		return
	}

	hm.alertManager.mu.RLock()
	defer hm.alertManager.mu.RUnlock()

	hm.mu.RLock()
	defer hm.mu.RUnlock()

	for _, rule := range hm.alertManager.alertRules {
		for serviceName, info := range hm.Services {
			if rule.Condition(info) {
				if time.Since(rule.LastTriggered) > rule.Cooldown {
					log.Warning("Alert triggered for service %s: %s", serviceName, rule.Name)
					err := rule.Action(info)
					if err != nil {
						log.Error("Failed to execute alert action for %s: %v", serviceName, err)
					}
					rule.LastTriggered = time.Now()
				}
			}
		}
	}
}
