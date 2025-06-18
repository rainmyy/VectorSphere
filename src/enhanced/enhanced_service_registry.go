package enhanced

import (
	"VectorSphere/src/library/logger"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// ServiceMetadata 增强的服务元数据
type ServiceMetadata struct {
	ServiceName   string                 `json:"service_name"`
	NodeID        string                 `json:"node_id"`
	Address       string                 `json:"address"`
	Port          int                    `json:"port"`
	NodeType      string                 `json:"node_type"`
	Status        string                 `json:"status"`
	Version       string                 `json:"version"`
	Region        string                 `json:"region"`
	Zone          string                 `json:"zone"`
	Tags          []string               `json:"tags"`
	Capabilities  []string               `json:"capabilities"`
	Load          float64                `json:"load"`
	HealthScore   float64                `json:"health_score"`
	LastHeartbeat time.Time              `json:"last_heartbeat"`
	StartTime     time.Time              `json:"start_time"`
	Metrics       map[string]interface{} `json:"metrics"`
	CustomData    map[string]string      `json:"custom_data"`
}

// ServiceRegistryConfig 服务注册配置
type ServiceRegistryConfig struct {
	BaseTTL              int64         `json:"base_ttl"`              // 基础TTL（秒）
	MinTTL               int64         `json:"min_ttl"`               // 最小TTL
	MaxTTL               int64         `json:"max_ttl"`               // 最大TTL
	HeartbeatInterval    time.Duration `json:"heartbeat_interval"`    // 心跳间隔
	HealthCheckInterval  time.Duration `json:"health_check_interval"` // 健康检查间隔
	RetryAttempts        int           `json:"retry_attempts"`        // 重试次数
	RetryInterval        time.Duration `json:"retry_interval"`        // 重试间隔
	EnableCache          bool          `json:"enable_cache"`          // 启用缓存
	CacheTTL             time.Duration `json:"cache_ttl"`             // 缓存TTL
	EnableNotification   bool          `json:"enable_notification"`   // 启用变更通知
	NotificationChannels []string      `json:"notification_channels"` // 通知渠道
}

// LeaseManager 租约管理器
type LeaseManager struct {
	client           *clientv3.Client
	leases           map[string]clientv3.LeaseID // serviceKey -> leaseID
	leaseStats       map[clientv3.LeaseID]*LeaseStats
	mu               sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
	config           *ServiceRegistryConfig
	adaptiveTTL      bool
	healthScores     map[string]float64 // serviceKey -> healthScore
	renewSuccessRate map[clientv3.LeaseID]float64
}

// LeaseStats 租约统计信息
type LeaseStats struct {
	LeaseID      clientv3.LeaseID `json:"lease_id"`
	ServiceKey   string           `json:"service_key"`
	TTL          int64            `json:"ttl"`
	CreatedAt    time.Time        `json:"created_at"`
	LastRenewed  time.Time        `json:"last_renewed"`
	RenewCount   int64            `json:"renew_count"`
	FailureCount int64            `json:"failure_count"`
	SuccessRate  float64          `json:"success_rate"`
	HealthScore  float64          `json:"health_score"`
}

// EnhancedServiceRegistry 增强的服务注册器
type EnhancedServiceRegistry struct {
	client         *clientv3.Client
	leaseManager   *LeaseManager
	config         *ServiceRegistryConfig
	serviceCache   map[string]*ServiceMetadata // serviceKey -> metadata
	cacheMu        sync.RWMutex
	watchers       map[string]context.CancelFunc
	watchersMu     sync.RWMutex
	notifyChannels map[string]chan *ServiceChangeEvent
	filters        map[string]ServiceFilter
	filtersMu      sync.RWMutex
	ctx            context.Context
	cancel         context.CancelFunc
	mu             sync.RWMutex
	LeaseId        clientv3.LeaseID
}

// ServiceChangeEvent 服务变更事件
type ServiceChangeEvent struct {
	Type        string           `json:"type"` // "register", "unregister", "update"
	ServiceKey  string           `json:"service_key"`
	Metadata    *ServiceMetadata `json:"metadata"`
	OldMetadata *ServiceMetadata `json:"old_metadata,omitempty"`
	Timestamp   time.Time        `json:"timestamp"`
}

// ServiceFilter 服务过滤器
type ServiceFilter struct {
	Name      string                      `json:"name"`
	Predicate func(*ServiceMetadata) bool `json:"-"`
	Tags      []string                    `json:"tags,omitempty"`
	Region    string                      `json:"region,omitempty"`
	Zone      string                      `json:"zone,omitempty"`
	MinHealth float64                     `json:"min_health,omitempty"`
}

// NewEnhancedServiceRegistry 创建增强的服务注册器
func NewEnhancedServiceRegistry(client *clientv3.Client, config *ServiceRegistryConfig) *EnhancedServiceRegistry {
	ctx, cancel := context.WithCancel(context.Background())

	if config == nil {
		config = &ServiceRegistryConfig{
			BaseTTL:             30,
			MinTTL:              10,
			MaxTTL:              300,
			HeartbeatInterval:   10 * time.Second,
			HealthCheckInterval: 30 * time.Second,
			RetryAttempts:       3,
			RetryInterval:       5 * time.Second,
			EnableCache:         true,
			CacheTTL:            60 * time.Second,
			EnableNotification:  true,
		}
	}

	leaseManager := &LeaseManager{
		client:           client,
		leases:           make(map[string]clientv3.LeaseID),
		leaseStats:       make(map[clientv3.LeaseID]*LeaseStats),
		ctx:              ctx,
		cancel:           cancel,
		config:           config,
		adaptiveTTL:      true,
		healthScores:     make(map[string]float64),
		renewSuccessRate: make(map[clientv3.LeaseID]float64),
	}

	return &EnhancedServiceRegistry{
		client:         client,
		leaseManager:   leaseManager,
		config:         config,
		serviceCache:   make(map[string]*ServiceMetadata),
		watchers:       make(map[string]context.CancelFunc),
		notifyChannels: make(map[string]chan *ServiceChangeEvent),
		filters:        make(map[string]ServiceFilter),
		ctx:            ctx,
		cancel:         cancel,
	}
}

// RegisterService 注册服务
func (esr *EnhancedServiceRegistry) RegisterService(ctx context.Context, metadata *ServiceMetadata) error {
	serviceKey := esr.buildServiceKey(metadata)
	logger.Info("Registering enhanced service: %s", serviceKey)

	// 计算初始TTL
	ttl := esr.calculateInitialTTL(metadata)

	// 创建租约
	leaseID, err := esr.leaseManager.CreateLease(ctx, serviceKey, ttl)
	if err != nil {
		return fmt.Errorf("创建租约失败: %v", err)
	}

	esr.LeaseId = leaseID
	// 序列化元数据
	metadataBytes, err := esr.serializeMetadata(metadata)
	if err != nil {
		return fmt.Errorf("序列化元数据失败: %v", err)
	}

	// 注册到etcd
	_, err = esr.client.Put(ctx, serviceKey, string(metadataBytes), clientv3.WithLease(leaseID))
	if err != nil {
		return fmt.Errorf("注册服务失败: %v", err)
	}

	// 更新缓存
	if esr.config.EnableCache {
		esr.updateCache(serviceKey, metadata)
	}

	// 发送通知
	if esr.config.EnableNotification {
		esr.notifyServiceChange(&ServiceChangeEvent{
			Type:       "register",
			ServiceKey: serviceKey,
			Metadata:   metadata,
			Timestamp:  time.Now(),
		})
	}

	// 启动租约续期
	go esr.leaseManager.StartKeepAlive(ctx, leaseID, serviceKey)

	logger.Info("Service registered successfully: %s with lease %d", serviceKey, leaseID)
	return nil
}

// UnregisterService 注销服务
func (esr *EnhancedServiceRegistry) UnregisterService(ctx context.Context, serviceKey string) error {
	logger.Info("Unregistering service: %s", serviceKey)

	// 获取旧的元数据
	oldMetadata := esr.getFromCache(serviceKey)

	// 撤销租约
	if err := esr.leaseManager.RevokeLease(ctx, serviceKey); err != nil {
		logger.Warning("撤销租约失败: %v", err)
	}

	// 从etcd删除
	_, err := esr.client.Delete(ctx, serviceKey)
	if err != nil {
		return fmt.Errorf("删除服务失败: %v", err)
	}

	// 更新缓存
	if esr.config.EnableCache {
		esr.removeFromCache(serviceKey)
	}

	// 发送通知
	if esr.config.EnableNotification {
		esr.notifyServiceChange(&ServiceChangeEvent{
			Type:        "unregister",
			ServiceKey:  serviceKey,
			OldMetadata: oldMetadata,
			Timestamp:   time.Now(),
		})
	}

	logger.Info("Service unregistered successfully: %s", serviceKey)
	return nil
}

// UpdateService 更新服务信息
func (esr *EnhancedServiceRegistry) UpdateService(ctx context.Context, metadata *ServiceMetadata) error {
	serviceKey := esr.buildServiceKey(metadata)
	logger.Info("Updating service: %s", serviceKey)

	// 获取旧的元数据
	oldMetadata := esr.getFromCache(serviceKey)

	// 序列化新元数据
	metadataBytes, err := esr.serializeMetadata(metadata)
	if err != nil {
		return fmt.Errorf("序列化元数据失败: %v", err)
	}

	// 更新etcd
	_, err = esr.client.Put(ctx, serviceKey, string(metadataBytes))
	if err != nil {
		return fmt.Errorf("更新服务失败: %v", err)
	}

	// 更新缓存
	if esr.config.EnableCache {
		esr.updateCache(serviceKey, metadata)
	}

	// 发送通知
	if esr.config.EnableNotification {
		esr.notifyServiceChange(&ServiceChangeEvent{
			Type:        "update",
			ServiceKey:  serviceKey,
			Metadata:    metadata,
			OldMetadata: oldMetadata,
			Timestamp:   time.Now(),
		})
	}

	logger.Info("Service updated successfully: %s", serviceKey)
	return nil
}

// DiscoverServices 发现服务
func (esr *EnhancedServiceRegistry) DiscoverServices(ctx context.Context, serviceName string, filter *ServiceFilter) ([]*ServiceMetadata, error) {
	logger.Debug("Discovering services: %s", serviceName)

	// 尝试从缓存获取
	if esr.config.EnableCache {
		if services := esr.getServicesFromCache(serviceName, filter); len(services) > 0 {
			logger.Debug("Found %d services in cache for %s", len(services), serviceName)
			return services, nil
		}
	}

	// 从etcd查询
	prefix := fmt.Sprintf("/vector_sphere/services/%s/", serviceName)
	resp, err := esr.client.Get(ctx, prefix, clientv3.WithPrefix())
	if err != nil {
		return nil, fmt.Errorf("查询服务失败: %v", err)
	}

	var services []*ServiceMetadata
	for _, kv := range resp.Kvs {
		metadata, err := esr.deserializeMetadata(kv.Value)
		if err != nil {
			logger.Warning("反序列化服务元数据失败: %v", err)
			continue
		}

		// 应用过滤器
		if filter != nil && !esr.applyFilter(metadata, filter) {
			continue
		}

		services = append(services, metadata)

		// 更新缓存
		if esr.config.EnableCache {
			serviceKey := string(kv.Key)
			esr.updateCache(serviceKey, metadata)
		}
	}

	logger.Debug("Found %d services for %s", len(services), serviceName)
	return services, nil
}

// 租约管理器方法

// CreateLease 创建租约
func (lm *LeaseManager) CreateLease(ctx context.Context, serviceKey string, ttl int64) (clientv3.LeaseID, error) {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	if lm.client == nil {
		return 0, fmt.Errorf("etcd client is nil, cannot create lease for service %s", serviceKey)
	}
	// 检查是否已存在租约
	if existingLeaseID, exists := lm.leases[serviceKey]; exists {
		logger.Warning("Service %s already has lease %d, revoking old lease", serviceKey, existingLeaseID)
		lm.client.Revoke(ctx, existingLeaseID)
		delete(lm.leases, serviceKey)
		delete(lm.leaseStats, existingLeaseID)
	}

	// 创建新租约
	leaseResp, err := lm.client.Grant(ctx, ttl)
	if err != nil {
		return 0, err
	}

	leaseID := leaseResp.ID
	lm.leases[serviceKey] = leaseID
	lm.leaseStats[leaseID] = &LeaseStats{
		LeaseID:     leaseID,
		ServiceKey:  serviceKey,
		TTL:         ttl,
		CreatedAt:   time.Now(),
		LastRenewed: time.Now(),
		HealthScore: 1.0,
	}

	logger.Debug("Created lease %d for service %s with TTL %d", leaseID, serviceKey, ttl)
	return leaseID, nil
}

// RevokeLease 撤销租约
func (lm *LeaseManager) RevokeLease(ctx context.Context, serviceKey string) error {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	leaseID, exists := lm.leases[serviceKey]
	if !exists {
		return fmt.Errorf("service %s has no active lease", serviceKey)
	}

	_, err := lm.client.Revoke(ctx, leaseID)
	if err != nil {
		return err
	}

	delete(lm.leases, serviceKey)
	delete(lm.leaseStats, leaseID)
	delete(lm.healthScores, serviceKey)
	delete(lm.renewSuccessRate, leaseID)

	logger.Debug("Revoked lease %d for service %s", leaseID, serviceKey)
	return nil
}

// StartKeepAlive 启动租约续期
func (lm *LeaseManager) StartKeepAlive(ctx context.Context, leaseID clientv3.LeaseID, serviceKey string) {
	ch, kaerr := lm.client.KeepAlive(ctx, leaseID)
	if kaerr != nil {
		logger.Error("KeepAlive failed for lease %d: %v", leaseID, kaerr)
		return
	}

	successCount := int64(0)
	totalCount := int64(0)

	for {
		select {
		case ka := <-ch:
			totalCount++
			if ka != nil {
				successCount++
				lm.updateLeaseStats(leaseID, true)

				// 自适应TTL调整
				if lm.adaptiveTTL {
					lm.adjustTTL(serviceKey, leaseID)
				}
			} else {
				lm.updateLeaseStats(leaseID, false)
				logger.Warning("KeepAlive response is nil for lease %d", leaseID)
			}

			// 更新成功率
			lm.mu.Lock()
			lm.renewSuccessRate[leaseID] = float64(successCount) / float64(totalCount)
			lm.mu.Unlock()

		case <-ctx.Done():
			logger.Info("KeepAlive stopped for lease %d due to context cancellation", leaseID)
			return
		case <-lm.ctx.Done():
			logger.Info("KeepAlive stopped for lease %d due to lease manager shutdown", leaseID)
			return
		}
	}
}

// 辅助方法

// buildServiceKey 构建服务键
func (esr *EnhancedServiceRegistry) buildServiceKey(metadata *ServiceMetadata) string {
	return fmt.Sprintf("/vector_sphere/services/%s/%s", metadata.ServiceName, metadata.NodeID)
}

// serializeMetadata 序列化元数据
func (esr *EnhancedServiceRegistry) serializeMetadata(metadata *ServiceMetadata) ([]byte, error) {
	metadata.LastHeartbeat = time.Now()
	return json.Marshal(metadata)
}

// deserializeMetadata 反序列化元数据
func (esr *EnhancedServiceRegistry) deserializeMetadata(data []byte) (*ServiceMetadata, error) {
	var metadata ServiceMetadata
	err := json.Unmarshal(data, &metadata)
	return &metadata, err
}

// calculateInitialTTL 计算初始TTL
func (esr *EnhancedServiceRegistry) calculateInitialTTL(metadata *ServiceMetadata) int64 {
	baseTTL := esr.config.BaseTTL

	// 根据服务类型调整TTL
	if metadata.NodeType == "master" {
		return baseTTL * 2 // master节点使用更长的TTL
	}

	// 根据健康评分调整TTL
	if metadata.HealthScore > 0.8 {
		return baseTTL + 10
	} else if metadata.HealthScore < 0.5 {
		return baseTTL - 5
	}

	return baseTTL
}

// applyFilter 应用过滤器
func (esr *EnhancedServiceRegistry) applyFilter(metadata *ServiceMetadata, filter *ServiceFilter) bool {
	if filter == nil {
		return true
	}

	// 自定义谓词过滤
	if filter.Predicate != nil && !filter.Predicate(metadata) {
		return false
	}

	// 区域过滤
	if filter.Region != "" && metadata.Region != filter.Region {
		return false
	}

	// 可用区过滤
	if filter.Zone != "" && metadata.Zone != filter.Zone {
		return false
	}

	// 健康评分过滤
	if filter.MinHealth > 0 && metadata.HealthScore < filter.MinHealth {
		return false
	}

	// 标签过滤
	if len(filter.Tags) > 0 {
		tagMap := make(map[string]bool)
		for _, tag := range metadata.Tags {
			tagMap[tag] = true
		}
		for _, requiredTag := range filter.Tags {
			if !tagMap[requiredTag] {
				return false
			}
		}
	}

	return true
}

// 缓存相关方法

// updateCache 更新缓存
func (esr *EnhancedServiceRegistry) updateCache(serviceKey string, metadata *ServiceMetadata) {
	esr.cacheMu.Lock()
	defer esr.cacheMu.Unlock()
	esr.serviceCache[serviceKey] = metadata
}

// getFromCache 从缓存获取
func (esr *EnhancedServiceRegistry) getFromCache(serviceKey string) *ServiceMetadata {
	esr.cacheMu.RLock()
	defer esr.cacheMu.RUnlock()
	return esr.serviceCache[serviceKey]
}

// removeFromCache 从缓存移除
func (esr *EnhancedServiceRegistry) removeFromCache(serviceKey string) {
	esr.cacheMu.Lock()
	defer esr.cacheMu.Unlock()
	delete(esr.serviceCache, serviceKey)
}

// getServicesFromCache 从缓存获取服务列表
func (esr *EnhancedServiceRegistry) getServicesFromCache(serviceName string, filter *ServiceFilter) []*ServiceMetadata {
	esr.cacheMu.RLock()
	defer esr.cacheMu.RUnlock()

	var services []*ServiceMetadata
	prefix := fmt.Sprintf("/vector_sphere/services/%s/", serviceName)

	for key, metadata := range esr.serviceCache {
		if strings.HasPrefix(key, prefix) {
			if filter == nil || esr.applyFilter(metadata, filter) {
				services = append(services, metadata)
			}
		}
	}

	return services
}

// 通知相关方法

// notifyServiceChange 发送服务变更通知
func (esr *EnhancedServiceRegistry) notifyServiceChange(event *ServiceChangeEvent) {
	esr.watchersMu.RLock()
	defer esr.watchersMu.RUnlock()

	for channelName, ch := range esr.notifyChannels {
		select {
		case ch <- event:
			logger.Debug("Sent service change notification to channel %s", channelName)
		default:
			logger.Warning("Failed to send notification to channel %s (channel full)", channelName)
		}
	}
}

// 租约统计更新

// updateLeaseStats 更新租约统计
func (lm *LeaseManager) updateLeaseStats(leaseID clientv3.LeaseID, success bool) {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	stats, exists := lm.leaseStats[leaseID]
	if !exists {
		return
	}

	stats.LastRenewed = time.Now()
	if success {
		stats.RenewCount++
	} else {
		stats.FailureCount++
	}

	// 计算成功率
	total := stats.RenewCount + stats.FailureCount
	if total > 0 {
		stats.SuccessRate = float64(stats.RenewCount) / float64(total)
	}
}

// adjustTTL 自适应TTL调整
func (lm *LeaseManager) adjustTTL(serviceKey string, leaseID clientv3.LeaseID) {
	lm.mu.RLock()
	stats := lm.leaseStats[leaseID]
	healthScore := lm.healthScores[serviceKey]
	successRate := lm.renewSuccessRate[leaseID]
	lm.mu.RUnlock()

	if stats == nil {
		return
	}

	// 基于健康评分和续期成功率调整TTL
	currentTTL := stats.TTL
	newTTL := currentTTL

	// 健康评分高且续期成功率高，增加TTL
	if healthScore > 0.8 && successRate > 0.95 {
		newTTL = int64(math.Min(float64(currentTTL)*1.2, float64(lm.config.MaxTTL)))
	} else if healthScore < 0.5 || successRate < 0.8 {
		// 健康评分低或续期失败率高，减少TTL
		newTTL = int64(math.Max(float64(currentTTL)*0.8, float64(lm.config.MinTTL)))
	}

	if newTTL != currentTTL {
		logger.Debug("Adjusting TTL for service %s from %d to %d (health: %.2f, success: %.2f)",
			serviceKey, currentTTL, newTTL, healthScore, successRate)
		stats.TTL = newTTL
	}
}
