package server

import (
	"VectorSphere/src/library/entity"
	"context"
	"encoding/json"
	"fmt"
	"github.com/cenkalti/backoff/v4"
	etcdv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"sync"
	"time"
)

// EnhancedServiceHub 增强的服务中心
type EnhancedServiceHub struct {
	client          *etcdv3.Client
	heartbeat       int64
	leaseID         etcdv3.LeaseID
	session         *concurrency.Session
	watched         sync.Map
	loadBalancer    Balancer
	serviceRegistry map[string]*ServiceInfo
	mu              sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
	retryPolicy     *backoff.ExponentialBackOff
}

// ServiceInfo 服务信息
type ServiceInfo struct {
	ServiceName string            `json:"serviceName"`
	Endpoint    *entity.EndPoint  `json:"endpoint"`
	Metadata    map[string]string `json:"metadata"`
	Version     string            `json:"version"`
	Weight      int               `json:"weight"`
	Healthy     bool              `json:"healthy"`
	RegistTime  time.Time         `json:"registTime"`
	LeaseID     etcdv3.LeaseID    `json:"leaseID"`

	Status      string          `json:"status"`
	LastSeen    time.Time       `json:"last_seen"`
	HealthCheck HealthCheckInfo `json:"health_check"`
}

// NewEnhancedServiceHub 创建增强的服务中心
func NewEnhancedServiceHub(endpoints []string, heartbeat int64) (*EnhancedServiceHub, error) {
	ctx, cancel := context.WithCancel(context.Background())

	client, err := etcdv3.New(etcdv3.Config{
		Endpoints:            endpoints,
		DialTimeout:          time.Duration(heartbeat) * time.Second,
		DialKeepAliveTime:    3 * time.Second,
		DialKeepAliveTimeout: 3 * time.Second,
		AutoSyncInterval:     30 * time.Second, // 自动同步集群信息
	})
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create etcd client: %w", err)
	}

	session, err := concurrency.NewSession(client, concurrency.WithTTL(int(heartbeat)))
	if err != nil {
		client.Close()
		cancel()
		return nil, fmt.Errorf("failed to create etcd session: %w", err)
	}

	// 配置重试策略
	retryPolicy := backoff.NewExponentialBackOff()
	retryPolicy.InitialInterval = 1 * time.Second
	retryPolicy.MaxInterval = 30 * time.Second
	retryPolicy.MaxElapsedTime = 5 * time.Minute

	hub := &EnhancedServiceHub{
		client:          client,
		heartbeat:       heartbeat,
		session:         session,
		watched:         sync.Map{},
		loadBalancer:    LoadBalanceFactory(WeightedRoundRobin),
		serviceRegistry: make(map[string]*ServiceInfo),
		ctx:             ctx,
		cancel:          cancel,
		retryPolicy:     retryPolicy,
	}

	// 启动心跳保持
	go hub.keepAliveLoop()

	return hub, nil
}

// RegisterServiceWithMetadata 注册服务（带元数据）
func (h *EnhancedServiceHub) RegisterServiceWithMetadata(serviceName string, endpoint *entity.EndPoint, metadata map[string]string, version string, weight int) error {
	operation := func() error {
		lease := etcdv3.NewLease(h.client)
		leaseResp, err := lease.Grant(h.ctx, h.heartbeat)
		if err != nil {
			return fmt.Errorf("failed to grant lease: %w", err)
		}

		serviceInfo := &ServiceInfo{
			ServiceName: serviceName,
			Endpoint:    endpoint,
			Metadata:    metadata,
			Version:     version,
			Weight:      weight,
			Healthy:     true,
			RegistTime:  time.Now(),
			LeaseID:     leaseResp.ID,
		}

		key := fmt.Sprintf("%s/%s/%s:%d", ServiceRootPath, serviceName, endpoint.Ip, endpoint.Port)
		val, _ := json.Marshal(serviceInfo)

		_, err = h.client.Put(h.ctx, key, string(val), etcdv3.WithLease(leaseResp.ID))
		if err != nil {
			return fmt.Errorf("failed to register service: %w", err)
		}

		h.mu.Lock()
		h.serviceRegistry[key] = serviceInfo
		h.leaseID = leaseResp.ID
		h.mu.Unlock()

		return nil
	}

	return backoff.Retry(operation, backoff.WithContext(h.retryPolicy, h.ctx))
}

// keepAliveLoop 心跳保持循环
func (h *EnhancedServiceHub) keepAliveLoop() {
	for {
		select {
		case <-h.ctx.Done():
			return
		default:
			h.mu.RLock()
			leaseID := h.leaseID
			h.mu.RUnlock()

			if leaseID > 0 {
				ch, kaerr := h.client.KeepAlive(h.ctx, leaseID)
				if kaerr != nil {
					// 重新注册服务
					go h.reregisterServices()
					continue
				}

				go func() {
					for ka := range ch {
						if ka != nil {
							// 心跳成功
							continue
						}
					}
				}()
			}
			time.Sleep(time.Duration(h.heartbeat/3) * time.Second)
		}
	}
}

// reregisterServices 重新注册所有服务
func (h *EnhancedServiceHub) reregisterServices() {
	h.mu.RLock()
	services := make(map[string]*ServiceInfo)
	for k, v := range h.serviceRegistry {
		services[k] = v
	}
	h.mu.RUnlock()

	for _, service := range services {
		h.RegisterServiceWithMetadata(
			service.ServiceName,
			service.Endpoint,
			service.Metadata,
			service.Version,
			service.Weight,
		)
	}
}

// WatchServiceChanges 监听服务变化
func (h *EnhancedServiceHub) WatchServiceChanges(serviceName string, callback func([]ServiceInfo)) {
	if _, loaded := h.watched.LoadOrStore(serviceName, true); loaded {
		return
	}

	go func() {
		prefix := fmt.Sprintf("%s/%s/", ServiceRootPath, serviceName)
		watchChan := h.client.Watch(h.ctx, prefix, etcdv3.WithPrefix())

		for watchResp := range watchChan {
			if watchResp.Err() != nil {
				continue
			}

			services := h.GetServiceInfos(serviceName)
			callback(services)
		}
	}()
}

// GetServiceInfos 获取服务信息列表
func (h *EnhancedServiceHub) GetServiceInfos(serviceName string) []ServiceInfo {
	prefix := fmt.Sprintf("%s/%s/", ServiceRootPath, serviceName)
	resp, err := h.client.Get(h.ctx, prefix, etcdv3.WithPrefix())
	if err != nil {
		return nil
	}

	var services []ServiceInfo
	for _, kv := range resp.Kvs {
		var service ServiceInfo
		if err := json.Unmarshal(kv.Value, &service); err == nil {
			services = append(services, service)
		}
	}
	return services
}
