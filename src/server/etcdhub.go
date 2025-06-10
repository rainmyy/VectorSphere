package server

import (
	"VectorSphere/src/library/log"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/cenkalti/backoff/v4"
	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	etcdv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health/grpc_health_v1"
	"net"
	"strconv"
	"sync"
	"time"
)

type ServiceHub interface {
	GetClient() *etcdv3.Client
	RegisterService(serviceName string, endpoint *EndPoint, leaseId etcdv3.LeaseID) (etcdv3.LeaseID, error)
	UnRegisterService(serviceName string, endpoint *EndPoint) error
	GetServiceEndpoints(serviceName string) []EndPoint
	GetServiceEndpoint(serviceName string) EndPoint
	StartHeartbeat(ctx context.Context, serviceName string, endpoint *EndPoint) error
	StopHeartbeat(serviceName string, endpoint *EndPoint)
	WatchServiceChanges(ctx context.Context, serviceName string, callback func([]EndPoint))
	Close()
	RefreshEndpoints(serviceName string)
}

type EtcdServiceHub struct {
	client    *etcdv3.Client // etcd客户端，用于与etcd进行操作
	heartbeat int64          // 服务续约的心跳频率，单位：秒
	leaseId   etcdv3.LeaseID
	session   *concurrency.Session
	election  *concurrency.Election

	watched      sync.Map // 存储已经监视的服务，以避免重复监视
	loadBalancer Balancer // 负载均衡策略的接口，支持多种负载均衡实现

	// 新增字段用于心跳管理
	heartbeatMap    sync.Map // map[string]context.CancelFunc 存储每个服务的心跳取消函数
	leaseMap        sync.Map // map[string]etcdv3.LeaseID 存储每个服务的租约ID
	serviceWatchers sync.Map // map[string]context.CancelFunc 存储服务监听的取消函数

	// 添加版本信息字段
	version string
}

var (
	etcdServiceHub *EtcdServiceHub
	hubOnce        sync.Once
)

// HealthCheckInfo 健康检查信息
type HealthCheckInfo struct {
	Enabled  bool          `json:"enabled"`
	Interval time.Duration `json:"interval"`
	Timeout  time.Duration `json:"timeout"`
	Path     string        `json:"path"`
}

const ServiceRootPath = "/opt/vector_sphere/services"

func init() {

}

func GetHub(endPoints []EndPoint, heartbeat int64, serviceName string) (error, *EtcdServiceHub) {
	if etcdServiceHub != nil {
		return nil, etcdServiceHub
	}

	var endPointIp []string
	for _, endPoint := range endPoints {
		endPointIp = append(endPointIp, endPoint.Ip+":"+strconv.Itoa(endPoint.Port))
	}
	var er error
	hubOnce.Do(func() {
		client, err := etcdv3.New(etcdv3.Config{
			Endpoints:            endPointIp,
			DialTimeout:          time.Duration(heartbeat) * time.Second,
			DialKeepAliveTime:    3 * time.Second,
			DialKeepAliveTimeout: 3 * time.Second,
		})
		if err != nil {
			er = err
			return
		}

		session, err := concurrency.NewSession(client, concurrency.WithTTL(3))
		if err != nil {
			er = err
			return
		}
		defer session.Close()
		election := concurrency.NewElection(session, serviceName)
		etcdServiceHub = &EtcdServiceHub{
			client:       client,
			heartbeat:    heartbeat,
			watched:      sync.Map{},
			session:      session,
			election:     election,
			loadBalancer: LoadBalanceFactory(WeightedRoundRobin),
		}
	})

	return er, etcdServiceHub
}

// StartHeartbeat 启动服务心跳续约
func (etcd *EtcdServiceHub) StartHeartbeat(ctx context.Context, serviceName string, endpoint *EndPoint) error {
	key := fmt.Sprintf("%s/%s", serviceName, endpoint.Ip)

	// 如果已经存在心跳，先停止
	if cancel, exists := etcd.heartbeatMap.Load(key); exists {
		cancel.(context.CancelFunc)()
	}

	// 创建租约
	lease := etcdv3.NewLease(etcd.client)
	leaseResp, err := lease.Grant(ctx, etcd.heartbeat)
	if err != nil {
		return fmt.Errorf("failed to grant lease: %w", err)
	}

	// 注册服务时添加健康状态信息
	serviceInfo := ServiceInfo{
		Endpoint: endpoint,
		Status:   "healthy",
		LastSeen: time.Now(),
		Version:  etcd.version,
		Metadata: endpoint.Tags,
		HealthCheck: HealthCheckInfo{
			Enabled:  true,
			Interval: 30 * time.Second,
			Timeout:  5 * time.Second,
			Path:     "/health",
		},
	}

	serviceKey := ServiceRootPath + "/" + serviceName + "/" + endpoint.Ip
	val, _ := json.Marshal(serviceInfo)
	_, err = etcd.client.Put(ctx, serviceKey, string(val), etcdv3.WithLease(leaseResp.ID))
	if err != nil {
		return fmt.Errorf("failed to register service: %w", err)
	}

	// 启动心跳续约
	heartbeatCtx, cancel := context.WithCancel(ctx)
	etcd.heartbeatMap.Store(key, cancel)
	etcd.leaseMap.Store(key, leaseResp.ID)

	go etcd.keepAliveWithRetry(heartbeatCtx, lease, leaseResp.ID, serviceName, endpoint)
	go etcd.monitorServiceHealth(heartbeatCtx, serviceName, endpoint)

	return nil
}

// monitorServiceHealth 监控服务健康状态
func (etcd *EtcdServiceHub) monitorServiceHealth(ctx context.Context, serviceName string, endpoint *EndPoint) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := etcd.performHealthCheck(serviceName, endpoint); err != nil {
				log.Warning("Health check failed for service %s (%s:%d): %v",
					serviceName, endpoint.Ip, endpoint.Port, err)
				// 可以选择更新服务状态为不健康
				etcd.updateServiceStatus(serviceName, endpoint, "unhealthy")
			} else {
				etcd.updateServiceStatus(serviceName, endpoint, "healthy")
			}
		}
	}
}

// performHealthCheck 执行健康检查
func (etcd *EtcdServiceHub) performHealthCheck(serviceName string, endpoint *EndPoint) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, fmt.Sprintf("%s:%d", endpoint.Ip, endpoint.Port),
		grpc.WithInsecure(),
		grpc.WithBlock(),
	)
	if err != nil {
		return err
	}
	defer conn.Close()

	client := grpc_health_v1.NewHealthClient(conn)
	_, err = client.Check(ctx, &grpc_health_v1.HealthCheckRequest{
		Service: serviceName,
	})

	return err
}

// updateServiceStatus 更新服务状态
func (etcd *EtcdServiceHub) updateServiceStatus(serviceName string, endpoint *EndPoint, status string) {
	serviceKey := ServiceRootPath + "/" + serviceName + "/" + endpoint.Ip

	// 获取当前服务信息
	resp, err := etcd.client.Get(context.Background(), serviceKey)
	if err != nil || len(resp.Kvs) == 0 {
		return
	}

	var serviceInfo ServiceInfo
	if err := json.Unmarshal(resp.Kvs[0].Value, &serviceInfo); err != nil {
		return
	}

	// 更新状态和时间戳
	serviceInfo.Status = status
	serviceInfo.LastSeen = time.Now()

	val, _ := json.Marshal(serviceInfo)
	etcd.client.Put(context.Background(), serviceKey, string(val))
}

// keepAliveWithRetry 带重试机制的心跳续约
func (etcd *EtcdServiceHub) keepAliveWithRetry(ctx context.Context, lease etcdv3.Lease, leaseID etcdv3.LeaseID, serviceName string, endpoint *EndPoint) {
	bo := backoff.NewExponentialBackOff()
	bo.MaxElapsedTime = 0 // 无限重试
	bo.MaxInterval = 30 * time.Second

	backoff.RetryNotify(func() error {
		select {
		case <-ctx.Done():
			return backoff.Permanent(ctx.Err())
		default:
		}

		keepaliveResp, err := lease.KeepAlive(ctx, leaseID)
		if err != nil {
			return fmt.Errorf("keepalive failed: %w", err)
		}

		// 处理心跳响应
		go func() {
			for ka := range keepaliveResp {
				if ka == nil {
					log.Warning("Lease %d expired for service %s", leaseID, serviceName)
					return
				}
				log.Info("Heartbeat success for service %s, TTL: %d", serviceName, ka.TTL)
			}
		}()

		return nil
	}, backoff.WithContext(bo, ctx), func(err error, duration time.Duration) {
		log.Info("Heartbeat failed for service %s, retrying in %v: %v", serviceName, duration, err)
	})
}

// StopHeartbeat 停止服务心跳
func (etcd *EtcdServiceHub) StopHeartbeat(serviceName string, endpoint *EndPoint) {
	key := fmt.Sprintf("%s/%s", serviceName, endpoint.Ip)

	if cancel, exists := etcd.heartbeatMap.Load(key); exists {
		cancel.(context.CancelFunc)()
		etcd.heartbeatMap.Delete(key)
	}

	if leaseID, exists := etcd.leaseMap.Load(key); exists {
		// 撤销租约
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, err := etcd.client.Revoke(ctx, leaseID.(etcdv3.LeaseID))
		if err != nil {
			log.Error("Failed to revoke lease for service %s: %v", serviceName, err)
		}
		etcd.leaseMap.Delete(key)
	}
}

// WatchServiceChanges 监听服务变化
func (etcd *EtcdServiceHub) WatchServiceChanges(ctx context.Context, serviceName string, callback func([]EndPoint)) {
	// 检查是否已经在监听
	if cancel, exists := etcd.serviceWatchers.Load(serviceName); exists {
		cancel.(context.CancelFunc)()
	}

	watchCtx, cancel := context.WithCancel(ctx)
	etcd.serviceWatchers.Store(serviceName, cancel)

	go func() {
		defer cancel()

		prefix := ServiceRootPath + "/" + serviceName + "/"
		watchChan := etcd.client.Watch(watchCtx, prefix, etcdv3.WithPrefix())

		// 初始获取当前服务列表
		callback(etcd.GetServiceEndpoints(serviceName))

		for watchResp := range watchChan {
			if watchResp.Err() != nil {
				log.Error("Watch error for service %s: %v", serviceName, watchResp.Err())
				continue
			}

			// 服务列表发生变化，通知回调
			callback(etcd.GetServiceEndpoints(serviceName))
		}
	}()
}

func (etcd *EtcdServiceHub) RegisterService(service string, endpoint *EndPoint, leaseId etcdv3.LeaseID) (etcdv3.LeaseID, error) {
	if etcd.client == nil {
		return 0, errors.New("etcd client is nil")
	}

	if leaseId <= 0 {
		lease := etcdv3.NewLease(etcd.client)
		leaseResp, err := lease.Grant(context.Background(), 3)
		if err != nil {
			return 0, err
		}
		key := ServiceRootPath + "/" + service + "/" + endpoint.Ip
		val, _ := json.Marshal(endpoint)
		_, err = etcd.client.Put(context.Background(), key, string(val), etcdv3.WithLease(leaseResp.ID))
		if err != nil {
			return 0, err
		}
		return leaseResp.ID, nil
	}
	_, err := etcd.client.KeepAliveOnce(context.Background(), leaseId)
	if errors.Is(err, rpctypes.ErrLeaseNotFound) {
		return etcd.RegisterService(service, endpoint, 0)
	}

	etcd.leaseId = leaseId
	return leaseId, nil
}

func (etcd *EtcdServiceHub) UnRegisterService(serviceName string, endpoint *EndPoint) error {
	key := ServiceRootPath + "/" + serviceName + "/" + endpoint.Ip
	_, err := etcd.client.Delete(context.Background(), key)
	if err != nil {
		return err
	}

	_, err = etcd.client.Revoke(context.Background(), etcd.leaseId)
	return err
}

func (etcd *EtcdServiceHub) GetServiceEndpoints(serviceName string) []EndPoint {
	prefix := ServiceRootPath + "/" + serviceName + "/"
	resp, err := etcd.client.Get(context.Background(), prefix, etcdv3.WithPrefix())
	if err != nil {
		return nil
	}
	endpoints := make([]EndPoint, 0, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		var ep EndPoint
		if err := json.Unmarshal(kv.Value, &ep); err == nil {
			endpoints = append(endpoints, ep)
		}
	}
	return endpoints
}

//func (etcd *EtcdServiceHub) GetServiceEndpoint(serviceName string) EndPoint {
//	endpoints := etcd.GetServiceEndpoints(serviceName)
//	etcd.loadBalancer.Set(endpoints...)
//	return etcd.loadBalancer.Take()
//}

// GetServiceEndpoint 支持更多负载均衡策略
func (etcd *EtcdServiceHub) GetServiceEndpoint(serviceName string) EndPoint {
	return etcd.GetServiceEndpointWithStrategy(serviceName, "")
}

// GetServiceEndpointWithStrategy 根据策略获取服务端点
func (etcd *EtcdServiceHub) GetServiceEndpointWithStrategy(serviceName, strategy string) EndPoint {
	endpoints := etcd.GetServiceEndpoints(serviceName)

	// 根据策略选择负载均衡器
	var balancer Balancer
	switch strategy {
	case "random":
		balancer = LoadBalanceFactory(Random)
	case "round_robin":
		balancer = LoadBalanceFactory(RoundRobin)
	case "weighted":
		balancer = LoadBalanceFactory(WeightedRoundRobin)
	case "least_conn":
		balancer = LoadBalanceFactory(LeastConnections)
	case "consistent_hash":
		balancer = LoadBalanceFactory(ConsistentHash)
	case "response_time":
		balancer = LoadBalanceFactory(ResponseTimeWeighted)
	case "adaptive":
		balancer = LoadBalanceFactory(AdaptiveRoundRobin)
	case "AdaptiveWeighted":
		balancer = LoadBalanceFactory(AdaptiveWeighted)
	default:
		balancer = etcd.loadBalancer
	}

	balancer.Set(endpoints...)
	return balancer.Take()
}

// GetServiceEndpointWithClientIP 根据客户端IP获取服务端点（用于源IP哈希）
func (etcd *EtcdServiceHub) GetServiceEndpointWithClientIP(serviceName, clientIP string) EndPoint {
	endpoints := etcd.GetServiceEndpoints(serviceName)
	balancer := NewSourceIPHashBalancer()
	balancer.Set(endpoints...)
	return balancer.TakeWithContext(clientIP)
}

// GetHealthyEndpoints 获取健康的服务端点
func (etcd *EtcdServiceHub) GetHealthyEndpoints(serviceName string) []EndPoint {
	allEndpoints := etcd.GetServiceEndpoints(serviceName)
	var healthyEndpoints []EndPoint

	for _, endpoint := range allEndpoints {
		if etcd.isEndpointHealthy(endpoint) {
			healthyEndpoints = append(healthyEndpoints, endpoint)
		}
	}

	return healthyEndpoints
}

// isEndpointHealthy 检查端点是否健康
func (etcd *EtcdServiceHub) isEndpointHealthy(endpoint EndPoint) bool {
	// 实现健康检查逻辑
	// 这里可以通过HTTP健康检查、TCP连接测试等方式
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", endpoint.Ip, endpoint.Port), 3*time.Second)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

func (etcd *EtcdServiceHub) Close() {
	etcd.client.Close()
}

func (etcd *EtcdServiceHub) RefreshEndpoints(serviceName string) {
	// Implementation of RefreshEndpoints method
}
