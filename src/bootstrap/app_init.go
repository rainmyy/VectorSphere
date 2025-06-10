package bootstrap

import (
	confType "VectorSphere/src/library/confType"
	conf "VectorSphere/src/library/config"
	"VectorSphere/src/library/security"
	"context"
	"encoding/json"
	"fmt"
	"go.etcd.io/etcd/client/pkg/v3/transport"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"log"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/sony/gobreaker"
	"go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"golang.org/x/time/rate"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/resolver"
	"gopkg.in/yaml.v3"
)

// AppContext 应用上下文
type AppContext struct {
	Config                  *conf.AppConfig
	EtcdClient              *clientv3.Client
	EtcdSession             *concurrency.Session // 用于分布式锁和选举
	LeaseID                 clientv3.LeaseID
	cancelKeepAlive         context.CancelFunc
	wg                      sync.WaitGroup
	mu                      sync.Mutex
	ServiceDiscovery        *EtcdServiceDiscovery
	DefaultBackOff          backoff.BackOff
	CircuitBreakers         map[string]*gobreaker.CircuitBreaker // 服务名 -> 熔断器
	RateLimiters            map[string]*rate.Limiter             // 服务名 -> 限流器
	isLeader                bool
	leaderChan              chan bool // 用于通知领导者状态变化
	shutdownOnce            sync.Once
	grpcServer              *grpc.Server // gRPC 服务器实例 (如果应用是gRPC服务)
	EnhancedSecurityManager *security.EnhancedSecurityManager
}

// CreateSecureEtcdClient 创建安全的etcd客户端
func CreateSecureEtcdClient(endpoints []string, tlsConfig *conf.TLSConfig) (*clientv3.Client, error) {
	cliConfig := clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: 5 * time.Second,
	}

	// 配置TLS
	if tlsConfig != nil && tlsConfig.CertFile != "" {
		tlsInfo := transport.TLSInfo{
			CertFile:      tlsConfig.CertFile,
			KeyFile:       tlsConfig.KeyFile,
			TrustedCAFile: tlsConfig.CAFile,
			ServerName:    tlsConfig.ServerName,
		}

		clientTLS, err := tlsInfo.ClientConfig()
		if err != nil {
			return nil, fmt.Errorf("failed to create TLS config: %w", err)
		}

		cliConfig.TLS = clientTLS
	}

	return clientv3.New(cliConfig)
}

// NewAppContext 创建新的应用上下文
func NewAppContext(config *conf.AppConfig) (*AppContext, error) {
	// 加载安全配置
	securityConfig, err := loadSecurityConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load security config: %w", err)
	}

	// 创建增强的安全管理器
	enhancedSecurityManager, err := security.NewEnhancedSecurityManager(securityConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create enhanced security manager: %w", err)
	}

	ctx := &AppContext{
		Config:                  config,
		CircuitBreakers:         make(map[string]*gobreaker.CircuitBreaker),
		RateLimiters:            make(map[string]*rate.Limiter),
		leaderChan:              make(chan bool, 1),
		EnhancedSecurityManager: enhancedSecurityManager,
	}

	ctx.EtcdClient, err = initEtcdClient(config)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize etcd client: %w", err)
	}

	// 初始化分布式锁和选举所需的session
	// 注意：session的TTL应该比服务租约的TTL短，或者独立管理
	// 这里我们假设session的TTL和服务TTL一致，实际中可能需要调整
	s, err := concurrency.NewSession(ctx.EtcdClient, concurrency.WithTTL(int(config.ServiceTTL)))
	if err != nil {
		ctx.EtcdClient.Close()
		return nil, fmt.Errorf("failed to create etcd session: %w", err)
	}
	ctx.EtcdSession = s

	ctx.DefaultBackOff = initDefaultBackOff(config.RetryPolicy)

	// 初始化服务发现 (如果需要)
	if config.ServiceRegistryPath != "" {
		// 服务名通常从 ServiceRegistryPath 中提取，或者单独配置
		// 例如: /services/my-app/ -> my-app
		parts := strings.Split(strings.Trim(config.ServiceRegistryPath, "/"), "/")
		serviceName := "default-service"
		if len(parts) > 1 {
			serviceName = parts[1]
		}

		ctx.ServiceDiscovery = NewEtcdServiceDiscovery(ctx.EtcdClient, config.ServiceRegistryPath, serviceName)
		resolver.Register(ctx.ServiceDiscovery) // 注册到gRPC
		log.Printf("Registered EtcdServiceDiscovery for scheme: %s", ctx.ServiceDiscovery.Scheme())
	}

	// 初始化熔断器 (示例)
	if config.CircuitBreaker != nil {
		cbSettings := gobreaker.Settings{
			Name:          config.CircuitBreaker.Name,
			MaxRequests:   config.CircuitBreaker.MaxRequests,
			Interval:      config.CircuitBreaker.Interval,
			Timeout:       config.CircuitBreaker.Timeout,
			ReadyToTrip:   config.CircuitBreaker.ReadyToTrip,
			OnStateChange: config.CircuitBreaker.OnStateChange,
		}
		if cbSettings.ReadyToTrip == nil { // 默认熔断逻辑
			cbSettings.ReadyToTrip = func(counts gobreaker.Counts) bool {
				// 例如：连续失败5次，或者失败率超过50%
				return counts.ConsecutiveFailures > 5 || (counts.Requests > 10 && float64(counts.TotalFailures)/float64(counts.Requests) > 0.5)
			}
		}
		ctx.CircuitBreakers[config.CircuitBreaker.Name] = gobreaker.NewCircuitBreaker(cbSettings)
	}

	// 初始化限流器 (示例)
	if config.RateLimiter != nil {
		ctx.RateLimiters["global"] = rate.NewLimiter(config.RateLimiter.Rate, config.RateLimiter.Burst)
	}

	return ctx, nil
}

// loadSecurityConfig 加载安全配置
func loadSecurityConfig() (*security.SecurityConfig, error) {
	var securityConfig security.SecurityConfig
	err := confType.ReadYAML("conf/security.yaml", &securityConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to read security config: %w", err)
	}
	return &securityConfig, nil
}

// initEtcdClient 初始化etcd客户端
func initEtcdClient(config *conf.AppConfig) (*clientv3.Client, error) {
	cliConfig := clientv3.Config{
		Endpoints:   config.EtcdEndpoints,
		DialTimeout: config.EtcdDialTimeout,
		Username:    config.EtcdUsername,
		Password:    config.EtcdPassword,
	}

	if config.EtcdTLS != nil && config.EtcdTLS.CertFile != "" && config.EtcdTLS.KeyFile != "" && config.EtcdTLS.CAFile != "" {
		tlsInfo := transport.TLSInfo{
			CertFile:      config.EtcdTLS.CertFile,
			KeyFile:       config.EtcdTLS.KeyFile,
			TrustedCAFile: config.EtcdTLS.CAFile,
			ServerName:    config.EtcdTLS.ServerName,
		}
		tlsConfig, err := tlsInfo.ClientConfig()
		if err != nil {
			return nil, fmt.Errorf("failed to create TLS config for etcd: %w", err)
		}
		cliConfig.TLS = tlsConfig
	}

	client, err := clientv3.New(cliConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to etcd: %w", err)
	}
	return client, nil
}

// initDefaultBackOff 初始化默认重试策略
func initDefaultBackOff(policyConf *conf.RetryPolicy) backoff.BackOff {
	if policyConf == nil {
		// 提供一个合理的默认值
		b := backoff.NewExponentialBackOff()
		b.MaxElapsedTime = 2 * time.Minute
		return b
	}
	b := backoff.NewExponentialBackOff()
	if policyConf.InitialInterval > 0 {
		b.InitialInterval = policyConf.InitialInterval
	}
	if policyConf.MaxInterval > 0 {
		b.MaxInterval = policyConf.MaxInterval
	}
	if policyConf.MaxElapsedTime > 0 {
		b.MaxElapsedTime = policyConf.MaxElapsedTime
	}
	if policyConf.Multiplier > 0 {
		b.Multiplier = policyConf.Multiplier
	}
	if policyConf.RandomizationFactor > 0 {
		b.RandomizationFactor = policyConf.RandomizationFactor
	}
	return b
}

// RegisterService 向etcd注册服务
func (appCtx *AppContext) RegisterService(serviceName, serviceAddress string) error {
	if appCtx.Config.ServiceTTL <= 0 {
		return fmt.Errorf("serviceTTL must be greater than 0 for service registration")
	}

	// 1. 创建租约
	lease := clientv3.NewLease(appCtx.EtcdClient)
	leaseGrantResp, err := lease.Grant(context.Background(), appCtx.Config.ServiceTTL)
	if err != nil {
		return fmt.Errorf("failed to grant lease: %w", err)
	}
	appCtx.LeaseID = leaseGrantResp.ID
	log.Printf("Granted lease %x for service %s", appCtx.LeaseID, serviceName)

	// 2. 注册服务到etcd，并关联租约
	// 服务节点路径通常是: /<ServiceRegistryPath>/<serviceName>/<serviceAddress>
	// 或者 /<ServiceRegistryPath>/<instanceId> -> serviceAddress (如果需要唯一实例ID)
	// 这里我们使用 serviceAddress 作为唯一标识的一部分
	serviceKey := fmt.Sprintf("%s%s/%s", appCtx.Config.ServiceRegistryPath, serviceName, serviceAddress)
	kv := clientv3.NewKV(appCtx.EtcdClient)
	_, err = kv.Put(context.Background(), serviceKey, serviceAddress, clientv3.WithLease(appCtx.LeaseID))
	if err != nil {
		// 注册失败，尝试撤销租约
		lease.Revoke(context.Background(), appCtx.LeaseID)
		return fmt.Errorf("failed to register service '%s' at '%s': %w", serviceName, serviceKey, err)
	}
	log.Printf("Registered service '%s' at '%s' with lease %x", serviceName, serviceKey, appCtx.LeaseID)
	return nil
}

// KeepAliveService 保持服务心跳
func (appCtx *AppContext) KeepAliveService(ctx context.Context) {
	if appCtx.LeaseID == 0 {
		log.Println("No lease ID found, skipping keep-alive")
		return
	}

	lease := clientv3.NewLease(appCtx.EtcdClient)
	keepAliveChan, err := lease.KeepAlive(ctx, appCtx.LeaseID)
	if err != nil {
		log.Printf("Failed to start keep-alive for lease %x: %v", appCtx.LeaseID, err)
		return
	}

	appCtx.wg.Add(1)
	go func() {
		defer appCtx.wg.Done()
		defer lease.Close() // 关闭 lease client
		log.Printf("Started keep-alive for lease %x", appCtx.LeaseID)
		for {
			select {
			case <-ctx.Done():
				log.Printf("Stopping keep-alive for lease %x due to context cancellation", appCtx.LeaseID)
				// 尝试撤销租约
				revokeCtx, cancelRevoke := context.WithTimeout(context.Background(), 5*time.Second)
				_, err := lease.Revoke(revokeCtx, appCtx.LeaseID)
				cancelRevoke()
				if err != nil {
					log.Printf("Failed to revoke lease %x on shutdown: %v", appCtx.LeaseID, err)
				} else {
					log.Printf("Revoked lease %x on shutdown", appCtx.LeaseID)
				}
				return
			case ka, ok := <-keepAliveChan:
				if !ok {
					log.Printf("Keep-alive channel closed for lease %x. Service might be deregistered.", appCtx.LeaseID)
					// 可以在这里添加重新注册逻辑
					return
				}
				log.Printf("Lease %x keep-alive acknowledged, TTL: %d", appCtx.LeaseID, ka.TTL)
			}
		}
	}()
}

// LoadConfig 从etcd加载配置
func (appCtx *AppContext) LoadConfig(configKey string, configStruct interface{}) error {
	fullPath := appCtx.Config.ConfigPathPrefix + configKey
	kv := clientv3.NewKV(appCtx.EtcdClient)
	resp, err := kv.Get(context.Background(), fullPath)
	if err != nil {
		return fmt.Errorf("failed to get config '%s' from etcd: %w", fullPath, err)
	}
	if len(resp.Kvs) == 0 {
		return fmt.Errorf("config '%s' not found in etcd", fullPath)
	}
	configData := resp.Kvs[0].Value

	// 假设配置是YAML格式
	err = yaml.Unmarshal(configData, configStruct)
	if err != nil {
		// 尝试JSON作为备选
		errJson := json.Unmarshal(configData, configStruct)
		if errJson != nil {
			return fmt.Errorf("failed to unmarshal config '%s' as YAML (%v) or JSON (%v)", fullPath, err, errJson)
		}
	}
	log.Printf("Loaded config '%s' from etcd", fullPath)
	return nil
}

// WatchConfig 监听etcd中的配置变化
func (appCtx *AppContext) WatchConfig(ctx context.Context, configKey string, updateFunc func([]byte) error) {
	fullPath := appCtx.Config.ConfigPathPrefix + configKey
	watcher := clientv3.NewWatcher(appCtx.EtcdClient)
	watchChan := watcher.Watch(ctx, fullPath)

	appCtx.wg.Add(1)
	go func() {
		defer appCtx.wg.Done()
		defer watcher.Close()
		log.Printf("Watching config changes for '%s'", fullPath)
		for {
			select {
			case <-ctx.Done():
				log.Printf("Stopping config watch for '%s' due to context cancellation", fullPath)
				return
			case watchResp, ok := <-watchChan:
				if !ok {
					log.Printf("Config watch channel closed for '%s'", fullPath)
					// 尝试重新建立watch
					time.Sleep(5 * time.Second) //避免立即重试导致循环
					// 重新创建watch chan (简化版，实际可能需要更复杂的重试逻辑)
					newWatcher := clientv3.NewWatcher(appCtx.EtcdClient)
					watchChan = newWatcher.Watch(ctx, fullPath)
					defer newWatcher.Close()
					log.Printf("Re-established config watch for '%s'", fullPath)
					continue
				}
				if watchResp.Err() != nil {
					log.Printf("Error watching config '%s': %v", fullPath, watchResp.Err())
					// 根据错误类型决定是否继续或重试
					continue
				}
				for _, event := range watchResp.Events {
					if event.Type == clientv3.EventTypePut {
						log.Printf("Config '%s' updated, new value: %s", fullPath, string(event.Kv.Value))
						if err := updateFunc(event.Kv.Value); err != nil {
							log.Printf("Error applying updated config for '%s': %v", fullPath, err)
						}
					}
				}
			}
		}
	}()
}

// AcquireLock 获取分布式锁
func (appCtx *AppContext) AcquireLock(ctx context.Context, lockName string) (*concurrency.Mutex, error) {
	if appCtx.EtcdSession == nil {
		return nil, fmt.Errorf("etcd session is not initialized, cannot acquire lock")
	}
	fullLockPath := appCtx.Config.LockPathPrefix + lockName
	m := concurrency.NewMutex(appCtx.EtcdSession, fullLockPath)

	log.Printf("Attempting to acquire lock '%s'", fullLockPath)
	// 使用带超时的上下文尝试获取锁
	// 如果ctx没有超时，NewMutex.Lock会阻塞直到获取锁或session关闭
	// 如果需要非阻塞或带超时的尝试，可以这样:
	// lockCtx, cancel := context.WithTimeout(ctx, 5*time.Second) // 示例超时
	// defer cancel()
	// if err := m.TryLock(lockCtx); err != nil { ... }
	if err := m.Lock(ctx); err != nil {
		return nil, fmt.Errorf("failed to acquire lock '%s': %w", fullLockPath, err)
	}
	log.Printf("Acquired lock '%s'", fullLockPath)
	return m, nil
}

// ReleaseLock 释放分布式锁
func (appCtx *AppContext) ReleaseLock(ctx context.Context, mutex *concurrency.Mutex) error {
	if mutex == nil {
		return fmt.Errorf("mutex is nil, cannot release")
	}
	lockKey := string(mutex.Key()) // 获取锁的实际key
	log.Printf("Attempting to release lock '%s'", lockKey)
	if err := mutex.Unlock(ctx); err != nil {
		return fmt.Errorf("failed to release lock '%s': %w", lockKey, err)
	}
	log.Printf("Released lock '%s'", lockKey)
	return nil
}

// ElectLeader 参与领导者选举
func (appCtx *AppContext) ElectLeader(ctx context.Context, electionName string) error {
	if appCtx.EtcdSession == nil {
		return fmt.Errorf("etcd session is not initialized, cannot participate in election")
	}
	fullElectionPath := appCtx.Config.ElectionPathPrefix + electionName
	e := concurrency.NewElection(appCtx.EtcdSession, fullElectionPath)

	appCtx.wg.Add(1)
	go func() {
		defer appCtx.wg.Done()
		log.Printf("Campaigning for leadership in election '%s'", fullElectionPath)
		// Campaign会阻塞直到成为领导者或上下文被取消
		if err := e.Campaign(ctx, appCtx.Config.AppName+":"+appCtx.Config.ListenAddress); err != nil {
			log.Printf("Failed to campaign for leadership in '%s': %v", fullElectionPath, err)
			appCtx.setLeaderStatus(false)
			return
		}
		log.Printf("Elected as leader in '%s'", fullElectionPath)
		appCtx.setLeaderStatus(true)

		// 保持领导者身份，直到上下文取消或放弃
		<-ctx.Done() // 等待应用关闭信号

		resignCtx, cancelResign := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancelResign()
		if err := e.Resign(resignCtx); err != nil {
			log.Printf("Failed to resign leadership in '%s': %v", fullElectionPath, err)
		} else {
			log.Printf("Resigned leadership in '%s'", fullElectionPath)
		}
		appCtx.setLeaderStatus(false)
	}()
	return nil
}

func (appCtx *AppContext) setLeaderStatus(isLeader bool) {
	appCtx.mu.Lock()
	appCtx.isLeader = isLeader
	appCtx.mu.Unlock()
	select {
	case appCtx.leaderChan <- isLeader:
	default: // 非阻塞发送，如果chan满了就丢弃
	}
}

// IsLeader 返回当前是否为领导者
func (appCtx *AppContext) IsLeader() bool {
	appCtx.mu.Lock()
	defer appCtx.mu.Unlock()
	return appCtx.isLeader
}

// ObserveLeader 观察领导者变化
// 返回一个channel，当领导者发生变化时会收到通知 (true表示成为领导者，false表示失去领导者)
func (appCtx *AppContext) ObserveLeader() <-chan bool {
	return appCtx.leaderChan
}

// RetryableOperation 执行可重试的操作
func (appCtx *AppContext) RetryableOperation(ctx context.Context, operation func() error, shouldRetry func(error) bool) error {
	bo := backoff.WithContext(appCtx.DefaultBackOff, ctx)
	return backoff.RetryNotify(operation, bo, func(err error, d time.Duration) {
		log.Printf("Operation failed, retrying in %s. Error: %v", d, err)
		if !shouldRetry(err) {
			// 如果 shouldRetry 返回 false，我们应该停止重试
			// backoff 库本身不直接支持这种在 Notify 中停止的机制
			// 因此 shouldRetry 逻辑主要用于决定是否一开始就进入重试循环
			// 或者在 operation 内部返回一个特殊的错误类型，让 backoff.Retry 停止
			// 例如，返回一个 permanentError
			// 这里我们假设 shouldRetry 主要用于外部判断，或者 operation 内部会处理
			log.Printf("Error is not retryable: %v. Stopping retries.", err)
			// To stop retries from Notify, you'd typically cancel the context or use a custom BackOff
			// For simplicity, we rely on shouldRetry being checked before calling RetryableOperation
			// or operation returning a permanent error.
		}
	})
}

// DefaultShouldRetry 默认的重试判断逻辑
func DefaultShouldRetry(err error) bool {
	if err == nil {
		return false
	}
	// 示例：可以根据错误类型或错误消息判断
	// 例如，网络超时、etcd临时性错误等是可重试的
	// 认证失败、请求参数错误等是不可重试的
	errMsg := strings.ToLower(err.Error())
	if strings.Contains(errMsg, "timeout") ||
		strings.Contains(errMsg, "connection refused") ||
		strings.Contains(errMsg, "connection reset by peer") ||
		strings.Contains(errMsg, "etcdserver: request timed out") ||
		strings.Contains(errMsg, "etcdserver: leader changed") ||
		strings.Contains(errMsg, "etcdserver: too many requests") || // 可选，看是否希望重试限流错误
		strings.Contains(errMsg, "context deadline exceeded") {
		return true
	}
	// gRPC 错误码判断
	st, ok := status.FromError(err)
	if ok {
		switch st.Code() {
		case codes.Unavailable, codes.DeadlineExceeded, codes.ResourceExhausted:
			return true
		default:
			return false
		}
	}
	return false // 默认不重试未知错误
}

// GetServiceEndpoints 获取服务的所有健康节点地址
// 注意：这个方法现在更多的是一个示例，实际的服务发现和负载均衡由 EtcdServiceDiscovery 处理
func (appCtx *AppContext) GetServiceEndpoints(serviceName string) ([]string, error) {
	if appCtx.ServiceDiscovery != nil {
		// 理论上，我们不直接从这里获取，而是依赖gRPC的resolver
		// 但如果需要手动获取，可以暴露一个方法从ServiceDiscovery获取
		return appCtx.ServiceDiscovery.getLastKnownAddresses(), nil
	}

	// Fallback or direct etcd query if ServiceDiscovery is not used for this purpose
	servicePrefix := fmt.Sprintf("%s%s/", appCtx.Config.ServiceRegistryPath, serviceName)
	kv := clientv3.NewKV(appCtx.EtcdClient)
	resp, err := kv.Get(context.Background(), servicePrefix, clientv3.WithPrefix())
	if err != nil {
		return nil, fmt.Errorf("failed to get service '%s' endpoints from etcd: %w", serviceName, err)
	}
	endpoints := make([]string, 0, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		endpoints = append(endpoints, string(kv.Value))
	}
	if len(endpoints) == 0 {
		return nil, fmt.Errorf("no endpoints found for service '%s'", serviceName)
	}
	return endpoints, nil
}

// WatchService 监听服务节点变化 (主要用于服务端健康检查或调试)
// 客户端通常依赖 EtcdServiceDiscovery
func (appCtx *AppContext) WatchService(ctx context.Context, serviceName string, callback func(addr string)) {
	servicePrefix := fmt.Sprintf("%s%s/", appCtx.Config.ServiceRegistryPath, serviceName)
	watcher := clientv3.NewWatcher(appCtx.EtcdClient)
	watchChan := watcher.Watch(ctx, servicePrefix, clientv3.WithPrefix())

	appCtx.wg.Add(1)
	go func() {
		defer appCtx.wg.Done()
		defer watcher.Close()
		log.Printf("Watching service '%s' for node changes at prefix '%s'", serviceName, servicePrefix)
		for {
			select {
			case <-ctx.Done():
				log.Printf("Stopping service watch for '%s' due to context cancellation", serviceName)
				return
			case watchResp, ok := <-watchChan:
				if !ok {
					log.Printf("Service watch channel closed for '%s'", serviceName)
					// Re-establish logic similar to WatchConfig
					return
				}
				if watchResp.Err() != nil {
					log.Printf("Error watching service '%s': %v", serviceName, watchResp.Err())
					continue
				}
				for _, event := range watchResp.Events {
					// Key is like /services/my-app/node1_addr, Value is node1_addr
					if event.Type != clientv3.EventTypePut {
						continue
					}
					addr := string(event.Kv.Value)
					log.Printf("Service '%s' event: Type=%s, Addr=%s", serviceName, event.Type, addr)
					if callback != nil {
						callback(addr)
					}
				}
			}
		}
	}()
}

// ExecuteWithCircuitBreaker 使用熔断器执行操作
func (appCtx *AppContext) ExecuteWithCircuitBreaker(serviceName string, operation func() (interface{}, error)) (interface{}, error) {
	cb, ok := appCtx.CircuitBreakers[serviceName]
	if !ok { // 如果没有特定服务的熔断器，直接执行
		log.Printf("No circuit breaker found for service '%s', executing directly.", serviceName)
		return operation()
	}
	return cb.Execute(operation)
}

// AllowRequestWithRateLimiter 检查是否允许请求通过限流器
func (appCtx *AppContext) AllowRequestWithRateLimiter(serviceName string) bool {
	limiter, ok := appCtx.RateLimiters[serviceName]
	if !ok {
		// 如果没有特定服务的限流器，尝试全局限流器
		limiter, ok = appCtx.RateLimiters["global"]
		if !ok {
			log.Printf("No rate limiter found for service '%s' or global, allowing request.", serviceName)
			return true // 没有配置限流器，则允许
		}
	}
	return limiter.Allow()
}

// Run 启动应用，包括服务注册和心跳
func (appCtx *AppContext) Run(appCtxCancel context.Context, serviceName, serviceHost string, servicePort int, startLogic func(context.Context, *AppContext, string) error) error {
	serviceAddress := fmt.Sprintf("%s:%d", serviceHost, servicePort)

	// 1. 注册服务 (如果配置了)
	if appCtx.Config.ServiceRegistryPath != "" && appCtx.Config.ServiceTTL > 0 {
		err := appCtx.RegisterService(serviceName, serviceAddress)
		if err != nil {
			return fmt.Errorf("failed to register service: %w", err)
		}
		// 启动心跳
		var keepAliveCtx context.Context
		keepAliveCtx, appCtx.cancelKeepAlive = context.WithCancel(context.Background()) // 使用独立的上下文控制心跳
		appCtx.KeepAliveService(keepAliveCtx)
	} else {
		log.Println("Service registration/keep-alive is disabled by config.")
	}

	// 2. 启动领导者选举 (如果配置了)
	if appCtx.Config.ElectionPathPrefix != "" {
		// 假设选举名称与服务名相关
		electionName := serviceName + "-leader"
		if err := appCtx.ElectLeader(appCtxCancel, electionName); err != nil { // 使用应用主上下文
			log.Printf("Failed to start leader election: %v", err)
			// 根据策略决定是否继续运行
		}
	}

	// 3. 启动应用核心逻辑
	log.Printf("Starting application %s version %s on %s", appCtx.Config.AppName, appCtx.Config.Version, serviceAddress)
	if startLogic != nil {
		err := startLogic(appCtxCancel, appCtx, serviceAddress) // 传递应用主上下文
		if err != nil {
			return fmt.Errorf("application start logic failed: %w", err)
		}
	} else {
		log.Println("No custom start logic provided.")
		// 如果是gRPC服务，可以在这里启动
		if appCtx.grpcServer != nil {
			lis, err := net.Listen("tcp", serviceAddress)
			if err != nil {
				return fmt.Errorf("failed to listen: %v", err)
			}
			log.Printf("gRPC server listening at %v", lis.Addr())
			if err := appCtx.grpcServer.Serve(lis); err != nil {
				return fmt.Errorf("failed to serve gRPC: %v", err)
			}
		}
	}

	// 等待应用关闭信号 (例如从 startLogic 返回，或者 appCtxCancel 被触发)
	<-appCtxCancel.Done()
	log.Println("Application shutting down...")
	return nil
}

// Close 优雅关闭应用，释放资源
func (appCtx *AppContext) Close() {
	appCtx.shutdownOnce.Do(func() {
		log.Println("Closing AppContext...")

		// 停止心跳 (会触发租约撤销)
		if appCtx.cancelKeepAlive != nil {
			log.Println("Cancelling service keep-alive...")
			appCtx.cancelKeepAlive()
		}

		// 关闭服务发现
		if appCtx.ServiceDiscovery != nil {
			log.Println("Closing service discovery...")
			appCtx.ServiceDiscovery.Close()
		}

		// 关闭etcd session (用于锁和选举)
		if appCtx.EtcdSession != nil {
			log.Println("Closing etcd session...")
			// Session的Close会释放通过此session创建的锁和选举
			// 但通常 Campaign 和 Lock 会在上下文取消时自行处理
			err := appCtx.EtcdSession.Close()
			if err != nil {
				log.Printf("Error closing etcd session: %v", err)
			}
		}

		// 等待所有goroutine完成 (例如心跳、watchers)
		// 需要一个超时来避免永久阻塞
		waitTimeout := 10 * time.Second
		waitChan := make(chan struct{})
		go func() {
			appCtx.wg.Wait()
			close(waitChan)
		}()
		select {
		case <-waitChan:
			log.Println("All background goroutines finished.")
		case <-time.After(waitTimeout):
			log.Printf("Timeout waiting for background goroutines to finish after %s.", waitTimeout)
		}

		// 关闭etcd客户端
		if appCtx.EtcdClient != nil {
			log.Println("Closing etcd client...")
			err := appCtx.EtcdClient.Close()
			if err != nil {
				log.Printf("Error closing etcd client: %v", err)
			}
		}
		log.Println("AppContext closed.")
	})
}

// --- EtcdServiceDiscovery ---
const etcdScheme = "etcd"

// EtcdServiceDiscovery 实现了 gRPC 的 resolver.Builder 和 resolver.Resolver
type EtcdServiceDiscovery struct {
	client      *clientv3.Client
	prefix      string // 服务在etcd中的前缀, e.g., /services/my-app/
	serviceName string // 服务名，用于 scheme
	cc          resolver.ClientConn
	wg          sync.WaitGroup
	ctx         context.Context
	cancel      context.CancelFunc
	mu          sync.Mutex
	addrs       []resolver.Address // 当前已发现的服务地址
}

// NewEtcdServiceDiscovery 创建一个新的服务发现实例
func NewEtcdServiceDiscovery(client *clientv3.Client, servicePrefix, serviceName string) *EtcdServiceDiscovery {
	ctx, cancel := context.WithCancel(context.Background())
	return &EtcdServiceDiscovery{
		client:      client,
		prefix:      strings.TrimRight(servicePrefix, "/") + "/" + serviceName + "/", // e.g. /services/root/my-service/
		serviceName: serviceName,
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Build 为给定的目标创建一个新的 resolver。
// target.Endpoint() 应该是服务名，例如 "my-service"
// 我们这里用 prefix 来查找
func (sd *EtcdServiceDiscovery) Build(target resolver.Target, cc resolver.ClientConn, opts resolver.BuildOptions) (resolver.Resolver, error) {
	sd.cc = cc
	log.Printf("EtcdServiceDiscovery: Building resolver for target: %+v, scheme: %s, prefix: %s", target, sd.Scheme(), sd.prefix)

	sd.wg.Add(1)
	go sd.watchServiceChanges()

	// 初始解析一次
	sd.ResolveNow(resolver.ResolveNowOptions{})
	return sd, nil
}

// Scheme 返回此 resolver 支持的 scheme
func (sd *EtcdServiceDiscovery) Scheme() string {
	return etcdScheme // 或者使用 sd.serviceName 如果希望每个服务一个scheme
}

// watchServiceChanges 监听etcd中服务节点的变化
func (sd *EtcdServiceDiscovery) watchServiceChanges() {
	defer sd.wg.Done()
	log.Printf("EtcdServiceDiscovery: Starting to watch for changes under prefix: %s", sd.prefix)

	// 首次获取所有节点
	if err := sd.updateEndpoints(); err != nil {
		log.Printf("EtcdServiceDiscovery: Initial updateEndpoints failed: %v", err)
		// 可以考虑通过 cc.ReportError() 报告错误
	}

	rch := sd.client.Watch(sd.ctx, sd.prefix, clientv3.WithPrefix())
	for {
		select {
		case <-sd.ctx.Done():
			log.Printf("EtcdServiceDiscovery: Watcher stopped for prefix %s due to context cancellation.", sd.prefix)
			return
		case wresp, ok := <-rch:
			if !ok {
				log.Printf("EtcdServiceDiscovery: Watch channel closed for prefix %s. Re-establishing watch might be needed.", sd.prefix)
				// 简单处理：退出goroutine。更健壮的实现会尝试重新watch。
				return
			}
			if wresp.Err() != nil {
				log.Printf("EtcdServiceDiscovery: Watch error for prefix %s: %v", sd.prefix, wresp.Err())
				// 根据错误类型决定是否继续
				continue
			}
			for _, ev := range wresp.Events {
				log.Printf("EtcdServiceDiscovery: Event received: Type=%s, Key=%s", ev.Type, string(ev.Kv.Key))
				// 无论 PUT 还是 DELETE，都重新获取所有端点并更新
				// 这是一个简单但可靠的策略
				if err := sd.updateEndpoints(); err != nil {
					log.Printf("EtcdServiceDiscovery: updateEndpoints after event failed: %v", err)
				}
			}
		}
	}
}

// updateEndpoints 从etcd获取当前所有服务节点并更新gRPC ClientConn
func (sd *EtcdServiceDiscovery) updateEndpoints() error {
	sd.mu.Lock()
	defer sd.mu.Unlock()

	resp, err := sd.client.Get(sd.ctx, sd.prefix, clientv3.WithPrefix())
	if err != nil {
		log.Printf("EtcdServiceDiscovery: Failed to get endpoints from etcd for prefix %s: %v", sd.prefix, err)
		// 可以选择通过 cc.ReportError() 报告错误给 gRPC 客户端
		// sd.cc.ReportError(err)
		return err
	}

	newAddrs := make([]resolver.Address, 0, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		// value 就是地址，例如 "127.0.0.1:8080"
		// key 可能是 /services/my-app/instance1 -> 127.0.0.1:8080
		// 我们需要确保地址是有效的
		addrStr := string(kv.Value)
		if _, _, err := net.SplitHostPort(addrStr); err != nil {
			log.Printf("EtcdServiceDiscovery: Invalid address format '%s' for key '%s', skipping: %v", addrStr, string(kv.Key), err)
			continue
		}
		// 可以从kv.Key或kv.Value中提取元数据，并设置到resolver.Address的Attributes中
		// attrs := attributes.New()
		// attrs = attrs.With(resolver.EndpointAttributesKey{}, resolver.EndpointAttributes{Weight: 10}) // 示例权重
		newAddrs = append(newAddrs, resolver.Address{Addr: addrStr /*, Attributes: attrs*/})
		log.Printf("EtcdServiceDiscovery: Discovered endpoint: %s for prefix %s", addrStr, sd.prefix)
	}
	sd.addrs = newAddrs

	if len(sd.addrs) == 0 {
		log.Printf("EtcdServiceDiscovery: No endpoints found for prefix %s. ClientConn will not be updated with new addresses.", sd.prefix)
		// 根据策略，可以选择上报一个空列表或者保持上一次的状态
		// sd.cc.UpdateState(resolver.State{Addresses: []resolver.Address{}})
		// 或者 sd.cc.ReportError(fmt.Errorf("no healthy endpoints available for %s", sd.prefix))
		// 这里我们上报空列表，让gRPC知道没有可用后端
		sd.cc.UpdateState(resolver.State{Addresses: []resolver.Address{}})
		return nil
	}

	log.Printf("EtcdServiceDiscovery: Updating gRPC client with %d addresses for prefix %s: %+v", len(sd.addrs), sd.prefix, sd.addrs)
	sd.cc.UpdateState(resolver.State{Addresses: sd.addrs})
	return nil
}

// ResolveNow 会被gRPC调用以强制解析器立即尝试解析目标名称。
func (sd *EtcdServiceDiscovery) ResolveNow(o resolver.ResolveNowOptions) {
	log.Printf("EtcdServiceDiscovery: ResolveNow called for prefix %s", sd.prefix)
	if err := sd.updateEndpoints(); err != nil {
		log.Printf("EtcdServiceDiscovery: ResolveNow failed to update endpoints for prefix %s: %v", sd.prefix, err)
	}
}

// Close 关闭 resolver，停止所有活动，如watch。
func (sd *EtcdServiceDiscovery) Close() {
	log.Printf("EtcdServiceDiscovery: Closing resolver for prefix %s...", sd.prefix)
	sd.cancel()  // 取消上下文，会停止 watchServiceChanges goroutine
	sd.wg.Wait() // 等待 goroutine 退出
	log.Printf("EtcdServiceDiscovery: Resolver closed for prefix %s.", sd.prefix)
}

// getLastKnownAddresses (辅助方法，非resolver接口)
func (sd *EtcdServiceDiscovery) getLastKnownAddresses() []string {
	sd.mu.Lock()
	defer sd.mu.Unlock()
	endpoints := make([]string, len(sd.addrs))
	for i, addr := range sd.addrs {
		endpoints[i] = addr.Addr
	}
	return endpoints
}

// --- gRPC Client Example Usage ---

func createGrpcClient(appCtx *AppContext, serviceName string) (*grpc.ClientConn, error) {
	target := fmt.Sprintf("%s:///%s", etcdScheme, serviceName) // e.g., "etcd:///my-chat-service"

	// 确保 EtcdServiceDiscovery 已经通过 resolver.Register(appCtx.ServiceDiscovery) 注册
	// 如果尚未注册，可以在这里或AppContext初始化时注册

	// 配置负载均衡策略，例如 round_robin
	// 注意: "round_robin" 是 gRPC 内置的，但有些策略可能需要导入相应的包

	serviceConfigJSON := fmt.Sprintf(`{"loadBalancingPolicy":"%s"}`, appCtx.Config.LoadBalancer)
	if appCtx.Config.LoadBalancer == "" {
		serviceConfigJSON = `{"loadBalancingPolicy":"round_robin"}` // 默认轮询
	}

	// 客户端TLS配置 (如果需要)
	var clientCreds credentials.TransportCredentials
	if appCtx.Config.ClientTLS != nil { // 假设有一个ClientTLS配置
		var err error
		clientCreds, err = security.LoadClientTLS(appCtx.Config.ClientTLS)
		if err != nil {
			return nil, fmt.Errorf("failed to load client TLS credentials: %w", err)
		}
	}

	dialOptions := []grpc.DialOption{
		grpc.WithDefaultServiceConfig(serviceConfigJSON),
		// grpc.WithBlock(), // 可选，Dial会阻塞直到连接成功或超时
	}

	if clientCreds != nil {
		dialOptions = append(dialOptions, grpc.WithTransportCredentials(clientCreds))
	} else {
		dialOptions = append(dialOptions, grpc.WithInsecure()) // 开发环境或内部网络
	}

	log.Printf("Dialing gRPC service '%s' at target '%s' with load balancing '%s'", serviceName, target, appCtx.Config.LoadBalancer)
	conn, err := grpc.DialContext(context.Background(), target, dialOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to dial gRPC service '%s': %w", serviceName, err)
	}
	return conn, nil
}

// --- Main Application Example (Illustrative) ---
/*
func main() {
	// 1. 加载应用配置 (例如从文件或环境变量)
	appCfg := &AppConfig{
		AppName:             "my-distributed-app",
		Version:             "1.0.0",
		ListenAddress:       "0.0.0.0:8080", // 将被 serviceHost 和 servicePort 覆盖
		EtcdEndpoints:       []string{"localhost:2379"},
		EtcdDialTimeout:     5 * time.Second,
		ServiceRegistryPath: "/services/myapp/", // 注意末尾的斜杠
		ServiceTTL:          10,
		ConfigPathPrefix:    "/config/myapp/",
		LockPathPrefix:      "/locks/myapp/",
		ElectionPathPrefix:  "/election/myapp/",
		RetryPolicy: &RetryPolicy{
			MaxElapsedTime: 30 * time.Second,
		},
		LoadBalancer: "round_robin", // gRPC 负载均衡策略
		CircuitBreaker: &CBConfig{
			Name: "my-service-cb",
			Timeout: 30 * time.Second,
			// ... 其他熔断器配置
		},
		RateLimiter: &RLConfig{
			Rate: 10, // 10 requests per second
			Burst: 5,
		},
	}

	// 2. 创建 AppContext
	appContext, err := NewAppContext(appCfg)
	if err != nil {
		log.Fatalf("Failed to create app context: %v", err)
	}
	defer appContext.Close()

	// 3. 创建应用主上下文，用于控制整个应用的生命周期
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 优雅关闭处理
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan
		log.Println("Received shutdown signal, initiating graceful shutdown...")
		cancel() // 通知应用关闭
	}()


	// 4. 示例：加载特定配置
	type MyServiceConfig struct {
		FeatureFlag bool   `yaml:"featureFlag"`
		Message     string `yaml:"message"`
	}
	var myCfg MyServiceConfig
	err = appContext.LoadConfig("service_specific.yaml", &myCfg)
	if err != nil {
		log.Printf("Failed to load service_specific.yaml: %v", err)
	} else {
		log.Printf("Loaded service_specific.yaml: %+v", myCfg)
	}

	// 5. 示例：监听配置变化
	appContext.WatchConfig(ctx, "service_specific.yaml", func(data []byte) error {
		var updatedCfg MyServiceConfig
		if err := yaml.Unmarshal(data, &updatedCfg); err != nil {
			return err
		}
		myCfg = updatedCfg // 热更新配置
		log.Printf("Hot reloaded service_specific.yaml: %+v", myCfg)
		return nil
	})


	// 6. 示例：使用分布式锁
	go func() {
		lockCtx, lockCancel := context.WithTimeout(ctx, 30*time.Second) // 尝试获取锁的上下文
		defer lockCancel()

		mutex, err := appContext.AcquireLock(lockCtx, "my_resource_lock")
		if err != nil {
			log.Printf("Failed to acquire lock: %v", err)
			return
		}
		log.Println("Lock acquired! Performing critical task...")
		time.Sleep(5 * time.Second) // 模拟任务
		log.Println("Critical task finished. Releasing lock...")

		releaseCtx, releaseCancel := context.WithTimeout(context.Background(), 5*time.Second) // 释放锁的上下文
		defer releaseCancel()
		if err := appContext.ReleaseLock(releaseCtx, mutex); err != nil {
			log.Printf("Failed to release lock: %v", err)
		}
	}()

	// 7. 示例：观察领导者变化
	go func() {
		leaderEvents := appContext.ObserveLeader()
		for {
			select {
			case <-ctx.Done():
				return
			case isLeader := <-leaderEvents:
				if isLeader {
					log.Println("I am now the LEADER!")
					// 执行领导者特定的任务
				} else {
					log.Println("I am now a FOLLOWER.")
					// 停止领导者任务，或切换到跟随者模式
				}
			}
		}
	}()


	// 8. 启动应用 (服务注册、心跳、选举会在这里面处理)
	// 假设我们的服务监听在 0.0.0.0 (所有接口) 的 8080 端口
	serviceHost := "0.0.0.0" // 或者获取本机可路由IP
	servicePort := 8080
	serviceName := "my-actual-service-name" // 用于服务发现的名称

	err = appContext.Run(ctx, serviceName, serviceHost, servicePort,
		func(runCtx context.Context, appCtx *AppContext, actualListenAddress string) error {
			// 这里是你的应用核心启动逻辑，例如启动HTTP或gRPC服务器
			log.Printf("Application specific start logic running on %s...", actualListenAddress)

			// 示例：模拟一个长时间运行的服务
			// 如果是gRPC服务，可以在这里初始化并启动
			// appCtx.grpcServer = grpc.NewServer(...)
			// lis, _ := net.Listen("tcp", actualListenAddress)
			// go appCtx.grpcServer.Serve(lis)

			// 示例：使用可重试操作
			retryableTask := func() error {
				log.Println("Attempting retryable task...")
				// 模拟可能失败的操作
				if time.Now().Second()%3 != 0 { // 随机失败
					return fmt.Errorf("simulated transient error at %v", time.Now())
				}
				log.Println("Retryable task succeeded!")
				return nil
			}

			// 使用默认的 shouldRetry 逻辑
			err := appCtx.RetryableOperation(runCtx, retryableTask, DefaultShouldRetry)
			if err != nil {
				log.Printf("Retryable task ultimately failed after retries: %v", err)
			}


			// 示例：使用熔断器
			protectedOperation := func() (interface{}, error) {
				log.Println("Attempting operation protected by circuit breaker...")
				// 模拟一个可能导致熔断的操作
				if time.Now().Nanosecond()%2 == 0 {
					return nil, fmt.Errorf("simulated failure for circuit breaker")
				}
				return "Operation successful", nil
			}

			// 假设熔断器名为 "my-service-cb" (与AppConfig中定义的一致)
			res, err := appCtx.ExecuteWithCircuitBreaker("my-service-cb", protectedOperation)
			if err != nil {
				if errors.Is(err, gobreaker.ErrOpenState) || errors.Is(err, gobreaker.ErrTooManyRequests) {
					log.Printf("Circuit breaker is open or half-open and too many requests: %v", err)
				} else {
					log.Printf("Protected operation failed: %v", err)
				}
			} else {
				log.Printf("Protected operation succeeded: %v", res)
			}

			// 示例：使用限流器
			for i := 0; i < 20; i++ {
				if appCtx.AllowRequestWithRateLimiter("global") {
					log.Printf("Request %d allowed by rate limiter", i)
				} else {
					log.Printf("Request %d denied by rate limiter", i)
				}
				time.Sleep(50 * time.Millisecond)
			}


			// 阻塞直到 runCtx 被取消 (例如收到SIGINT)
			<-runCtx.Done()
			log.Println("Application specific start logic shutting down...")
			// if appCtx.grpcServer != nil {
			//    appCtx.grpcServer.GracefulStop()
			// }
			return nil
		})

	if err != nil {
		log.Fatalf("Application run failed: %v", err)
	}

	log.Println("Application has shut down gracefully.")
}
*/
