package distributed

import (
	"VectorSphere/src/library/logger"
	"VectorSphere/src/proto/serverProto"
	"context"
	"fmt"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

// CommunicationService 通信服务
type CommunicationService struct {
	mutex       sync.RWMutex
	connections map[string]*grpc.ClientConn // slave地址 -> 连接
	clients     map[string]interface{}      // slave地址 -> 客户端
	timeout     time.Duration
	etcdClient  *clientv3.Client
	serviceName string
}

const (
	defaultClient = iota
	shardStoreClient
)

// NewCommunicationService 创建通信服务
func NewCommunicationService(etcdClient *clientv3.Client, serviceName string) *CommunicationService {
	return &CommunicationService{
		connections: make(map[string]*grpc.ClientConn),
		clients:     make(map[string]interface{}),
		timeout:     30 * time.Second, // 默认超时时间
		etcdClient:  etcdClient,
		serviceName: serviceName,
	}
}

// GetSlaveClient 获取slave客户端连接
func (cs *CommunicationService) GetSlaveClient(slaveAddr string) (serverProto.IndexServiceClient, error) {
	cs.mutex.RLock()
	client, exists := cs.clients[slaveAddr]
	cs.mutex.RUnlock()

	if exists {
		return client.(serverProto.IndexServiceClient), nil
	}

	indexClient, err := cs.createSlaveConnection(slaveAddr, defaultClient)
	if err != nil {
		return nil, err
	}

	// 创建新连接
	return indexClient.(serverProto.IndexServiceClient), err
}

func (cs *CommunicationService) GetSShardStoreSlaveClient(slaveAddr string) (serverProto.DistributedStorageServiceClient, error) {
	cs.mutex.RLock()
	client, exists := cs.clients[slaveAddr]
	cs.mutex.RUnlock()

	if exists {
		return client.(serverProto.DistributedStorageServiceClient), nil
	}

	shardIndexClient, err := cs.createSlaveConnection(slaveAddr, shardStoreClient)

	// 创建新连接
	return shardIndexClient.(serverProto.DistributedStorageServiceClient), err
}

// createSlaveConnection 创建slave连接
func (cs *CommunicationService) createSlaveConnection(slaveAddr string, clientType int) (interface{}, error) {
	cs.mutex.Lock()
	defer cs.mutex.Unlock()

	// 双重检查
	if client, exists := cs.clients[slaveAddr]; exists {
		return client, nil
	}

	logger.Info("Creating connection to slave: %s", slaveAddr)

	// 创建gRPC连接
	conn, err := grpc.Dial(slaveAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                10 * time.Second,
			Timeout:             3 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(64*1024*1024), // 64MB
			grpc.MaxCallSendMsgSize(64*1024*1024), // 64MB
		),
	)
	if err != nil {
		return nil, fmt.Errorf("连接slave失败 %s: %v", slaveAddr, err)
	}

	// 创建客户端
	var client interface{}
	if clientType == shardStoreClient {
		client = serverProto.NewDistributedStorageServiceClient(conn)
	} else {
		client = serverProto.NewIndexServiceClient(conn)
	}
	// 存储连接和客户端
	cs.connections[slaveAddr] = conn
	cs.clients[slaveAddr] = client

	logger.Info("Successfully connected to slave: %s", slaveAddr)
	return client, nil
}

// RemoveSlaveConnection 移除slave连接
func (cs *CommunicationService) RemoveSlaveConnection(slaveAddr string) {
	cs.mutex.Lock()
	defer cs.mutex.Unlock()

	if conn, exists := cs.connections[slaveAddr]; exists {
		err := conn.Close()
		if err != nil {
			logger.Error("close conn failed:%v", err)
		}
		delete(cs.connections, slaveAddr)
		delete(cs.clients, slaveAddr)
		logger.Info("Removed connection to slave: %s", slaveAddr)
	}
}

// BroadcastToSlaves 向所有slave广播请求
func (cs *CommunicationService) BroadcastToSlaves(ctx context.Context, slaveAddrs []string, requestFunc func(client serverProto.IndexServiceClient) error) map[string]error {
	results := make(map[string]error)
	var wg sync.WaitGroup
	var resultMutex sync.Mutex

	for _, addr := range slaveAddrs {
		wg.Add(1)
		go func(slaveAddr string) {
			defer wg.Done()

			client, err := cs.GetSlaveClient(slaveAddr)
			if err != nil {
				resultMutex.Lock()
				results[slaveAddr] = fmt.Errorf("获取客户端失败: %v", err)
				resultMutex.Unlock()
				return
			}

			// 创建带超时的上下文
			_, cancel := context.WithTimeout(ctx, cs.timeout)
			defer cancel()

			// 执行请求
			err = requestFunc(client)

			resultMutex.Lock()
			results[slaveAddr] = err
			resultMutex.Unlock()

			// 如果请求失败，移除连接以便下次重新建立
			if err != nil {
				logger.Warning("Request to slave %s failed: %v", slaveAddr, err)
				cs.RemoveSlaveConnection(slaveAddr)
			}
		}(addr)
	}

	wg.Wait()
	return results
}

// SendToSlave 向指定slave发送请求
func (cs *CommunicationService) SendToSlave(ctx context.Context, slaveAddr string, requestFunc func(client serverProto.IndexServiceClient) error) error {
	client, err := cs.GetSlaveClient(slaveAddr)
	if err != nil {
		return fmt.Errorf("获取slave客户端失败: %v", err)
	}

	// 创建带超时的上下文
	_, cancel := context.WithTimeout(ctx, cs.timeout)
	defer cancel()

	// 执行请求
	err = requestFunc(client)
	if err != nil {
		logger.Warning("Request to slave %s failed: %v", slaveAddr, err)
		cs.RemoveSlaveConnection(slaveAddr)
		return err
	}

	return nil
}

// CreateTableOnSlaves 在所有slave上创建表
func (cs *CommunicationService) CreateTableOnSlaves(ctx context.Context, slaveAddrs []string, request *serverProto.CreateTableRequest) map[string]error {
	return cs.BroadcastToSlaves(ctx, slaveAddrs, func(client serverProto.IndexServiceClient) error {
		_, err := client.CreateTable(ctx, request)
		return err
	})
}

// DeleteTableOnSlaves 在所有slave上删除表
func (cs *CommunicationService) DeleteTableOnSlaves(ctx context.Context, slaveAddrs []string, request *serverProto.TableRequest) map[string]error {
	return cs.BroadcastToSlaves(ctx, slaveAddrs, func(client serverProto.IndexServiceClient) error {
		_, err := client.DeleteTable(ctx, request)
		return err
	})
}

// AddDocumentToSlaves 向所有slave添加文档
func (cs *CommunicationService) AddDocumentToSlaves(ctx context.Context, slaveAddrs []string, request *serverProto.AddDocumentRequest) map[string]error {
	return cs.BroadcastToSlaves(ctx, slaveAddrs, func(client serverProto.IndexServiceClient) error {
		_, err := client.AddDocumentToTable(ctx, request)
		return err
	})
}

// DeleteDocumentFromSlaves 从所有slave删除文档
func (cs *CommunicationService) DeleteDocumentFromSlaves(ctx context.Context, slaveAddrs []string, request *serverProto.DeleteDocumentRequest) map[string]error {
	return cs.BroadcastToSlaves(ctx, slaveAddrs, func(client serverProto.IndexServiceClient) error {
		_, err := client.DeleteDocumentFromTable(ctx, request)
		return err
	})
}

// SearchOnSlaves 在所有slave上搜索
func (cs *CommunicationService) SearchOnSlaves(ctx context.Context, slaveAddrs []string, request *serverProto.SearchRequest) map[string]*serverProto.SearchResult {
	results := make(map[string]*serverProto.SearchResult)
	var wg sync.WaitGroup
	var resultMutex sync.Mutex

	for _, addr := range slaveAddrs {
		wg.Add(1)
		go func(slaveAddr string) {
			defer wg.Done()

			client, err := cs.GetSlaveClient(slaveAddr)
			if err != nil {
				logger.Error("获取slave客户端失败 %s: %v", slaveAddr, err)
				return
			}

			// 创建带超时的上下文
			requestCtx, cancel := context.WithTimeout(ctx, cs.timeout)
			defer cancel()

			// 执行搜索
			result, err := client.SearchTable(requestCtx, request)
			if err != nil {
				logger.Error("Search on slave %s failed: %v", slaveAddr, err)
				cs.RemoveSlaveConnection(slaveAddr)
				return
			}

			resultMutex.Lock()
			results[slaveAddr] = result
			resultMutex.Unlock()
		}(addr)
	}

	wg.Wait()
	return results
}

// HealthCheckSlaves 检查所有slave的健康状态
func (cs *CommunicationService) HealthCheckSlaves(ctx context.Context, slaveAddrs []string) map[string]bool {
	results := make(map[string]bool)
	var wg sync.WaitGroup
	var resultMutex sync.Mutex

	for _, addr := range slaveAddrs {
		wg.Add(1)
		go func(slaveAddr string) {
			defer wg.Done()

			client, err := cs.GetSlaveClient(slaveAddr)
			if err != nil {
				resultMutex.Lock()
				results[slaveAddr] = false
				resultMutex.Unlock()
				return
			}

			// 创建带超时的上下文
			requestCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()

			// 执行健康检查
			_, err = client.HealthCheck(requestCtx, &serverProto.HealthCheckRequest{})

			resultMutex.Lock()
			results[slaveAddr] = err == nil
			resultMutex.Unlock()

			if err != nil {
				cs.RemoveSlaveConnection(slaveAddr)
			}
		}(addr)
	}

	wg.Wait()
	return results
}

// Close 关闭所有连接
func (cs *CommunicationService) Close() {
	cs.mutex.Lock()
	defer cs.mutex.Unlock()

	for addr, conn := range cs.connections {
		conn.Close()
		logger.Info("Closed connection to slave: %s", addr)
	}

	cs.connections = make(map[string]*grpc.ClientConn)
	cs.clients = make(map[string]interface{})
}

// GetActiveConnections 获取活跃连接数
func (cs *CommunicationService) GetActiveConnections() []string {
	cs.mutex.RLock()
	defer cs.mutex.RUnlock()

	var addrs []string
	for addr := range cs.connections {
		addrs = append(addrs, addr)
	}
	return addrs
}
