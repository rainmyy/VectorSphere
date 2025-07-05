package distributed

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"context"
	"encoding/json"
	"fmt"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

const (
	// ServiceRootPath etcd路径前缀
	ServiceRootPath = "/vector_sphere/services"
	MasterPath      = "/vector_sphere/master"
	SlavePath       = "/vector_sphere/slaves"
	ElectionPath    = "/vector_sphere/election"
)

// ServiceInfo 服务信息
type ServiceInfo struct {
	ServiceName string            `json:"service_name"`
	NodeID      string            `json:"node_id"`
	Address     string            `json:"address"`
	Port        int               `json:"port"`
	NodeType    string            `json:"node_type"` // "master" or "slave"
	Status      string            `json:"status"`    // "active", "inactive"
	Metadata    map[string]string `json:"metadata"`
	LastSeen    time.Time         `json:"last_seen"`
	Version     string            `json:"version"`
}

// ServiceDiscovery 服务发现组件
type ServiceDiscovery struct {
	client      *clientv3.Client
	serviceName string
	leaseID     clientv3.LeaseID
	mutex       sync.RWMutex

	// 服务缓存
	masterInfo *ServiceInfo
	slaveInfos map[string]*ServiceInfo       // nodeID -> ServiceInfo
	watchers   map[string]context.CancelFunc // 监听器取消函数

	// 回调函数
	onMasterChange func(*ServiceInfo)
	onSlaveChange  func(map[string]*ServiceInfo)

	stopCh chan struct{}
}

// NewServiceDiscovery 创建服务发现组件
func NewServiceDiscovery(client *clientv3.Client, serviceName string) *ServiceDiscovery {
	return &ServiceDiscovery{
		client:      client,
		serviceName: serviceName,
		slaveInfos:  make(map[string]*ServiceInfo),
		watchers:    make(map[string]context.CancelFunc),
		stopCh:      make(chan struct{}),
	}
}

// RegisterService 注册服务
func (sd *ServiceDiscovery) RegisterService(ctx context.Context, info *ServiceInfo, ttl int64) error {
	logger.Info("Registering service: %s, type: %s, address: %s", info.ServiceName, info.NodeType, info.Address)

	// 构建key
	var key string
	if info.NodeType == "master" {
		key = path.Join(MasterPath, info.ServiceName)
	} else {
		key = path.Join(SlavePath, info.ServiceName, info.NodeID)
	}

	// 检查服务是否已存在
	resp, err := sd.client.Get(ctx, key)
	if err != nil {
		return fmt.Errorf("检查服务是否存在失败: %v", err)
	}

	// 如果服务已存在，先删除
	if len(resp.Kvs) > 0 {
		logger.Info("Service already exists, deleting before re-registration: %s", key)
		_, err = sd.client.Delete(ctx, key)
		if err != nil {
			return fmt.Errorf("删除已存在服务失败: %v", err)
		}
		
		// 如果有旧的租约，撤销它
		if sd.leaseID != 0 {
			sd.client.Revoke(ctx, sd.leaseID)
			sd.leaseID = 0
		}
		logger.Info("Existing service deleted successfully: %s", key)
	}

	// 创建新租约
	leaseResp, err := sd.client.Grant(ctx, ttl)
	if err != nil {
		return fmt.Errorf("创建租约失败: %v", err)
	}
	sd.leaseID = leaseResp.ID

	// 序列化服务信息
	infoBytes, err := json.Marshal(info)
	if err != nil {
		return fmt.Errorf("序列化服务信息失败: %v", err)
	}

	// 注册服务
	_, err = sd.client.Put(ctx, key, string(infoBytes), clientv3.WithLease(sd.leaseID))
	if err != nil {
		return fmt.Errorf("注册服务失败: %v", err)
	}

	// 启动租约续期
	go sd.keepAlive(ctx)

	logger.Info("Service registered successfully: %s", key)
	return nil
}

// UnregisterService 注销服务
func (sd *ServiceDiscovery) UnregisterService(ctx context.Context, nodeType, nodeID string) error {
	var key string
	if nodeType == "master" {
		key = path.Join(MasterPath, sd.serviceName)
	} else {
		key = path.Join(SlavePath, sd.serviceName, nodeID)
	}

	_, err := sd.client.Delete(ctx, key)
	if err != nil {
		return fmt.Errorf("注销服务失败: %v", err)
	}

	// 撤销租约
	if sd.leaseID != 0 {
		sd.client.Revoke(ctx, sd.leaseID)
	}

	logger.Info("Service unregistered: %s", key)
	return nil
}

// keepAlive 保持租约活跃
func (sd *ServiceDiscovery) keepAlive(ctx context.Context) {
	ch, kaerr := sd.client.KeepAlive(ctx, sd.leaseID)
	if kaerr != nil {
		logger.Error("KeepAlive failed: %v", kaerr)
		return
	}

	for {
		select {
		case ka := <-ch:
			if ka == nil {
				logger.Warning("KeepAlive channel closed")
				return
			}
			// log.Debug("KeepAlive response: %v", ka)
		case <-sd.stopCh:
			logger.Info("KeepAlive stopped")
			return
		case <-ctx.Done():
			logger.Info("KeepAlive context cancelled")
			return
		}
	}
}

// DiscoverMaster 发现master节点
func (sd *ServiceDiscovery) DiscoverMaster(ctx context.Context) (*ServiceInfo, error) {
	key := path.Join(MasterPath, sd.serviceName)
	resp, err := sd.client.Get(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("查询master失败: %v", err)
	}

	if len(resp.Kvs) == 0 {
		return nil, fmt.Errorf("未找到master节点")
	}

	var info ServiceInfo
	if err := json.Unmarshal(resp.Kvs[0].Value, &info); err != nil {
		return nil, fmt.Errorf("解析master信息失败: %v", err)
	}

	return &info, nil
}

// DiscoverSlaves 发现所有slave节点
func (sd *ServiceDiscovery) DiscoverSlaves(ctx context.Context) (map[string]*ServiceInfo, error) {
	key := path.Join(SlavePath, sd.serviceName) + "/"
	resp, err := sd.client.Get(ctx, key, clientv3.WithPrefix())
	if err != nil {
		return nil, fmt.Errorf("查询slaves失败: %v", err)
	}

	slaves := make(map[string]*ServiceInfo)
	for _, kv := range resp.Kvs {
		var info ServiceInfo
		if err := json.Unmarshal(kv.Value, &info); err != nil {
			logger.Warning("解析slave信息失败: %v", err)
			continue
		}
		slaves[info.NodeID] = &info
	}

	return slaves, nil
}

// WatchMaster 监听master变化
func (sd *ServiceDiscovery) WatchMaster(ctx context.Context, callback func(*ServiceInfo)) {
	sd.onMasterChange = callback

	key := path.Join(MasterPath, sd.serviceName)
	watchCtx, cancel := context.WithCancel(ctx)
	sd.watchers["master"] = cancel

	go func() {
		defer cancel()

		// 首先检查当前是否已有master节点
		sd.checkInitialMasterState(ctx, key, callback)

		// 启动持续监听
		sd.startMasterWatch(watchCtx, key, callback)
	}()
}

// checkInitialMasterState 检查初始master状态
func (sd *ServiceDiscovery) checkInitialMasterState(ctx context.Context, key string, callback func(*ServiceInfo)) {
	resp, err := sd.client.Get(ctx, key)
	if err != nil {
		logger.Error("Failed to check initial master state: %v", err)
		return
	}

	if len(resp.Kvs) > 0 {
		var info ServiceInfo
		if err := json.Unmarshal(resp.Kvs[0].Value, &info); err != nil {
			logger.Error("Failed to parse initial master info: %v", err)
			return
		}

		sd.mutex.Lock()
		sd.masterInfo = &info
		sd.mutex.Unlock()

		logger.Info("Initial master found: %s:%d", info.Address, info.Port)
		if callback != nil {
			callback(&info)
		}
	} else {
		logger.Info("No initial master found")
		if callback != nil {
			callback(nil)
		}
	}
}

// startMasterWatch 启动master监听
func (sd *ServiceDiscovery) startMasterWatch(ctx context.Context, key string, callback func(*ServiceInfo)) {
	for {
		select {
		case <-sd.stopCh:
			return
		case <-ctx.Done():
			return
		default:
			// 创建watch
			watchCh := sd.client.Watch(ctx, key)

			sd.processMasterWatchEvents(ctx, watchCh, callback)

			// 如果watch断开，等待一段时间后重试
			select {
			case <-time.After(2 * time.Second):
				logger.Info("Reconnecting master watch...")
			case <-sd.stopCh:
				return
			case <-ctx.Done():
				return
			}
		}
	}
}

// processMasterWatchEvents 处理master监听事件
func (sd *ServiceDiscovery) processMasterWatchEvents(ctx context.Context, watchCh clientv3.WatchChan, callback func(*ServiceInfo)) {
	for {
		select {
		case watchResp := <-watchCh:
			if watchResp.Err() != nil {
				logger.Error("Watch master error: %v", watchResp.Err())
				return
			}

			for _, event := range watchResp.Events {
				switch event.Type {
				case clientv3.EventTypePut:
					var info ServiceInfo
					if err := json.Unmarshal(event.Kv.Value, &info); err != nil {
						logger.Error("Failed to parse master info: %v", err)
						continue
					}
					sd.mutex.Lock()
					sd.masterInfo = &info
					sd.mutex.Unlock()
					logger.Info("Master changed: %s:%d", info.Address, info.Port)
					if callback != nil {
						callback(&info)
					}
				case clientv3.EventTypeDelete:
					sd.mutex.Lock()
					sd.masterInfo = nil
					sd.mutex.Unlock()
					logger.Info("Master removed")
					if callback != nil {
						callback(nil)
					}
				}
			}
		case <-sd.stopCh:
			return
		case <-ctx.Done():
			return
		}
	}
}

// WatchSlaves 监听slaves变化
func (sd *ServiceDiscovery) WatchSlaves(ctx context.Context, callback func(map[string]*ServiceInfo)) {
	sd.onSlaveChange = callback

	key := path.Join(SlavePath, sd.serviceName) + "/"
	watchCtx, cancel := context.WithCancel(ctx)
	sd.watchers["slaves"] = cancel

	go func() {
		defer cancel()
		watchCh := sd.client.Watch(watchCtx, key, clientv3.WithPrefix())

		for {
			select {
			case watchResp := <-watchCh:
				if watchResp.Err() != nil {
					logger.Error("Watch slaves error: %v", watchResp.Err())
					return
				}

				sd.mutex.Lock()
				for _, event := range watchResp.Events {
					switch event.Type {
					case clientv3.EventTypePut:
						var info ServiceInfo
						if err := json.Unmarshal(event.Kv.Value, &info); err != nil {
							logger.Error("解析slave信息失败: %v", err)
							continue
						}
						sd.slaveInfos[info.NodeID] = &info
						logger.Info("Slave added/updated: %s", info.Address)
					case clientv3.EventTypeDelete:
						// 从key中提取nodeID
						keyStr := string(event.Kv.Key)
						parts := strings.Split(keyStr, "/")
						if len(parts) > 0 {
							nodeID := parts[len(parts)-1]
							delete(sd.slaveInfos, nodeID)
							logger.Info("Slave removed: %s", nodeID)
						}
					}
				}
				slavesCopy := make(map[string]*ServiceInfo)
				for k, v := range sd.slaveInfos {
					slavesCopy[k] = v
				}
				sd.mutex.Unlock()

				if callback != nil {
					callback(slavesCopy)
				}
			case <-sd.stopCh:
				return
			case <-watchCtx.Done():
				return
			}
		}
	}()
}

// GetMaster 获取当前master信息
func (sd *ServiceDiscovery) GetMaster() *ServiceInfo {
	sd.mutex.RLock()
	defer sd.mutex.RUnlock()
	return sd.masterInfo
}

// GetSlaves 获取当前所有slave信息
func (sd *ServiceDiscovery) GetSlaves() map[string]*ServiceInfo {
	sd.mutex.RLock()
	defer sd.mutex.RUnlock()

	slaves := make(map[string]*ServiceInfo)
	for k, v := range sd.slaveInfos {
		slaves[k] = v
	}
	return slaves
}

// GetSlaveAddresses 获取所有slave地址列表
func (sd *ServiceDiscovery) GetSlaveAddresses() []string {
	sd.mutex.RLock()
	defer sd.mutex.RUnlock()

	var addresses []string
	for _, info := range sd.slaveInfos {
		addresses = append(addresses, info.Address+":"+strconv.Itoa(info.Port))
	}
	return addresses
}

// GetHealthySlaveAddresses 获取健康的slave地址列表
func (sd *ServiceDiscovery) GetHealthySlaveAddresses() []string {
	sd.mutex.RLock()
	defer sd.mutex.RUnlock()

	var addresses []string
	for _, info := range sd.slaveInfos {
		if info.Status == "active" {
			addresses = append(addresses, info.Address+":"+strconv.Itoa(info.Port))
		}
	}
	if len(addresses) == 0 {
		addresses = append(addresses, "127.0.0.1:2379")
	}
	return addresses
}

// GetSlaveInfo 获取指定slave节点的信息
func (sd *ServiceDiscovery) GetSlaveInfo(nodeID string) (*ServiceInfo, error) {
	sd.mutex.RLock()
	defer sd.mutex.RUnlock()

	info, exists := sd.slaveInfos[nodeID]
	if !exists {
		return nil, fmt.Errorf("slave节点 %s 不存在", nodeID)
	}

	return info, nil
}

// UpdateServiceStatus 更新服务状态
func (sd *ServiceDiscovery) UpdateServiceStatus(ctx context.Context, nodeType, nodeID, status string) error {
	var key string
	if nodeType == "master" {
		key = path.Join(MasterPath, sd.serviceName)
	} else {
		key = path.Join(SlavePath, sd.serviceName, nodeID)
	}

	// 获取当前信息
	resp, err := sd.client.Get(ctx, key)
	if err != nil {
		return fmt.Errorf("获取服务信息失败: %v", err)
	}

	if len(resp.Kvs) == 0 {
		return fmt.Errorf("服务不存在")
	}

	var info ServiceInfo
	if err := json.Unmarshal(resp.Kvs[0].Value, &info); err != nil {
		return fmt.Errorf("解析服务信息失败: %v", err)
	}

	// 更新状态和时间
	info.Status = status
	info.LastSeen = time.Now()

	// 序列化并更新
	infoBytes, err := json.Marshal(info)
	if err != nil {
		return fmt.Errorf("序列化服务信息失败: %v", err)
	}

	_, err = sd.client.Put(ctx, key, string(infoBytes), clientv3.WithLease(sd.leaseID))
	if err != nil {
		return fmt.Errorf("更新服务状态失败: %v", err)
	}

	return nil
}

// Stop 停止服务发现
func (sd *ServiceDiscovery) Stop() {
	close(sd.stopCh)

	// 取消所有监听器
	for name, cancel := range sd.watchers {
		cancel()
		logger.Info("Stopped watcher: %s", name)
	}

	// 撤销租约
	if sd.leaseID != 0 {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		sd.client.Revoke(ctx, sd.leaseID)
	}

	logger.Info("Service discovery stopped")
}

// CreateServiceInfo 创建服务信息
func CreateServiceInfo(serviceName, nodeID, address string, port int, nodeType string) *ServiceInfo {
	return &ServiceInfo{
		ServiceName: serviceName,
		NodeID:      nodeID,
		Address:     address,
		Port:        port,
		NodeType:    nodeType,
		Status:      "active",
		Metadata:    make(map[string]string),
		LastSeen:    time.Now(),
		Version:     "1.0.0",
	}
}

// ConvertToEndPoints 转换为EndPoint列表
func (sd *ServiceDiscovery) ConvertToEndPoints() []entity.EndPoint {
	sd.mutex.RLock()
	defer sd.mutex.RUnlock()

	var endpoints []entity.EndPoint
	for _, info := range sd.slaveInfos {
		if info.Status == "active" {
			endpoints = append(endpoints, entity.EndPoint{
				Ip:   info.Address,
				Port: info.Port,
			})
		}
	}
	return endpoints
}
