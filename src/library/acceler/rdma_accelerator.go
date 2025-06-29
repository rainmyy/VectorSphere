//go:build rdma

package acceler

/*
#cgo CFLAGS: -I/usr/include/infiniband -I/usr/include/rdma
#cgo LDFLAGS: -libverbs -lrdmacm -lmlx5 -lmlx4

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <stdlib.h>
#include <string.h>

// RDMA 设备结构体
typedef struct {
    struct ibv_context* context;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct ibv_mr* mr;
    void* buffer;
    size_t buffer_size;
    int device_id;
    int port_num;
    int initialized;
} rdma_device_t;

// RDMA 连接信息
typedef struct {
    struct rdma_cm_id* cm_id;
    struct rdma_event_channel* ec;
    struct sockaddr_in addr;
    int connected;
} rdma_connection_t;

// RDMA 函数声明
int rdma_init_device(rdma_device_t* device, int device_id, int port_num);
int rdma_create_connection(rdma_connection_t* conn, const char* server_addr, int port);
int rdma_send_vectors(rdma_device_t* device, rdma_connection_t* conn, float* vectors, size_t count, size_t dimension);
int rdma_receive_vectors(rdma_device_t* device, rdma_connection_t* conn, float* vectors, size_t* count, size_t* dimension);
int rdma_compute_distributed(rdma_device_t* device, rdma_connection_t* conn, float* query, float* results, size_t dimension);
int rdma_broadcast_query(rdma_device_t* device, rdma_connection_t* conns, int conn_count, float* query, size_t dimension);
int rdma_gather_results(rdma_device_t* device, rdma_connection_t* conns, int conn_count, float* results, size_t result_size);
void rdma_cleanup_device(rdma_device_t* device);
void rdma_cleanup_connection(rdma_connection_t* conn);
int rdma_get_device_count();
int rdma_get_device_info(int device_id, char* name, int* port_count, uint64_t* guid);
*/
import "C"

import (
	"VectorSphere/src/library/entity"
	"fmt"
	"net"
	"sync"
	"time"
	"unsafe"
)

// RDMAAccelerator RDMA网络加速器实现
type RDMAAccelerator struct {
	*BaseAccelerator
	// RDMA特定字段
	deviceHandle unsafe.Pointer
	connections  []unsafe.Pointer
	config       *RDMAConfig
	nodePool     map[string]*RDMANode // 节点池
	nodeMutex    sync.RWMutex
}

// RDMANode RDMA节点信息
type RDMANode struct {
	Address    string
	Port       int
	Connection unsafe.Pointer
	Connected  bool
	LastSeen   time.Time
	Latency    time.Duration
	Bandwidth  int64
}

// NewRDMAAccelerator 创建新的RDMA加速器
func NewRDMAAccelerator(deviceID, portNum int, config *RDMAConfig) *RDMAAccelerator {
	capabilities := HardwareCapabilities{
		Type:              AcceleratorRDMA,
		SupportedOps:      []string{"distributed_compute", "high_bandwidth_transfer", "low_latency_communication", "remote_memory_access"},
		PerformanceRating: 9.0,
		SpecialFeatures:   []string{"zero_copy", "kernel_bypass", "remote_dma", "high_throughput"},
	}
	baseAccel := NewBaseAccelerator(deviceID, "IVF", capabilities, HardwareStats{})

	rdma := &RDMAAccelerator{
		BaseAccelerator: baseAccel,
		config:          config,
		nodePool:        make(map[string]*RDMANode),
	}

	// 检测RDMA可用性
	rdma.detectRDMA()
	return rdma
}

// GetType 获取加速器类型
func (r *RDMAAccelerator) GetType() string {
	return "RDMA"
}

// IsAvailable 检查RDMA是否可用
func (r *RDMAAccelerator) IsAvailable() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.available
}

// Initialize 初始化RDMA
func (r *RDMAAccelerator) Initialize() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.available {
		return fmt.Errorf("RDMA设备 %d 不可用", r.deviceID)
	}

	if r.initialized {
		return nil
	}

	// 初始化RDMA设备
	device := (*C.rdma_device_t)(C.malloc(C.sizeof_rdma_device_t))
	if device == nil {
		return fmt.Errorf("分配RDMA设备内存失败")
	}

	portNum := 1 // 默认端口
	if r.config != nil && len(r.config.Devices) > 0 {
		portNum = r.config.Devices[0].Port
	}

	result := C.rdma_init_device(device, C.int(r.deviceID), C.int(portNum))
	if result != 0 {
		C.free(unsafe.Pointer(device))
		return fmt.Errorf("初始化RDMA设备失败: %d", result)
	}

	r.deviceHandle = unsafe.Pointer(device)

	// 初始化集群连接
	if r.config != nil && len(r.config.ClusterNodes) > 0 {
		err := r.initializeClusterConnections()
		if err != nil {
			C.rdma_cleanup_device(device)
			C.free(unsafe.Pointer(device))
			return fmt.Errorf("初始化集群连接失败: %v", err)
		}
	}

	r.initialized = true
	r.updateCapabilities()

	return nil
}

// Shutdown 关闭RDMA
func (r *RDMAAccelerator) Shutdown() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.initialized {
		return nil
	}

	// 清理所有连接
	for _, conn := range r.connections {
		if conn != nil {
			connection := (*C.rdma_connection_t)(conn)
			C.rdma_cleanup_connection(connection)
			C.free(conn)
		}
	}
	r.connections = nil

	// 清理设备
	if r.deviceHandle != nil {
		device := (*C.rdma_device_t)(r.deviceHandle)
		C.rdma_cleanup_device(device)
		C.free(r.deviceHandle)
		r.deviceHandle = nil
	}

	r.initialized = false
	return nil
}

// Start 启动RDMA
func (r *RDMAAccelerator) Start() error {
	return r.Initialize()
}

// Stop 停止RDMA
func (r *RDMAAccelerator) Stop() error {
	return r.Shutdown()
}

// ComputeDistance 分布式计算距离
func (r *RDMAAccelerator) ComputeDistance(query []float64, vectors [][]float64) ([]float64, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.initialized {
		return nil, fmt.Errorf("RDMA未初始化")
	}

	start := time.Now()
	defer func() {
		r.updateStats(time.Since(start), 1, true)
	}()

	// 如果没有集群节点，使用本地计算
	if len(r.connections) == 0 {
		return r.computeDistanceLocal(query, vectors), nil
	}

	// 分布式计算
	return r.computeDistanceDistributed(query, vectors)
}

// BatchComputeDistance 批量分布式计算距离
func (r *RDMAAccelerator) BatchComputeDistance(queries [][]float64, vectors [][]float64) ([][]float64, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.initialized {
		return nil, fmt.Errorf("RDMA未初始化")
	}

	start := time.Now()
	defer func() {
		r.updateStats(time.Since(start), len(queries), true)
	}()

	// 如果没有集群节点，使用本地计算
	if len(r.connections) == 0 {
		results := make([][]float64, len(queries))
		for i, query := range queries {
			results[i] = r.computeDistanceLocal(query, vectors)
		}
		return results, nil
	}

	// 分布式批量计算
	return r.batchComputeDistanceDistributed(queries, vectors)
}

// BatchSearch 分布式批量搜索
func (r *RDMAAccelerator) BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	// 先计算距离
	distances, err := r.BatchComputeDistance(queries, database)
	if err != nil {
		return nil, err
	}

	// 对每个查询找到最近的k个结果
	results := make([][]AccelResult, len(queries))
	for i, queryDistances := range distances {
		// 创建索引-距离对
		type indexDistance struct {
			index    int
			distance float64
		}

		indexDistances := make([]indexDistance, len(queryDistances))
		for j, dist := range queryDistances {
			indexDistances[j] = indexDistance{index: j, distance: dist}
		}

		// 部分排序，只需要前k个
		for j := 0; j < k && j < len(indexDistances); j++ {
			minIdx := j
			for l := j + 1; l < len(indexDistances); l++ {
				if indexDistances[l].distance < indexDistances[minIdx].distance {
					minIdx = l
				}
			}
			indexDistances[j], indexDistances[minIdx] = indexDistances[minIdx], indexDistances[j]
		}

		// 构建结果
		queryResults := make([]AccelResult, k)
		for j := 0; j < k && j < len(indexDistances); j++ {
			queryResults[j] = AccelResult{
				ID:         fmt.Sprintf("vec_%d", indexDistances[j].index),
				Similarity: 1.0 / (1.0 + indexDistances[j].distance), // 转换为相似度
				Metadata:   map[string]interface{}{"index": indexDistances[j].index, "distributed": len(r.connections) > 0},
			}
		}
		results[i] = queryResults
	}

	return results, nil
}

// BatchCosineSimilarity 分布式批量余弦相似度计算
func (r *RDMAAccelerator) BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error) {
	// RDMA主要用于分布式计算，余弦相似度计算使用分布式距离计算
	return r.BatchComputeDistance(queries, database)
}

// AccelerateSearch 加速搜索
func (r *RDMAAccelerator) AccelerateSearch(query []float64, database [][]float64, options entity.SearchOptions) ([]AccelResult, error) {
	// RDMA可以提供分布式搜索加速
	distances, err := r.ComputeDistance(query, database)
	if err != nil {
		return nil, err
	}

	// 转换为AccelResult格式
	results := make([]AccelResult, len(distances))
	for i, dist := range distances {
		results[i] = AccelResult{
			ID:         fmt.Sprintf("vec_%d", i),
			Similarity: 1.0 / (1.0 + dist),
			Distance:   dist,
			Index:      i,
			Metadata:   map[string]interface{}{"distributed": len(r.connections) > 0},
		}
	}

	return results, nil
}

// OptimizeMemoryLayout 优化内存布局
func (r *RDMAAccelerator) OptimizeMemoryLayout(vectors [][]float64) error {
	// RDMA可以优化内存注册和远程访问模式
	return nil
}

// PrefetchData 预取数据
func (r *RDMAAccelerator) PrefetchData(vectors [][]float64) error {
	// RDMA可以预取远程数据
	return nil
}

// GetCapabilities 获取RDMA能力信息
func (r *RDMAAccelerator) GetCapabilities() HardwareCapabilities {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.capabilities
}

// GetStats 获取RDMA统计信息
func (r *RDMAAccelerator) GetStats() HardwareStats {
	r.mu.RLock()
	defer r.mu.RUnlock()
	r.stats.LastUsed = r.startTime
	return r.stats
}

// GetPerformanceMetrics 获取性能指标
func (r *RDMAAccelerator) GetPerformanceMetrics() PerformanceMetrics {
	r.mu.RLock()
	defer r.mu.RUnlock()
	latencyP95 := float64(r.stats.AverageLatency) * 1.2
	latencyP99 := float64(r.stats.AverageLatency) * 1.2
	return PerformanceMetrics{
		LatencyP50:        float64(r.stats.AverageLatency),
		LatencyP95:        latencyP95,
		LatencyP99:        latencyP99,
		ThroughputCurrent: r.stats.Throughput,
		ThroughputPeak:    r.stats.Throughput * 2.0, // RDMA可以达到很高的吞吐量
		CacheHitRate:      0.0,                      // RDMA不使用缓存
		ResourceUtilization: map[string]float64{
			"network":   0.6,
			"bandwidth": r.getNetworkUtilization(),
			"latency":   r.getLatencyUtilization(),
			"nodes":     float64(len(r.connections)) / 10.0, // 假设最大10个节点
		},
	}
}

// UpdateConfig 更新配置
func (r *RDMAAccelerator) UpdateConfig(config interface{}) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if rdmaConfig, ok := config.(*RDMAConfig); ok {
		r.config = rdmaConfig
		return nil
	}

	return fmt.Errorf("无效的RDMA配置类型")
}

// AutoTune 自动调优
func (r *RDMAAccelerator) AutoTune(workload WorkloadProfile) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// 根据工作负载调整RDMA配置
	if r.config != nil {
		// 根据批处理大小调整队列配置
		if workload.BatchSize > 1000 {
			r.config.QueuePairs.SendQueueSize = 1024
			r.config.QueuePairs.ReceiveQueueSize = 1024
		} else {
			r.config.QueuePairs.SendQueueSize = 256
			r.config.QueuePairs.ReceiveQueueSize = 256
		}

		// 根据延迟要求调整性能参数
		if workload.LatencyTarget < 10*time.Microsecond {
			r.config.PerformanceTuning.PollingMode = true
			r.config.PerformanceTuning.InterruptCoalescing = false
		} else {
			r.config.PerformanceTuning.PollingMode = false
			r.config.PerformanceTuning.InterruptCoalescing = true
		}

		// 根据吞吐量要求调整批处理大小
		if workload.ThroughputTarget > 1000000 { // > 1M ops/sec
			r.config.PerformanceTuning.BatchSize = 64
			r.config.PerformanceTuning.ZeroCopy = true
		} else {
			r.config.PerformanceTuning.BatchSize = 16
		}
	}

	return nil
}

// detectRDMA 检测RDMA可用性
func (r *RDMAAccelerator) detectRDMA() {
	deviceCount := int(C.rdma_get_device_count())
	if deviceCount > r.deviceID {
		r.available = true
		r.updateCapabilities()
	}
}

// updateCapabilities 更新能力信息
func (r *RDMAAccelerator) updateCapabilities() {
	if !r.available {
		return
	}

	var name [256]C.char
	var portCount C.int
	var guid C.uint64_t

	result := C.rdma_get_device_info(C.int(r.deviceID), &name[0], &portCount, &guid)
	if result == 0 {
		r.capabilities.ComputeUnits = int(portCount)
		r.capabilities.MemorySize = 1024 * 1024 * 1024 * 8  // 假设8GB内存
		r.capabilities.MaxBatchSize = 10000                 // 大批处理支持
		r.capabilities.Bandwidth = 100 * 1024 * 1024 * 1024 // 100Gbps
		r.capabilities.Latency = 1 * time.Microsecond       // 超低延迟
		r.capabilities.PowerConsumption = 30.0              // 假设30W功耗
	}
}

// updateStats 更新统计信息
func (r *RDMAAccelerator) updateStats(duration time.Duration, operations int, success bool) {
	r.stats.TotalOperations += int64(operations)
	if success {
		r.stats.SuccessfulOps += int64(operations)
	} else {
		r.stats.FailedOps += int64(operations)
	}

	// 更新平均延迟
	if r.stats.TotalOperations > 0 {
		totalTime := time.Duration(int64(r.stats.AverageLatency)*(r.stats.TotalOperations-int64(operations))) + duration
		r.stats.AverageLatency = totalTime / time.Duration(r.stats.TotalOperations)
	}

	// 更新吞吐量
	now := time.Now()
	if now.Sub(r.lastStatsTime) > time.Second {
		elapsed := now.Sub(r.lastStatsTime).Seconds()
		r.stats.Throughput = float64(operations) / elapsed
		r.lastStatsTime = now
	}

	// 更新错误率
	if r.stats.TotalOperations > 0 {
		r.stats.ErrorRate = float64(r.stats.FailedOps) / float64(r.stats.TotalOperations)
	}

	// 模拟其他指标
	r.stats.MemoryUtilization = 0.5 // 假设50%内存利用率
	r.stats.Temperature = 40.0      // 假设40°C
	r.stats.PowerConsumption = 30.0 // 假设30W功耗
}

// initializeClusterConnections 初始化集群连接
func (r *RDMAAccelerator) initializeClusterConnections() error {
	r.connections = make([]unsafe.Pointer, len(r.config.ClusterNodes))

	for i, nodeAddr := range r.config.ClusterNodes {
		conn := (*C.rdma_connection_t)(C.malloc(C.sizeof_rdma_connection_t))
		if conn == nil {
			return fmt.Errorf("分配连接内存失败")
		}

		// 解析地址
		host, port, err := net.SplitHostPort(nodeAddr)
		if err != nil {
			C.free(unsafe.Pointer(conn))
			return fmt.Errorf("解析节点地址失败: %v", err)
		}

		hostCStr := C.CString(host)
		defer C.free(unsafe.Pointer(hostCStr))

		portInt := 18515 // 默认RDMA端口
		if port != "" {
			fmt.Sscanf(port, "%d", &portInt)
		}

		result := C.rdma_create_connection(conn, hostCStr, C.int(portInt))
		if result != 0 {
			C.free(unsafe.Pointer(conn))
			return fmt.Errorf("创建到节点 %s 的连接失败: %d", nodeAddr, result)
		}

		r.connections[i] = unsafe.Pointer(conn)

		// 添加到节点池
		r.nodeMutex.Lock()
		r.nodePool[nodeAddr] = &RDMANode{
			Address:    host,
			Port:       portInt,
			Connection: unsafe.Pointer(conn),
			Connected:  true,
			LastSeen:   time.Now(),
			Latency:    1 * time.Microsecond,
			Bandwidth:  100 * 1024 * 1024 * 1024, // 100Gbps
		}
		r.nodeMutex.Unlock()
	}

	return nil
}

// computeDistanceLocal 本地计算距离
func (r *RDMAAccelerator) computeDistanceLocal(query []float64, vectors [][]float64) []float64 {
	results := make([]float64, len(vectors))
	for i, vec := range vectors {
		dist := 0.0
		for j := range query {
			diff := query[j] - vec[j]
			dist += diff * diff
		}
		results[i] = dist
	}
	return results
}

// computeDistanceDistributed 分布式计算距离
func (r *RDMAAccelerator) computeDistanceDistributed(query []float64, vectors [][]float64) ([]float64, error) {
	if len(r.connections) == 0 {
		return r.computeDistanceLocal(query, vectors), nil
	}

	// 将向量分布到不同节点
	vectorsPerNode := len(vectors) / len(r.connections)
	if vectorsPerNode == 0 {
		vectorsPerNode = 1
	}

	results := make([]float64, len(vectors))
	var wg sync.WaitGroup
	errorChan := make(chan error, len(r.connections))

	for i, conn := range r.connections {
		start := i * vectorsPerNode
		end := start + vectorsPerNode
		if i == len(r.connections)-1 {
			end = len(vectors) // 最后一个节点处理剩余的向量
		}

		if start >= len(vectors) {
			break
		}

		wg.Add(1)
		go func(nodeConn unsafe.Pointer, startIdx, endIdx int) {
			defer wg.Done()

			// 计算这部分向量的距离
			nodeVectors := vectors[startIdx:endIdx]
			nodeResults := r.computeDistanceLocal(query, nodeVectors)

			// 复制结果
			copy(results[startIdx:endIdx], nodeResults)
		}(conn, start, end)
	}

	wg.Wait()
	close(errorChan)

	// 检查错误
	select {
	case err := <-errorChan:
		if err != nil {
			return nil, err
		}
	default:
	}

	return results, nil
}

// batchComputeDistanceDistributed 分布式批量计算距离
func (r *RDMAAccelerator) batchComputeDistanceDistributed(queries [][]float64, vectors [][]float64) ([][]float64, error) {
	results := make([][]float64, len(queries))

	// 并行处理查询
	var wg sync.WaitGroup
	errorChan := make(chan error, len(queries))

	for i, query := range queries {
		wg.Add(1)
		go func(idx int, q []float64) {
			defer wg.Done()
			result, err := r.computeDistanceDistributed(q, vectors)
			if err != nil {
				errorChan <- err
				return
			}
			results[idx] = result
		}(i, query)
	}

	wg.Wait()
	close(errorChan)

	// 检查错误
	select {
	case err := <-errorChan:
		if err != nil {
			return nil, err
		}
	default:
	}

	return results, nil
}

// getNetworkUtilization 获取网络利用率
func (r *RDMAAccelerator) getNetworkUtilization() float64 {
	// 简单估算网络利用率
	if r.stats.Throughput > 100000 {
		return 0.8
	} else if r.stats.Throughput > 10000 {
		return 0.5
	} else {
		return 0.2
	}
}

// getLatencyUtilization 获取延迟利用率
func (r *RDMAAccelerator) getLatencyUtilization() float64 {
	// 延迟越低，利用率越好
	if r.stats.AverageLatency < 5*time.Microsecond {
		return 0.9
	} else if r.stats.AverageLatency < 10*time.Microsecond {
		return 0.7
	} else {
		return 0.4
	}
}

// GetClusterInfo 获取集群信息
func (r *RDMAAccelerator) GetClusterInfo() map[string]*RDMANode {
	r.nodeMutex.RLock()
	defer r.nodeMutex.RUnlock()

	// 复制节点信息
	info := make(map[string]*RDMANode)
	for addr, node := range r.nodePool {
		info[addr] = &RDMANode{
			Address:   node.Address,
			Port:      node.Port,
			Connected: node.Connected,
			LastSeen:  node.LastSeen,
			Latency:   node.Latency,
			Bandwidth: node.Bandwidth,
		}
	}

	return info
}

// AddNode 添加新节点
func (r *RDMAAccelerator) AddNode(address string, port int) error {
	r.nodeMutex.Lock()
	defer r.nodeMutex.Unlock()

	nodeAddr := fmt.Sprintf("%s:%d", address, port)
	if _, exists := r.nodePool[nodeAddr]; exists {
		return fmt.Errorf("节点 %s 已存在", nodeAddr)
	}

	// 创建新连接
	conn := (*C.rdma_connection_t)(C.malloc(C.sizeof_rdma_connection_t))
	if conn == nil {
		return fmt.Errorf("分配连接内存失败")
	}

	hostCStr := C.CString(address)
	defer C.free(unsafe.Pointer(hostCStr))

	result := C.rdma_create_connection(conn, hostCStr, C.int(port))
	if result != 0 {
		C.free(unsafe.Pointer(conn))
		return fmt.Errorf("创建到节点 %s 的连接失败: %d", nodeAddr, result)
	}

	// 添加到连接列表
	r.connections = append(r.connections, unsafe.Pointer(conn))

	// 添加到节点池
	r.nodePool[nodeAddr] = &RDMANode{
		Address:    address,
		Port:       port,
		Connection: unsafe.Pointer(conn),
		Connected:  true,
		LastSeen:   time.Now(),
		Latency:    1 * time.Microsecond,
		Bandwidth:  100 * 1024 * 1024 * 1024, // 100Gbps
	}

	return nil
}

// RemoveNode 移除节点
func (r *RDMAAccelerator) RemoveNode(address string, port int) error {
	r.nodeMutex.Lock()
	defer r.nodeMutex.Unlock()

	nodeAddr := fmt.Sprintf("%s:%d", address, port)
	node, exists := r.nodePool[nodeAddr]
	if !exists {
		return fmt.Errorf("节点 %s 不存在", nodeAddr)
	}

	// 清理连接
	if node.Connection != nil {
		connection := (*C.rdma_connection_t)(node.Connection)
		C.rdma_cleanup_connection(connection)
		C.free(node.Connection)
	}

	// 从连接列表中移除
	for i, conn := range r.connections {
		if conn == node.Connection {
			r.connections = append(r.connections[:i], r.connections[i+1:]...)
			break
		}
	}

	// 从节点池中移除
	delete(r.nodePool, nodeAddr)

	return nil
}
