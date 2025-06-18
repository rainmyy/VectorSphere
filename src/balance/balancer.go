package balance

import (
	"VectorSphere/src/library/entity"
	"fmt"
	"hash/crc32"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const (
	Random = iota
	RoundRobin
	WeightedRoundRobin
	ConsistentHash
	LeastConnections     // 新增：最少连接数
	SourceIPHash         // 新增：源IP哈希
	ResponseTimeWeighted // 新增：响应时间加权
	AdaptiveRoundRobin   // 新增：自适应轮询
	AdaptiveWeighted
)

func LoadBalanceFactory(lbType int) Balancer {
	switch lbType {
	case Random:
		return &RandomBalancer{}
	case RoundRobin:
		return &RoundRobinBalancer{}
	case WeightedRoundRobin:
		return &WeightRandomBalance{}
	case ConsistentHash:
		return NewConsistentHashBalancer(10, nil)
	case LeastConnections:
		return &LeastConnBalancer{}
	case SourceIPHash:
		return NewSourceIPHashBalancer()
	case ResponseTimeWeighted:
		return NewResponseTimeWeightedBalancer()
	case AdaptiveRoundRobin:
		return NewAdaptiveRoundRobinBalancer()
	case AdaptiveWeighted:
		return NewAdaptiveWeightedBalancer()
	default:
		return &RoundRobinBalancer{}
	}
}

type Balancer interface {
	Take() entity.EndPoint
	Set(endpoints ...entity.EndPoint) bool
}

type RandomBalancer struct {
	endpoints []entity.EndPoint
}

func (r *RandomBalancer) Set(endpoints ...entity.EndPoint) bool {
	r.endpoints = endpoints
	return true
}

func (r *RandomBalancer) Take() entity.EndPoint {
	if len(r.endpoints) == 0 {
		return entity.EndPoint{}
	}
	index := rand.Intn(len(r.endpoints))
	return r.endpoints[index]
}

type WeightRandomBalance struct {
	adders []entity.EndPoint
	totals []int
	max    int
}

func (b *WeightedBalancer) Take() entity.EndPoint {
	if len(b.endpoints) == 0 {
		return entity.EndPoint{}
	}
	total := 0
	for _, w := range b.weights {
		total += w
	}
	if total == 0 {
		return b.endpoints[0]
	}
	r := rand.Intn(total)
	for i, w := range b.weights {
		if r < w {
			return b.endpoints[i]
		}
		r -= w
	}
	return b.endpoints[0]
}

type WeightedBalancer struct {
	endpoints []entity.EndPoint
	weights   []int
}

func (b *WeightedBalancer) Set(eps ...entity.EndPoint) bool {
	b.endpoints = eps
	b.weights = make([]int, len(eps))
	for i, ep := range eps {
		if ep.Weight > 0 {
			b.weights[i] = int(ep.Weight)
		} else {
			b.weights[i] = 1
		}
	}
	return true
}

type LeastConnBalancer struct {
	endpoints []entity.EndPoint
	connMap   map[string]int
}

func (b *LeastConnBalancer) Set(eps ...entity.EndPoint) bool {
	b.endpoints = eps
	if b.connMap == nil {
		b.connMap = make(map[string]int)
	}
	return true
}
func (b *LeastConnBalancer) Take() entity.EndPoint {
	if len(b.endpoints) == 0 {
		return entity.EndPoint{}
	}
	minIdx := 0
	minConn := b.connMap[b.endpoints[0].Ip]
	for i, ep := range b.endpoints {
		if b.connMap[ep.Ip] < minConn {
			minConn = b.connMap[ep.Ip]
			minIdx = i
		}
	}
	b.connMap[b.endpoints[minIdx].Ip]++
	return b.endpoints[minIdx]
}

func (w *WeightRandomBalance) Set(endpoints ...entity.EndPoint) bool {
	if w == nil {
		return false
	}
	sort.Slice(endpoints, func(i, j int) bool {
		return endpoints[i].Weight < endpoints[j].Weight
	})
	totals := make([]int, len(endpoints))
	runningTotal := 0
	for i, e := range endpoints {
		runningTotal += int(e.Weight)
		totals[i] = runningTotal
	}
	w.adders = endpoints
	w.totals = totals
	w.max = runningTotal
	return true
}

func (w *WeightRandomBalance) Take() entity.EndPoint {
	r := rand.Intn(w.max) + 1
	i := sort.SearchInts(w.totals, r)
	return w.adders[i]
}

type RoundRobinBalancer struct {
	endpoints []entity.EndPoint
	acc       int64
}

func (r *RoundRobinBalancer) Set(endpoints ...entity.EndPoint) bool {
	r.endpoints = endpoints
	return true
}

func (r *RoundRobinBalancer) Take() entity.EndPoint {
	if len(r.endpoints) == 0 {
		return entity.EndPoint{}
	}
	n := atomic.AddInt64(&r.acc, 1)
	current := r.endpoints[r.acc]
	r.acc = int64(int(n) % len(r.endpoints))

	return current
}

type HashFunc func(data []byte) uint32

type Uint32Slice []uint32

func (s Uint32Slice) Len() int {
	return len(s)
}

func (s Uint32Slice) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s Uint32Slice) Less(i, j int) bool {
	return s[i] < s[j]
}

type ConsistentHashBalancer struct {
	mux      sync.RWMutex
	hash     HashFunc
	replicas int
	keys     Uint32Slice
	hashMap  map[uint32]entity.EndPoint
}

func NewConsistentHashBalancer(replicas int, hash HashFunc) *ConsistentHashBalancer {
	c := &ConsistentHashBalancer{
		replicas: replicas,
		hash:     hash,
		hashMap:  make(map[uint32]entity.EndPoint),
	}
	if c.hash == nil {
		c.hash = crc32.ChecksumIEEE
	}

	return c
}

func (c *ConsistentHashBalancer) Set(points ...entity.EndPoint) bool {
	if len(points) == 0 {
		return false
	}
	endpoint := points[0]
	c.mux.Lock()
	defer c.mux.Unlock()
	for i := 0; i < c.replicas; i++ {
		hash := c.hash([]byte(strconv.Itoa(i) + endpoint.Ip))
		c.keys = append(c.keys, hash)
		c.hashMap[hash] = endpoint
	}

	sort.Sort(c.keys)
	return true
}

func (c *ConsistentHashBalancer) IsEmpty() bool {
	return len(c.keys) == 0
}

func (c *ConsistentHashBalancer) Take() entity.EndPoint {
	if c.IsEmpty() {
		return entity.EndPoint{}
	}
	hash := c.hash([]byte(""))
	idx := sort.Search(len(c.keys), func(i int) bool { return c.keys[i] >= hash })
	if idx == len(c.keys) {
		idx = 0
	}
	c.mux.RLock()
	defer c.mux.RUnlock()

	return c.hashMap[c.keys[idx]]
}

// 源IP哈希负载均衡器
type SourceIPHashBalancer struct {
	endpoints []entity.EndPoint
	mu        sync.RWMutex
}

func NewSourceIPHashBalancer() *SourceIPHashBalancer {
	return &SourceIPHashBalancer{}
}

func (s *SourceIPHashBalancer) Set(endpoints ...entity.EndPoint) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.endpoints = endpoints
	return true
}

func (s *SourceIPHashBalancer) Take() entity.EndPoint {
	return s.TakeWithContext("")
}

func (s *SourceIPHashBalancer) TakeWithContext(clientIP string) entity.EndPoint {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.endpoints) == 0 {
		return entity.EndPoint{}
	}

	if clientIP == "" {
		// 如果没有客户端IP，回退到轮询
		return s.endpoints[0]
	}

	hash := crc32.ChecksumIEEE([]byte(clientIP))
	index := int(hash) % len(s.endpoints)
	return s.endpoints[index]
}

// 响应时间加权负载均衡器
type ResponseTimeWeightedBalancer struct {
	endpoints     []entity.EndPoint
	responseTimes []int64 // 平均响应时间（毫秒）
	requestCounts []int64 // 请求计数
	mu            sync.RWMutex
}

func NewResponseTimeWeightedBalancer() *ResponseTimeWeightedBalancer {
	return &ResponseTimeWeightedBalancer{}
}

func (r *ResponseTimeWeightedBalancer) Set(endpoints ...entity.EndPoint) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.endpoints = endpoints
	r.responseTimes = make([]int64, len(endpoints))
	r.requestCounts = make([]int64, len(endpoints))

	// 初始化响应时间为100ms
	for i := range r.responseTimes {
		r.responseTimes[i] = 100
	}
	return true
}

func (r *ResponseTimeWeightedBalancer) Take() entity.EndPoint {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.endpoints) == 0 {
		return entity.EndPoint{}
	}

	// 计算权重（响应时间越低权重越高）
	var totalWeight float64
	weights := make([]float64, len(r.endpoints))

	for i, responseTime := range r.responseTimes {
		if responseTime > 0 {
			weights[i] = 1000.0 / float64(responseTime) // 倒数作为权重
		} else {
			weights[i] = 10.0 // 默认权重
		}
		totalWeight += weights[i]
	}

	// 加权随机选择
	random := rand.Float64() * totalWeight
	var currentWeight float64

	for i, weight := range weights {
		currentWeight += weight
		if random <= currentWeight {
			return r.endpoints[i]
		}
	}

	return r.endpoints[0]
}

// 记录响应时间
func (r *ResponseTimeWeightedBalancer) RecordResponseTime(endpoint entity.EndPoint, responseTime time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for i, ep := range r.endpoints {
		if ep.Ip == endpoint.Ip && ep.Port == endpoint.Port {
			// 使用移动平均计算响应时间
			count := atomic.LoadInt64(&r.requestCounts[i])
			currentAvg := atomic.LoadInt64(&r.responseTimes[i])

			newAvg := (currentAvg*count + responseTime.Milliseconds()) / (count + 1)
			atomic.StoreInt64(&r.responseTimes[i], newAvg)
			atomic.AddInt64(&r.requestCounts[i], 1)
			break
		}
	}
}

// 自适应轮询负载均衡器
type AdaptiveRoundRobinBalancer struct {
	endpoints      []entity.EndPoint
	current        int64
	loadFactors    []float64 // 负载因子
	mu             sync.RWMutex
	lastUpdate     time.Time
	updateInterval time.Duration
}

func NewAdaptiveRoundRobinBalancer() *AdaptiveRoundRobinBalancer {
	return &AdaptiveRoundRobinBalancer{
		updateInterval: 30 * time.Second,
		lastUpdate:     time.Now(),
	}
}

func (a *AdaptiveRoundRobinBalancer) Set(endpoints ...entity.EndPoint) bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.endpoints = endpoints
	a.loadFactors = make([]float64, len(endpoints))

	// 初始化负载因子为1.0
	for i := range a.loadFactors {
		a.loadFactors[i] = 1.0
	}
	return true
}

func (a *AdaptiveRoundRobinBalancer) Take() entity.EndPoint {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if len(a.endpoints) == 0 {
		return entity.EndPoint{}
	}

	// 根据负载因子调整选择
	minLoadIndex := 0
	minLoad := a.loadFactors[0]

	for i, load := range a.loadFactors {
		if load < minLoad {
			minLoad = load
			minLoadIndex = i
		}
	}

	return a.endpoints[minLoadIndex]
}

// UpdateLoadFactor 更新负载因子
func (a *AdaptiveRoundRobinBalancer) UpdateLoadFactor(endpoint entity.EndPoint, loadFactor float64) {
	a.mu.Lock()
	defer a.mu.Unlock()

	for i, ep := range a.endpoints {
		if ep.Ip == endpoint.Ip && ep.Port == endpoint.Port {
			a.loadFactors[i] = loadFactor
			break
		}
	}
	a.lastUpdate = time.Now()
}

// HealthAwareBalancer 健康感知负载均衡器
type HealthAwareBalancer struct {
	endpoints    []entity.EndPoint
	healthStatus map[string]bool
	mu           sync.RWMutex
	balancer     Balancer
}

// NewHealthAwareBalancer 创建健康感知负载均衡器
func NewHealthAwareBalancer(balancer Balancer) *HealthAwareBalancer {
	return &HealthAwareBalancer{
		balancer:     balancer,
		healthStatus: make(map[string]bool),
	}
}

// Set 设置端点
func (hab *HealthAwareBalancer) Set(endpoints ...entity.EndPoint) bool {
	hab.mu.Lock()
	defer hab.mu.Unlock()

	// 过滤健康的端点
	var healthyEndpoints []entity.EndPoint
	for _, ep := range endpoints {
		key := fmt.Sprintf("%s:%d", ep.Ip, ep.Port)
		if healthy, exists := hab.healthStatus[key]; !exists || healthy {
			healthyEndpoints = append(healthyEndpoints, ep)
		}
	}

	hab.endpoints = healthyEndpoints
	return hab.balancer.Set(healthyEndpoints...)
}

// Take 获取端点
func (hab *HealthAwareBalancer) Take() entity.EndPoint {
	return hab.balancer.Take()
}

// UpdateHealth 更新健康状态
func (hab *HealthAwareBalancer) UpdateHealth(endpoint string, healthy bool) {
	hab.mu.Lock()
	hab.healthStatus[endpoint] = healthy
	hab.mu.Unlock()

	// 重新设置端点
	hab.Set(hab.endpoints...)
}

// AdaptiveWeightedBalancer 自适应加权负载均衡器
type AdaptiveWeightedBalancer struct {
	endpoints     []entity.EndPoint
	weights       []int64
	responseTime  []int64 // 响应时间（毫秒）
	errorCount    []int64 // 错误计数
	totalRequests []int64 // 总请求数
	mu            sync.RWMutex
	current       int64
}

// NewAdaptiveWeightedBalancer 创建自适应加权负载均衡器
func NewAdaptiveWeightedBalancer() *AdaptiveWeightedBalancer {
	return &AdaptiveWeightedBalancer{}
}

// Set 设置端点
func (awb *AdaptiveWeightedBalancer) Set(endpoints ...entity.EndPoint) bool {
	awb.mu.Lock()
	defer awb.mu.Unlock()

	awb.endpoints = endpoints
	awb.weights = make([]int64, len(endpoints))
	awb.responseTime = make([]int64, len(endpoints))
	awb.errorCount = make([]int64, len(endpoints))
	awb.totalRequests = make([]int64, len(endpoints))

	// 初始化权重
	for i, ep := range endpoints {
		if ep.Weight > 0 {
			awb.weights[i] = int64(ep.Weight)
		} else {
			awb.weights[i] = 100 // 默认权重
		}
	}

	return true
}

// Take 获取端点
func (awb *AdaptiveWeightedBalancer) Take() entity.EndPoint {
	awb.mu.RLock()
	defer awb.mu.RUnlock()

	if len(awb.endpoints) == 0 {
		return entity.EndPoint{}
	}

	// 计算动态权重
	dynamicWeights := awb.calculateDynamicWeights()

	// 加权随机选择
	totalWeight := int64(0)
	for _, w := range dynamicWeights {
		totalWeight += w
	}

	if totalWeight == 0 {
		return awb.endpoints[0]
	}

	r := rand.Int63n(totalWeight)
	for i, w := range dynamicWeights {
		if r < w {
			return awb.endpoints[i]
		}
		r -= w
	}

	return awb.endpoints[0]
}

// calculateDynamicWeights 计算动态权重
func (awb *AdaptiveWeightedBalancer) calculateDynamicWeights() []int64 {
	dynamicWeights := make([]int64, len(awb.endpoints))

	for i := range awb.endpoints {
		baseWeight := awb.weights[i]
		responseTime := atomic.LoadInt64(&awb.responseTime[i])
		errorCount := atomic.LoadInt64(&awb.errorCount[i])
		totalRequests := atomic.LoadInt64(&awb.totalRequests[i])

		// 计算错误率
		errorRate := float64(0)
		if totalRequests > 0 {
			errorRate = float64(errorCount) / float64(totalRequests)
		}

		// 根据响应时间和错误率调整权重
		weightFactor := float64(1)

		// 响应时间因子（响应时间越长，权重越低）
		if responseTime > 0 {
			weightFactor *= math.Max(0.1, 1.0/(1.0+float64(responseTime)/1000.0))
		}

		// 错误率因子（错误率越高，权重越低）
		if errorRate > 0 {
			weightFactor *= math.Max(0.1, 1.0-errorRate)
		}

		dynamicWeights[i] = int64(float64(baseWeight) * weightFactor)
		if dynamicWeights[i] < 1 {
			dynamicWeights[i] = 1
		}
	}

	return dynamicWeights
}

// RecordResponse 记录响应
func (awb *AdaptiveWeightedBalancer) RecordResponse(endpoint entity.EndPoint, responseTime time.Duration, success bool) {
	awb.mu.RLock()
	defer awb.mu.RUnlock()

	for i, ep := range awb.endpoints {
		if ep.Ip == endpoint.Ip && ep.Port == endpoint.Port {
			atomic.StoreInt64(&awb.responseTime[i], responseTime.Milliseconds())
			atomic.AddInt64(&awb.totalRequests[i], 1)
			if !success {
				atomic.AddInt64(&awb.errorCount[i], 1)
			}
			break
		}
	}
}
