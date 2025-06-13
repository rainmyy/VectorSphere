package optimization

import (
	"VectorSphere/src/enhanced"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/log"
	"VectorSphere/src/vector"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// HighThroughputOptimizer 高吞吐量优化器
type HighThroughputOptimizer struct {
	vectorDB           *vector.VectorDB
	config             *HighThroughputConfig
	workerPool         *WorkerPool
	cacheOptimizer     *CacheOptimizer
	resultCache        *ResultCache
	metrics            *PerformanceMetrics
	circuitBreaker     *CircuitBreaker
	loadBalancer       *LoadBalancer
	requestMerger      *RequestMerger
	queryOptimizer     *QueryOptimizer
	performanceMonitor *PerformanceMonitor
	mu                 sync.RWMutex
	isInitialized      bool
	totalQueries       atomic.Int64
	totalBatches       atomic.Int64
	totalCacheHits     atomic.Int64
	totalErrors        atomic.Int64
	lastOptimizeTime   time.Time
	deadlockDetector   *enhanced.DeadlockDetector
}

// ResultCache 结果缓存
type ResultCache struct {
	cache      map[string][][]entity.Result
	expiration map[string]time.Time
	frequency  map[string]int
	maxEntries int
	ttl        time.Duration
	mu         sync.RWMutex
}

// CircuitBreaker 熔断器
type CircuitBreaker struct {
	state            int // 0: 关闭, 1: 半开, 2: 打开
	failureThreshold float64
	failureCount     int
	totalCount       int
	lastFailureTime  time.Time
	recoveryTime     time.Duration
	mu               sync.Mutex // 使用单一锁类型
}

// LoadBalancer 负载均衡器
type LoadBalancer struct {
	strategy    string
	workerLoads []int
	mu          sync.Mutex // 使用单一锁类型
}

// RequestMerger 请求合并器
type RequestMerger struct {
	mergeWindow     time.Duration
	pendingRequests map[string]*MergedRequest
	mu              sync.Mutex // 使用单一锁类型
}

// MergedRequest 合并请求
type MergedRequest struct {
	vectors    [][]float64
	indices    []int
	k          int
	options    *SearchOptions
	resultChan chan MergedResult
	createTime time.Time
}

// MergedResult 合并结果
type MergedResult struct {
	results []entity.Result
	index   int
	err     error
}

// QueryOptimizer 查询优化器
type QueryOptimizer struct {
	queryPlanCache map[string]string
	cacheSize      int
	mu             sync.Mutex // 使用单一锁类型
}

// NewHighThroughputOptimizer 创建高吞吐量优化器
func NewHighThroughputOptimizer(vectorDB *vector.VectorDB, config *HighThroughputConfig) *HighThroughputOptimizer {
	if config == nil {
		config = GetDefaultHighThroughputConfig()
	} else {
		config = config.MergeWithDefault()
		config.Validate()
	}

	// 根据硬件优化配置
	if config.EnableHardwareAwareOptimization {
		cpuCores := runtime.NumCPU()
		memoryGB := 16 // 默认假设16GB内存
		hasGPU := config.EnableGPU
		gpuMemoryGB := 8 // 默认假设8GB GPU内存

		config = config.OptimizeForHardware(cpuCores, memoryGB, hasGPU, gpuMemoryGB)
	}

	hto := &HighThroughputOptimizer{
		vectorDB:         vectorDB,
		config:           config,
		isInitialized:    false,
		lastOptimizeTime: time.Now(),
	}

	return hto
}

// Initialize 初始化优化器
func (hto *HighThroughputOptimizer) Initialize() error {
	hto.mu.Lock()
	defer hto.mu.Unlock()

	if hto.isInitialized {
		return nil
	}

	// 创建工作池
	hto.workerPool = NewWorkerPool(hto.config.WorkerPoolSize, hto.config.TaskQueueCapacity)
	hto.workerPool.Start()

	// 创建缓存优化器
	if hto.config.EnableCache {
		hto.cacheOptimizer = NewCacheOptimizer(hto.vectorDB.MultiCache)
	}

	// 创建结果缓存
	if hto.config.EnableResultCache {
		hto.resultCache = &ResultCache{
			cache:      make(map[string][][]entity.Result),
			expiration: make(map[string]time.Time),
			frequency:  make(map[string]int),
			maxEntries: hto.config.ResultCacheMaxEntries,
			ttl:        hto.config.ResultCacheTTL,
		}
	}

	// 创建熔断器
	if hto.config.EnableCircuitBreaker {
		hto.circuitBreaker = &CircuitBreaker{
			state:            0, // 关闭状态
			failureThreshold: hto.config.CircuitBreakerThreshold,
			recoveryTime:     hto.config.CircuitBreakerRecoveryTime,
		}
	}

	// 创建负载均衡器
	if hto.config.EnableLoadBalancing {
		hto.loadBalancer = &LoadBalancer{
			strategy:    hto.config.LoadBalancingStrategy,
			workerLoads: make([]int, hto.config.WorkerPoolSize),
		}
	}

	// 创建请求合并器
	if hto.config.EnableRequestMerging {
		hto.requestMerger = &RequestMerger{
			mergeWindow:     hto.config.RequestMergeWindow,
			pendingRequests: make(map[string]*MergedRequest),
		}
	}

	// 创建查询优化器
	if hto.config.EnableQueryOptimization {
		hto.queryOptimizer = &QueryOptimizer{
			queryPlanCache: make(map[string]string),
			cacheSize:      hto.config.QueryPlanCacheSize,
		}
	}

	// 创建性能指标
	hto.metrics = NewPerformanceMetrics()

	// 创建性能监控器
	hto.performanceMonitor = NewPerformanceMonitor(nil)
	hto.performanceMonitor.Start()

	hto.isInitialized = true
	log.Info("高吞吐量优化器初始化完成")
	return nil
}

// OptimizedSearch 优化搜索
func (hto *HighThroughputOptimizer) OptimizedSearch(ctx context.Context, vector []float64, k int, options *SearchOptions) ([]entity.Result, error) {
	if !hto.isInitialized {
		if err := hto.Initialize(); err != nil {
			return nil, err
		}
	}

	// 记录查询计数
	hto.totalQueries.Add(1)

	// 检查熔断器状态
	if hto.circuitBreaker != nil && hto.isCircuitBreakerOpen() {
		return nil, fmt.Errorf("circuit breaker is open")
	}

	// 创建带超时的上下文
	var cancel context.CancelFunc
	if options.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, options.Timeout)
		defer cancel()
	} else if hto.config.BaseTimeout > 0 {
		// 使用默认超时
		ctx, cancel = context.WithTimeout(ctx, hto.config.BaseTimeout)
		defer cancel()
	}

	// 检查缓存
	var cacheKey string
	if options.EnableCache && hto.resultCache != nil {
		cacheKey = hto.generateCacheKey(vector, k, options)
		if cachedResults := hto.checkResultCache(cacheKey); cachedResults != nil {
			hto.totalCacheHits.Add(1)
			return cachedResults, nil
		}
	}

	// 为当前搜索操作生成唯一ID
	searchID := fmt.Sprintf("search_%d", time.Now().UnixNano())

	// 选择最优策略
	strategy := hto.selectOptimalStrategy(vector, k, options)

	// 添加到死锁检测器
	hto.deadlockDetector.AddLockOwner(searchID, "search_thread")

	// 执行搜索
	results, err := hto.executeSearch(ctx, vector, k, options, strategy)

	// 从死锁检测器移除
	hto.deadlockDetector.RemoveLockOwner(searchID)

	// 更新熔断器状态
	if hto.circuitBreaker != nil {
		hto.updateCircuitBreaker(err == nil)
	}

	// 如果发生错误，记录错误计数
	if err != nil {
		hto.totalErrors.Add(1)
		return nil, err
	}

	// 缓存结果
	if options.EnableCache && hto.resultCache != nil && cacheKey != "" {
		hto.cacheResults(cacheKey, results)
	}

	// 定期优化
	hto.periodicOptimize()

	return results, nil
}

// BatchSearch 批量搜索
func (hto *HighThroughputOptimizer) BatchSearch(ctx context.Context, vectors [][]float64, k int, options *SearchOptions) ([][]entity.Result, error) {
	if !hto.isInitialized {
		if err := hto.Initialize(); err != nil {
			return nil, err
		}
	}

	// 记录查询计数和批次计数
	queryCount := len(vectors)
	hto.totalQueries.Add(int64(queryCount))
	hto.totalBatches.Add(1)

	// 检查熔断器状态
	if hto.circuitBreaker != nil && hto.isCircuitBreakerOpen() {
		return nil, fmt.Errorf("circuit breaker is open")
	}

	// 创建带超时的上下文
	var cancel context.CancelFunc
	if options.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, options.Timeout)
		defer cancel()
	} else if hto.config.BaseTimeout > 0 {
		// 使用默认超时
		ctx, cancel = context.WithTimeout(ctx, hto.config.BaseTimeout)
		defer cancel()
	}

	// 为当前批处理操作生成唯一ID
	batchID := fmt.Sprintf("batch_%d", time.Now().UnixNano())

	// 添加到死锁检测器
	hto.deadlockDetector.AddLockOwner(batchID, "batch_thread")

	// 检查缓存
	if options.EnableCache && hto.resultCache != nil {
		cachedResults := make([][]entity.Result, queryCount)
		cacheMissIndices := make([]int, 0, queryCount)
		cacheMissVectors := make([][]float64, 0, queryCount)

		// 检查每个查询的缓存
		for i, vector := range vectors {
			cacheKey := hto.generateCacheKey(vector, k, options)
			if results := hto.checkResultCache(cacheKey); results != nil {
				cachedResults[i] = results
				hto.totalCacheHits.Add(1)
			} else {
				cacheMissIndices = append(cacheMissIndices, i)
				cacheMissVectors = append(cacheMissVectors, vector)
			}
		}

		// 如果全部命中缓存，直接返回
		if len(cacheMissVectors) == 0 {
			// 从死锁检测器移除
			hto.deadlockDetector.RemoveLockOwner(batchID)
			return cachedResults, nil
		}

		// 只处理缓存未命中的查询
		if len(cacheMissVectors) < queryCount {
			missResults, err := hto.processBatches(ctx, cacheMissVectors, k, options)
			if err != nil {
				// 从死锁检测器移除
				hto.deadlockDetector.RemoveLockOwner(batchID)
				return nil, err
			}

			// 合并缓存结果和新查询结果
			for i, missIndex := range cacheMissIndices {
				cachedResults[missIndex] = missResults[i]

				// 缓存新结果
				cacheKey := hto.generateCacheKey(vectors[missIndex], k, options)
				hto.cacheResults(cacheKey, missResults[i])
			}

			// 从死锁检测器移除
			hto.deadlockDetector.RemoveLockOwner(batchID)
			return cachedResults, nil
		}
	}

	// 处理所有批次
	results, err := hto.processBatches(ctx, vectors, k, options)

	// 从死锁检测器移除
	hto.deadlockDetector.RemoveLockOwner(batchID)

	// 更新熔断器状态
	if hto.circuitBreaker != nil {
		hto.updateCircuitBreaker(err == nil)
	}

	// 如果发生错误，记录错误计数
	if err != nil {
		hto.totalErrors.Add(1)
		return nil, err
	}

	// 缓存结果
	if options.EnableCache && hto.resultCache != nil {
		for i, vector := range vectors {
			cacheKey := hto.generateCacheKey(vector, k, options)
			hto.cacheResults(cacheKey, results[i])
		}
	}

	// 定期优化
	hto.periodicOptimize()

	return results, nil
}

// processBatches 处理批次
func (hto *HighThroughputOptimizer) processBatches(ctx context.Context, vectors [][]float64, k int, options *SearchOptions) ([][]entity.Result, error) {
	queryCount := len(vectors)
	startTime := time.Now()

	// 计算最优批处理大小
	batchSize := hto.config.BatchSize
	if hto.config.AdaptiveBatchSize {
		batchSize = calculateOptimalBatchSize(queryCount, len(vectors[0]), options.EnableGPU)
		if options.BatchSize > 0 {
			// 如果选项中指定了批处理大小，使用较小的值
			if options.BatchSize < batchSize {
				batchSize = options.BatchSize
			}
		}
	}

	// 确保批处理大小在配置范围内
	if batchSize < hto.config.MinBatchSize {
		batchSize = hto.config.MinBatchSize
	} else if batchSize > hto.config.MaxBatchSize {
		batchSize = hto.config.MaxBatchSize
	}

	// 计算批次数
	numBatches := (queryCount + batchSize - 1) / batchSize

	// 创建结果数组
	allResults := make([][]entity.Result, queryCount)

	// 创建错误通道和结果通道
	errChan := make(chan error, numBatches)
	completedChan := make(chan int, numBatches) // 用于跟踪完成的批次

	// 创建等待组
	var wg sync.WaitGroup
	wg.Add(numBatches)

	// 启动监控协程
	go func() {
		completed := 0
		totalBatches := numBatches

		for {
			select {
			case <-completedChan:
				completed++
				if completed%5 == 0 || completed == totalBatches {
					progress := float64(completed) / float64(totalBatches) * 100
					elapsed := time.Since(startTime)
					log.Trace("批处理进度: %.1f%% (%d/%d), 耗时: %v",
						progress, completed, totalBatches, elapsed)
				}
			case <-ctx.Done():
				// 上下文取消或超时
				return
			}
		}
	}()

	// 处理每个批次
	for i := 0; i < numBatches; i++ {
		startIdx := i * batchSize
		endIdx := startIdx + batchSize
		if endIdx > queryCount {
			endIdx = queryCount
		}

		// 提取当前批次的向量
		batchVectors := vectors[startIdx:endIdx]

		// 为每个批次创建唯一ID
		batchTaskID := fmt.Sprintf("batch_task_%d_%d", i, time.Now().UnixNano())

		// 创建批处理任务
		task := &BatchSearchTask{
			ctx:           ctx,
			vectorDB:      hto.vectorDB,
			vectors:       batchVectors,
			k:             k,
			options:       options,
			startIndex:    startIdx,
			results:       allResults,
			errChan:       errChan,
			completedChan: completedChan,
			wg:            &wg,
			enableGPU:     options.EnableGPU,
			batchIndex:    i,
			totalBatches:  numBatches,
		}

		// 添加到死锁检测器
		hto.deadlockDetector.AddWaitingRelation(batchTaskID, "worker_pool")

		// 提交任务到工作池
		hto.workerPool.SubmitTask(task)
	}

	// 等待所有批次完成或上下文取消
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	// 等待完成或超时
	select {
	case <-done:
		// 所有批次已完成
	case <-ctx.Done():
		// 超时或取消
		return nil, ctx.Err()
	}

	// 检查是否有错误
	select {
	case err := <-errChan:
		return nil, err
	default:
		// 没有错误，继续
	}

	// 记录性能指标
	elapsed := time.Since(startTime)
	hto.recordBatchPerformance(queryCount, elapsed)
	log.Debug("批处理完成: %d个查询, %d个批次, 耗时: %v, 平均每查询: %.2fms",
		queryCount, numBatches, elapsed, float64(elapsed.Milliseconds())/float64(queryCount))

	return allResults, nil
}

// isCircuitBreakerOpen 检查熔断器是否打开
func (hto *HighThroughputOptimizer) isCircuitBreakerOpen() bool {
	hto.circuitBreaker.mu.Unlock()
	defer hto.circuitBreaker.mu.Unlock()

	// 如果熔断器处于打开状态
	if hto.circuitBreaker.state == 2 {
		// 检查是否已经过了恢复时间
		if time.Since(hto.circuitBreaker.lastFailureTime) > hto.circuitBreaker.recoveryTime {
			// 切换到半开状态
			hto.circuitBreaker.mu.Unlock()
			hto.circuitBreaker.mu.Lock()
			hto.circuitBreaker.state = 1 // 半开
			hto.circuitBreaker.failureCount = 0
			hto.circuitBreaker.totalCount = 0
			hto.circuitBreaker.mu.Unlock()
			hto.circuitBreaker.mu.Lock()

			return false
		}
		return true
	}

	return false
}

// updateCircuitBreaker 更新熔断器状态
func (hto *HighThroughputOptimizer) updateCircuitBreaker(success bool) {
	hto.circuitBreaker.mu.Lock()
	defer hto.circuitBreaker.mu.Unlock()

	hto.circuitBreaker.totalCount++

	if !success {
		hto.circuitBreaker.failureCount++
		hto.circuitBreaker.lastFailureTime = time.Now()
	}

	// 计算失败率
	failureRate := float64(hto.circuitBreaker.failureCount) / float64(hto.circuitBreaker.totalCount)

	// 根据熔断器状态更新
	switch hto.circuitBreaker.state {
	case 0: // 关闭状态
		// 如果失败率超过阈值，切换到打开状态
		if failureRate >= hto.circuitBreaker.failureThreshold && hto.circuitBreaker.totalCount >= 10 {
			hto.circuitBreaker.state = 2 // 打开
			log.Warning("熔断器已打开，失败率: %.2f%%", failureRate*100)
		}
	case 1: // 半开状态
		if success {
			// 成功请求，切换到关闭状态
			hto.circuitBreaker.state = 0 // 关闭
			hto.circuitBreaker.failureCount = 0
			hto.circuitBreaker.totalCount = 0
			log.Info("熔断器已关闭")
		} else {
			// 失败请求，切换回打开状态
			hto.circuitBreaker.state = 2 // 打开
			log.Warning("熔断器重新打开")
		}
	}
}

// checkResultCache 检查结果缓存
func (hto *HighThroughputOptimizer) checkResultCache(key string) []entity.Result {
	hto.resultCache.mu.RLock()
	defer hto.resultCache.mu.RUnlock()

	// 检查缓存是否存在
	results, exists := hto.resultCache.cache[key]
	if !exists || len(results) == 0 {
		return nil
	}

	// 检查是否过期
	expireTime, exists := hto.resultCache.expiration[key]
	if !exists || time.Now().After(expireTime) {
		return nil
	}

	// 更新访问频率
	hto.resultCache.mu.RUnlock()
	hto.resultCache.mu.Lock()
	hto.resultCache.frequency[key]++

	// 如果频率达到阈值，延长过期时间
	if hto.resultCache.frequency[key] > 5 {
		// 热点数据，延长过期时间
		extendedExpiration := time.Now().Add(hto.resultCache.ttl * 2)
		if extendedExpiration.After(hto.resultCache.expiration[key]) {
			hto.resultCache.expiration[key] = extendedExpiration
			log.Trace("热点缓存延期: 键=%s, 频率=%d", key, hto.resultCache.frequency[key])
		}
	}
	hto.resultCache.mu.Unlock()
	hto.resultCache.mu.RLock()

	return results[0]
}

// cacheResults 缓存结果
func (hto *HighThroughputOptimizer) cacheResults(key string, results []entity.Result) {
	hto.resultCache.mu.Lock()
	defer hto.resultCache.mu.Unlock()

	// 检查缓存大小
	if len(hto.resultCache.cache) >= hto.resultCache.maxEntries {
		// 清理过期缓存
		hto.cleanExpiredCache()

		// 如果仍然超过大小限制，删除最旧的条目
		if len(hto.resultCache.cache) >= hto.resultCache.maxEntries {
			hto.evictOldestCache()
		}
	}

	// 添加到缓存
	hto.resultCache.cache[key] = [][]entity.Result{results}
	hto.resultCache.expiration[key] = time.Now().Add(hto.resultCache.ttl)
}

// cleanExpiredCache 清理过期缓存
func (hto *HighThroughputOptimizer) cleanExpiredCache() {
	now := time.Now()
	for key, expireTime := range hto.resultCache.expiration {
		if now.After(expireTime) {
			delete(hto.resultCache.cache, key)
			delete(hto.resultCache.expiration, key)
			delete(hto.resultCache.frequency, key)
		}
	}
}

// evictOldestCache 淘汰最旧的缓存
func (hto *HighThroughputOptimizer) evictOldestCache() {
	// 已经在调用方法中获取了锁，这里不需要再加锁
	// 使用LRU策略淘汰缓存
	// 1. 按访问频率和时间综合评分
	type cacheScore struct {
		key   string
		score float64
	}

	scores := make([]cacheScore, 0, len(hto.resultCache.expiration))
	now := time.Now()

	// 计算每个缓存项的评分
	for key, expireTime := range hto.resultCache.expiration {
		// 获取访问频率（如果有的话）
		frequency := hto.resultCache.frequency[key]
		if frequency == 0 {
			frequency = 1 // 避免除以零
		}

		// 计算时间因子（越接近过期时间评分越低）
		timeLeft := expireTime.Sub(now)
		timeFactor := float64(timeLeft) / float64(hto.resultCache.ttl)

		// 综合评分：访问频率和剩余时间的加权和
		// 高频率、高剩余时间 = 高分（保留）
		// 低频率、低剩余时间 = 低分（淘汰）
		score := float64(frequency)*0.7 + timeFactor*0.3

		scores = append(scores, cacheScore{key: key, score: score})
	}

	// 按评分排序（升序，最低分在前面）
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score < scores[j].score
	})

	// 淘汰评分最低的10%缓存项，或至少1个
	evictCount := len(scores) / 10
	if evictCount < 1 {
		evictCount = 1
	}

	// 执行淘汰
	for i := 0; i < evictCount && i < len(scores); i++ {
		key := scores[i].key
		delete(hto.resultCache.cache, key)
		delete(hto.resultCache.expiration, key)
		delete(hto.resultCache.frequency, key)
		log.Trace("缓存淘汰: 键=%s, 评分=%.2f", key, scores[i].score)
	}

	log.Debug("缓存淘汰完成: 淘汰%d项, 剩余%d项", evictCount, len(hto.resultCache.cache))
}

// generateCacheKey 生成缓存键
func (hto *HighThroughputOptimizer) generateCacheKey(vector []float64, k int, options *SearchOptions) string {
	// 生成向量的哈希值 - 使用更高效的哈希算法
	vectorHash := hashVectorFNV(vector)

	// 构建缓存键
	strategy := "auto"
	if options.ForceStrategy != "" {
		strategy = options.ForceStrategy
	}

	// 使用字符串构建器提高性能
	var sb strings.Builder
	sb.WriteString(vectorHash)
	sb.WriteByte(':')
	sb.WriteString(strconv.Itoa(k))
	sb.WriteByte(':')
	sb.WriteString(strategy)
	sb.WriteByte(':')
	sb.WriteString(fmt.Sprintf("%.2f", options.QualityLevel))
	sb.WriteByte(':')
	sb.WriteString(strconv.FormatBool(options.EnableGPU))
	sb.WriteByte(':')
	sb.WriteString(strconv.Itoa(options.Nprobe))

	return sb.String()
}

// hashVectorFNV 使用FNV哈希算法哈希向量
func hashVectorFNV(vector []float64) string {
	// 使用FNV-1a哈希算法
	h := fnv.New64a()

	// 将向量数据写入哈希器
	for _, v := range vector {
		binary.Write(h, binary.LittleEndian, v)
	}

	// 返回哈希值的十六进制表示
	return fmt.Sprintf("%x", h.Sum64())
}

// periodicOptimize 定期优化
func (hto *HighThroughputOptimizer) periodicOptimize() {
	// 检查是否需要优化
	if time.Since(hto.lastOptimizeTime) < hto.config.AdaptiveOptimizationInterval {
		return
	}

	hto.mu.Lock()
	defer hto.mu.Unlock()

	// 再次检查，防止并发优化
	if time.Since(hto.lastOptimizeTime) < hto.config.AdaptiveOptimizationInterval {
		return
	}

	// 更新最后优化时间
	hto.lastOptimizeTime = time.Now()

	// 优化缓存
	if hto.config.EnableCache && hto.cacheOptimizer != nil {
		hto.cacheOptimizer.Optimize()
	}

	// 优化批处理大小
	if hto.config.AdaptiveBatchSize {
		hto.optimizeBatchSize()
	}
}

// optimizeBatchSize 优化批处理大小
func (hto *HighThroughputOptimizer) optimizeBatchSize() {
	// 获取性能指标
	totalQueries := hto.totalQueries.Load()
	totalBatches := hto.totalBatches.Load()

	if totalBatches == 0 || totalQueries == 0 {
		return
	}

	// 计算平均批处理大小
	avgBatchSize := float64(totalQueries) / float64(totalBatches)

	// 如果平均批处理大小小于当前批处理大小的一半，减小批处理大小
	if avgBatchSize < float64(hto.config.BatchSize)/2 {
		newBatchSize := int(float64(hto.config.BatchSize) / hto.config.BatchSizeAdjustFactor)
		if newBatchSize >= hto.config.MinBatchSize {
			hto.config.BatchSize = newBatchSize
			log.Info("批处理大小已调整为 %d", hto.config.BatchSize)
		}
	} else if avgBatchSize > float64(hto.config.BatchSize)*0.9 {
		// 如果平均批处理大小接近当前批处理大小，增大批处理大小
		newBatchSize := int(float64(hto.config.BatchSize) * hto.config.BatchSizeAdjustFactor)
		if newBatchSize <= hto.config.MaxBatchSize {
			hto.config.BatchSize = newBatchSize
			log.Info("批处理大小已调整为 %d", hto.config.BatchSize)
		}
	}

	// 重置计数器
	hto.totalQueries.Store(0)
	hto.totalBatches.Store(0)
}

// GetStats 获取统计信息
func (hto *HighThroughputOptimizer) GetStats() map[string]interface{} {
	stats := make(map[string]interface{})

	// 基本统计
	stats["total_queries"] = hto.totalQueries.Load()
	stats["total_batches"] = hto.totalBatches.Load()
	stats["total_cache_hits"] = hto.totalCacheHits.Load()
	stats["total_errors"] = hto.totalErrors.Load()

	// 缓存统计
	if hto.resultCache != nil {
		hto.resultCache.mu.Lock()
		stats["cache_size"] = len(hto.resultCache.cache)
		stats["cache_max_entries"] = hto.resultCache.maxEntries
		hto.resultCache.mu.Unlock()
	}

	// 熔断器统计
	if hto.circuitBreaker != nil {
		hto.circuitBreaker.mu.Lock()
		stats["circuit_breaker_state"] = hto.circuitBreaker.state
		stats["circuit_breaker_failure_count"] = hto.circuitBreaker.failureCount
		stats["circuit_breaker_total_count"] = hto.circuitBreaker.totalCount
		hto.circuitBreaker.mu.Unlock()
	}

	// 工作池统计
	if hto.workerPool != nil {
		stats["worker_pool_size"] = hto.workerPool.size
		stats["worker_pool_queue_size"] = len(hto.workerPool.taskQueue)
	}

	// 配置统计
	stats["batch_size"] = hto.config.BatchSize
	stats["min_batch_size"] = hto.config.MinBatchSize
	stats["max_batch_size"] = hto.config.MaxBatchSize
	stats["enable_gpu"] = hto.config.EnableGPU
	stats["enable_cache"] = hto.config.EnableCache

	// 死锁检测器统计
	hto.deadlockDetector.Mu.Lock()
	stats["deadlock_detector_wait_graph_size"] = len(hto.deadlockDetector.WaitGraph)
	stats["deadlock_detector_lock_owners_size"] = len(hto.deadlockDetector.LockOwners)
	hto.deadlockDetector.Mu.Unlock()

	return stats
}

// Close 关闭优化器
func (hto *HighThroughputOptimizer) Close() error {
	hto.mu.Lock()
	defer hto.mu.Unlock()

	if !hto.isInitialized {
		return nil
	}

	// 创建带超时的上下文，确保关闭操作不会无限期阻塞
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// 停止工作池
	if hto.workerPool != nil {
		go func() {
			hto.workerPool.Stop()
			cancel() // 工作池停止后取消上下文
		}()

		// 等待工作池停止或超时
		select {
		case <-ctx.Done():
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				log.Warning("关闭工作池超时")
			}
		}
	}

	// 停止性能监控器
	if hto.performanceMonitor != nil {
		hto.performanceMonitor.Stop()
	}

	hto.isInitialized = false
	log.Info("高吞吐量优化器已关闭")
	return nil
}

// calculateOptimalBatchSize 计算最优批处理大小
func calculateOptimalBatchSize(queryCount, vectorDim int, enableGPU bool) int {
	// 获取系统资源信息
	cpuCores := runtime.NumCPU()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	availableMemoryGB := float64(m.TotalAlloc-m.Alloc) / (1024 * 1024 * 1024)

	// 基础批处理大小 - 根据CPU核心数调整
	baseBatchSize := cpuCores * 8

	// 根据向量维度调整
	dimFactor := 1.0
	if vectorDim > 1000 {
		// 高维向量，减小批处理大小
		dimFactor = 0.5
	} else if vectorDim > 500 {
		dimFactor = 0.75
	} else if vectorDim < 100 {
		// 低维向量，增大批处理大小
		dimFactor = 1.5
	}

	// 应用维度因子
	baseBatchSize = int(float64(baseBatchSize) * dimFactor)

	// 根据可用内存调整
	memFactor := 1.0
	if availableMemoryGB < 2 {
		// 内存不足，减小批处理大小
		memFactor = 0.5
	} else if availableMemoryGB > 16 {
		// 内存充足，增大批处理大小
		memFactor = 1.25
	}

	// 应用内存因子
	baseBatchSize = int(float64(baseBatchSize) * memFactor)

	// 根据GPU可用性调整
	if enableGPU {
		// GPU通常可以处理更大的批次
		gpuFactor := 2.0
		// 如果向量维度很高，GPU优势更明显
		if vectorDim > 1000 {
			gpuFactor = 3.0
		}
		baseBatchSize = int(float64(baseBatchSize) * gpuFactor)
	}

	// 根据查询数量调整
	if queryCount < baseBatchSize {
		// 如果查询数量小于基础批处理大小，使用查询数量
		return queryCount
	} else if queryCount < baseBatchSize*2 {
		// 如果查询数量小于基础批处理大小的两倍，使用一半查询数量
		return queryCount / 2
	}

	// 确保批处理大小在合理范围内
	if baseBatchSize < 16 {
		baseBatchSize = 16 // 最小批处理大小
	} else if baseBatchSize > 1024 {
		baseBatchSize = 1024 // 最大批处理大小
	}

	log.Trace("计算最优批处理大小: %d (查询数=%d, 维度=%d, GPU=%v, CPU核心=%d, 可用内存=%.2fGB)",
		baseBatchSize, queryCount, vectorDim, enableGPU, cpuCores, availableMemoryGB)

	return baseBatchSize
}

// selectOptimalStrategy 选择最优策略
func (hto *HighThroughputOptimizer) selectOptimalStrategy(data []float64, k int, options *SearchOptions) string {
	// 如果强制使用特定策略，直接返回
	if options.ForceStrategy != "" {
		return options.ForceStrategy
	}

	// 构建搜索上下文
	searchCtx := vector.SearchContext{
		QueryVector:  data,
		K:            k,
		QualityLevel: options.QualityLevel,
		Timeout:      options.Timeout,
	}

	// 使用向量数据库的自适应索引选择器
	strategy := hto.vectorDB.SelectOptimalIndexStrategy(searchCtx)

	// 将内部策略转换为字符串
	strategyStr := "brute_force"
	switch strategy {
	case vector.StrategyBruteForce:
		strategyStr = "brute_force"
	case vector.StrategyIVF:
		strategyStr = "ivf"
	case vector.StrategyHNSW:
		strategyStr = "hnsw"
	case vector.StrategyPQ:
		strategyStr = "pq"
	case vector.StrategyHybrid:
		strategyStr = "hybrid"
	case vector.StrategyEnhancedIVF:
		strategyStr = "enhanced_ivf"
	case vector.StrategyEnhancedLSH:
		strategyStr = "enhanced_lsh"
	}

	return strategyStr
}

// executeSearch 执行搜索
func (hto *HighThroughputOptimizer) executeSearch(ctx context.Context, data []float64, k int, options *SearchOptions, strategy string) ([]entity.Result, error) {
	// 构建搜索上下文
	searchCtx := vector.SearchContext{
		QueryVector:  data,
		K:            k,
		QualityLevel: options.QualityLevel,
		Timeout:      options.Timeout,
	}

	// 根据策略设置搜索参数
	switch strategy {
	case "ivf":
		searchCtx.UseIVF = true
		searchCtx.Nprobe = options.Nprobe
	case "hnsw":
		searchCtx.UseHNSW = true
	case "pq":
		searchCtx.UsePQ = true
	case "hybrid":
		searchCtx.UseHybrid = true
	case "enhanced_ivf":
		searchCtx.UseEnhancedIVF = true
	case "enhanced_lsh":
		searchCtx.UseEnhancedLSH = true
	}

	// 设置GPU选项
	if options.EnableGPU {
		searchCtx.UseGPU = true
	}

	// 创建与SearchContext对应的SearchOptions
	searchOptions := vector.SearchOptions{
		Nprobe:        searchCtx.Nprobe,
		SearchTimeout: searchCtx.Timeout,
		QualityLevel:  searchCtx.QualityLevel,
		UseCache:      options.EnableCache,
		MaxCandidates: k * 2, // 设置一个合理的候选数量
	}

	// 执行优化搜索
	return hto.vectorDB.OptimizedSearch(data, k, searchOptions)
}
