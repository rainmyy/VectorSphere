package optimization

import (
	"context"
	"fmt"
	"sync"
	"time"

	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/vector"
)

// BatchSearchTask 表示一个批量搜索任务
type BatchSearchTask struct {
	ctx           context.Context
	vectorDB      *vector.VectorDB
	vectors       [][]float64
	k             int
	options       *SearchOptions
	startIndex    int
	results       [][]entity.Result
	errChan       chan error
	completedChan chan int
	wg            *sync.WaitGroup
	enableGPU     bool
	batchIndex    int
	totalBatches  int
}

// Execute 执行批量搜索任务
func (task *BatchSearchTask) Execute() {
	defer task.wg.Done()
	defer func() {
		// 通知任务完成
		task.completedChan <- task.batchIndex
	}()

	// 记录开始时间
	startTime := time.Now()

	// 检查上下文是否已取消
	if task.ctx.Err() != nil {
		task.errChan <- task.ctx.Err()
		return
	}

	// 创建搜索选项
	searchOptions := entity.SearchOptions{
		Nprobe:        task.options.Nprobe,
		SearchTimeout: task.options.Timeout,
		QualityLevel:  task.options.QualityLevel,
		UseCache:      task.options.EnableCache,
		MaxCandidates: 100, // 设置一个合理的默认值
	}

	// 根据任务选项设置其他搜索参数
	if task.options.ForceStrategy == "lsh" {
		searchOptions.UseANN = true
		searchOptions.NumHashTables = 10 // 默认值，可以根据需要调整
	}

	// 执行批量搜索
	results, err := task.vectorDB.OptimizedBatchSearch(task.vectors, task.k, searchOptions)

	// 检查错误
	if err != nil {
		task.errChan <- fmt.Errorf("批次 %d/%d 搜索失败: %w", task.batchIndex+1, task.totalBatches, err)
		return
	}

	// 将结果复制到结果数组中
	for i, result := range results {
		task.results[task.startIndex+i] = result
	}

	// 记录批次完成时间
	elapsed := time.Since(startTime)
	logger.Trace("批次 %d/%d 完成, 处理了 %d 个查询, 耗时: %v",
		task.batchIndex+1, task.totalBatches, len(task.vectors), elapsed)
}

// GetID 获取任务ID
func (task *BatchSearchTask) GetID() string {
	return fmt.Sprintf("batch-search-%d", task.batchIndex)
}

// Task 表示一个可执行的任务
type Task interface {
	// Execute 执行任务
	Execute()
	// GetID 获取任务ID
	GetID() string
}

// WorkerPool 表示一个工作池
type WorkerPool struct {
	workers   []*Worker
	taskQueue chan Task
	shutdown  chan struct{}
	wg        sync.WaitGroup
	isRunning bool
	mu        sync.Mutex
	stats     PoolStats
	size      int
}

// Worker 表示一个工作者
type Worker struct {
	id        int
	taskQueue chan Task
	shutdown  chan struct{}
	pool      *WorkerPool
}

// PoolStats 表示工作池的统计信息
type PoolStats struct {
	totalTasks      int64
	completedTasks  int64
	failedTasks     int64
	averageTaskTime time.Duration
	totalTaskTime   time.Duration
	mu              sync.Mutex
}

// NewWorkerPool 创建一个新的工作池
func NewWorkerPool(size, queueSize int) *WorkerPool {
	if size <= 0 {
		size = 1
	}
	if queueSize <= 0 {
		queueSize = 100
	}

	pool := &WorkerPool{
		workers:   make([]*Worker, size),
		taskQueue: make(chan Task, queueSize),
		shutdown:  make(chan struct{}),
		isRunning: false,
		stats: PoolStats{
			totalTasks:      0,
			completedTasks:  0,
			failedTasks:     0,
			averageTaskTime: 0,
			totalTaskTime:   0,
		},
		size: size,
	}

	// 创建工作者
	for i := 0; i < size; i++ {
		pool.workers[i] = &Worker{
			id:        i,
			taskQueue: pool.taskQueue,
			shutdown:  pool.shutdown,
			pool:      pool,
		}
	}

	return pool
}

// Start 启动工作池
func (wp *WorkerPool) Start() {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	if wp.isRunning {
		return
	}

	wp.isRunning = true

	// 启动所有工作者
	for _, worker := range wp.workers {
		wp.wg.Add(1)
		go worker.start(&wp.wg)
	}

	logger.Info("工作池已启动，工作者数量: %d", len(wp.workers))
}

// Stop 停止工作池
func (wp *WorkerPool) Stop() {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	if !wp.isRunning {
		return
	}

	// 关闭关闭通道，通知所有工作者停止
	close(wp.shutdown)

	// 等待所有工作者完成
	wp.wg.Wait()

	wp.isRunning = false
	logger.Info("工作池已停止")
}

// SubmitTask 提交任务到工作池
func (wp *WorkerPool) SubmitTask(task Task) error {
	wp.mu.Lock()
	defer wp.mu.Unlock()

	if !wp.isRunning {
		return fmt.Errorf("工作池未启动")
	}

	// 更新统计信息
	wp.stats.mu.Lock()
	wp.stats.totalTasks++
	wp.stats.mu.Unlock()

	// 提交任务到队列
	select {
	case wp.taskQueue <- task:
		return nil
	default:
		return fmt.Errorf("任务队列已满")
	}
}

// GetStats 获取工作池统计信息
func (wp *WorkerPool) GetStats() PoolStats {
	wp.stats.mu.Lock()
	defer wp.stats.mu.Unlock()

	return wp.stats
}

// start 启动工作者
func (w *Worker) start(wg *sync.WaitGroup) {
	defer wg.Done()

	logger.Debug("工作者 %d 已启动", w.id)

	for {
		select {
		case task := <-w.taskQueue:
			// 记录开始时间
			startTime := time.Now()

			// 执行任务
			func() {
				defer func() {
					if r := recover(); r != nil {
						logger.Error("工作者 %d 执行任务 %s 时发生panic: %v", w.id, task.GetID(), r)

						// 更新统计信息
						w.pool.stats.mu.Lock()
						w.pool.stats.failedTasks++
						w.pool.stats.mu.Unlock()
					}
				}()

				task.Execute()

				// 计算任务执行时间
				elapsed := time.Since(startTime)

				// 更新统计信息
				w.pool.stats.mu.Lock()
				w.pool.stats.completedTasks++
				w.pool.stats.totalTaskTime += elapsed
				w.pool.stats.averageTaskTime = w.pool.stats.totalTaskTime / time.Duration(w.pool.stats.completedTasks)
				w.pool.stats.mu.Unlock()

				logger.Trace("工作者 %d 完成任务 %s, 耗时: %v", w.id, task.GetID(), elapsed)
			}()

		case <-w.shutdown:
			logger.Debug("工作者 %d 已停止", w.id)
			return
		}
	}
}
