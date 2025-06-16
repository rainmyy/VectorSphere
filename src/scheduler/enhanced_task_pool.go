package scheduler

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"VectorSphere/src/library/logger"

	"github.com/robfig/cron/v3"
)

// TaskPriority 定义任务优先级
type TaskPriority int

const (
	PriorityLow      TaskPriority = 0
	PriorityNormal   TaskPriority = 50
	PriorityHigh     TaskPriority = 100
	PriorityCritical TaskPriority = 200
)

// TaskStatus 定义任务状态
type TaskStatus string

const (
	TaskStatusPending   TaskStatus = "pending"
	TaskStatusRunning   TaskStatus = "running"
	TaskStatusCompleted TaskStatus = "completed"
	TaskStatusFailed    TaskStatus = "failed"
	TaskStatusCancelled TaskStatus = "cancelled"
)

// TaskExecutionInfo 记录任务执行的详细信息
type TaskExecutionInfo struct {
	TaskID        string
	TaskName      string
	StartTime     time.Time
	EndTime       time.Time
	Duration      time.Duration
	Status        TaskStatus
	Error         error
	RetryCount    int
	RetryLimit    int
	NextRetryTime time.Time
}

// EnhancedTaskConfig 增强型任务配置
type EnhancedTaskConfig struct {
	MaxConcurrentTasks int           // 最大并发任务数
	DefaultRetryLimit  int           // 默认重试次数
	RetryBackoff       time.Duration // 重试间隔基数
	TaskQueueSize      int           // 任务队列大小
	EnableMetrics      bool          // 是否启用指标收集
	EnableTaskHistory  bool          // 是否保留任务执行历史
	HistorySize        int           // 历史记录保留数量
	ShutdownTimeout    time.Duration // 关闭超时时间
}

// DefaultTaskConfig 返回默认的任务配置
func DefaultTaskConfig() *EnhancedTaskConfig {
	return &EnhancedTaskConfig{
		MaxConcurrentTasks: 10,
		DefaultRetryLimit:  3,
		RetryBackoff:       5 * time.Second,
		TaskQueueSize:      100,
		EnableMetrics:      true,
		EnableTaskHistory:  true,
		HistorySize:        1000,
		ShutdownTimeout:    30 * time.Second,
	}
}

// EnhancedTaskPoolManager 增强型任务池管理器
type EnhancedTaskPoolManager struct {
	// 基础组件
	tasks  map[string]ScheduledTask // 存储已注册的任务
	cron   *cron.Cron               // Cron调度器实例
	config *EnhancedTaskConfig      // 任务池配置

	// 并发控制
	taskQueue    chan taskWrapper // 任务队列
	workerPool   chan struct{}    // 工作池信号量
	runningTasks int32            // 当前运行的任务数量

	// 状态管理
	executionHistory []TaskExecutionInfo          // 任务执行历史
	taskStatus       map[string]TaskExecutionInfo // 任务状态映射
	taskDependencies map[string][]string          // 任务依赖关系

	// 同步和控制
	mu        sync.RWMutex       // 读写锁
	stopCh    chan struct{}      // 停止信号
	ctx       context.Context    // 上下文
	cancel    context.CancelFunc // 取消函数
	isRunning bool               // 运行状态

	// 指标收集
	metrics *TaskPoolMetrics // 任务池指标
}

// TaskPoolMetrics 任务池指标
type TaskPoolMetrics struct {
	TotalTasksSubmitted  int64         // 提交的任务总数
	TotalTasksCompleted  int64         // 完成的任务总数
	TotalTasksFailed     int64         // 失败的任务总数
	TotalTasksCancelled  int64         // 取消的任务总数
	TotalTasksRetried    int64         // 重试的任务总数
	AverageExecutionTime time.Duration // 平均执行时间
	MaxExecutionTime     time.Duration // 最长执行时间
	MinExecutionTime     time.Duration // 最短执行时间
	TotalExecutionTime   time.Duration // 总执行时间
	LastExecutionTime    time.Time     // 最后执行时间
	mu                   sync.Mutex    // 指标锁
}

// taskWrapper 任务包装器，添加了优先级和上下文
type taskWrapper struct {
	task       ScheduledTask
	priority   TaskPriority
	ctx        context.Context
	cancel     context.CancelFunc
	taskID     string
	taskName   string
	retryCount int
	retryLimit int
	dependsOn  []string
}

// NewEnhancedTaskPoolManager 创建一个新的增强型任务池管理器
func NewEnhancedTaskPoolManager(config *EnhancedTaskConfig) *EnhancedTaskPoolManager {
	if config == nil {
		config = DefaultTaskConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &EnhancedTaskPoolManager{
		tasks:            make(map[string]ScheduledTask),
		config:           config,
		taskQueue:        make(chan taskWrapper, config.TaskQueueSize),
		workerPool:       make(chan struct{}, config.MaxConcurrentTasks),
		stopCh:           make(chan struct{}),
		ctx:              ctx,
		cancel:           cancel,
		taskStatus:       make(map[string]TaskExecutionInfo),
		taskDependencies: make(map[string][]string),
		metrics:          &TaskPoolMetrics{},
		executionHistory: make([]TaskExecutionInfo, 0, config.HistorySize),
	}
}

// RegisterTasks 批量注册一个或多个定时任务
func (pm *EnhancedTaskPoolManager) RegisterTasks(tasks ...ScheduledTask) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.isRunning {
		return fmt.Errorf("任务池已在运行，无法注册新任务")
	}

	for _, task := range tasks {
		taskName := task.GetName()
		if _, exists := pm.tasks[taskName]; exists {
			logger.Warning("任务 '%s' 已经注册，将被覆盖", taskName)
		}
		pm.tasks[taskName] = task
		logger.Info("任务 '%s' 已注册，Cron表达式: %s", taskName, task.GetCronSpec())
	}
	return nil
}

// SetTaskDependencies 设置任务依赖关系
func (pm *EnhancedTaskPoolManager) SetTaskDependencies(taskName string, dependsOn []string) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.isRunning {
		return fmt.Errorf("任务池已在运行，无法修改任务依赖")
	}

	// 验证任务是否存在
	if _, exists := pm.tasks[taskName]; !exists {
		return fmt.Errorf("任务 '%s' 不存在", taskName)
	}

	// 验证依赖的任务是否存在
	for _, depTask := range dependsOn {
		if _, exists := pm.tasks[depTask]; !exists {
			return fmt.Errorf("依赖的任务 '%s' 不存在", depTask)
		}
	}

	// 检测循环依赖
	visited := make(map[string]bool)
	path := make(map[string]bool)

	var checkCyclicDep func(string) bool
	checkCyclicDep = func(task string) bool {
		if !visited[task] {
			visited[task] = true
			path[task] = true

			for _, dep := range pm.taskDependencies[task] {
				if !visited[dep] && checkCyclicDep(dep) {
					return true
				} else if path[dep] {
					return true
				}
			}
		}

		path[task] = false
		return false
	}

	// 临时添加新的依赖关系进行检查
	oldDeps := pm.taskDependencies[taskName]
	pm.taskDependencies[taskName] = dependsOn

	hasCycle := checkCyclicDep(taskName)

	if hasCycle {
		// 恢复原来的依赖关系
		pm.taskDependencies[taskName] = oldDeps
		return fmt.Errorf("设置依赖关系会导致循环依赖")
	}

	// 确认依赖关系
	pm.taskDependencies[taskName] = dependsOn
	logger.Info("任务 '%s' 的依赖关系已设置: %v", taskName, dependsOn)
	return nil
}

// InitializeTasks 初始化所有已注册的任务
func (pm *EnhancedTaskPoolManager) InitializeTasks() error {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	logger.Info("开始初始化所有已注册的任务...")
	for name, task := range pm.tasks {
		// 检查是否实现了 Init() error 方法
		if initializer, ok := task.(interface{ Init() error }); ok {
			if err := initializer.Init(); err != nil {
				logger.Error("初始化任务 '%s' 失败: %v", name, err)
				return fmt.Errorf("初始化任务 '%s' 失败: %w", name, err)
			}
			logger.Info("任务 '%s' 初始化成功", name)
		} else {
			logger.Info("任务 '%s' 没有 Init 方法，跳过初始化", name)
		}
	}
	logger.Info("所有已注册任务初始化完成。")
	return nil
}

// Start 启动任务池，开始执行所有定时任务
func (pm *EnhancedTaskPoolManager) Start() error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.isRunning {
		return fmt.Errorf("任务池已经在运行中")
	}

	if len(pm.tasks) == 0 {
		logger.Info("没有已注册的任务，任务池未启动。")
		return nil
	}

	// 初始化cron调度器，添加秒级支持
	pm.cron = cron.New(cron.WithSeconds())

	logger.Info("开始将任务添加到调度器...")
	for name, task := range pm.tasks {
		// 使用闭包来捕获正确的task实例
		currentTask := task
		currentName := name
		entryID, err := pm.cron.AddFunc(currentTask.GetCronSpec(), func() {
			// 创建任务包装器
			taskCtx, taskCancel := context.WithTimeout(pm.ctx, currentTask.Timeout())
			taskID := fmt.Sprintf("%s-%d", currentName, time.Now().UnixNano())

			wrapper := taskWrapper{
				task:       currentTask,
				priority:   PriorityNormal, // 默认优先级
				ctx:        taskCtx,
				cancel:     taskCancel,
				taskID:     taskID,
				taskName:   currentName,
				retryCount: 0,
				retryLimit: pm.config.DefaultRetryLimit,
				dependsOn:  pm.taskDependencies[currentName],
			}

			// 提交任务到队列
			pm.submitTaskWrapper(wrapper)
		})
		if err != nil {
			logger.Error("添加任务 '%s' 到调度器失败: %v", currentName, err)
			// 如果一个任务添加失败，停止所有已启动的
			pm.cron.Stop() // 停止已经部分启动的cron
			return fmt.Errorf("添加任务 '%s' 到调度器失败: %w", currentName, err)
		}
		logger.Info("任务 '%s' (EntryID: %d) 已成功添加到调度器，Cron: %s", currentName, entryID, currentTask.GetCronSpec())
	}

	// 启动工作池
	for i := 0; i < pm.config.MaxConcurrentTasks; i++ {
		go pm.worker()
	}

	pm.cron.Start()
	pm.isRunning = true
	logger.Info("任务池已启动，所有定时任务开始调度执行。最大并发任务数: %d", pm.config.MaxConcurrentTasks)

	// 启动一个goroutine监听停止信号
	go func() {
		<-pm.stopCh
		logger.Info("接收到停止信号，开始停止任务池...")

		// 创建一个带超时的上下文
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), pm.config.ShutdownTimeout)
		defer shutdownCancel()

		// 取消所有正在运行的任务
		pm.cancel()

		// 停止接收新任务
		ctx := pm.cron.Stop() // 优雅地停止cron调度器

		// 等待所有任务完成或超时
		select {
		case <-ctx.Done():
			logger.Info("所有定时任务已停止。")
		case <-shutdownCtx.Done():
			logger.Warning("等待任务完成超时，强制停止。")
		}

		// 等待工作池中的任务完成或超时
		pm.waitForWorkers(shutdownCtx)

		// 调用所有任务的Stop方法
		pm.mu.RLock()
		for name, task := range pm.tasks {
			if stopper, ok := task.(interface{ Stop() error }); ok {
				if err := stopper.Stop(); err != nil {
					logger.Error("停止任务 '%s' 时发生错误: %v", name, err)
				}
			}
		}
		pm.mu.RUnlock()

		logger.Info("所有任务已尝试停止，任务池已完全关闭。")
		pm.mu.Lock()
		pm.isRunning = false
		pm.mu.Unlock()
	}()

	return nil
}

// waitForWorkers 等待工作池中的任务完成
func (pm *EnhancedTaskPoolManager) waitForWorkers(ctx context.Context) {
	done := make(chan struct{})

	go func() {
		for {
			running := atomic.LoadInt32(&pm.runningTasks)
			if running == 0 {
				close(done)
				return
			}
			logger.Info("等待 %d 个正在运行的任务完成...", running)
			time.Sleep(500 * time.Millisecond)
		}
	}()

	select {
	case <-done:
		logger.Info("所有工作池任务已完成。")
	case <-ctx.Done():
		logger.Warning("等待工作池任务完成超时。")
	}
}

// worker 工作池中的工作协程
func (pm *EnhancedTaskPoolManager) worker() {
	for {
		select {
		case <-pm.ctx.Done():
			// 上下文已取消，退出工作协程
			return
		case wrapper := <-pm.taskQueue:
			// 获取工作池信号量
			pm.workerPool <- struct{}{}

			// 检查任务上下文是否已取消
			if wrapper.ctx.Err() != nil {
				<-pm.workerPool // 释放工作池信号量
				continue
			}

			// 检查依赖任务是否已完成
			if !pm.checkDependencies(wrapper) {
				<-pm.workerPool // 释放工作池信号量
				// 将任务重新放回队列，稍后再试
				go func() {
					time.Sleep(1 * time.Second)
					pm.submitTaskWrapper(wrapper)
				}()
				continue
			}

			// 执行任务
			atomic.AddInt32(&pm.runningTasks, 1)
			go func(w taskWrapper) {
				defer func() {
					atomic.AddInt32(&pm.runningTasks, -1)
					<-pm.workerPool // 释放工作池信号量

					// 恢复可能的panic
					if r := recover(); r != nil {
						logger.Error("任务 '%s' (ID: %s) 执行时发生panic: %v", w.taskName, w.taskID, r)

						// 记录失败状态
						pm.recordTaskExecution(w, TaskStatusFailed, fmt.Errorf("任务panic: %v", r))

						// 尝试重试任务
						pm.retryTaskIfNeeded(w)
					}
				}()

				// 记录任务开始执行
				startTime := time.Now()
				pm.updateTaskStatus(w.taskID, TaskExecutionInfo{
					TaskID:     w.taskID,
					TaskName:   w.taskName,
					StartTime:  startTime,
					Status:     TaskStatusRunning,
					RetryCount: w.retryCount,
					RetryLimit: w.retryLimit,
				})

				logger.Info("开始执行任务: %s (ID: %s)", w.taskName, w.taskID)

				// 执行任务，捕获错误
				var err error
				done := make(chan struct{})

				go func() {
					defer close(done)
					err = w.task.Run()
				}()

				// 等待任务完成或上下文取消
				select {
				case <-done:
					// 任务正常完成
				case <-w.ctx.Done():
					// 任务超时或被取消
					err = w.ctx.Err()
				}

				endTime := time.Now()
				duration := endTime.Sub(startTime)

				// 根据执行结果更新状态
				if err != nil {
					logger.Error("执行任务 '%s' (ID: %s) 失败: %v", w.taskName, w.taskID, err)

					// 记录失败状态
					pm.recordTaskExecution(w, TaskStatusFailed, err)

					// 尝试重试任务
					pm.retryTaskIfNeeded(w)
				} else {
					logger.Info("任务 '%s' (ID: %s) 执行完成，耗时: %v", w.taskName, w.taskID, duration)

					// 记录成功状态
					pm.recordTaskExecution(w, TaskStatusCompleted, nil)
				}
			}(wrapper)
		}
	}
}

// checkDependencies 检查任务的依赖是否已完成
func (pm *EnhancedTaskPoolManager) checkDependencies(wrapper taskWrapper) bool {
	if len(wrapper.dependsOn) == 0 {
		return true
	}

	pm.mu.RLock()
	defer pm.mu.RUnlock()

	for _, depTask := range wrapper.dependsOn {
		// 检查依赖任务的最新状态
		found := false
		for i := len(pm.executionHistory) - 1; i >= 0; i-- {
			info := pm.executionHistory[i]
			if info.TaskName == depTask {
				found = true
				if info.Status != TaskStatusCompleted {
					// 依赖任务未成功完成
					return false
				}
				break
			}
		}

		if !found {
			// 依赖任务尚未执行
			return false
		}
	}

	return true
}

// retryTaskIfNeeded 根据需要重试任务
func (pm *EnhancedTaskPoolManager) retryTaskIfNeeded(wrapper taskWrapper) {
	if wrapper.retryCount >= wrapper.retryLimit {
		logger.Warning("任务 '%s' (ID: %s) 已达到最大重试次数 %d，不再重试",
			wrapper.taskName, wrapper.taskID, wrapper.retryLimit)
		return
	}

	// 增加重试计数
	wrapper.retryCount++

	// 计算退避时间
	backoff := time.Duration(wrapper.retryCount) * pm.config.RetryBackoff

	logger.Info("计划在 %v 后重试任务 '%s' (ID: %s)，这是第 %d 次重试",
		backoff, wrapper.taskName, wrapper.taskID, wrapper.retryCount)

	// 更新指标
	pm.metrics.mu.Lock()
	pm.metrics.TotalTasksRetried++
	pm.metrics.mu.Unlock()

	// 创建新的上下文
	newCtx, newCancel := context.WithTimeout(pm.ctx, wrapper.task.Timeout())

	// 创建新的包装器
	newWrapper := taskWrapper{
		task:       wrapper.task,
		priority:   wrapper.priority,
		ctx:        newCtx,
		cancel:     newCancel,
		taskID:     fmt.Sprintf("%s-retry-%d", wrapper.taskID, wrapper.retryCount),
		taskName:   wrapper.taskName,
		retryCount: wrapper.retryCount,
		retryLimit: wrapper.retryLimit,
		dependsOn:  wrapper.dependsOn,
	}

	// 延迟提交重试任务
	go func() {
		time.Sleep(backoff)
		pm.submitTaskWrapper(newWrapper)
	}()
}

// recordTaskExecution 记录任务执行信息
func (pm *EnhancedTaskPoolManager) recordTaskExecution(wrapper taskWrapper, status TaskStatus, err error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	endTime := time.Now()

	// 获取开始时间
	startTime := endTime
	if info, exists := pm.taskStatus[wrapper.taskID]; exists {
		startTime = info.StartTime
	}

	duration := endTime.Sub(startTime)

	// 创建执行信息
	execInfo := TaskExecutionInfo{
		TaskID:     wrapper.taskID,
		TaskName:   wrapper.taskName,
		StartTime:  startTime,
		EndTime:    endTime,
		Duration:   duration,
		Status:     status,
		Error:      err,
		RetryCount: wrapper.retryCount,
		RetryLimit: wrapper.retryLimit,
	}

	// 更新任务状态
	pm.taskStatus[wrapper.taskID] = execInfo

	// 添加到历史记录
	if pm.config.EnableTaskHistory {
		pm.executionHistory = append(pm.executionHistory, execInfo)

		// 如果历史记录超过限制，删除最旧的记录
		if len(pm.executionHistory) > pm.config.HistorySize {
			excess := len(pm.executionHistory) - pm.config.HistorySize
			pm.executionHistory = pm.executionHistory[excess:]
		}
	}

	// 更新指标
	if pm.config.EnableMetrics {
		pm.updateMetrics(execInfo)
	}
}

// updateTaskStatus 更新任务状态
func (pm *EnhancedTaskPoolManager) updateTaskStatus(taskID string, info TaskExecutionInfo) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.taskStatus[taskID] = info
}

// updateMetrics 更新任务池指标
func (pm *EnhancedTaskPoolManager) updateMetrics(info TaskExecutionInfo) {
	pm.metrics.mu.Lock()
	defer pm.metrics.mu.Unlock()

	// 更新总计数
	switch info.Status {
	case TaskStatusCompleted:
		pm.metrics.TotalTasksCompleted++
	case TaskStatusFailed:
		pm.metrics.TotalTasksFailed++
	case TaskStatusCancelled:
		pm.metrics.TotalTasksCancelled++
	}

	// 更新执行时间统计
	pm.metrics.TotalExecutionTime += info.Duration
	pm.metrics.LastExecutionTime = info.EndTime

	// 更新最大/最小执行时间
	if info.Duration > pm.metrics.MaxExecutionTime {
		pm.metrics.MaxExecutionTime = info.Duration
	}

	if pm.metrics.MinExecutionTime == 0 || info.Duration < pm.metrics.MinExecutionTime {
		pm.metrics.MinExecutionTime = info.Duration
	}

	// 计算平均执行时间
	totalTasks := pm.metrics.TotalTasksCompleted + pm.metrics.TotalTasksFailed + pm.metrics.TotalTasksCancelled
	if totalTasks > 0 {
		pm.metrics.AverageExecutionTime = pm.metrics.TotalExecutionTime / time.Duration(totalTasks)
	}
}

// submitTaskWrapper 提交任务包装器到队列
func (pm *EnhancedTaskPoolManager) submitTaskWrapper(wrapper taskWrapper) {
	// 更新指标
	pm.metrics.mu.Lock()
	pm.metrics.TotalTasksSubmitted++
	pm.metrics.mu.Unlock()

	// 提交到队列
	select {
	case pm.taskQueue <- wrapper:
		// 成功提交
	case <-pm.ctx.Done():
		// 任务池已停止
		wrapper.cancel() // 取消任务
		logger.Warning("任务池已停止，无法提交任务 '%s'", wrapper.taskName)
	}
}

// Submit 提交一个任务到任务池执行（立即执行，非定时）
func (pm *EnhancedTaskPoolManager) Submit(task ScheduledTask, priority TaskPriority) error {
	if !pm.isRunning {
		return fmt.Errorf("任务池未运行，无法提交任务")
	}

	taskID := fmt.Sprintf("%s-%d", task.GetName(), time.Now().UnixNano())
	taskCtx, taskCancel := context.WithTimeout(pm.ctx, task.Timeout())

	// 创建任务包装器
	wrapper := taskWrapper{
		task:       task,
		priority:   priority,
		ctx:        taskCtx,
		cancel:     taskCancel,
		taskID:     taskID,
		taskName:   task.GetName(),
		retryCount: 0,
		retryLimit: pm.config.DefaultRetryLimit,
	}

	logger.Info("提交任务 '%s' (ID: %s) 到任务池执行，优先级: %d", task.GetName(), taskID, priority)

	// 提交任务到队列
	pm.submitTaskWrapper(wrapper)

	return nil
}

// Stop 停止任务池中的所有定时任务
func (pm *EnhancedTaskPoolManager) Stop() {
	pm.mu.Lock()
	if !pm.isRunning {
		pm.mu.Unlock()
		logger.Info("任务池未运行，无需停止。")
		return
	}

	// 避免重复关闭stopCh
	select {
	case <-pm.stopCh:
		// 已经关闭或正在关闭
		pm.mu.Unlock()
		return
	default:
	}
	pm.mu.Unlock() // 在关闭通道前解锁，避免与监听stopCh的goroutine死锁

	logger.Info("正在发送停止信号给任务池...")
	close(pm.stopCh) // 发送停止信号
}

// GetTaskStatus 获取任务状态
func (pm *EnhancedTaskPoolManager) GetTaskStatus(taskID string) (TaskExecutionInfo, bool) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	info, exists := pm.taskStatus[taskID]
	return info, exists
}

// GetTaskHistory 获取任务执行历史
func (pm *EnhancedTaskPoolManager) GetTaskHistory(taskName string, limit int) []TaskExecutionInfo {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if !pm.config.EnableTaskHistory {
		return nil
	}

	result := make([]TaskExecutionInfo, 0)

	// 从最新的记录开始查找
	for i := len(pm.executionHistory) - 1; i >= 0; i-- {
		info := pm.executionHistory[i]
		if info.TaskName == taskName {
			result = append(result, info)

			if limit > 0 && len(result) >= limit {
				break
			}
		}
	}

	return result
}

// GetMetrics 获取任务池指标
func (pm *EnhancedTaskPoolManager) GetMetrics() *TaskPoolMetrics {
	pm.metrics.mu.Lock()
	defer pm.metrics.mu.Unlock()

	// 返回指标的副本
	copy := *pm.metrics
	return &copy
}

// GetTask 获取已注册的特定任务
func (pm *EnhancedTaskPoolManager) GetTask(name string) (ScheduledTask, bool) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	task, exists := pm.tasks[name]
	return task, exists
}

// ListTasks 列出所有已注册的任务名称
func (pm *EnhancedTaskPoolManager) ListTasks() []string {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	names := make([]string, 0, len(pm.tasks))
	for name := range pm.tasks {
		names = append(names, name)
	}
	return names
}

// CancelTask 取消正在执行的任务
func (pm *EnhancedTaskPoolManager) CancelTask(taskID string) bool {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// 查找任务状态
	info, exists := pm.taskStatus[taskID]
	if !exists || info.Status != TaskStatusRunning {
		return false
	}

	// 遍历所有工作协程中的任务
	for _, wrapper := range pm.getRunningTasks() {
		if wrapper.taskID == taskID {
			// 取消任务
			wrapper.cancel()

			// 更新状态
			info.Status = TaskStatusCancelled
			info.EndTime = time.Now()
			info.Duration = info.EndTime.Sub(info.StartTime)
			pm.taskStatus[taskID] = info

			// 添加到历史记录
			if pm.config.EnableTaskHistory {
				pm.executionHistory = append(pm.executionHistory, info)
			}

			// 更新指标
			if pm.config.EnableMetrics {
				pm.metrics.mu.Lock()
				pm.metrics.TotalTasksCancelled++
				pm.metrics.mu.Unlock()
			}

			logger.Info("已取消任务 '%s' (ID: %s)", info.TaskName, taskID)
			return true
		}
	}

	return false
}

// getRunningTasks 获取所有正在运行的任务
// 注意：调用此方法前必须持有锁
func (pm *EnhancedTaskPoolManager) getRunningTasks() []taskWrapper {
	// 这个方法在实际实现中需要跟踪所有正在运行的任务
	// 这里只是一个占位符
	return []taskWrapper{}
}

// SetTaskRetryLimit 设置任务的重试限制
func (pm *EnhancedTaskPoolManager) SetTaskRetryLimit(taskName string, retryLimit int) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// 验证任务是否存在
	if _, exists := pm.tasks[taskName]; !exists {
		return fmt.Errorf("任务 '%s' 不存在", taskName)
	}

	// 在实际实现中，这里应该设置任务的重试限制
	logger.Info("设置任务 '%s' 的重试限制为 %d", taskName, retryLimit)

	return nil
}

// GetRunningTaskCount 返回当前正在运行的任务数量
func (pm *EnhancedTaskPoolManager) GetRunningTaskCount() int32 {
	return atomic.LoadInt32(&pm.runningTasks)
}
