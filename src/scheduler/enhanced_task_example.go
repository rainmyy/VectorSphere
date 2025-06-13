package scheduler

import (
	"VectorSphere/src/library/log"
	"fmt"
	"math/rand"
	"time"
)

// EnhancedSampleTask 增强版示例任务
type EnhancedSampleTask struct {
	Name        string
	Description string
	CronSpec    string
	taskID      string
	taskTarget  string
	timeoutDur  time.Duration
	params      map[string]interface{}
	failRate    float64 // 模拟失败的概率，用于测试重试机制
}

// NewEnhancedSampleTask 创建一个新的增强版示例任务
func NewEnhancedSampleTask(name, description, cronSpec string, timeout time.Duration, failRate float64) *EnhancedSampleTask {
	return &EnhancedSampleTask{
		Name:        name,
		Description: description,
		CronSpec:    cronSpec,
		timeoutDur:  timeout,
		failRate:    failRate,
		params:      make(map[string]interface{}),
	}
}

// Run 实现 ScheduledTask 接口的 Run 方法
func (t *EnhancedSampleTask) Run() error {
	log.Info("任务 '%s' 正在运行... 描述: %s, 时间戳: %s", t.Name, t.Description, time.Now().String())

	// 模拟任务执行时间
	executionTime := time.Duration(rand.Intn(1000)) * time.Millisecond
	time.Sleep(executionTime)

	// 模拟随机失败，用于测试重试机制
	if rand.Float64() < t.failRate {
		err := fmt.Errorf("任务 '%s' 随机失败，用于测试重试机制", t.Name)
		log.Warning("任务执行失败: %v", err)
		return err
	}

	log.Info("任务 '%s' 执行完成，耗时: %v", t.Name, executionTime)
	return nil
}

// GetCronSpec 实现 ScheduledTask 接口的 GetCronSpec 方法
func (t *EnhancedSampleTask) GetCronSpec() string {
	return t.CronSpec
}

// GetName 实现 ScheduledTask 接口的 GetName 方法
func (t *EnhancedSampleTask) GetName() string {
	return t.Name
}

// Init 实现 ScheduledTask 接口的 Init 方法
func (t *EnhancedSampleTask) Init() error {
	log.Info("初始化任务 '%s'...", t.Name)
	// 初始化任务参数
	t.params["initialized_at"] = time.Now()
	t.params["description"] = t.Description
	return nil
}

// Stop 实现 ScheduledTask 接口的 Stop 方法
func (t *EnhancedSampleTask) Stop() error {
	log.Info("停止任务 '%s'...", t.Name)
	return nil
}

// Params 实现 ScheduledTask 接口的 Params 方法
func (t *EnhancedSampleTask) Params() map[string]interface{} {
	return t.params
}

// Timeout 实现 ScheduledTask 接口的 Timeout 方法
func (t *EnhancedSampleTask) Timeout() time.Duration {
	return t.timeoutDur
}

// Clone 实现 ScheduledTask 接口的 Clone 方法
func (t *EnhancedSampleTask) Clone() ScheduledTask {
	cloned := &EnhancedSampleTask{
		Name:        t.Name,
		Description: t.Description,
		CronSpec:    t.CronSpec,
		taskID:      t.taskID,
		taskTarget:  t.taskTarget,
		timeoutDur:  t.timeoutDur,
		failRate:    t.failRate,
		params:      make(map[string]interface{}),
	}

	// 复制参数
	for k, v := range t.params {
		cloned.params[k] = v
	}

	return cloned
}

// SetID 实现 ScheduledTask 接口的 SetID 方法
func (t *EnhancedSampleTask) SetID(id string) {
	t.taskID = id
}

// SetTarget 实现 ScheduledTask 接口的 SetTarget 方法
func (t *EnhancedSampleTask) SetTarget(target string) {
	t.taskTarget = target
}

// 示例：如何使用增强版任务池
func EnhancedTaskPoolExample() {
	// 创建任务池配置
	config := &EnhancedTaskConfig{
		MaxConcurrentTasks: 5,
		DefaultRetryLimit:  3,
		RetryBackoff:       2 * time.Second,
		TaskQueueSize:      50,
		EnableMetrics:      true,
		EnableTaskHistory:  true,
		HistorySize:        100,
		ShutdownTimeout:    10 * time.Second,
	}

	// 创建增强版任务池管理器
	manager := NewEnhancedTaskPoolManager(config)

	// 创建任务实例
	task1 := NewEnhancedSampleTask(
		"DataProcessor",
		"处理数据的任务",
		"*/30 * * * * *", // 每30秒执行一次
		10*time.Second,   // 超时时间
		0.2,             // 20%的失败率
	)

	task2 := NewEnhancedSampleTask(
		"ReportGenerator",
		"生成报告的任务",
		"0 */1 * * * *", // 每分钟执行一次
		30*time.Second,  // 超时时间
		0.1,            // 10%的失败率
	)

	task3 := NewEnhancedSampleTask(
		"DataCleanup",
		"清理数据的任务",
		"0 0 */1 * * *", // 每小时执行一次
		1*time.Minute,   // 超时时间
		0.05,           // 5%的失败率
	)

	// 注册任务
	if err := manager.RegisterTasks(task1, task2, task3); err != nil {
		log.Fatal("注册任务失败: %v", err)
	}

	// 设置任务依赖关系
	// ReportGenerator 依赖于 DataProcessor
	if err := manager.SetTaskDependencies("ReportGenerator", []string{"DataProcessor"}); err != nil {
		log.Warning("设置任务依赖关系失败: %v", err)
	}

	// DataCleanup 依赖于 ReportGenerator
	if err := manager.SetTaskDependencies("DataCleanup", []string{"ReportGenerator"}); err != nil {
		log.Warning("设置任务依赖关系失败: %v", err)
	}

	// 初始化任务
	if err := manager.InitializeTasks(); err != nil {
		log.Fatal("初始化任务失败: %v", err)
	}

	// 启动任务池
	if err := manager.Start(); err != nil {
		log.Fatal("启动任务池失败: %v", err)
	}

	log.Info("增强版任务池已启动，按 CTRL+C 退出")

	// 立即提交一个高优先级任务
	immediate := NewEnhancedSampleTask(
		"ImmediateTask",
		"立即执行的高优先级任务",
		"", // 不需要Cron表达式
		5*time.Second,
		0.0, // 不会失败
	)

	if err := manager.Submit(immediate, PriorityHigh); err != nil {
		log.Error("提交立即执行任务失败: %v", err)
	}

	// 等待一段时间，让任务执行
	time.Sleep(2 * time.Minute)

	// 获取并打印指标
	metrics := manager.GetMetrics()
	log.Info("任务池指标统计:")
	log.Info("- 提交的任务总数: %d", metrics.TotalTasksSubmitted)
	log.Info("- 完成的任务总数: %d", metrics.TotalTasksCompleted)
	log.Info("- 失败的任务总数: %d", metrics.TotalTasksFailed)
	log.Info("- 重试的任务总数: %d", metrics.TotalTasksRetried)
	log.Info("- 平均执行时间: %v", metrics.AverageExecutionTime)

	// 停止任务池
	log.Info("准备停止任务池...")
	manager.Stop()

	log.Info("程序退出。")
}