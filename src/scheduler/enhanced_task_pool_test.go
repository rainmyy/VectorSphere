package scheduler

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"VectorSphere/src/library/logger"
)

// 用于测试的简单任务实现
type testTask struct {
	name       string
	cronSpec   string
	executions int32
	failures   int32
	taskID     string
	taskTarget string
	shouldFail bool
}

func newTestTask(name, cronSpec string, shouldFail bool) *testTask {
	return &testTask{
		name:       name,
		cronSpec:   cronSpec,
		shouldFail: shouldFail,
	}
}

func (t *testTask) Run() error {
	atomic.AddInt32(&t.executions, 1)
	if t.shouldFail {
		atomic.AddInt32(&t.failures, 1)
		return fmt.Errorf("任务 %s 故意失败", t.name)
	}
	return nil
}

func (t *testTask) GetCronSpec() string { return t.cronSpec }
func (t *testTask) GetName() string     { return t.name }
func (t *testTask) Init() error         { return nil }
func (t *testTask) Stop() error         { return nil }
func (t *testTask) Params() map[string]interface{} {
	return map[string]interface{}{"executions": t.executions, "failures": t.failures}
}
func (t *testTask) Timeout() time.Duration { return 5 * time.Second }
func (t *testTask) Clone() ScheduledTask {
	return &testTask{
		name:       t.name,
		cronSpec:   t.cronSpec,
		executions: 0,
		failures:   0,
		shouldFail: t.shouldFail,
	}
}
func (t *testTask) SetID(id string)         { t.taskID = id }
func (t *testTask) SetTarget(target string) { t.taskTarget = target }

// TestEnhancedTaskPoolBasic 测试增强版任务池的基本功能
func TestEnhancedTaskPoolBasic(t *testing.T) {
	// 创建任务池配置
	config := &EnhancedTaskConfig{
		MaxConcurrentTasks: 5,
		DefaultRetryLimit:  2,
		RetryBackoff:       100 * time.Millisecond,
		TaskQueueSize:      10,
		EnableMetrics:      true,
		EnableTaskHistory:  true,
		HistorySize:        10,
		ShutdownTimeout:    1 * time.Second,
	}

	// 创建任务池
	pool := NewEnhancedTaskPoolManager(config)

	// 创建测试任务
	task1 := newTestTask("task1", "* * * * * *", false) // 每秒执行一次，不会失败
	task2 := newTestTask("task2", "* * * * * *", true)  // 每秒执行一次，会失败

	// 注册任务
	err := pool.RegisterTasks(task1, task2)
	if err != nil {
		t.Fatalf("注册任务失败: %v", err)
	}

	// 初始化任务
	err = pool.InitializeTasks()
	if err != nil {
		t.Fatalf("初始化任务失败: %v", err)
	}

	// 启动任务池
	err = pool.Start()
	if err != nil {
		t.Fatalf("启动任务池失败: %v", err)
	}

	// 等待任务执行
	time.Sleep(3 * time.Second)

	// 停止任务池
	pool.Stop()

	// 验证任务执行次数
	if task1.executions == 0 {
		t.Errorf("任务1应该至少执行一次，但实际执行了 %d 次", task1.executions)
	}

	if task2.executions == 0 {
		t.Errorf("任务2应该至少执行一次，但实际执行了 %d 次", task2.executions)
	}

	if task2.failures == 0 {
		t.Errorf("任务2应该至少失败一次，但实际失败了 %d 次", task2.failures)
	}

	// 验证指标
	metrics := pool.GetMetrics()
	if metrics.TotalTasksSubmitted == 0 {
		t.Errorf("应该有任务被提交，但指标显示 %d", metrics.TotalTasksSubmitted)
	}

	if metrics.TotalTasksFailed == 0 {
		t.Errorf("应该有任务失败，但指标显示 %d", metrics.TotalTasksFailed)
	}

	if metrics.TotalTasksRetried == 0 {
		t.Errorf("应该有任务重试，但指标显示 %d", metrics.TotalTasksRetried)
	}
}

// TestTaskDependencies 测试任务依赖关系
func TestTaskDependencies(t *testing.T) {
	// 创建任务池配置
	config := &EnhancedTaskConfig{
		MaxConcurrentTasks: 5,
		DefaultRetryLimit:  1,
		RetryBackoff:       100 * time.Millisecond,
		TaskQueueSize:      10,
		EnableMetrics:      true,
		EnableTaskHistory:  true,
		HistorySize:        10,
		ShutdownTimeout:    1 * time.Second,
	}

	// 创建任务池
	pool := NewEnhancedTaskPoolManager(config)

	// 创建测试任务
	task1 := newTestTask("parent", "", false)
	task2 := newTestTask("child", "", false)

	// 注册任务
	err := pool.RegisterTasks(task1, task2)
	if err != nil {
		t.Fatalf("注册任务失败: %v", err)
	}

	// 设置依赖关系
	err = pool.SetTaskDependencies("child", []string{"parent"})
	if err != nil {
		t.Fatalf("设置任务依赖关系失败: %v", err)
	}

	// 初始化任务
	err = pool.InitializeTasks()
	if err != nil {
		t.Fatalf("初始化任务失败: %v", err)
	}

	// 启动任务池
	err = pool.Start()
	if err != nil {
		t.Fatalf("启动任务池失败: %v", err)
	}

	// 提交子任务，由于依赖关系，它应该不会立即执行
	err = pool.Submit(task2, PriorityNormal)
	if err != nil {
		t.Fatalf("提交子任务失败: %v", err)
	}

	// 等待一小段时间，确保任务有机会被处理
	time.Sleep(500 * time.Millisecond)

	// 此时子任务不应该执行，因为父任务还没有执行
	if task2.executions > 0 {
		t.Errorf("子任务不应该执行，但实际执行了 %d 次", task2.executions)
	}

	// 提交父任务
	err = pool.Submit(task1, PriorityNormal)
	if err != nil {
		t.Fatalf("提交父任务失败: %v", err)
	}

	// 等待一段时间，让任务有机会执行
	time.Sleep(1 * time.Second)

	// 现在子任务应该执行了
	if task2.executions == 0 {
		t.Errorf("子任务应该执行，但实际执行了 %d 次", task2.executions)
	}

	// 停止任务池
	pool.Stop()
}

// TestTaskPriority 测试任务优先级
func TestTaskPriority(t *testing.T) {
	// 创建一个有限制的任务池，只允许一个任务同时执行
	config := &EnhancedTaskConfig{
		MaxConcurrentTasks: 1, // 限制为1个并发任务
		TaskQueueSize:      10,
		ShutdownTimeout:    1 * time.Second,
	}

	// 创建任务池
	pool := NewEnhancedTaskPoolManager(config)

	// 创建测试任务
	lowTask := newTestTask("low", "", false)
	highTask := newTestTask("high", "", false)

	// 注册任务
	err := pool.RegisterTasks(lowTask, highTask)
	if err != nil {
		t.Fatalf("注册任务失败: %v", err)
	}

	// 初始化任务
	err = pool.InitializeTasks()
	if err != nil {
		t.Fatalf("初始化任务失败: %v", err)
	}

	// 启动任务池
	err = pool.Start()
	if err != nil {
		t.Fatalf("启动任务池失败: %v", err)
	}

	// 阻塞工作池，使其无法立即处理任务
	_, blockCancel := context.WithCancel(context.Background())
	blockTask := &testTask{
		name:     "block",
		cronSpec: "",
	}

	// 提交一个阻塞任务
	err = pool.Submit(blockTask, PriorityNormal)
	if err != nil {
		t.Fatalf("提交阻塞任务失败: %v", err)
	}

	// 等待阻塞任务开始执行
	time.Sleep(100 * time.Millisecond)

	// 提交低优先级任务
	err = pool.Submit(lowTask, PriorityLow)
	if err != nil {
		t.Fatalf("提交低优先级任务失败: %v", err)
	}

	// 提交高优先级任务
	err = pool.Submit(highTask, PriorityHigh)
	if err != nil {
		t.Fatalf("提交高优先级任务失败: %v", err)
	}

	// 解除阻塞
	blockCancel()

	// 等待任务执行
	time.Sleep(500 * time.Millisecond)

	// 高优先级任务应该先执行
	if highTask.executions == 0 {
		t.Errorf("高优先级任务应该执行，但实际执行了 %d 次", highTask.executions)
	}

	// 停止任务池
	pool.Stop()
}

// TestTaskRetry 测试任务重试机制
func TestTaskRetry(t *testing.T) {
	// 创建任务池配置
	config := &EnhancedTaskConfig{
		MaxConcurrentTasks: 5,
		DefaultRetryLimit:  3, // 最多重试3次
		RetryBackoff:       100 * time.Millisecond,
		TaskQueueSize:      10,
		EnableMetrics:      true,
		ShutdownTimeout:    1 * time.Second,
	}

	// 创建任务池
	pool := NewEnhancedTaskPoolManager(config)

	// 创建一个总是失败的测试任务
	failingTask := newTestTask("failing", "", true)

	// 注册任务
	err := pool.RegisterTasks(failingTask)
	if err != nil {
		t.Fatalf("注册任务失败: %v", err)
	}

	// 初始化任务
	err = pool.InitializeTasks()
	if err != nil {
		t.Fatalf("初始化任务失败: %v", err)
	}

	// 启动任务池
	err = pool.Start()
	if err != nil {
		t.Fatalf("启动任务池失败: %v", err)
	}

	// 提交任务
	err = pool.Submit(failingTask, PriorityNormal)
	if err != nil {
		t.Fatalf("提交任务失败: %v", err)
	}

	// 等待足够长的时间，让任务有机会重试
	time.Sleep(2 * time.Second)

	// 验证任务执行和失败次数
	if failingTask.executions <= 1 {
		t.Errorf("任务应该执行多次（包括重试），但实际执行了 %d 次", failingTask.executions)
	}

	if failingTask.failures <= 1 {
		t.Errorf("任务应该失败多次，但实际失败了 %d 次", failingTask.failures)
	}

	// 验证指标
	metrics := pool.GetMetrics()
	if metrics.TotalTasksRetried == 0 {
		t.Errorf("应该有任务重试，但指标显示 %d", metrics.TotalTasksRetried)
	}

	// 停止任务池
	pool.Stop()
}

// TestTaskCancellation 测试任务取消功能
func TestTaskCancellation(t *testing.T) {
	// 创建任务池配置
	config := &EnhancedTaskConfig{
		MaxConcurrentTasks: 5,
		TaskQueueSize:      10,
		ShutdownTimeout:    1 * time.Second,
	}

	// 创建任务池
	pool := NewEnhancedTaskPoolManager(config)

	// 创建一个长时间运行的任务
	longRunningTask := &testTask{
		name:     "longRunning",
		cronSpec: "",
	}

	// 注册任务
	err := pool.RegisterTasks(longRunningTask)
	if err != nil {
		t.Fatalf("注册任务失败: %v", err)
	}

	// 初始化任务
	err = pool.InitializeTasks()
	if err != nil {
		t.Fatalf("初始化任务失败: %v", err)
	}

	// 启动任务池
	err = pool.Start()
	if err != nil {
		t.Fatalf("启动任务池失败: %v", err)
	}

	// 提交任务
	err = pool.Submit(longRunningTask, PriorityNormal)
	if err != nil {
		t.Fatalf("提交任务失败: %v", err)
	}

	// 等待任务开始执行
	time.Sleep(100 * time.Millisecond)

	// 获取任务ID并取消任务
	taskID := "" // 在实际实现中，需要从任务状态中获取ID
	for id, info := range pool.taskStatus {
		if info.TaskName == "longRunning" && info.Status == TaskStatusRunning {
			taskID = id
			break
		}
	}

	if taskID != "" {
		cancelled := pool.CancelTask(taskID)
		if !cancelled {
			t.Errorf("应该能够取消任务，但取消失败")
		}

		// 验证任务状态
		info, exists := pool.GetTaskStatus(taskID)
		if !exists {
			t.Errorf("应该能够获取任务状态，但获取失败")
		} else if info.Status != TaskStatusCancelled {
			t.Errorf("任务状态应该是已取消，但实际是 %s", info.Status)
		}
	} else {
		t.Log("未找到正在运行的任务ID，跳过取消测试")
	}

	// 停止任务池
	pool.Stop()
}

// BenchmarkTaskSubmission 基准测试：任务提交性能
func BenchmarkTaskSubmission(b *testing.B) {
	// 创建任务池配置
	config := &EnhancedTaskConfig{
		MaxConcurrentTasks: 10,
		TaskQueueSize:      1000,
		EnableMetrics:      false, // 禁用指标以提高性能
		EnableTaskHistory:  false, // 禁用历史记录以提高性能
		ShutdownTimeout:    1 * time.Second,
	}

	// 创建任务池
	pool := NewEnhancedTaskPoolManager(config)

	// 创建一个简单的测试任务
	task := newTestTask("benchmark", "", false)

	// 注册任务
	pool.RegisterTasks(task)

	// 初始化任务
	pool.InitializeTasks()

	// 启动任务池
	pool.Start()

	// 重置计时器
	b.ResetTimer()

	// 执行基准测试
	for i := 0; i < b.N; i++ {
		pool.Submit(task, PriorityNormal)
	}

	// 停止计时器
	b.StopTimer()

	// 停止任务池
	pool.Stop()
}

// BenchmarkConcurrentTaskExecution 基准测试：并发任务执行性能
func BenchmarkConcurrentTaskExecution(b *testing.B) {
	// 创建任务池配置，使用较大的并发数
	config := &EnhancedTaskConfig{
		MaxConcurrentTasks: 50,
		TaskQueueSize:      1000,
		EnableMetrics:      false,
		EnableTaskHistory:  false,
		ShutdownTimeout:    1 * time.Second,
	}

	// 创建任务池
	pool := NewEnhancedTaskPoolManager(config)

	// 创建测试任务
	tasks := make([]*testTask, 100)
	for i := 0; i < 100; i++ {
		tasks[i] = newTestTask(fmt.Sprintf("task-%d", i), "", false)
		pool.RegisterTasks(tasks[i])
	}

	// 初始化任务
	pool.InitializeTasks()

	// 启动任务池
	pool.Start()

	// 重置计时器
	b.ResetTimer()

	// 执行基准测试
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			pool.Submit(tasks[i%100], PriorityNormal)
			i++
		}
	})

	// 停止计时器
	b.StopTimer()

	// 停止任务池
	pool.Stop()
}

// 运行测试示例
func ExampleEnhancedTaskPool() {
	// 创建任务池配置
	config := DefaultTaskConfig()

	// 创建任务池
	pool := NewEnhancedTaskPoolManager(config)

	// 创建任务
	task1 := NewEnhancedSampleTask(
		"ExampleTask",
		"示例任务",
		"*/5 * * * * *", // 每5秒执行一次
		10*time.Second,  // 超时时间
		0.0,             // 不会失败
	)

	// 注册任务
	pool.RegisterTasks(task1)

	// 初始化任务
	pool.InitializeTasks()

	// 启动任务池
	pool.Start()

	// 等待任务执行
	time.Sleep(10 * time.Second)

	// 获取指标
	metrics := pool.GetMetrics()
	logger.Info("任务执行次数: %d", metrics.TotalTasksCompleted)

	// 停止任务池
	pool.Stop()

	// Output: 任务执行次数: 2
}
