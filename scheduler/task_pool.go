package scheduler

import (
	"fmt"
	"sync"

	"seetaSearch/library/log"

	"github.com/robfig/cron/v3"
)

// ScheduledTask 定义了定时任务需要实现的接口
type ScheduledTask interface {
	Run() error          // 执行任务的逻辑
	GetCronSpec() string // 获取任务的Cron表达式
	GetName() string     // 获取任务的名称，用于日志和管理
	Init() error         // 可选：任务初始化逻辑
	Stop() error         // 可选：任务停止前的清理逻辑
}

// TaskPoolManager 管理一组定时任务
type TaskPoolManager struct {
	tasks     map[string]ScheduledTask // 存储已注册的任务，key为任务名称
	cron      *cron.Cron               // Cron调度器实例
	mu        sync.RWMutex
	stopCh    chan struct{} // 用于优雅停止所有任务的信号
	IsRunning bool
}

// NewTaskPoolManager 创建一个新的任务池管理器
func NewTaskPoolManager() *TaskPoolManager {
	return &TaskPoolManager{
		tasks:  make(map[string]ScheduledTask),
		stopCh: make(chan struct{}),
	}
}

// RegisterTasks 批量注册一个或多个定时任务
func (pm *TaskPoolManager) RegisterTasks(tasks ...ScheduledTask) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.IsRunning {
		return fmt.Errorf("任务池已在运行，无法注册新任务")
	}

	for _, task := range tasks {
		taskName := task.GetName()
		if _, exists := pm.tasks[taskName]; exists {
			log.Warning("任务 '%s' 已经注册，将被覆盖", taskName)
		}
		pm.tasks[taskName] = task
		log.Info("任务 '%s' 已注册，Cron表达式: %s", taskName, task.GetCronSpec())
	}
	return nil
}

// InitializeTasks 初始化所有已注册的任务（如果它们实现了Init方法）
func (pm *TaskPoolManager) InitializeTasks() error {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	log.Info("开始初始化所有已注册的任务...")
	for name, task := range pm.tasks {
		// 检查是否实现了 Init() error 方法
		if initializer, ok := task.(interface{ Init() error }); ok {
			if err := initializer.Init(); err != nil {
				log.Error("初始化任务 '%s' 失败: %v", name, err)
				return fmt.Errorf("初始化任务 '%s' 失败: %w", name, err)
			}
			log.Info("任务 '%s' 初始化成功", name)
		} else {
			log.Info("任务 '%s' 没有 Init 方法，跳过初始化", name)
		}
	}
	log.Info("所有已注册任务初始化完成。")
	return nil
}

// Start 启动任务池，开始执行所有定时任务
func (pm *TaskPoolManager) Start() error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.IsRunning {
		return fmt.Errorf("任务池已经在运行中")
	}

	if len(pm.tasks) == 0 {
		log.Info("没有已注册的任务，任务池未启动。")
		return nil
	}

	// 初始化cron调度器，可以添加秒级支持（如果需要）
	// pm.cron = cron.New(cron.WithSeconds())
	pm.cron = cron.New()

	log.Info("开始将任务添加到调度器...")
	for name, task := range pm.tasks {
		// 使用闭包来捕获正确的task实例
		currentTask := task
		currentName := name
		entryID, err := pm.cron.AddFunc(currentTask.GetCronSpec(), func() {
			log.Info("开始执行定时任务: %s", currentName)
			if err := currentTask.Run(); err != nil {
				log.Error("执行定时任务 '%s' 失败: %v", currentName, err)
			} else {
				log.Info("定时任务 '%s' 执行完成", currentName)
			}
		})
		if err != nil {
			log.Error("添加任务 '%s' 到调度器失败: %v", currentName, err)
			// 如果一个任务添加失败，可以选择停止所有已启动的，或者继续尝试其他任务
			// 这里选择停止并返回错误
			pm.cron.Stop() // 停止已经部分启动的cron
			return fmt.Errorf("添加任务 '%s' 到调度器失败: %w", currentName, err)
		}
		log.Info("任务 '%s' (EntryID: %d) 已成功添加到调度器，Cron: %s", currentName, entryID, currentTask.GetCronSpec())
	}

	pm.cron.Start()
	pm.IsRunning = true
	log.Info("任务池已启动，所有定时任务开始调度执行。")

	// 启动一个goroutine监听停止信号
	go func() {
		<-pm.stopCh
		log.Info("接收到停止信号，开始停止任务池...")
		ctx := pm.cron.Stop() // 优雅地停止cron调度器，等待正在执行的任务完成
		<-ctx.Done()          // 等待所有任务完成
		log.Info("Cron调度器已停止。")

		// 可选：调用所有任务的Stop方法
		pm.mu.RLock()
		for name, task := range pm.tasks {
			if stopper, ok := task.(interface{ Stop() error }); ok {
				if err := stopper.Stop(); err != nil {
					log.Error("停止任务 '%s' 时发生错误: %v", name, err)
				}
			}
		}
		pm.mu.RUnlock()
		log.Info("所有任务已尝试停止，任务池已完全关闭。")
		pm.mu.Lock()
		pm.IsRunning = false
		pm.mu.Unlock()
	}()

	return nil
}

// Stop 停止任务池中的所有定时任务
func (pm *TaskPoolManager) Stop() {
	pm.mu.Lock()
	if !pm.IsRunning {
		pm.mu.Unlock()
		log.Info("任务池未运行，无需停止。")
		return
	}
	// 避免重复关闭stopCh
	// select {
	// case <-pm.stopCh:
	// 	// 已经关闭或正在关闭
	// 	pm.mu.Unlock()
	// 	return
	// default:
	// }
	pm.mu.Unlock() // Unlock before closing channel to avoid deadlock with the goroutine listening on stopCh

	log.Info("正在发送停止信号给任务池...")
	close(pm.stopCh) // 发送停止信号
}

// GetTask 获取已注册的特定任务
func (pm *TaskPoolManager) GetTask(name string) (ScheduledTask, bool) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	task, exists := pm.tasks[name]
	return task, exists
}

// ListTasks 列出所有已注册的任务名称
func (pm *TaskPoolManager) ListTasks() []string {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	names := make([]string, 0, len(pm.tasks))
	for name := range pm.tasks {
		names = append(names, name)
	}
	return names
}
