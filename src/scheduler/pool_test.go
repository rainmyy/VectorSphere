package scheduler

import (
	"VectorSphere/src/library/logger"
	"time"
)

func test() {
	// 创建任务池管理器
	manager := NewTaskPoolManager()

	// 创建任务实例
	task1 := &MySampleTask{Name: "ReportGenerator"}
	task2 := &MySampleTask{Name: "DataCleanup"}
	task3, _ := NewPQTrainingScheduler(nil)
	// 注册任务
	if err := manager.RegisterTasks(task1, task2, task3); err != nil {
		logger.Fatal("注册任务失败: %v", err)
	}

	// 初始化任务 (可选)
	if err := manager.InitializeTasks(); err != nil {
		logger.Fatal("初始化任务失败: %v", err)
	}

	// 启动任务池
	if err := manager.Start(); err != nil {
		logger.Fatal("启动任务池失败: %v", err)
	}

	logger.Info("任务池已启动。按 CTRL+C 退出。")

	// 模拟程序运行一段时间
	time.Sleep(30 * time.Second)

	// 停止任务池
	logger.Info("准备停止任务池...")
	manager.Stop()

	// 等待一段时间确保所有任务都已停止（或者在Stop中加入等待逻辑）
	time.Sleep(5 * time.Second)
	logger.Info("程序退出。")
}
