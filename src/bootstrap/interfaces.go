package bootstrap

import (
	"VectorSphere/src/library/logger"
	PoolLib "VectorSphere/src/library/pool"
	scheduler2 "VectorSphere/src/scheduler"
	"context"
	"time"
)

// RunnableService 定义了可由任务池管理的服务的基本行为
type RunnableService interface {
	GetName() string                                        // 获取服务名称
	GetDependencies() []string                              // 获取服务依赖的任务名称列表
	Start(ctx context.Context) error                        // 启动服务
	Stop(ctx context.Context) error                         // 停止服务
	GetTaskSpec() *scheduler2.TaskSpec                      // 获取任务规格，用于高级调度（如重试、超时）
	ToPoolTask() *PoolLib.Queue                             // 转换为与 PoolLib.Task 兼容的任务
	SetTaskPoolManager(manager *scheduler2.TaskPoolManager) // 设置任务池管理器
}

// ServiceTask 是 RunnableService 的一个包装器，使其与 PoolLib.Task 兼容
type ServiceTask struct {
	service RunnableService
	manager *scheduler2.TaskPoolManager // 添加 manager 字段
}

// NewServiceTask 创建一个新的 ServiceTask
func NewServiceTask(service RunnableService, manager *scheduler2.TaskPoolManager) *ServiceTask {
	service.SetTaskPoolManager(manager) // 在创建时设置 manager
	return &ServiceTask{
		service: service,
		manager: manager,
	}
}

// Execute 执行服务启动逻辑
func (st *ServiceTask) Execute(ctx context.Context) error {
	// 在这里可以根据需要访问 st.manager
	return st.service.Start(ctx)
}

// GetName 返回任务名称
func (st *ServiceTask) GetName() string {
	return st.service.GetName()
}

// GetDependencies 返回任务依赖
func (st *ServiceTask) GetDependencies() []string {
	return st.service.GetDependencies()
}

// GetRetries 返回任务重试次数
func (st *ServiceTask) GetRetries() int {
	if spec := st.service.GetTaskSpec(); spec != nil {
		return spec.RetryCount
	}
	return 0 // 默认不重试
}

// GetTimeout 返回任务超时时间
func (st *ServiceTask) GetTimeout() time.Duration {
	if spec := st.service.GetTaskSpec(); spec != nil {
		return spec.Timeout
	}
	return 0 // 默认无超时
}

// OnError 是一个可选方法，用于处理任务执行错误
func (st *ServiceTask) OnError(err error) {
	logger.Error("Task %s failed: %v", st.GetName(), err)
	// 这里可以添加更复杂的错误处理逻辑，例如通知、回滚等
}

// ToPoolTask 将 ServiceTask 转换为 PoolLib.Queue
func (st *ServiceTask) ToPoolTask() *PoolLib.Queue {
	startWrapper := func() error {
		logger.Info("ServiceTask for %s is about to run Execute()", st.GetName())
		// 创建一个新的上下文，或者使用背景上下文
		ctx := context.Background()
		err := st.Execute(ctx)
		if err != nil {
			logger.Error("ServiceTask for %s failed to Execute: %v", st.GetName(), err)
			// 调用错误处理方法
			st.OnError(err)
		}
		return err
	}

	return PoolLib.QueryInit(st.GetName(), startWrapper)
}
