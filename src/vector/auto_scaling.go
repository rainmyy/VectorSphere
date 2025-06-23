package vector

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// AutoScaler 自动扩缩容器
type AutoScaler struct {
	metrics         *MetricsCollector
	scalingPolicies []*ScalingPolicy
	resourceManager *ResourceManager
	alertManager    *AlertManager
	mu              sync.RWMutex
	running         bool
	stopCh          chan struct{}
}

// ScalingPolicy 扩缩容策略
type ScalingPolicy struct {
	Name           string
	MetricType     MetricType
	Threshold      float64
	Direction      ScalingDirection
	CooldownPeriod time.Duration
	MinInstances   int
	MaxInstances   int
	ScalingFactor  float64
	LastScaleTime  time.Time
	Enabled        bool
}

// MetricType 指标类型
type MetricType string

const (
	CPUUtilization    MetricType = "cpu_utilization"
	MemoryUtilization MetricType = "memory_utilization"
	QueryLatency      MetricType = "query_latency"
	QueryThroughput   MetricType = "query_throughput"
	ErrorRate         MetricType = "error_rate"
	QueueLength       MetricType = "queue_length"
	GPUUtilization    MetricType = "gpu_utilization"
)

// ScalingDirection 扩缩容方向
type ScalingDirection string

const (
	ScaleUp   ScalingDirection = "scale_up"
	ScaleDown ScalingDirection = "scale_down"
)

// ResourceManager 资源管理器
type ResourceManager struct {
	currentInstances int
	instancePool     []*Instance
	mu               sync.RWMutex
}

// Instance 实例
type Instance struct {
	ID      string
	Type    InstanceType
	Status  InstanceStatus
	CPU     float64
	Memory  float64
	GPU     bool
	Created time.Time
	mu      sync.RWMutex
}

// InstanceType 实例类型
type InstanceType string

const (
	ComputeInstance InstanceType = "compute"
	StorageInstance InstanceType = "storage"
	ProxyInstance   InstanceType = "proxy"
	GPUInstance     InstanceType = "gpu"
)

// InstanceStatus 实例状态
type InstanceStatus string

const (
	InstancePending     InstanceStatus = "pending"
	InstanceRunning     InstanceStatus = "running"
	InstanceTerminating InstanceStatus = "terminating"
	InstanceTerminated  InstanceStatus = "terminated"
)

// MetricsCollector 指标收集器
type MetricsCollector struct {
	metrics map[MetricType]*MetricValue
	mu      sync.RWMutex
}

// MetricValue 指标值
type MetricValue struct {
	Value     float64
	Timestamp time.Time
	Tags      map[string]string
}

// ScalingEvent 扩缩容事件
type ScalingEvent struct {
	Timestamp    time.Time
	PolicyName   string
	MetricType   MetricType
	MetricValue  float64
	Threshold    float64
	Direction    ScalingDirection
	OldInstances int
	NewInstances int
	Reason       string
}

// NewAutoScaler 创建自动扩缩容器
func NewAutoScaler(alertManager *AlertManager) *AutoScaler {
	return &AutoScaler{
		metrics:         NewMetricsCollector(),
		scalingPolicies: make([]*ScalingPolicy, 0),
		resourceManager: NewResourceManager(),
		alertManager:    alertManager,
		stopCh:          make(chan struct{}),
	}
}

// NewMetricsCollector 创建指标收集器
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		metrics: make(map[MetricType]*MetricValue),
	}
}

// NewResourceManager 创建资源管理器
func NewResourceManager() *ResourceManager {
	return &ResourceManager{
		currentInstances: 1, // 默认一个实例
		instancePool:     make([]*Instance, 0),
	}
}

// AddScalingPolicy 添加扩缩容策略
func (as *AutoScaler) AddScalingPolicy(policy *ScalingPolicy) {
	as.mu.Lock()
	defer as.mu.Unlock()
	as.scalingPolicies = append(as.scalingPolicies, policy)
}

// Start 启动自动扩缩容
func (as *AutoScaler) Start(ctx context.Context) {
	as.mu.Lock()
	if as.running {
		as.mu.Unlock()
		return
	}
	as.running = true
	as.mu.Unlock()

	ticker := time.NewTicker(30 * time.Second) // 每30秒检查一次
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			as.evaluateScalingPolicies()
		case <-ctx.Done():
			as.Stop()
			return
		case <-as.stopCh:
			return
		}
	}
}

// Stop 停止自动扩缩容
func (as *AutoScaler) Stop() {
	as.mu.Lock()
	defer as.mu.Unlock()

	if !as.running {
		return
	}

	as.running = false
	close(as.stopCh)
}

// evaluateScalingPolicies 评估扩缩容策略
func (as *AutoScaler) evaluateScalingPolicies() {
	as.mu.RLock()
	policies := make([]*ScalingPolicy, len(as.scalingPolicies))
	copy(policies, as.scalingPolicies)
	as.mu.RUnlock()

	for _, policy := range policies {
		if !policy.Enabled {
			continue
		}

		// 检查冷却期
		if time.Since(policy.LastScaleTime) < policy.CooldownPeriod {
			continue
		}

		// 获取当前指标值
		metricValue := as.metrics.GetMetric(policy.MetricType)
		if metricValue == nil {
			continue
		}

		// 评估是否需要扩缩容
		if as.shouldScale(policy, metricValue.Value) {
			as.executeScaling(policy, metricValue.Value)
		}
	}
}

// shouldScale 判断是否应该扩缩容
func (as *AutoScaler) shouldScale(policy *ScalingPolicy, currentValue float64) bool {
	switch policy.Direction {
	case ScaleUp:
		return currentValue > policy.Threshold
	case ScaleDown:
		return currentValue < policy.Threshold
	default:
		return false
	}
}

// executeScaling 执行扩缩容
func (as *AutoScaler) executeScaling(policy *ScalingPolicy, metricValue float64) {
	currentInstances := as.resourceManager.GetCurrentInstances()
	var newInstances int

	switch policy.Direction {
	case ScaleUp:
		newInstances = int(float64(currentInstances) * (1 + policy.ScalingFactor))
		if newInstances > policy.MaxInstances {
			newInstances = policy.MaxInstances
		}
	case ScaleDown:
		newInstances = int(float64(currentInstances) * (1 - policy.ScalingFactor))
		if newInstances < policy.MinInstances {
			newInstances = policy.MinInstances
		}
	}

	if newInstances == currentInstances {
		return
	}

	// 执行扩缩容操作
	err := as.resourceManager.ScaleInstances(newInstances)
	if err != nil {
		// 发送告警
		// 记录扩缩容失败日志
		fmt.Printf("[ERROR] 扩缩容失败: %v\n", err)
		return
	}

	// 记录扩缩容事件
	event := &ScalingEvent{
		Timestamp:    time.Now(),
		PolicyName:   policy.Name,
		MetricType:   policy.MetricType,
		MetricValue:  metricValue,
		Threshold:    policy.Threshold,
		Direction:    policy.Direction,
		OldInstances: currentInstances,
		NewInstances: newInstances,
		Reason:       fmt.Sprintf("指标 %s 值 %.2f 超过阈值 %.2f", policy.MetricType, metricValue, policy.Threshold),
	}

	// 更新策略的最后扩缩容时间
	policy.LastScaleTime = time.Now()

	// 记录扩缩容成功日志
	fmt.Printf("[INFO] 自动扩缩容成功: %s\n", event.Reason)
}

// UpdateMetric 更新指标
func (mc *MetricsCollector) UpdateMetric(metricType MetricType, value float64, tags map[string]string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.metrics[metricType] = &MetricValue{
		Value:     value,
		Timestamp: time.Now(),
		Tags:      tags,
	}
}

// GetMetric 获取指标
func (mc *MetricsCollector) GetMetric(metricType MetricType) *MetricValue {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	return mc.metrics[metricType]
}

// GetCurrentInstances 获取当前实例数
func (rm *ResourceManager) GetCurrentInstances() int {
	rm.mu.RLock()
	defer rm.mu.RUnlock()
	return rm.currentInstances
}

// ScaleInstances 扩缩容实例
func (rm *ResourceManager) ScaleInstances(targetInstances int) error {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	currentInstances := rm.currentInstances
	if targetInstances == currentInstances {
		return nil
	}

	if targetInstances > currentInstances {
		// 扩容
		for i := currentInstances; i < targetInstances; i++ {
			instance := &Instance{
				ID:      fmt.Sprintf("instance-%d", i),
				Type:    ComputeInstance,
				Status:  InstancePending,
				CPU:     2.0,
				Memory:  4.0,
				GPU:     false,
				Created: time.Now(),
			}

			// 模拟实例启动
			go func(inst *Instance) {
				time.Sleep(30 * time.Second) // 模拟启动时间
				inst.mu.Lock()
				inst.Status = InstanceRunning
				inst.mu.Unlock()
			}(instance)

			rm.instancePool = append(rm.instancePool, instance)
		}
	} else {
		// 缩容
		instancesToRemove := currentInstances - targetInstances
		for i := 0; i < instancesToRemove && i < len(rm.instancePool); i++ {
			instance := rm.instancePool[len(rm.instancePool)-1-i]
			instance.mu.Lock()
			instance.Status = InstanceTerminating
			instance.mu.Unlock()

			// 模拟实例终止
			go func(inst *Instance) {
				time.Sleep(10 * time.Second) // 模拟终止时间
				inst.mu.Lock()
				inst.Status = InstanceTerminated
				inst.mu.Unlock()
			}(instance)
		}

		// 从池中移除实例
		rm.instancePool = rm.instancePool[:len(rm.instancePool)-instancesToRemove]
	}

	rm.currentInstances = targetInstances
	return nil
}

// GetInstances 获取所有实例
func (rm *ResourceManager) GetInstances() []*Instance {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	instances := make([]*Instance, len(rm.instancePool))
	copy(instances, rm.instancePool)
	return instances
}

// GetRunningInstances 获取运行中的实例
func (rm *ResourceManager) GetRunningInstances() []*Instance {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	runningInstances := make([]*Instance, 0)
	for _, instance := range rm.instancePool {
		instance.mu.RLock()
		if instance.Status == InstanceRunning {
			runningInstances = append(runningInstances, instance)
		}
		instance.mu.RUnlock()
	}

	return runningInstances
}

// CreateDefaultScalingPolicies 创建默认扩缩容策略
func CreateDefaultScalingPolicies() []*ScalingPolicy {
	return []*ScalingPolicy{
		{
			Name:           "cpu_scale_up",
			MetricType:     CPUUtilization,
			Threshold:      70.0, // CPU使用率超过70%时扩容
			Direction:      ScaleUp,
			CooldownPeriod: 5 * time.Minute,
			MinInstances:   1,
			MaxInstances:   10,
			ScalingFactor:  0.5, // 扩容50%
			Enabled:        true,
		},
		{
			Name:           "cpu_scale_down",
			MetricType:     CPUUtilization,
			Threshold:      30.0, // CPU使用率低于30%时缩容
			Direction:      ScaleDown,
			CooldownPeriod: 10 * time.Minute,
			MinInstances:   1,
			MaxInstances:   10,
			ScalingFactor:  0.3, // 缩容30%
			Enabled:        true,
		},
		{
			Name:           "latency_scale_up",
			MetricType:     QueryLatency,
			Threshold:      200.0, // 查询延迟超过200ms时扩容
			Direction:      ScaleUp,
			CooldownPeriod: 3 * time.Minute,
			MinInstances:   1,
			MaxInstances:   15,
			ScalingFactor:  0.6, // 扩容60%
			Enabled:        true,
		},
		{
			Name:           "memory_scale_up",
			MetricType:     MemoryUtilization,
			Threshold:      80.0, // 内存使用率超过80%时扩容
			Direction:      ScaleUp,
			CooldownPeriod: 5 * time.Minute,
			MinInstances:   1,
			MaxInstances:   12,
			ScalingFactor:  0.4, // 扩容40%
			Enabled:        true,
		},
	}
}
