package acceler

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// RecoveryStrategy 恢复策略
type RecoveryStrategy int

const (
	RecoveryStrategyRestart RecoveryStrategy = iota
	RecoveryStrategyReinitialize
	RecoveryStrategyFallback
	RecoveryStrategyDisable
)

// String 返回恢复策略的字符串表示
func (rs RecoveryStrategy) String() string {
	switch rs {
	case RecoveryStrategyRestart:
		return "重启"
	case RecoveryStrategyReinitialize:
		return "重新初始化"
	case RecoveryStrategyFallback:
		return "回退"
	case RecoveryStrategyDisable:
		return "禁用"
	default:
		return "未知"
	}
}

// RecoveryAction 恢复动作
type RecoveryAction struct {
	AcceleratorType string           `json:"accelerator_type"`
	Strategy        RecoveryStrategy `json:"strategy"`
	Timestamp       time.Time        `json:"timestamp"`
	Reason          string           `json:"reason"`
	Success         bool             `json:"success"`
	ErrorMessage    string           `json:"error_message,omitempty"`
}

// RecoveryConfig 恢复配置
type RecoveryConfig struct {
	MaxRetries          int           `json:"max_retries"`
	RetryInterval       time.Duration `json:"retry_interval"`
	HealthCheckInterval time.Duration `json:"health_check_interval"`
	AutoRecoveryEnabled bool          `json:"auto_recovery_enabled"`
	FallbackEnabled     bool          `json:"fallback_enabled"`
}

// DefaultRecoveryConfig 默认恢复配置
func DefaultRecoveryConfig() *RecoveryConfig {
	return &RecoveryConfig{
		MaxRetries:          3,
		RetryInterval:       30 * time.Second,
		HealthCheckInterval: 60 * time.Second,
		AutoRecoveryEnabled: true,
		FallbackEnabled:     true,
	}
}

// RecoveryManager 恢复管理器
type RecoveryManager struct {
	mutex           sync.RWMutex
	hardwareManager *HardwareManager
	config          *RecoveryConfig
	recoveryHistory []RecoveryAction
	retryCount      map[string]int
	lastRecovery    map[string]time.Time
	stopCh          chan struct{}
	running         bool
}

// NewRecoveryManager 创建新的恢复管理器
func NewRecoveryManager(hm *HardwareManager, config *RecoveryConfig) *RecoveryManager {
	if config == nil {
		config = DefaultRecoveryConfig()
	}

	return &RecoveryManager{
		hardwareManager: hm,
		config:          config,
		retryCount:      make(map[string]int),
		lastRecovery:    make(map[string]time.Time),
		stopCh:          make(chan struct{}),
	}
}

// Start 启动恢复管理器
func (rm *RecoveryManager) Start() {
	rm.mutex.Lock()
	defer rm.mutex.Unlock()

	if rm.running || !rm.config.AutoRecoveryEnabled {
		return
	}

	rm.running = true
	go rm.recoveryLoop()
}

// Stop 停止恢复管理器
func (rm *RecoveryManager) Stop() {
	rm.mutex.Lock()
	defer rm.mutex.Unlock()

	if !rm.running {
		return
	}

	rm.running = false
	close(rm.stopCh)
}

// recoveryLoop 恢复循环
func (rm *RecoveryManager) recoveryLoop() {
	ticker := time.NewTicker(rm.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			rm.checkAndRecover()
		case <-rm.stopCh:
			return
		}
	}
}

// checkAndRecover 检查并恢复异常的加速器
func (rm *RecoveryManager) checkAndRecover() {
	reports := rm.hardwareManager.GetAllHealthReports()
	if reports == nil {
		return
	}

	for acceleratorType, report := range reports {
		if report.Status == HealthStatusCritical {
			rm.attemptRecovery(acceleratorType, report)
		}
	}
}

// attemptRecovery 尝试恢复指定的加速器
func (rm *RecoveryManager) attemptRecovery(acceleratorType string, report *HealthReport) {
	rm.mutex.Lock()
	defer rm.mutex.Unlock()

	// 检查是否超过最大重试次数
	if rm.retryCount[acceleratorType] >= rm.config.MaxRetries {
		log.Printf("加速器 %s 已达到最大重试次数，停止恢复尝试", acceleratorType)
		return
	}

	// 检查距离上次恢复的时间间隔
	if lastRecovery, exists := rm.lastRecovery[acceleratorType]; exists {
		if time.Since(lastRecovery) < rm.config.RetryInterval {
			return // 还未到重试时间
		}
	}

	// 确定恢复策略
	strategy := rm.determineRecoveryStrategy(acceleratorType, report)

	// 执行恢复
	action := RecoveryAction{
		AcceleratorType: acceleratorType,
		Strategy:        strategy,
		Timestamp:       time.Now(),
		Reason:          report.Message,
	}

	err := rm.executeRecovery(acceleratorType, strategy)
	if err != nil {
		action.Success = false
		action.ErrorMessage = err.Error()
		rm.retryCount[acceleratorType]++
		log.Printf("恢复加速器 %s 失败: %v", acceleratorType, err)
	} else {
		action.Success = true
		rm.retryCount[acceleratorType] = 0 // 重置重试计数
		log.Printf("成功恢复加速器 %s，使用策略: %s", acceleratorType, strategy)
	}

	rm.lastRecovery[acceleratorType] = action.Timestamp
	rm.recoveryHistory = append(rm.recoveryHistory, action)

	// 限制历史记录长度
	if len(rm.recoveryHistory) > 100 {
		rm.recoveryHistory = rm.recoveryHistory[1:]
	}
}

// determineRecoveryStrategy 确定恢复策略
func (rm *RecoveryManager) determineRecoveryStrategy(acceleratorType string, report *HealthReport) RecoveryStrategy {
	retryCount := rm.retryCount[acceleratorType]

	// 根据重试次数和错误类型确定策略
	switch {
	case retryCount == 0:
		// 第一次尝试：重新初始化
		return RecoveryStrategyReinitialize
	case retryCount == 1:
		// 第二次尝试：重启
		return RecoveryStrategyRestart
	case retryCount == 2 && rm.config.FallbackEnabled:
		// 第三次尝试：回退到其他加速器
		return RecoveryStrategyFallback
	default:
		// 最后尝试：禁用
		return RecoveryStrategyDisable
	}
}

// executeRecovery 执行恢复操作
func (rm *RecoveryManager) executeRecovery(acceleratorType string, strategy RecoveryStrategy) error {
	switch strategy {
	case RecoveryStrategyReinitialize:
		return rm.reinitializeAccelerator(acceleratorType)
	case RecoveryStrategyRestart:
		return rm.restartAccelerator(acceleratorType)
	case RecoveryStrategyFallback:
		return rm.fallbackAccelerator(acceleratorType)
	case RecoveryStrategyDisable:
		return rm.disableAccelerator(acceleratorType)
	default:
		return fmt.Errorf("未知的恢复策略: %v", strategy)
	}
}

// reinitializeAccelerator 重新初始化加速器
func (rm *RecoveryManager) reinitializeAccelerator(acceleratorType string) error {
	acc, ok := rm.hardwareManager.GetAccelerator(acceleratorType)
	if !ok {
		return fmt.Errorf("找不到加速器: %s", acceleratorType)
	}

	// 关闭现有加速器
	if err := acc.Shutdown(); err != nil {
		log.Printf("关闭加速器 %s 时出错: %v", acceleratorType, err)
	}

	// 重新初始化
	if err := acc.Initialize(); err != nil {
		return fmt.Errorf("重新初始化加速器 %s 失败: %v", acceleratorType, err)
	}

	return nil
}

// restartAccelerator 重启加速器
func (rm *RecoveryManager) restartAccelerator(acceleratorType string) error {
	// 移除现有加速器
	err := rm.hardwareManager.RemoveAccelerator(acceleratorType)
	if err != nil {
		return err
	}

	// 尝试重新注册加速器
	err = rm.hardwareManager.ReRegisterAccelerator(acceleratorType)
	if err != nil {
		return fmt.Errorf("重新注册加速器 %s 失败: %v", acceleratorType, err)
	}
	return nil
}

// fallbackAccelerator 回退到其他加速器
func (rm *RecoveryManager) fallbackAccelerator(acceleratorType string) error {
	// 检查主加速器是否已存在且健康
	if rm.hardwareManager.healthMonitor.IsHealthy(acceleratorType) {
		fmt.Printf("主加速器 %s 已存在且健康，无需回退。\n", acceleratorType)
		return nil
	}

	// 尝试重新注册主加速器
	err := rm.hardwareManager.ReRegisterAccelerator(acceleratorType)
	if err != nil {
		return fmt.Errorf("回退到主加速器 %s 失败: %v", acceleratorType, err)
	}
	return nil
}

// disableAccelerator 禁用加速器
func (rm *RecoveryManager) disableAccelerator(acceleratorType string) error {
	acc, ok := rm.hardwareManager.GetAccelerator(acceleratorType)
	if !ok {
		return fmt.Errorf("找不到加速器: %s", acceleratorType)
	}

	// 关闭加速器
	if err := acc.Shutdown(); err != nil {
		log.Printf("关闭加速器 %s 时出错: %v", acceleratorType, err)
	}

	// 从管理器中移除
	err := rm.hardwareManager.RemoveAccelerator(acceleratorType)
	if err != nil {
		return err
	}

	log.Printf("已禁用加速器: %s", acceleratorType)
	return nil
}

// GetRecoveryHistory 获取恢复历史
func (rm *RecoveryManager) GetRecoveryHistory() []RecoveryAction {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()

	history := make([]RecoveryAction, len(rm.recoveryHistory))
	copy(history, rm.recoveryHistory)
	return history
}

// GetRetryCount 获取重试次数
func (rm *RecoveryManager) GetRetryCount(acceleratorType string) int {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()

	return rm.retryCount[acceleratorType]
}

// ResetRetryCount 重置重试次数
func (rm *RecoveryManager) ResetRetryCount(acceleratorType string) {
	rm.mutex.Lock()
	defer rm.mutex.Unlock()

	delete(rm.retryCount, acceleratorType)
	delete(rm.lastRecovery, acceleratorType)
}

// UpdateConfig 更新恢复配置
func (rm *RecoveryManager) UpdateConfig(config *RecoveryConfig) {
	rm.mutex.Lock()
	defer rm.mutex.Unlock()

	rm.config = config
}

// GetConfig 获取当前配置
func (rm *RecoveryManager) GetConfig() *RecoveryConfig {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()

	return rm.config
}
