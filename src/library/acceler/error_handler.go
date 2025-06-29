package acceler

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"
)

// ErrorType 错误类型枚举
type ErrorType int

const (
	ErrorTypeInitialization ErrorType = iota
	ErrorTypeNilPointer
	ErrorTypeNotAvailable
	ErrorTypeNotInitialized
	ErrorTypeInvalidConfig
	ErrorTypeMemoryAllocation
	ErrorTypeDeviceError
	ErrorTypeNetworkError
	ErrorTypeTimeout
	ErrorTypeUnknown
)

// AcceleratorError 硬件加速器错误结构
type AcceleratorError struct {
	Type        ErrorType
	Accelerator string
	Operation   string
	Message     string
	Cause       error
	Timestamp   time.Time
	StackTrace  string
}

// Error 实现error接口
func (e *AcceleratorError) Error() string {
	return fmt.Sprintf("[%s] %s操作失败: %s", e.Accelerator, e.Operation, e.Message)
}

// Unwrap 支持错误链
func (e *AcceleratorError) Unwrap() error {
	return e.Cause
}

// ErrorHandler 错误处理器
type ErrorHandler struct {
	mutex      sync.RWMutex
	errorCount map[string]int
	lastErrors map[string]*AcceleratorError
	maxRetries int
	retryDelay time.Duration

	mu            sync.RWMutex
	errorCounts   map[string]int
	retryInterval time.Duration
}

// NewErrorHandler 创建新的错误处理器
func NewErrorHandler() *ErrorHandler {
	return &ErrorHandler{
		errorCount: make(map[string]int),
		lastErrors: make(map[string]*AcceleratorError),
		maxRetries: 3,
		retryDelay: time.Second,
	}
}

// HandleError 处理错误
func (eh *ErrorHandler) HandleError(acceleratorType, operation string, err error) *AcceleratorError {
	eh.mutex.Lock()
	defer eh.mutex.Unlock()

	// 创建加速器错误
	accelErr := &AcceleratorError{
		Type:        eh.classifyError(err),
		Accelerator: acceleratorType,
		Operation:   operation,
		Message:     err.Error(),
		Cause:       err,
		Timestamp:   time.Now(),
		StackTrace:  eh.getStackTrace(),
	}

	// 更新错误统计
	key := fmt.Sprintf("%s_%s", acceleratorType, operation)
	eh.errorCount[key]++
	eh.lastErrors[key] = accelErr

	// 记录日志
	log.Printf("硬件加速器错误: %s", accelErr.Error())

	return accelErr
}

// classifyError 分类错误
func (eh *ErrorHandler) classifyError(err error) ErrorType {
	if err == nil {
		return ErrorTypeUnknown
	}

	errorMsg := err.Error()
	switch {
	case contains(errorMsg, "not initialized", "未初始化"):
		return ErrorTypeNotInitialized
	case contains(errorMsg, "not available", "不可用"):
		return ErrorTypeNotAvailable
	case contains(errorMsg, "nil pointer", "空指针"):
		return ErrorTypeNilPointer
	case contains(errorMsg, "invalid config", "无效配置"):
		return ErrorTypeInvalidConfig
	case contains(errorMsg, "memory", "内存"):
		return ErrorTypeMemoryAllocation
	case contains(errorMsg, "device", "设备"):
		return ErrorTypeDeviceError
	case contains(errorMsg, "network", "网络", "connection", "连接"):
		return ErrorTypeNetworkError
	case contains(errorMsg, "timeout", "超时"):
		return ErrorTypeTimeout
	case contains(errorMsg, "initialize", "初始化"):
		return ErrorTypeInitialization
	default:
		return ErrorTypeUnknown
	}
}

// contains 检查字符串是否包含任一关键词
func contains(str string, keywords ...string) bool {
	for _, keyword := range keywords {
		if len(str) >= len(keyword) {
			for i := 0; i <= len(str)-len(keyword); i++ {
				if str[i:i+len(keyword)] == keyword {
					return true
				}
			}
		}
	}
	return false
}

// getStackTrace 获取堆栈跟踪
func (eh *ErrorHandler) getStackTrace() string {
	buf := make([]byte, 1024)
	n := runtime.Stack(buf, false)
	return string(buf[:n])
}

// GetErrorCount 获取错误计数
func (eh *ErrorHandler) GetErrorCount(acceleratorType, operation string) int {
	eh.mutex.RLock()
	defer eh.mutex.RUnlock()

	key := fmt.Sprintf("%s_%s", acceleratorType, operation)
	return eh.errorCount[key]
}

// GetLastError 获取最后一个错误
func (eh *ErrorHandler) GetLastError(acceleratorType, operation string) *AcceleratorError {
	eh.mutex.RLock()
	defer eh.mutex.RUnlock()

	key := fmt.Sprintf("%s_%s", acceleratorType, operation)
	return eh.lastErrors[key]
}

// ShouldRetry 判断是否应该重试
func (eh *ErrorHandler) ShouldRetry(acceleratorType, operation string) bool {
	eh.mutex.RLock()
	defer eh.mutex.RUnlock()

	key := fmt.Sprintf("%s_%s", acceleratorType, operation)
	count := eh.errorCount[key]
	return count < eh.maxRetries
}

// Reset 重置错误计数
func (eh *ErrorHandler) Reset(acceleratorType, operation string) {
	eh.mutex.Lock()
	defer eh.mutex.Unlock()

	key := fmt.Sprintf("%s_%s", acceleratorType, operation)
	delete(eh.errorCount, key)
	delete(eh.lastErrors, key)
}

// GetAllErrors 获取所有错误统计
func (eh *ErrorHandler) GetAllErrors() map[string]int {
	eh.mutex.RLock()
	defer eh.mutex.RUnlock()

	result := make(map[string]int)
	for k, v := range eh.errorCount {
		result[k] = v
	}
	return result
}

// SafeCall 安全调用函数，带错误处理和重试
func (eh *ErrorHandler) SafeCall(acceleratorType, operation string, fn func() error) error {
	var lastErr error

	for i := 0; i < eh.maxRetries; i++ {
		if err := fn(); err != nil {
			lastErr = err
			accelErr := eh.HandleError(acceleratorType, operation, err)

			// 如果是致命错误，不重试
			if accelErr.Type == ErrorTypeNilPointer || accelErr.Type == ErrorTypeInvalidConfig {
				return accelErr
			}

			// 等待后重试
			if i < eh.maxRetries-1 {
				time.Sleep(eh.retryDelay * time.Duration(i+1))
			}
		} else {
			// 成功，重置错误计数
			eh.Reset(acceleratorType, operation)
			return nil
		}
	}

	return eh.HandleError(acceleratorType, operation, lastErr)
}

// 全局错误处理器实例
var globalErrorHandler = NewErrorHandler()

// HandleAcceleratorError 全局错误处理函数
func HandleAcceleratorError(acceleratorType, operation string, err error) *AcceleratorError {
	return globalErrorHandler.HandleError(acceleratorType, operation, err)
}

// SafeAcceleratorCall 全局安全调用函数
func SafeAcceleratorCall(acceleratorType, operation string, fn func() error) error {
	return globalErrorHandler.SafeCall(acceleratorType, operation, fn)
}
