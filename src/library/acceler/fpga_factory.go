package acceler

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"
)

// FPGAFactory FPGA加速器工厂
type FPGAFactory struct{}

// NewFPGAFactory 创建FPGA工厂实例
func NewFPGAFactory() *FPGAFactory {
	return &FPGAFactory{}
}

// CreateFPGAAccelerator 创建FPGA加速器实例
// 根据环境自动选择真实FPGA或模拟器
func (factory *FPGAFactory) CreateFPGAAccelerator(config *FPGAConfig) (UnifiedAccelerator, error) {
	// 检查是否强制使用模拟器
	if factory.shouldUseStub() {
		return NewFPGAAcceleratorStub(config)
	}

	// 尝试创建真实FPGA加速器
	fpgaAccel := NewFPGAAccelerator(0, config) // 修复：添加deviceID参数
	// 检查FPGA是否真正可用
	if !fpgaAccel.IsAvailable() {
		// FPGA不可用，使用模拟器
		return NewFPGAAcceleratorStub(config)
	}

	return fpgaAccel, nil
}

// shouldUseStub 判断是否应该使用模拟器
func (factory *FPGAFactory) shouldUseStub() bool {
	// 检查环境变量
	if useStub := os.Getenv("FPGA_USE_STUB"); useStub != "" {
		return strings.ToLower(useStub) == "true" || useStub == "1"
	}

	// 检查是否在测试环境
	if factory.isTestEnvironment() {
		return true
	}

	// 检查操作系统兼容性
	if !factory.isFPGACompatibleOS() {
		return true
	}

	return false
}

// isTestEnvironment 检查是否在测试环境
func (factory *FPGAFactory) isTestEnvironment() bool {
	// 检查是否在运行测试
	for _, arg := range os.Args {
		if strings.Contains(arg, "test") || strings.Contains(arg, ".test") {
			return true
		}
	}

	// 检查测试相关环境变量
	if testEnv := os.Getenv("GO_TEST"); testEnv != "" {
		return true
	}

	return false
}

// isFPGACompatibleOS 检查操作系统是否支持FPGA
func (factory *FPGAFactory) isFPGACompatibleOS() bool {
	switch runtime.GOOS {
	case "linux":
		return true
	case "windows":
		// Windows需要特定的FPGA驱动
		return factory.hasWindowsFPGADriver()
	case "darwin":
		// macOS通常不支持FPGA
		return false
	default:
		return false
	}
}

// hasWindowsFPGADriver 检查Windows是否有FPGA驱动
func (factory *FPGAFactory) hasWindowsFPGADriver() bool {
	// 检查常见的FPGA驱动路径
	driverPaths := []string{
		"C:\\Xilinx\\Vivado",
		"C:\\Intel\\Quartus",
		"C:\\Program Files\\Xilinx",
		"C:\\Program Files\\Intel\\Quartus",
	}

	for _, path := range driverPaths {
		if _, err := os.Stat(path); err == nil {
			return true
		}
	}

	return false
}

// GetRecommendedConfig 获取推荐的FPGA配置
func (factory *FPGAFactory) GetRecommendedConfig() *FPGAConfig {
	return &FPGAConfig{
		Enable:         true,
		DeviceIDs:      []int{0},
		ClockFrequency: 200, // 200MHz
		PipelineDepth:  8,
		Parallelism: FPGAParallelismConfig{
			ComputeUnits:   4,
			VectorWidth:    128,
			UnrollFactor:   4,
			PipelineStages: 8,
		},
		Optimization: FPGAOptimizationConfig{
			ResourceSharing:    true,
			MemoryOptimization: true,
			TimingOptimization: true,
			PowerOptimization:  false,
			AreaOptimization:   false,
		},
		Reconfiguration: FPGAReconfigurationConfig{
			Enable:                 false,
			PartialReconfiguration: false,
			ReconfigurationTime:    100 * time.Millisecond,
			BitstreamCache:         true,
			HotSwap:                false,
		},
	}
}

// ValidateConfig 验证FPGA配置
func (factory *FPGAFactory) ValidateConfig(config *FPGAConfig) error {
	if config == nil {
		return fmt.Errorf("配置不能为空")
	}

	if config.ClockFrequency <= 0 || config.ClockFrequency > 1000 {
		return fmt.Errorf("时钟频率必须在1-1000MHz之间")
	}

	if config.Parallelism.ComputeUnits <= 0 || config.Parallelism.ComputeUnits > 32 {
		return fmt.Errorf("计算单元数量必须在1-32之间")
	}

	if config.PipelineDepth <= 0 || config.PipelineDepth > 64 {
		return fmt.Errorf("流水线深度必须在1-64之间")
	}

	return nil
}

// NewFPGAAcceleratorStub 创建FPGA模拟器实例
func NewFPGAAcceleratorStub(config *FPGAConfig) (UnifiedAccelerator, error) {
	deviceID := 0
	if config != nil && len(config.DeviceIDs) > 0 {
		deviceID = config.DeviceIDs[0]
	}
	return NewFPGAAccelerator(deviceID, config), nil
}
