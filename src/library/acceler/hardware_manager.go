package acceler

import (
	"VectorSphere/src/library/entity"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// HardwareManager 硬件管理器（整合原有代码）
type HardwareManager struct {
	accelerators    map[string]UnifiedAccelerator
	mutex           sync.RWMutex
	defaultType     string
	manager         *AcceleratorManager
	config          *HardwareConfig
	errorHandler    *ErrorHandler
	healthMonitor   *HealthMonitor
	recoveryManager *RecoveryManager
}

// GPUConfig GPU配置
type GPUConfig struct {
	Enable      bool   `json:"enable" yaml:"enable"`
	DeviceIDs   []int  `json:"device_ids" yaml:"device_ids"`
	MemoryLimit int64  `json:"memory_limit" yaml:"memory_limit"`
	BatchSize   int    `json:"batch_size" yaml:"batch_size"`
	Precision   string `json:"precision" yaml:"precision"`
	IndexType   string `json:"index_type" yaml:"index_type"`
}

// HardwareConfig 硬件配置结构体
type HardwareConfig struct {
	CPU  CPUConfig  `json:"cpu" yaml:"cpu"`
	GPU  GPUConfig  `json:"gpu" yaml:"gpu"`
	FPGA FPGAConfig `json:"fpga" yaml:"fpga"`
	PMem PMemConfig `json:"pmem" yaml:"pmem"`
	RDMA RDMAConfig `json:"rdma" yaml:"rdma"`
}

// GetDefaultHardwareConfig 获取默认硬件配置
func GetDefaultHardwareConfig() *HardwareConfig {
	return &HardwareConfig{
		CPU: CPUConfig{
			Enable:      true,
			IndexType:   "IDMap,Flat",
			DeviceID:    0,
			Threads:     0, // 0表示使用所有可用线程
			VectorWidth: 256,
		},
		GPU: GPUConfig{
			Enable:      true,
			DeviceIDs:   []int{0},
			MemoryLimit: 8 * 1024 * 1024 * 1024, // 8GB
			BatchSize:   1000,
			Precision:   "float32",
			IndexType:   "IVF,Flat",
		},
		FPGA: FPGAConfig{
			Enable:         false,
			DeviceIDs:      []int{0},
			ClockFrequency: 200,
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
		},
		PMem: PMemConfig{
			Enable:     false,
			DevicePath: "/mnt/pmem0/vectors.db",
			PoolSize:   10 * 1024 * 1024 * 1024, // 10GB
			Mode:       "app_direct",
		},
		RDMA: RDMAConfig{
			Enable:    false,
			DeviceID:  0,
			PortNum:   1,
			QueueSize: 1024,
			Protocol:  "RoCE",
		},
	}
}

// NewHardwareManager 创建新的硬件管理器
func NewHardwareManager() *HardwareManager {
	return NewHardwareManagerWithConfig(GetDefaultHardwareConfig())
}

// NewHardwareManagerWithConfig 使用指定配置创建新的硬件管理器
func NewHardwareManagerWithConfig(config *HardwareConfig) *HardwareManager {
	hm := &HardwareManager{
		accelerators: make(map[string]UnifiedAccelerator),
		defaultType:  "cpu",
		manager:      NewAcceleratorManager(),
		config:       config,
		errorHandler: NewErrorHandler(),
	}

	// 初始化健康监控器
	hm.healthMonitor = NewHealthMonitor(hm)

	// 初始化恢复管理器
	hm.recoveryManager = NewRecoveryManager(hm, DefaultRecoveryConfig())

	// 注册可用的硬件加速器
	hm.registerAllAccelerators()

	// 配置默认策略
	hm.setupDefaultStrategies()

	return hm
}

// GetAcceleratorConfig 根据加速器类型获取其配置
func (hm *HardwareManager) GetAcceleratorConfig(acceleratorType string) (interface{}, error) {
	switch acceleratorType {
	case AcceleratorCPU:
		return hm.config.CPU, nil
	case AcceleratorGPU:
		return hm.config.GPU, nil
	case AcceleratorFPGA:
		return hm.config.FPGA, nil
	case AcceleratorPMem:
		return hm.config.PMem, nil
	case AcceleratorRDMA:
		return hm.config.RDMA, nil
	default:
		return nil, fmt.Errorf("未知加速器类型: %s", acceleratorType)
	}
}

// CreateAcceleratorFromConfig 根据配置创建加速器实例
func (hm *HardwareManager) CreateAcceleratorFromConfig(acceleratorType string, cfg interface{}) (UnifiedAccelerator, error) {
	switch acceleratorType {
	case AcceleratorCPU:
		cpuCfg, ok := cfg.(CPUConfig)
		if !ok {
			return nil, fmt.Errorf("配置类型不匹配 CPUConfig")
		}
		return NewCPUAccelerator(cpuCfg.DeviceID, cpuCfg.IndexType), nil
	case AcceleratorGPU:
		gpuCfg, ok := cfg.(GPUConfig)
		if !ok {
			return nil, fmt.Errorf("配置类型不匹配 GPUConfig")
		}
		// 使用第一个DeviceID来创建GPU加速器实例
		gpuAcc := NewGPUAccelerator(gpuCfg.DeviceIDs[0])
		return gpuAcc, nil
	case AcceleratorFPGA:
		fpgaCfg, ok := cfg.(FPGAConfig)
		if !ok {
			return nil, fmt.Errorf("配置类型不匹配 FPGAConfig")
		}
		// 假设我们只使用第一个DeviceID来创建实例
		return NewFPGAAccelerator(fpgaCfg.DeviceIDs[0], &fpgaCfg), nil
	case AcceleratorPMem:
		pmemCfg, ok := cfg.(PMemConfig)
		if !ok {
			return nil, fmt.Errorf("配置类型不匹配 PMemConfig")
		}
		return NewPMemAccelerator(&pmemCfg), nil
	case AcceleratorRDMA:
		rdmaCfg, ok := cfg.(RDMAConfig)
		if !ok {
			return nil, fmt.Errorf("配置类型不匹配 RDMAConfig")
		}
		return NewRDMAAccelerator(rdmaCfg.DeviceID, rdmaCfg.PortNum, &rdmaCfg), nil
	default:
		return nil, fmt.Errorf("未知加速器类型: %s", acceleratorType)
	}
}

// ReRegisterAccelerator 重新注册加速器
func (hm *HardwareManager) ReRegisterAccelerator(acceleratorType string) error {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()

	// 1. 获取旧的加速器实例并关闭
	oldAcc, exists := hm.accelerators[acceleratorType]
	if exists && oldAcc != nil {
		if err := oldAcc.Shutdown(); err != nil {
			fmt.Printf("警告: 关闭旧加速器 %s 失败: %v\n", acceleratorType, err)
		}
		delete(hm.accelerators, acceleratorType)
	}

	// 2. 获取加速器配置
	config, err := hm.GetAcceleratorConfig(acceleratorType)
	if err != nil {
		return fmt.Errorf("获取加速器 %s 配置失败: %v", acceleratorType, err)
	}

	// 3. 根据配置创建新的加速器实例
	newAcc, err := hm.CreateAcceleratorFromConfig(acceleratorType, config)
	if err != nil {
		return fmt.Errorf("从配置创建加速器 %s 失败: %v", acceleratorType, err)
	}

	// 4. 初始化新的加速器
	if err := newAcc.Initialize(); err != nil {
		return fmt.Errorf("初始化新加速器 %s 失败: %v", acceleratorType, err)
	}

	// 5. 注册新的加速器
	hm.accelerators[acceleratorType] = newAcc
	fmt.Printf("成功重新注册加速器: %s\n", acceleratorType)
	return nil
}

// NewHardwareManagerFromFile 从配置文件创建硬件管理器
func NewHardwareManagerFromFile(configFilePath string) (*HardwareManager, error) {
	config, err := LoadConfigFromFile(configFilePath)
	if err != nil {
		return nil, fmt.Errorf("从文件加载配置失败: %v", err)
	}

	return NewHardwareManagerWithConfig(config), nil
}

// setupDefaultStrategies 配置默认策略
func (hm *HardwareManager) setupDefaultStrategies() {
	// 高吞吐量工作负载优先使用GPU
	hm.manager.SetStrategy("high_throughput", AcceleratorGPU)
	// 低延迟工作负载优先使用FPGA
	hm.manager.SetStrategy("low_latency", AcceleratorFPGA)
	// 分布式工作负载优先使用RDMA
	hm.manager.SetStrategy("distributed", AcceleratorRDMA)
	// 高维度向量优先使用GPU
	hm.manager.SetStrategy("high_dimension", AcceleratorGPU)
	// 默认使用CPU
	hm.manager.SetStrategy("default", AcceleratorCPU)
}

// registerAllAccelerators 根据配置注册所有可用的硬件加速器
func (hm *HardwareManager) registerAllAccelerators() {
	// 注册CPU加速器 (使用FAISS作为CPU实现)
	if hm.config.CPU.Enable {
		cpuAcc := NewCPUAccelerator(hm.config.CPU.DeviceID, hm.config.CPU.IndexType)
		if cpuAcc.IsAvailable() {
			hm.RegisterAccelerator(AcceleratorCPU, cpuAcc)
			if err := hm.manager.RegisterAccelerator(cpuAcc); err != nil {
				fmt.Printf("警告: 注册CPU加速器失败: %v\n", err)
			}

			// CPU配置通过构造函数设置，无需额外更新
		}
	}

	// 注册GPU加速器
	if hm.config.GPU.Enable {
		for _, deviceID := range hm.config.GPU.DeviceIDs {
			gpuAcc := NewGPUAccelerator(deviceID)
			if gpuAcc == nil {
				fmt.Printf("警告: 创建GPU加速器失败，设备ID: %d\n", deviceID)
				continue
			}

			if gpuAcc.IsAvailable() {
				name := fmt.Sprintf("%s_%d", AcceleratorGPU, deviceID)
				hm.RegisterAccelerator(name, gpuAcc)
				if err := hm.manager.RegisterAccelerator(gpuAcc); err != nil {
					fmt.Printf("警告: 注册GPU加速器 %s 失败: %v\n", name, err)
					continue
				}

				// GPU配置通过构造函数和AutoTune方法设置

				hm.defaultType = name // 优先使用GPU
				break                 // 只注册第一个可用的GPU
			}
		}
	}

	// 注册FPGA加速器
	if hm.config.FPGA.Enable {
		for _, deviceID := range hm.config.FPGA.DeviceIDs {
			config := &FPGAConfig{
				Enable:          true,
				DeviceIDs:       []int{deviceID},
				ClockFrequency:  hm.config.FPGA.ClockFrequency,
				PipelineDepth:   hm.config.FPGA.PipelineDepth,
				Parallelism:     hm.config.FPGA.Parallelism,
				Optimization:    hm.config.FPGA.Optimization,
				Reconfiguration: hm.config.FPGA.Reconfiguration,
			}
			fpgaAcc := NewFPGAAccelerator(deviceID, config)
			if fpgaAcc.IsAvailable() {
				name := fmt.Sprintf("%s_%d", AcceleratorFPGA, deviceID)
				hm.RegisterAccelerator(name, fpgaAcc)
				if err := hm.manager.RegisterAccelerator(fpgaAcc); err != nil {
					fmt.Printf("警告: 注册FPGA加速器 %s 失败: %v\n", name, err)
					continue
				}
				break
			}
		}
	}

	// 注册PMem加速器
	if hm.config.PMem.Enable {
		pmemConfig := &PMemConfig{
			DevicePath: hm.config.PMem.DevicePath,
			PoolSize:   hm.config.PMem.PoolSize,
		}
		pmemAcc := NewPMemAccelerator(pmemConfig)
		if pmemAcc.IsAvailable() {
			hm.RegisterAccelerator(AcceleratorPMem, pmemAcc)
			if err := hm.manager.RegisterAccelerator(pmemAcc); err != nil {
				fmt.Printf("警告: 注册PMem加速器失败: %v\n", err)
			}
		}
	}

	// 注册RDMA加速器
	if hm.config.RDMA.Enable {
		rdmaConfig := &RDMAConfig{
			DeviceID: hm.config.RDMA.DeviceID,
			PortNum:  hm.config.RDMA.PortNum,
		}
		rdmaAcc := NewRDMAAccelerator(hm.config.RDMA.DeviceID, hm.config.RDMA.PortNum, rdmaConfig)
		if rdmaAcc.IsAvailable() {
			hm.RegisterAccelerator(AcceleratorRDMA, rdmaAcc)
			if err := hm.manager.RegisterAccelerator(rdmaAcc); err != nil {
				fmt.Printf("警告: 注册RDMA加速器失败: %v\n", err)
			}
		}
	}
}

// RegisterAccelerator 注册硬件加速器
func (hm *HardwareManager) RegisterAccelerator(name string, accelerator UnifiedAccelerator) {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()
	hm.accelerators[name] = accelerator
}

// GetAccelerator 获取硬件加速器
func (hm *HardwareManager) GetAccelerator(name string) (UnifiedAccelerator, bool) {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()
	acc, exists := hm.accelerators[name]
	return acc, exists
}

// GetBestAccelerator 获取最佳硬件加速器
func (hm *HardwareManager) GetBestAccelerator(workloadType string) UnifiedAccelerator {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	var bestAcc UnifiedAccelerator
	var bestRating float64

	for _, acc := range hm.accelerators {
		if acc.IsAvailable() {
			caps := acc.GetCapabilities()
			rating := caps.PerformanceRating

			// 根据工作负载类型调整评分
			switch workloadType {
			case "low_latency":
				if caps.Type == AcceleratorFPGA {
					rating += 2.0
				}
			case "high_throughput":
				if caps.Type == AcceleratorGPU {
					rating += 2.0
				}
			case "distributed":
				if caps.Type == AcceleratorRDMA {
					rating += 3.0
				}
			case "persistent":
				if caps.Type == AcceleratorPMem {
					rating += 2.5
				}
			}

			if rating > bestRating {
				bestRating = rating
				bestAcc = acc
			}
		}
	}

	// 如果没有找到最佳加速器，返回默认加速器
	if bestAcc == nil {
		if defaultAcc, exists := hm.accelerators[hm.defaultType]; exists {
			return defaultAcc
		}
		// 返回第一个可用的加速器
		for _, acc := range hm.accelerators {
			if acc.IsAvailable() {
				return acc
			}
		}
	}

	return bestAcc
}

// GetOptimalAccelerator 使用统一管理器获取最优加速器
func (hm *HardwareManager) GetOptimalAccelerator(workload WorkloadProfile) UnifiedAccelerator {
	return hm.manager.GetOptimalAccelerator(workload)
}

// InitializeAll 初始化所有可用的硬件加速器
func (hm *HardwareManager) InitializeAll() error {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	var errors []string
	for name, acc := range hm.accelerators {
		if acc.IsAvailable() {
			if err := acc.Initialize(); err != nil {
				errorMsg := fmt.Sprintf("初始化硬件加速器 %s 失败: %v", name, err)
				errors = append(errors, errorMsg)
				fmt.Printf("警告: %s\n", errorMsg)
			}
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("部分加速器初始化失败: %v", errors)
	}

	return nil
}

// ShutdownAll 关闭所有硬件加速器
func (hm *HardwareManager) ShutdownAll() error {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	var errors []string
	for name, acc := range hm.accelerators {
		if err := acc.Shutdown(); err != nil {
			errorMsg := fmt.Sprintf("关闭硬件加速器 %s 失败: %v", name, err)
			errors = append(errors, errorMsg)
			fmt.Printf("警告: %s\n", errorMsg)
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("部分加速器关闭失败: %v", errors)
	}

	return nil
}

// GetAllStats 获取所有硬件加速器的统计信息
func (hm *HardwareManager) GetAllStats() map[string]HardwareStats {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	stats := make(map[string]HardwareStats)
	for name, acc := range hm.accelerators {
		stats[name] = acc.GetStats()
	}

	return stats
}

// GetAllCapabilities 获取所有硬件加速器的能力信息
func (hm *HardwareManager) GetAllCapabilities() map[string]HardwareCapabilities {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	caps := make(map[string]HardwareCapabilities)
	for name, acc := range hm.accelerators {
		caps[name] = acc.GetCapabilities()
	}

	return caps
}

// GetAllPerformanceMetrics 获取所有硬件加速器的性能指标
func (hm *HardwareManager) GetAllPerformanceMetrics() map[string]PerformanceMetrics {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	metrics := make(map[string]PerformanceMetrics)
	for name, acc := range hm.accelerators {
		metrics[name] = acc.GetPerformanceMetrics()
	}

	return metrics
}

// AutoTuneAll 自动调优所有硬件加速器
func (hm *HardwareManager) AutoTuneAll(workload WorkloadProfile) error {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	var errors []string
	for name, acc := range hm.accelerators {
		if acc.IsAvailable() {
			if err := acc.AutoTune(workload); err != nil {
				errorMsg := fmt.Sprintf("自动调优硬件加速器 %s 失败: %v", name, err)
				errors = append(errors, errorMsg)
				fmt.Printf("警告: %s\n", errorMsg)
			}
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("部分加速器自动调优失败: %v", errors)
	}

	return nil
}

// GetAvailableAccelerators 获取所有可用的硬件加速器列表
func (hm *HardwareManager) GetAvailableAccelerators() []string {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	var available []string
	for name, acc := range hm.accelerators {
		if acc.IsAvailable() {
			available = append(available, name)
		}
	}

	return available
}

// GetAcceleratorsByType 根据类型获取加速器
func (hm *HardwareManager) GetAcceleratorsByType(accType string) []UnifiedAccelerator {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	var accelerators []UnifiedAccelerator
	for _, acc := range hm.accelerators {
		if acc.GetType() == accType && acc.IsAvailable() {
			accelerators = append(accelerators, acc)
		}
	}

	return accelerators
}

// HealthCheck 健康检查
func (hm *HardwareManager) HealthCheck() map[string]bool {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	health := make(map[string]bool)
	for name, acc := range hm.accelerators {
		health[name] = acc.IsAvailable()
	}

	return health
}

// GetSystemInfo 获取系统信息
func (hm *HardwareManager) GetSystemInfo() map[string]interface{} {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	info := map[string]interface{}{
		"total_accelerators":     len(hm.accelerators),
		"available_accelerators": len(hm.GetAvailableAccelerators()),
		"default_type":           hm.defaultType,
		"timestamp":              time.Now(),
	}

	// 按类型统计
	typeCount := make(map[string]int)
	for _, acc := range hm.accelerators {
		if acc.IsAvailable() {
			typeCount[acc.GetType()]++
		}
	}
	info["type_distribution"] = typeCount

	return info
}

// SetDefaultAccelerator 设置默认加速器
func (hm *HardwareManager) SetDefaultAccelerator(name string) error {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()

	if _, exists := hm.accelerators[name]; !exists {
		return fmt.Errorf("加速器 %s 不存在", name)
	}

	if !hm.accelerators[name].IsAvailable() {
		return fmt.Errorf("加速器 %s 不可用", name)
	}

	hm.defaultType = name
	return nil
}

// GetDefaultAccelerator 获取默认加速器
func (hm *HardwareManager) GetDefaultAccelerator() UnifiedAccelerator {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	if acc, exists := hm.accelerators[hm.defaultType]; exists && acc.IsAvailable() {
		return acc
	}

	// 如果默认加速器不可用，返回第一个可用的
	for _, acc := range hm.accelerators {
		if acc.IsAvailable() {
			return acc
		}
	}

	return nil
}

// RemoveAccelerator 移除加速器
func (hm *HardwareManager) RemoveAccelerator(name string) error {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()

	if acc, exists := hm.accelerators[name]; exists {
		// 先关闭加速器
		if err := acc.Shutdown(); err != nil {
			fmt.Printf("警告: 关闭加速器 %s 时出错: %v\n", name, err)
		}
		// 从映射中删除
		delete(hm.accelerators, name)
		// 如果是默认加速器，重新选择默认加速器
		if hm.defaultType == name {
			hm.selectNewDefault()
		}
		return nil
	}

	return fmt.Errorf("加速器 %s 不存在", name)
}

// selectNewDefault 选择新的默认加速器
func (hm *HardwareManager) selectNewDefault() {
	// 优先级：GPU > FPGA > PMem > RDMA > CPU
	priority := []string{AcceleratorGPU, AcceleratorFPGA, AcceleratorPMem, AcceleratorRDMA, AcceleratorCPU}

	for _, accType := range priority {
		for name, acc := range hm.accelerators {
			if acc.GetType() == accType && acc.IsAvailable() {
				hm.defaultType = name
				return
			}
		}
	}

	// 如果没有找到，使用第一个可用的
	for name, acc := range hm.accelerators {
		if acc.IsAvailable() {
			hm.defaultType = name
			return
		}
	}

	hm.defaultType = "cpu" // 最后的备选
}

// GetManager 获取统一加速器管理器
func (hm *HardwareManager) GetManager() *AcceleratorManager {
	return hm.manager
}

// GetConfig 获取当前硬件配置
func (hm *HardwareManager) GetConfig() *HardwareConfig {
	return hm.config
}

// LoadConfigFromFile 从文件加载硬件配置
func LoadConfigFromFile(filePath string) (*HardwareConfig, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("读取配置文件失败: %v", err)
	}

	config := &HardwareConfig{}

	// 根据文件扩展名决定解析方式
	ext := strings.ToLower(filepath.Ext(filePath))
	switch ext {
	case ".json":
		err = json.Unmarshal(data, config)
	case ".yaml", ".yml":
		err = yaml.Unmarshal(data, config)
	default:
		return nil, fmt.Errorf("不支持的配置文件格式: %s", ext)
	}

	if err != nil {
		return nil, fmt.Errorf("解析配置文件失败: %v", err)
	}

	return config, nil
}

// SaveConfigToFile 保存硬件配置到文件
func (hm *HardwareManager) SaveConfigToFile(filePath string) error {
	// 根据文件扩展名决定序列化方式
	ext := strings.ToLower(filepath.Ext(filePath))

	var data []byte
	var err error

	switch ext {
	case ".json":
		data, err = json.MarshalIndent(hm.config, "", "  ")
	case ".yaml", ".yml":
		data, err = yaml.Marshal(hm.config)
	default:
		return fmt.Errorf("不支持的配置文件格式: %s", ext)
	}

	if err != nil {
		return fmt.Errorf("序列化配置失败: %v", err)
	}

	// 确保目录存在
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("创建目录失败: %v", err)
	}

	// 写入文件
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("写入配置文件失败: %v", err)
	}

	return nil
}

// UpdateConfig 更新硬件配置
func (hm *HardwareManager) UpdateConfig(config *HardwareConfig) error {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()

	// 保存旧配置，以便回滚
	oldConfig := hm.config

	// 更新配置
	hm.config = config

	// 关闭所有现有加速器
	if err := hm.ShutdownAll(); err != nil {
		fmt.Printf("警告: 关闭现有加速器时出错: %v\n", err)
	}

	// 清空现有加速器
	hm.accelerators = make(map[string]UnifiedAccelerator)
	hm.manager = NewAcceleratorManager()

	// 根据新配置重新注册加速器
	hm.registerAllAccelerators()

	// 重新配置策略
	hm.setupDefaultStrategies()

	// 初始化所有加速器
	if err := hm.InitializeAll(); err != nil {
		// 初始化失败，回滚配置
		fmt.Printf("错误: 初始化加速器失败，回滚配置: %v\n", err)
		hm.config = oldConfig
		hm.accelerators = make(map[string]UnifiedAccelerator)
		hm.manager = NewAcceleratorManager()
		hm.registerAllAccelerators()
		hm.setupDefaultStrategies()
		_ = hm.InitializeAll() // 尝试使用旧配置初始化
		return fmt.Errorf("更新配置失败: %v", err)
	}

	return nil
}

// ApplyVectorConfig 应用向量数据库硬件配置
func (hm *HardwareManager) ApplyVectorConfig(vectorConfig interface{}) error {
	// 尝试将接口转换为硬件加速配置
	hardwareConfig, ok := vectorConfig.(*HardwareConfig)
	if !ok {
		// 尝试从vector包的HardwareAccelerationConfig转换
		vectorHardwareConfig, ok := vectorConfig.(interface {
			GetGPUConfig() interface{}
			GetCPUConfig() interface{}
			GetFPGAConfig() interface{}
			GetPMemConfig() interface{}
			GetRDMAConfig() interface{}
		})
		if !ok {
			return fmt.Errorf("无法识别的配置类型")
		}

		// 创建新的硬件配置
		hardwareConfig = &HardwareConfig{}

		// 转换GPU配置
		if gpuConfig, ok := vectorHardwareConfig.GetGPUConfig().(interface {
			IsEnabled() bool
			GetDeviceIDs() []int
			GetMemoryLimit() int64
			GetBatchSize() int
			GetPrecision() string
		}); ok {
			hardwareConfig.GPU.Enable = gpuConfig.IsEnabled()
			hardwareConfig.GPU.DeviceIDs = gpuConfig.GetDeviceIDs()
			hardwareConfig.GPU.MemoryLimit = gpuConfig.GetMemoryLimit()
			hardwareConfig.GPU.BatchSize = gpuConfig.GetBatchSize()
			hardwareConfig.GPU.Precision = gpuConfig.GetPrecision()
		}

		// 转换CPU配置
		if cpuConfig, ok := vectorHardwareConfig.GetCPUConfig().(interface {
			IsEnabled() bool
			GetThreads() int
		}); ok {
			hardwareConfig.CPU.Enable = cpuConfig.IsEnabled()
			hardwareConfig.CPU.Threads = cpuConfig.GetThreads()
		}

		// 转换FPGA配置
		if fpgaConfig, ok := vectorHardwareConfig.GetFPGAConfig().(interface {
			IsEnabled() bool
			GetDeviceIDs() []int
		}); ok {
			hardwareConfig.FPGA.Enable = fpgaConfig.IsEnabled()
			hardwareConfig.FPGA.DeviceIDs = fpgaConfig.GetDeviceIDs()
		}

		// 转换PMem配置
		if pmemConfig, ok := vectorHardwareConfig.GetPMemConfig().(interface {
			IsEnabled() bool
			GetDevicePath() string
			GetPoolSize() int64
		}); ok {
			hardwareConfig.PMem.Enable = pmemConfig.IsEnabled()
			hardwareConfig.PMem.DevicePath = pmemConfig.GetDevicePath()
			hardwareConfig.PMem.PoolSize = uint64(pmemConfig.GetPoolSize())
		}

		// 转换RDMA配置
		if rdmaConfig, ok := vectorHardwareConfig.GetRDMAConfig().(interface {
			IsEnabled() bool
			GetDeviceID() int
			GetPortNum() int
		}); ok {
			hardwareConfig.RDMA.Enable = rdmaConfig.IsEnabled()
			hardwareConfig.RDMA.DeviceID = rdmaConfig.GetDeviceID()
			hardwareConfig.RDMA.PortNum = rdmaConfig.GetPortNum()
		}
	}

	// 应用配置
	return hm.UpdateConfig(hardwareConfig)
}

// BatchComputeDistance 使用最佳加速器进行批量距离计算
func (hm *HardwareManager) BatchComputeDistance(queries [][]float64, vectors [][]float64, workloadType string) ([][]float64, error) {
	acc := hm.GetBestAccelerator(workloadType)
	if acc == nil {
		return nil, fmt.Errorf("没有可用的硬件加速器")
	}

	return acc.BatchComputeDistance(queries, vectors)
}

// BatchSearch 使用最佳加速器进行批量搜索
func (hm *HardwareManager) BatchSearch(queries [][]float64, database [][]float64, k int, workloadType string) ([][]AccelResult, error) {
	acc := hm.GetBestAccelerator(workloadType)
	if acc == nil {
		return nil, fmt.Errorf("没有可用的硬件加速器")
	}

	return acc.BatchSearch(queries, database, k)
}

// AccelerateSearch 使用最佳加速器进行搜索加速
func (hm *HardwareManager) AccelerateSearch(query []float64, database [][]float64, options entity.SearchOptions, workloadType string) ([]AccelResult, error) {
	acc := hm.GetBestAccelerator(workloadType)
	if acc == nil {
		return nil, fmt.Errorf("no available accelerator")
	}

	return acc.AccelerateSearch(query, database, options)
}

// GetErrorStats 获取硬件加速器错误统计
func (hm *HardwareManager) GetErrorStats() map[string]int {
	return hm.errorHandler.GetAllErrors()
}

// GetLastError 获取指定加速器的最后一个错误
func (hm *HardwareManager) GetLastError(acceleratorType, operation string) *AcceleratorError {
	return hm.errorHandler.GetLastError(acceleratorType, operation)
}

// ResetErrorCount 重置错误计数
func (hm *HardwareManager) ResetErrorCount(acceleratorType, operation string) {
	hm.errorHandler.Reset(acceleratorType, operation)
}

// StartHealthMonitoring 启动健康监控
func (hm *HardwareManager) StartHealthMonitoring() {
	if hm.healthMonitor != nil {
		hm.healthMonitor.Start()
	}
}

// StopHealthMonitoring 停止健康监控
func (hm *HardwareManager) StopHealthMonitoring() {
	if hm.healthMonitor != nil {
		hm.healthMonitor.Stop()
	}
}

// GetHealthReport 获取指定加速器的健康报告
func (hm *HardwareManager) GetHealthReport(acceleratorType string) *HealthReport {
	if hm.healthMonitor != nil {
		return hm.healthMonitor.GetHealthReport(acceleratorType)
	}
	return nil
}

// GetAllHealthReports 获取所有加速器的健康报告
func (hm *HardwareManager) GetAllHealthReports() map[string]*HealthReport {
	if hm.healthMonitor != nil {
		return hm.healthMonitor.GetAllHealthReports()
	}
	return nil
}

// GetOverallHealth 获取整体健康状态
func (hm *HardwareManager) GetOverallHealth() HealthStatus {
	if hm.healthMonitor != nil {
		return hm.healthMonitor.GetOverallHealth()
	}
	return HealthStatusUnknown
}

// IsHealthy 检查指定加速器是否健康
func (hm *HardwareManager) IsHealthy(acceleratorType string) bool {
	report := hm.GetHealthReport(acceleratorType)
	if report == nil {
		return false
	}
	return report.Status == HealthStatusHealthy
}

// StartRecoveryManager 启动恢复管理器
func (hm *HardwareManager) StartRecoveryManager() {
	if hm.recoveryManager != nil {
		hm.recoveryManager.Start()
	}
}

// StopRecoveryManager 停止恢复管理器
func (hm *HardwareManager) StopRecoveryManager() {
	if hm.recoveryManager != nil {
		hm.recoveryManager.Stop()
	}
}

// GetRecoveryHistory 获取恢复历史
func (hm *HardwareManager) GetRecoveryHistory() []RecoveryAction {
	if hm.recoveryManager != nil {
		return hm.recoveryManager.GetRecoveryHistory()
	}
	return nil
}

// GetRetryCount 获取指定加速器的重试次数
func (hm *HardwareManager) GetRetryCount(acceleratorType string) int {
	if hm.recoveryManager != nil {
		return hm.recoveryManager.GetRetryCount(acceleratorType)
	}
	return 0
}

// ResetRetryCount 重置指定加速器的重试次数
func (hm *HardwareManager) ResetRetryCount(acceleratorType string) {
	if hm.recoveryManager != nil {
		hm.recoveryManager.ResetRetryCount(acceleratorType)
	}
}

// UpdateRecoveryConfig 更新恢复配置
func (hm *HardwareManager) UpdateRecoveryConfig(config *RecoveryConfig) {
	if hm.recoveryManager != nil {
		hm.recoveryManager.UpdateConfig(config)
	}
}

// GetGPUAccelerator 获取GPU加速器
func (hm *HardwareManager) GetGPUAccelerator() UnifiedAccelerator {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()

	// 首先检查直接的GPU加速器
	if acc, exists := hm.accelerators[AcceleratorGPU]; exists {
		if acc.IsAvailable() {
			return acc
		}
	}

	// 检查带设备ID的GPU加速器
	for name, acc := range hm.accelerators {
		if strings.HasPrefix(name, AcceleratorGPU+"_") && acc.IsAvailable() {
			return acc
		}
	}

	return nil
}

// RegisterGPUAccelerator 注册GPU加速器
func (hm *HardwareManager) RegisterGPUAccelerator(accelerator UnifiedAccelerator) error {
	if accelerator == nil {
		return fmt.Errorf("GPU加速器不能为空")
	}

	if !accelerator.IsAvailable() {
		return fmt.Errorf("GPU加速器不可用")
	}

	hm.mutex.Lock()
	defer hm.mutex.Unlock()

	hm.accelerators[AcceleratorGPU] = accelerator

	// 同时注册到AcceleratorManager
	if err := hm.manager.RegisterAccelerator(accelerator); err != nil {
		return fmt.Errorf("注册GPU加速器到管理器失败: %v", err)
	}

	return nil
}

// SafeGPUCall 安全调用GPU加速器方法
func (hm *HardwareManager) SafeGPUCall(operation string, fn func(UnifiedAccelerator) error) error {
	gpuAccel := hm.GetGPUAccelerator()
	if gpuAccel == nil {
		return fmt.Errorf("GPU加速器不可用，操作: %s", operation)
	}

	if !gpuAccel.IsAvailable() {
		return fmt.Errorf("GPU加速器状态不可用，操作: %s", operation)
	}

	return fn(gpuAccel)
}

// SafeGPUBatchSearch 安全的GPU批量搜索
func (hm *HardwareManager) SafeGPUBatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	var result [][]AccelResult
	var lastErr error

	err := hm.errorHandler.SafeCall(AcceleratorGPU, "BatchSearch", func() error {
		gpuAccel := hm.GetGPUAccelerator()
		if gpuAccel == nil {
			return fmt.Errorf("GPU加速器不可用")
		}

		if !gpuAccel.IsAvailable() {
			return fmt.Errorf("GPU加速器状态不可用")
		}

		// 输入验证
		if len(queries) == 0 || len(database) == 0 {
			return fmt.Errorf("查询或数据库为空")
		}

		if k <= 0 {
			return fmt.Errorf("k必须为正数")
		}

		var err error
		result, err = gpuAccel.BatchSearch(queries, database, k)
		lastErr = err
		return err
	})

	if err != nil {
		return nil, err
	}

	return result, lastErr
}

// SetDetectionInterval 设置硬件检测间隔
func (hm *HardwareManager) SetDetectionInterval(interval time.Duration) {
	if hm.healthMonitor != nil {
		hm.healthMonitor.SetDetectionInterval(interval)
	}
}

// SetGPUMemoryThreshold 设置GPU内存阈值
func (hm *HardwareManager) SetGPUMemoryThreshold(threshold float64) {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()
	
	if hm.config != nil {
		// 更新配置中的GPU内存限制
		if threshold > 0 && threshold <= 1.0 {
			// 假设threshold是百分比，转换为实际字节数
			if hm.config.GPU.MemoryLimit > 0 {
				hm.config.GPU.MemoryLimit = int64(float64(hm.config.GPU.MemoryLimit) * threshold)
			}
		}
	}
}

// SetCPUUsageThreshold 设置CPU使用率阈值
func (hm *HardwareManager) SetCPUUsageThreshold(threshold float64) {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()
	
	if hm.config != nil && threshold > 0 && threshold <= 1.0 {
		// 更新CPU配置中的线程数，基于阈值调整
		if threshold < 0.8 {
			// 如果阈值较低，减少线程数
			maxThreads := runtime.NumCPU()
			hm.config.CPU.Threads = int(float64(maxThreads) * threshold)
			if hm.config.CPU.Threads < 1 {
				hm.config.CPU.Threads = 1
			}
		}
	}
}

// SetAutoFallback 设置自动回退功能
func (hm *HardwareManager) SetAutoFallback(enabled bool) {
	if hm.recoveryManager != nil {
		hm.recoveryManager.SetAutoFallback(enabled)
	}
}

// CheckGPUHealth 检查GPU健康状态
func (hm *HardwareManager) CheckGPUHealth() error {
	gpuAccel := hm.GetGPUAccelerator()
	if gpuAccel == nil {
		return fmt.Errorf("GPU加速器不可用")
	}
	
	if !gpuAccel.IsAvailable() {
		return fmt.Errorf("GPU加速器状态异常")
	}
	
	// 检查GPU统计信息
	stats := gpuAccel.GetStats()
	if stats.ErrorCount > 0 {
		return fmt.Errorf("GPU存在错误，错误计数: %d", stats.ErrorCount)
	}
	
	return nil
}

// DisableGPU 禁用GPU加速
func (hm *HardwareManager) DisableGPU() error {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()
	
	if hm.config != nil {
		hm.config.GPU.Enable = false
	}
	
	// 关闭GPU加速器
	gpuAccel := hm.GetGPUAccelerator()
	if gpuAccel != nil {
		if err := gpuAccel.Shutdown(); err != nil {
			return fmt.Errorf("关闭GPU加速器失败: %v", err)
		}
	}
	
	return nil
}

// RecoverGPU 恢复GPU加速
func (hm *HardwareManager) RecoverGPU() error {
	hm.mutex.Lock()
	defer hm.mutex.Unlock()
	
	if hm.config != nil {
		hm.config.GPU.Enable = true
	}
	
	// 重新注册GPU加速器
	return hm.ReRegisterAccelerator(AcceleratorGPU)
}

// IsGPUEnabled 检查GPU是否启用
func (hm *HardwareManager) IsGPUEnabled() bool {
	hm.mutex.RLock()
	defer hm.mutex.RUnlock()
	
	return hm.config != nil && hm.config.GPU.Enable
}
