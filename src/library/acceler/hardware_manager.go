package acceler

import (
	"VectorSphere/src/library/entity"
	"fmt"
	"sync"
	"time"
)

// HardwareManager 硬件管理器（整合原有代码）
type HardwareManager struct {
	accelerators map[string]UnifiedAccelerator
	mutex        sync.RWMutex
	defaultType  string
	manager      *AcceleratorManager
}

// NewHardwareManager 创建新的硬件管理器
func NewHardwareManager() *HardwareManager {
	hm := &HardwareManager{
		accelerators: make(map[string]UnifiedAccelerator),
		defaultType:  "cpu",
		manager:      NewAcceleratorManager(),
	}

	// 注册可用的硬件加速器
	hm.registerAllAccelerators()

	// 配置默认策略
	hm.setupDefaultStrategies()

	return hm
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

// registerAllAccelerators 注册所有可用的硬件加速器
func (hm *HardwareManager) registerAllAccelerators() {
	// 注册CPU加速器 (使用FAISS作为CPU实现)
	cpuAcc := NewFAISSAccelerator(&FAISSConfig{DeviceID: 0, IndexType: "IDMap,Flat"})
	if cpuAcc.IsAvailable() {
		hm.RegisterAccelerator(AcceleratorCPU, cpuAcc)
		if err := hm.manager.RegisterAccelerator(cpuAcc); err != nil {
			fmt.Printf("警告: 注册CPU加速器失败: %v\n", err)
		}
	}

	// 注册GPU加速器
	for i := 0; i < 4; i++ { // 最多检测4个GPU
		gpuAcc := NewGPUAccelerator(&GPUConfig{DeviceID: i, IndexType: "IVF,Flat", NumClusters: 128})
		if gpuAcc.IsAvailable() {
			name := fmt.Sprintf("AcceleratorGPU_%d", i)
			hm.RegisterAccelerator(name, gpuAcc)
			if err := hm.manager.RegisterAccelerator(gpuAcc); err != nil {
				fmt.Printf("警告: 注册GPU加速器 %s 失败: %v\n", name, err)
				continue
			}
			hm.defaultType = name // 优先使用GPU
			break                 // 只注册第一个可用的GPU
		}
	}

	// 注册FPGA加速器
	for i := 0; i < 2; i++ { // 最多检测2个FPGA
		config := &FPGAConfig{
			DeviceID:      i,
			BitstreamPath: fmt.Sprintf("path/to/bitstream_%d.bin", i), // 动态路径
			BufferSize:    1024 * 1024 * 1024,                         // 1GB
		}
		fpgaAcc := NewFPGAAccelerator(0, config)
		if fpgaAcc.IsAvailable() {
			name := fmt.Sprintf("%s_%d", AcceleratorFPGA, i)
			hm.RegisterAccelerator(name, fpgaAcc)
			if err := hm.manager.RegisterAccelerator(fpgaAcc); err != nil {
				fmt.Printf("警告: 注册FPGA加速器 %s 失败: %v\n", name, err)
				continue
			}
			break
		}
	}

	// 注册PMem加速器
	pmemConfig := &PMemConfig{
		DevicePath: "/mnt/pmem0/vectors.db",
		PoolSize:   10 * 1024 * 1024 * 1024, // 10GB
	}
	pmemAcc := NewPMemAccelerator(pmemConfig)
	if pmemAcc.IsAvailable() {
		hm.RegisterAccelerator(AcceleratorPMem, pmemAcc)
		if err := hm.manager.RegisterAccelerator(pmemAcc); err != nil {
			fmt.Printf("警告: 注册PMem加速器失败: %v\n", err)
		}
	}

	// 注册RDMA加速器
	rdmaConfig := &RDMAConfig{
		DeviceID: 0,
		PortNum:  1,
	}
	rdmaAcc := NewRDMAAccelerator(rdmaConfig)
	if rdmaAcc.IsAvailable() {
		hm.RegisterAccelerator(AcceleratorRDMA, rdmaAcc)
		if err := hm.manager.RegisterAccelerator(rdmaAcc); err != nil {
			fmt.Printf("警告: 注册RDMA加速器失败: %v\n", err)
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
