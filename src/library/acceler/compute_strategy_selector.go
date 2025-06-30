package acceler

// ComputeStrategySelector 计算策略选择器
type ComputeStrategySelector struct {
	Detector        *HardwareDetector
	HardwareManager *HardwareManager
	GpuThreshold    int
	FpgaThreshold   int
	RdmaThreshold   int
}

// NewComputeStrategySelector 创建计算策略选择器
func NewComputeStrategySelector() *ComputeStrategySelector {
	return &ComputeStrategySelector{
		Detector:      &HardwareDetector{},
		GpuThreshold:  10000,  // 默认10000个向量以上使用GPU
		FpgaThreshold: 50000,  // 默认50000个向量以上考虑FPGA
		RdmaThreshold: 100000, // 默认100000个向量以上考虑RDMA
	}
}

// SetHardwareManager 设置硬件管理器
func (css *ComputeStrategySelector) SetHardwareManager(hm *HardwareManager) {
	css.HardwareManager = hm
}

func (css *ComputeStrategySelector) GetHardwareCapabilities() HardwareCapabilities {
	return css.Detector.GetHardwareCapabilities()
}

// SelectOptimalStrategy 选择最优计算策略
func (css *ComputeStrategySelector) SelectOptimalStrategy(dataSize int, vectorDim int) ComputeStrategy {
	caps := css.Detector.GetHardwareCapabilities()

	// 如果有硬件管理器，优先使用硬件管理器的信息
	if css.HardwareManager != nil {
		return css.selectStrategyWithHardwareManager(dataSize, vectorDim, caps)
	}

	// 回退到原有逻辑
	return css.selectStrategyFallback(dataSize, vectorDim, caps)
}

// selectStrategyWithHardwareManager 使用硬件管理器选择策略
func (css *ComputeStrategySelector) selectStrategyWithHardwareManager(dataSize int, vectorDim int, caps HardwareCapabilities) ComputeStrategy {
	// 超大数据量优先考虑RDMA分布式计算
	if dataSize >= css.RdmaThreshold {
		if rdmaAcc, exists := css.HardwareManager.GetAccelerator(AcceleratorRDMA); exists && rdmaAcc != nil && rdmaAcc.IsAvailable() {
			return StrategyRDMA
		}
	}

	// 大数据量优先考虑FPGA或GPU
	if dataSize >= css.FpgaThreshold {
		if fpgaAcc, exists := css.HardwareManager.GetAccelerator(AcceleratorFPGA); exists && fpgaAcc != nil && fpgaAcc.IsAvailable() {
			return StrategyFPGA
		}
	}

	// 中大数据量考虑GPU
	if dataSize >= css.GpuThreshold {
		if gpuAcc, exists := css.HardwareManager.GetAccelerator(AcceleratorGPU); exists && gpuAcc != nil && gpuAcc.IsAvailable() {
			return StrategyGPU
		}
	}

	// CPU加速器策略选择
	if cpuAcc, exists := css.HardwareManager.GetAccelerator(AcceleratorCPU); exists && cpuAcc != nil && cpuAcc.IsAvailable() {
		// 检查向量维度是否适合SIMD指令
		if vectorDim%8 == 0 {
			// 验证CPU是否真正支持AVX512
			if caps.HasAVX512 && css.VerifyCPUSupport(AcceleratorCPU, "avx512") {
				return StrategyAVX512
			}
			// 验证CPU是否真正支持AVX2
			if caps.HasAVX2 && css.VerifyCPUSupport(AcceleratorCPU, "avx2") {
				return StrategyAVX2
			}
		}
	}

	// 默认使用标准实现
	return StrategyStandard
}

// selectStrategyFallback 回退策略选择
func (css *ComputeStrategySelector) selectStrategyFallback(dataSize int, vectorDim int, caps HardwareCapabilities) ComputeStrategy {
	// 大数据量优先考虑GPU
	if dataSize >= css.GpuThreshold && caps.HasGPU {
		return StrategyGPU
	}

	// 中等数据量考虑AVX指令集
	if vectorDim%8 == 0 && caps.HasAVX512 {
		return StrategyAVX512
	}

	if vectorDim%8 == 0 && caps.HasAVX2 {
		return StrategyAVX2
	}

	// 默认使用标准实现
	return StrategyStandard
}

// VerifyCPUSupport 验证CPU是否真正支持指定的指令集
func (css *ComputeStrategySelector) VerifyCPUSupport(acceleratorType string, instructionSet string) bool {
	if css.HardwareManager == nil {
		return false
	}

	// 获取CPU加速器配置
	cfg, err := css.HardwareManager.GetAcceleratorConfig(acceleratorType)
	if err != nil {
		return false
	}

	cpuCfg, ok := cfg.(CPUConfig)
	if !ok {
		return false
	}

	// 检查CPU配置中是否启用了相应的指令集
	switch instructionSet {
	case "avx512":
		return cpuCfg.Enable && cpuCfg.VectorWidth >= 512
	case "avx2":
		return cpuCfg.Enable && cpuCfg.VectorWidth >= 256
	default:
		return cpuCfg.Enable
	}
}
