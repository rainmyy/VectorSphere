package acceler

// ComputeStrategySelector 计算策略选择器
type ComputeStrategySelector struct {
	detector     *HardwareDetector
	gpuThreshold int
}

// NewComputeStrategySelector 创建计算策略选择器
func NewComputeStrategySelector() *ComputeStrategySelector {
	return &ComputeStrategySelector{
		detector:     &HardwareDetector{},
		gpuThreshold: 10000, // 默认10000个向量以上使用GPU
	}
}

func (css *ComputeStrategySelector) GetHardwareCapabilities() HardwareCapabilities {
	return css.detector.GetHardwareCapabilities()
}

// SelectOptimalStrategy 选择最优计算策略
func (css *ComputeStrategySelector) SelectOptimalStrategy(dataSize int, vectorDim int) ComputeStrategy {
	caps := css.detector.GetHardwareCapabilities()

	// 大数据量优先考虑GPU
	if dataSize >= css.gpuThreshold && caps.HasGPU {
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
