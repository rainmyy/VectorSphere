package acceler

import (
	"fmt"
	"github.com/klauspost/cpuid"
	"runtime"
	"sync"
	"time"
)

// HardwareDetector 硬件检测器
type HardwareDetector struct {
	once         sync.Once
	capabilities HardwareCapabilities
	mu           sync.RWMutex
}

// GlobalHardwareDetector 全局硬件检测器实例
var GlobalHardwareDetector = &HardwareDetector{}

// DetectAllHardware 检测所有硬件能力
func (hd *HardwareDetector) DetectAllHardware() HardwareCapabilities {
	hd.once.Do(func() {
		hd.capabilities = HardwareCapabilities{
			HasAVX2:           cpuid.CPU.AVX2(),
			HasAVX512:         cpuid.CPU.AVX512F() && cpuid.CPU.AVX512DQ(),
			HasGPU:            hd.detectGPUSupport(),
			CPUCores:          runtime.NumCPU(),
			GPUDevices:        hd.getGPUDeviceCount(),
			Type:              "System",
			MemorySize:        hd.getSystemMemorySize(),
			ComputeUnits:      runtime.NumCPU(),
			MaxBatchSize:      10000,
			SupportedOps:      []string{"distance_compute", "batch_compute", "cosine_similarity"},
			PerformanceRating: hd.calculatePerformanceRating(),
			Bandwidth:         hd.getMemoryBandwidth(),
			Latency:           time.Microsecond * 100,
			PowerConsumption:  hd.estimatePowerConsumption(),
			SpecialFeatures:   hd.detectSpecialFeatures(),
		}
	})
	return hd.capabilities
}

// detectGPUSupport 检测GPU支持
func (hd *HardwareDetector) detectGPUSupport() bool {
	// 尝试检测CUDA设备
	if hd.detectCUDASupport() {
		return true
	}
	// 尝试检测OpenCL设备
	if hd.detectOpenCLSupport() {
		return true
	}
	return false
}

// detectCUDASupport 检测CUDA支持
func (hd *HardwareDetector) detectCUDASupport() bool {
	// 这里应该调用CUDA API检测
	// 简化实现，实际应该调用C.cudaGetDeviceCount()
	return false
}

// detectOpenCLSupport 检测OpenCL支持
func (hd *HardwareDetector) detectOpenCLSupport() bool {
	// 这里应该调用OpenCL API检测
	return false
}

// getGPUDeviceCount 获取GPU设备数量
func (hd *HardwareDetector) getGPUDeviceCount() int {
	// 这里需要调用CUDA API获取设备数量
	// 简化实现，实际应该调用C.cudaGetDeviceCount()
	return 0
}

// getSystemMemorySize 获取系统内存大小
func (hd *HardwareDetector) getSystemMemorySize() int64 {
	// 简化实现，实际应该读取系统信息
	return 8 * 1024 * 1024 * 1024 // 8GB
}

// calculatePerformanceRating 计算性能评级
func (hd *HardwareDetector) calculatePerformanceRating() float64 {
	rating := 5.0 // 基础评分

	if cpuid.CPU.AVX2() {
		rating += 1.0
	}
	if cpuid.CPU.AVX512F() {
		rating += 1.5
	}
	if hd.detectGPUSupport() {
		rating += 2.0
	}

	// 根据CPU核心数调整
	cores := runtime.NumCPU()
	if cores >= 8 {
		rating += 1.0
	} else if cores >= 4 {
		rating += 0.5
	}

	return rating
}

// getMemoryBandwidth 获取内存带宽
func (hd *HardwareDetector) getMemoryBandwidth() int64 {
	// 简化实现，实际应该通过基准测试获取
	return 25 * 1024 * 1024 * 1024 // 25GB/s
}

// estimatePowerConsumption 估算功耗
func (hd *HardwareDetector) estimatePowerConsumption() float64 {
	cores := runtime.NumCPU()
	basePower := 15.0                 // 基础功耗
	corePower := float64(cores) * 2.5 // 每核心2.5W

	if hd.detectGPUSupport() {
		corePower += 150.0 // GPU额外功耗
	}

	return basePower + corePower
}

// detectSpecialFeatures 检测特殊功能
func (hd *HardwareDetector) detectSpecialFeatures() []string {
	features := []string{}

	if cpuid.CPU.AVX2() {
		features = append(features, "avx2")
	}
	if cpuid.CPU.AVX512F() {
		features = append(features, "avx512")
	}
	if cpuid.CPU.FMA3() {
		features = append(features, "fma3")
	}
	if hd.detectGPUSupport() {
		features = append(features, "gpu_acceleration")
	}
	if hd.detectFPGASupport() {
		features = append(features, "fpga_acceleration")
	}
	if hd.detectRDMASupport() {
		features = append(features, "rdma_networking")
	}
	if hd.detectPMemSupport() {
		features = append(features, "persistent_memory")
	}

	return features
}

// detectFPGASupport 检测FPGA支持
func (hd *HardwareDetector) detectFPGASupport() bool {
	// 简化实现，实际应该检测FPGA设备
	return false
}

// detectRDMASupport 检测RDMA支持
func (hd *HardwareDetector) detectRDMASupport() bool {
	// 简化实现，实际应该检测InfiniBand或RoCE设备
	return false
}

// detectPMemSupport 检测持久内存支持
func (hd *HardwareDetector) detectPMemSupport() bool {
	// 简化实现，实际应该检测NVDIMM设备
	return false
}

// GetCapabilities 获取硬件能力（线程安全）
func (hd *HardwareDetector) GetCapabilities() HardwareCapabilities {
	hd.mu.RLock()
	defer hd.mu.RUnlock()
	return hd.capabilities
}

// RefreshCapabilities 刷新硬件能力检测
func (hd *HardwareDetector) RefreshCapabilities() {
	hd.mu.Lock()
	defer hd.mu.Unlock()
	hd.once = sync.Once{} // 重置once
	hd.DetectAllHardware()
}

// GetOptimalAcceleratorType 获取最优加速器类型
func (hd *HardwareDetector) GetOptimalAcceleratorType(workload WorkloadProfile) string {
	caps := hd.DetectAllHardware()

	// 根据工作负载特征选择最优加速器
	if workload.RequiresHighThroughput && caps.HasGPU {
		return AcceleratorGPU
	}

	if workload.RequiresLowLatency && hd.detectFPGASupport() {
		return AcceleratorFPGA
	}

	if workload.RequiresDistributed && hd.detectRDMASupport() {
		return AcceleratorRDMA
	}

	if workload.RequiresPersistence && hd.detectPMemSupport() {
		return AcceleratorPMem
	}

	// 默认使用CPU加速器
	return AcceleratorCPU
}

// ValidateHardwareRequirements 验证硬件需求
func (hd *HardwareDetector) ValidateHardwareRequirements(requirements HardwareCapabilities) error {
	caps := hd.DetectAllHardware()

	if requirements.HasAVX2 && !caps.HasAVX2 {
		return fmt.Errorf("需要AVX2支持，但当前系统不支持")
	}

	if requirements.HasAVX512 && !caps.HasAVX512 {
		return fmt.Errorf("需要AVX512支持，但当前系统不支持")
	}

	if requirements.HasGPU && !caps.HasGPU {
		return fmt.Errorf("需要GPU支持，但当前系统不支持")
	}

	if requirements.CPUCores > caps.CPUCores {
		return fmt.Errorf("需要%d个CPU核心，但当前系统只有%d个", requirements.CPUCores, caps.CPUCores)
	}

	if requirements.MemorySize > caps.MemorySize {
		return fmt.Errorf("需要%d字节内存，但当前系统只有%d字节", requirements.MemorySize, caps.MemorySize)
	}

	return nil
}

// GetHardwareRecommendations 获取硬件推荐
func (hd *HardwareDetector) GetHardwareRecommendations(workload WorkloadProfile) []string {
	recommendations := []string{}
	caps := hd.DetectAllHardware()

	if workload.RequiresHighThroughput && !caps.HasGPU {
		recommendations = append(recommendations, "建议添加GPU以提高吞吐量")
	}

	if workload.RequiresLowLatency && !hd.detectFPGASupport() {
		recommendations = append(recommendations, "建议添加FPGA以降低延迟")
	}

	if workload.DataSize > caps.MemorySize {
		recommendations = append(recommendations, "建议增加内存容量")
	}

	if workload.VectorDimension > 1024 && !caps.HasAVX512 {
		recommendations = append(recommendations, "建议使用支持AVX512的CPU以处理高维向量")
	}

	return recommendations
}
