package acceler

import "sync"

// Hash 简单哈希函数
func Hash(key interface{}) int {
	switch v := key.(type) {
	case string:
		h := 0
		for _, c := range v {
			h = 31*h + int(c)
		}
		return h
	case int:
		return v
	default:
		return 0
	}
}

// CalculateEuclideanDistance 计算欧几里得距离
// 注意：此函数返回平方距离，而不是实际距离，避免开方运算
// 保留此函数作为兼容性备份
func CalculateEuclideanDistance(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return float64(^uint(0) >> 1) // 返回最大float64值
	}

	sum := 0.0
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}
	return sum // 返回平方距离，避免开方运算
}

// CalculateDistanceWithCalculator 使用距离计算器计算向量距离
// 如果提供了计算器，则使用计算器；否则使用默认的欧几里得距离
// 注意：此函数返回平方距离，而不是实际距离，避免开方运算
func CalculateDistanceWithCalculator(v1, v2 []float64, calculator interface{}) float64 {
	// 尝试使用DistanceCalculator接口
	if calc, ok := calculator.(interface {
		CalculateSquared([]float64, []float64) float64
	}); ok {
		return calc.CalculateSquared(v1, v2)
	}

	// 回退到默认实现
	return CalculateEuclideanDistance(v1, v2)
}

// 全局距离计算器实例，用于在没有VectorDB实例的情况下访问距离计算器
var globalDistanceCalculator interface{}
var globalDistanceCalculatorMu sync.RWMutex

// SetGlobalDistanceCalculator 设置全局距离计算器
func SetGlobalDistanceCalculator(calculator interface{}) {
	globalDistanceCalculatorMu.Lock()
	defer globalDistanceCalculatorMu.Unlock()
	globalDistanceCalculator = calculator
}

// GetGlobalDistanceCalculator 获取全局距离计算器
func GetGlobalDistanceCalculator() (interface{}, bool) {
	globalDistanceCalculatorMu.RLock()
	defer globalDistanceCalculatorMu.RUnlock()
	return globalDistanceCalculator, globalDistanceCalculator != nil
}
