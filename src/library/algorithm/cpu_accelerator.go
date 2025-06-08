//go:build !gpu

package algorithm

import "C"
import "fmt"

func NewFAISSAccelerator(deviceID int, indexType string) *FAISSAccelerator {
	return &FAISSAccelerator{deviceID: deviceID, indexType: indexType}
}

func (c *FAISSAccelerator) Initialize() error {
	return fmt.Errorf("GPU加速功能未启用，请安装FAISS-GPU库")
}

func (c *FAISSAccelerator) BatchCosineSimilarity(queries [][]float64, database [][]float64) ([][]float64, error) {
	return nil, fmt.Errorf("GPU加速功能未启用")
}

func (c *FAISSAccelerator) BatchSearch(queries [][]float64, database [][]float64, k int) ([][]AccelResult, error) {
	return nil, fmt.Errorf("GPU加速功能未启用")
}

func (c *FAISSAccelerator) Cleanup() error {
	return nil
}

// CheckGPUAvailability 公共方法，供外部调用检查GPU可用性
func (c *FAISSAccelerator) CheckGPUAvailability() error {
	return fmt.Errorf("GPU加速功能未启用")
}

func (c *FAISSAccelerator) GetGPUMemoryInfo() (free, total uint64, err error) {
	// 模拟返回值
	return 8 * 1024 * 1024 * 1024, 12 * 1024 * 1024 * 1024, nil // 8GB free, 12GB total
}
