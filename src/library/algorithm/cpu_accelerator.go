//go:build !gpu

package algorithm

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
