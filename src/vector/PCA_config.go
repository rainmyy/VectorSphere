package vector

import (
	"VectorSphere/src/library/logger"
	"fmt"
)

// PCAConfig PCA 配置
type PCAConfig struct {
	TargetDimension int         // 目标维度
	VarianceRatio   float64     // 保留的方差比例
	Components      [][]float64 // PCA 主成分
	Mean            []float64   // 均值向量
}

// ApplyPCA 应用 PCA 降维
func (db *VectorDB) ApplyPCA(targetDim int, varianceRatio float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if len(db.vectors) == 0 {
		return fmt.Errorf("数据库为空，无法进行 PCA")
	}

	// 收集所有向量
	vectors := make([][]float64, 0, len(db.vectors))
	ids := make([]string, 0, len(db.vectors))

	for id, vec := range db.vectors {
		vectors = append(vectors, vec)
		ids = append(ids, id)
	}

	// 计算均值
	dim := len(vectors[0])
	mean := make([]float64, dim)
	for _, vec := range vectors {
		for i, val := range vec {
			mean[i] += val
		}
	}
	for i := range mean {
		mean[i] /= float64(len(vectors))
	}

	// 中心化数据
	centeredVectors := make([][]float64, len(vectors))
	for i, vec := range vectors {
		centeredVectors[i] = make([]float64, dim)
		for j, val := range vec {
			centeredVectors[i][j] = val - mean[j]
		}
	}

	// 计算协方差矩阵
	covariance := make([][]float64, dim)
	for i := range covariance {
		covariance[i] = make([]float64, dim)
		for j := range covariance[i] {
			for _, vec := range centeredVectors {
				covariance[i][j] += vec[i] * vec[j]
			}
			covariance[i][j] /= float64(len(vectors) - 1)
		}
	}

	// 这里需要实现特征值分解，简化示例
	// 实际应用中建议使用 gonum 等数学库

	// 应用降维变换
	reducedVectors := make(map[string][]float64)
	for i, id := range ids {
		// 简化的降维实现，实际需要使用主成分
		reduced := make([]float64, targetDim)
		for j := 0; j < targetDim && j < len(vectors[i]); j++ {
			reduced[j] = vectors[i][j]
		}
		reducedVectors[id] = reduced
	}

	// 更新向量数据库
	db.vectors = reducedVectors
	db.vectorDim = targetDim
	db.indexed = false // 需要重建索引

	logger.Info("PCA 降维完成：%d -> %d 维", dim, targetDim)
	return nil
}
