package vector

import (
	"VectorSphere/src/library/logger"
	"fmt"

	"gonum.org/v1/gonum/mat"
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

	// 将中心化数据转换为 gonum 矩阵
	data := mat.NewDense(len(centeredVectors), dim, nil)
	for i, vec := range centeredVectors {
		data.SetRow(i, vec)
	}

	// 计算协方差矩阵: C = (1/n-1) * X^T * X
	// 其中 X 是中心化数据矩阵 (n x d)
	var cov mat.SymDense
	// 使用正确的协方差矩阵计算：C = X^T * X / (n-1)
	var dataT mat.Dense
	dataT.CloneFrom(data.T())
	cov.SymOuterK(1.0/float64(len(vectors)-1), &dataT)

	// 特征值分解
	var eig mat.EigenSym
	if ok := eig.Factorize(&cov, true); !ok {
		return fmt.Errorf("无法进行特征值分解")
	}

	// 获取特征值和特征向量
	eigenvalues := eig.Values(nil)
	var eigenvectors = &mat.Dense{}
	eig.VectorsTo(eigenvectors)

	// 根据方差比例或目标维度选择主成分
	numComponents := 0
	if varianceRatio > 0 && varianceRatio < 1.0 {
		totalVariance := 0.0
		for _, val := range eigenvalues {
			if val > 0 { // 只考虑正特征值
				totalVariance += val
			}
		}
		if totalVariance == 0 {
			logger.Warning("总方差为0，无法根据方差比例选择主成分。将尝试使用targetDim。")
			// 如果总方差为0，则基于方差比例的计算无意义，此时依赖 targetDim
			if targetDim > 0 && targetDim <= dim {
				numComponents = targetDim
			} else {
				return fmt.Errorf("总方差为0且targetDim无效 (%d)，无法确定主成分数量", targetDim)
			}
		} else {
			currentVariance := 0.0
			// 特征值由 eig.Factorize 保证是升序排列的，所以从后往前取是取最大的
			for i := len(eigenvalues) - 1; i >= 0; i-- {
				if eigenvalues[i] <= 0 { // 忽略非正特征值
					continue
				}
				currentVariance += eigenvalues[i]
				numComponents++
				if currentVariance/totalVariance >= varianceRatio {
					break
				}
			}
		}
		// 如果同时指定了 targetDim，需要处理两种情况：
		// 1. 方差比例计算出的维度大于目标维度：使用目标维度
		// 2. 方差比例计算出的维度小于目标维度：使用目标维度（确保至少达到目标维度）
		if targetDim > 0 {
			if targetDim < numComponents {
				logger.Info("根据方差比例计算的主成分数 (%d) 大于目标维度 (%d)，将使用目标维度。", numComponents, targetDim)
				numComponents = targetDim
			} else if targetDim > numComponents {
				logger.Info("根据方差比例计算的主成分数 (%d) 小于目标维度 (%d)，将使用目标维度。", numComponents, targetDim)
				numComponents = targetDim
			}
		}
	} else if targetDim > 0 {
		numComponents = targetDim
	} else {
		return fmt.Errorf("必须指定有效的 targetDimension (>0) 或 varianceRatio (0 < ratio < 1)")
	}

	if numComponents <= 0 || numComponents > dim {
		return fmt.Errorf("计算出的主成分数量无效: %d (原始维度: %d)", numComponents, dim)
	}

	// 提取主成分 (最大的 numComponents 个特征向量)
	// eigenvectors 的列是特征向量，按特征值升序排列。
	// 我们需要与最大的 numComponents 个特征值相对应的特征向量，即 eigenvectors 的最后 numComponents 列。

	// 检查特征向量矩阵的维度
	eigRows, eigCols := eigenvectors.Dims()
	logger.Debug("特征向量矩阵维度: %d x %d, 原始维度: %d, 目标主成分数: %d", eigRows, eigCols, dim, numComponents)

	if eigCols != dim {
		return fmt.Errorf("特征向量矩阵列数 (%d) 与原始维度 (%d) 不匹配", eigCols, dim)
	}

	p := mat.NewDense(dim, numComponents, nil) // p 是主成分矩阵 W
	for j := 0; j < numComponents; j++ {
		// p 的第 j 列 (0-indexed) 对应 eigenvectors 的第 (dim - 1 - j) 列 (0-indexed)
		// 这是因为 eigenvectors 的列是按特征值升序排列的。
		columnToExtract := dim - 1 - j

		// 确保列索引在有效范围内
		if columnToExtract < 0 || columnToExtract >= eigCols {
			return fmt.Errorf("列索引 %d 超出特征向量矩阵范围 [0, %d)", columnToExtract, eigCols)
		}

		// mat.Col copies the column into a new []float64 slice.
		colData := make([]float64, dim)
		mat.Col(colData, columnToExtract, eigenvectors)
		p.SetCol(j, colData)
	}

	// 应用降维变换: Y = Xc * W (其中 Xc 是中心化数据, W 是主成分)
	reducedData := mat.NewDense(len(vectors), numComponents, nil)
	reducedData.Mul(data, p)

	// 存储PCA配置，这应该在所有计算成功后进行
	db.pcaConfig = &PCAConfig{
		TargetDimension: numComponents,
		VarianceRatio:   varianceRatio, // 记录用于计算的方差比
		Components:      matrixToSlice(p),
		Mean:            mean,
	}

	// 更新向量数据库
	reducedVectors := make(map[string][]float64)
	for i, id := range ids {
		reducedVectors[id] = reducedData.RawRowView(i)
	}

	db.vectors = reducedVectors
	db.vectorDim = numComponents
	db.indexed = false // 需要重建索引

	logger.Info("PCA 降维完成: %d -> %d 维 (保留方差比例目标: %.2f)", dim, numComponents, varianceRatio)
	return nil
}

// Helper function to convert gonum matrix to [][]float64
func matrixToSlice(m mat.Matrix) [][]float64 {
	r, c := m.Dims()
	slice := make([][]float64, r)
	for i := 0; i < r; i++ {
		slice[i] = make([]float64, c)
		for j := 0; j < c; j++ {
			slice[i][j] = m.At(i, j)
		}
	}
	return slice
}
