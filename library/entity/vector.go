package entity

import "seetaSearch/library/algorithm"

// CompressedVector 向量量化压缩
type CompressedVector struct {
	Data     []byte              // 压缩后的数据
	Codebook [][]algorithm.Point // 码本（可选，用于PQ压缩）
}
