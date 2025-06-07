package entity

// Point 代表一个数据点（向量）
type Point []float64

// CompressedVector 向量量化压缩
type CompressedVector struct {
	Data     []byte    // 压缩后的数据
	Codebook [][]Point // 码本（可选，用于PQ压缩）
}
