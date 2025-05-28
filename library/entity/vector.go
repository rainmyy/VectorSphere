package entity

// CompressedVector 向量量化压缩
type CompressedVector struct {
	Data     []byte    // 压缩后的数据
	Codebook []float64 // 码本（可选，用于PQ压缩）
}
