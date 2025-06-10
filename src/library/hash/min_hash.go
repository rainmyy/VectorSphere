package hash

import (
	"VectorSphere/src/library/log"
	"VectorSphere/src/vector"
	"math"
	"math/rand"
)

/*
这个 MinHash 实现包含以下特性：

1. 标准 MinHash 算法 ：使用多个哈希函数计算最小哈希值
2. 向量到特征集转换 ：将浮点向量转换为特征集合（非零元素）
3. 可配置的内部哈希函数数量 ：支持调整精度和性能平衡
4. Jaccard 相似度估算 ：提供了估算两个 MinHash 签名相似度的辅助函数
5. 错误处理 ：包含完善的输入验证和日志记录
6. 接口兼容 ：实现了 LSHHashFunction 接口
MinHash 特别适用于：

- 稀疏向量的相似性检索
- 文档去重
- 集合相似性计算
- 推荐系统中的用户/物品相似性计算
*/

// MinHash MinHash 哈希函数
type MinHash struct {
	A          []uint64             `json:"a"`          // 随机系数 a
	B          []uint64             `json:"b"`          // 随机系数 b
	P          uint64               `json:"p"`          // 大质数
	NumHashes  int                  `json:"num_hashes"` // 哈希函数数量
	FamilyType vector.LSHFamilyType `json:"family_type"`
}

// NewMinHash 创建新的 MinHash 实例
func NewMinHash(numHashes int) *MinHash {
	// 使用一个大质数
	p := uint64(2147483647) // 2^31 - 1

	a := make([]uint64, numHashes)
	b := make([]uint64, numHashes)

	// 生成随机系数
	for i := 0; i < numHashes; i++ {
		a[i] = uint64(rand.Int63n(int64(p-1))) + 1 // a 必须大于 0
		b[i] = uint64(rand.Int63n(int64(p)))
	}

	return &MinHash{
		A:          a,
		B:          b,
		P:          p,
		NumHashes:  numHashes,
		FamilyType: vector.LSHFamilyMinHash,
	}
}

// Hash 计算 MinHash 值
func (mh *MinHash) Hash(vector []float64) uint64 {
	if len(vector) == 0 {
		log.Warning("Empty vector provided for MinHash computation")
		return 0
	}

	// 将浮点向量转换为特征集合（非零元素的索引）
	features := make([]uint64, 0)
	for i, val := range vector {
		if math.Abs(val) > 1e-9 { // 避免浮点精度问题
			// 将索引和值组合作为特征
			feature := uint64(i)*1000000 + uint64(math.Abs(val)*1000000)
			features = append(features, feature)
		}
	}

	if len(features) == 0 {
		log.Warning("No non-zero features found in vector")
		return 0
	}

	// 计算 MinHash 签名
	minHashValues := make([]uint64, mh.NumHashes)
	for i := 0; i < mh.NumHashes; i++ {
		minHashValues[i] = math.MaxUint64

		// 对每个特征计算哈希值，取最小值
		for _, feature := range features {
			hashValue := (mh.A[i]*feature + mh.B[i]) % mh.P
			if hashValue < minHashValues[i] {
				minHashValues[i] = hashValue
			}
		}
	}

	// 组合多个 MinHash 值
	var combinedHash uint64 = 0
	for i, val := range minHashValues {
		// 使用位移和异或组合
		shiftBits := uint(i % 64)
		combinedHash ^= (val << shiftBits) | (val >> (64 - shiftBits))
	}

	return combinedHash
}

// GetType 获取哈希函数类型
func (mh *MinHash) GetType() vector.LSHFamilyType {
	return mh.FamilyType
}

// GetParameters 获取哈希函数参数
func (mh *MinHash) GetParameters() map[string]interface{} {
	return map[string]interface{}{
		"num_hashes":  mh.NumHashes,
		"p":           mh.P,
		"family_type": mh.FamilyType,
		"a_length":    len(mh.A),
		"b_length":    len(mh.B),
	}
}

// EstimateJaccardSimilarity 估算两个 MinHash 签名的 Jaccard 相似度
func EstimateJaccardSimilarity(sig1, sig2 []uint64) float64 {
	if len(sig1) != len(sig2) {
		return 0.0
	}

	matches := 0
	for i := 0; i < len(sig1); i++ {
		if sig1[i] == sig2[i] {
			matches++
		}
	}

	return float64(matches) / float64(len(sig1))
}
