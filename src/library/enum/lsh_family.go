package enum

// LSHFamilyType LSH 族类型
type LSHFamilyType int

const (
	LSHFamilyRandomProjection LSHFamilyType = iota // 随机投影
	LSHFamilyMinHash                               // MinHash
	LSHFamilyP2LSH                                 // p-stable LSH
	LSHFamilyAngular                               // 角度 LSH
	LSHFamilyEuclidean                             // 欧几里得 LSH
)
