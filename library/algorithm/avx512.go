package algorithm

// #cgo CFLAGS: -mavx512f -mavx512dq -march=native
// #include <immintrin.h>
// #include <stdint.h>
//
// // 使用 AVX512 指令集计算欧氏距离的平方
// double euclidean_distance_squared_avx512(const double* a, const double* b, int len) {
//     __m512d sum = _mm512_setzero_pd(); // 初始化为0
//
//     // 每次处理8个double (512位)
//     for (int i = 0; i < len; i += 8) {
//         __m512d va = _mm512_loadu_pd(a + i); // 加载8个double
//         __m512d vb = _mm512_loadu_pd(b + i); // 加载8个double
//         __m512d diff = _mm512_sub_pd(va, vb); // 计算差值
//         // 使用FMA指令计算平方和累加: sum += diff * diff
//         sum = _mm512_fmadd_pd(diff, diff, sum);
//     }
//
//     // 水平相加得到最终结果
//     return _mm512_reduce_add_pd(sum);
// }
//
// // 使用 AVX512 指令集查找最近的质心
// int find_nearest_centroid_avx512(const double* vec, const double** centroids, int num_centroids, int dim) {
//     double min_dist = 1e100; // 初始化为一个很大的值
//     int nearest_idx = -1;
//
//     for (int i = 0; i < num_centroids; i++) {
//         double dist = euclidean_distance_squared_avx512(vec, centroids[i], dim);
//         if (dist < min_dist) {
//             min_dist = dist;
//             nearest_idx = i;
//         }
//     }
//
//     return nearest_idx;
// }
import "C"

import (
	"math"
	"seetaSearch/library/entity"
	"unsafe"
)

// euclideanDistanceSquaredAVX512 使用 AVX512 指令集计算两个向量间的欧氏距离平方
func euclideanDistanceSquaredAVX512(a, b []float64) float64 {
	if len(a) != len(b) {
		return -1
	}

	// 确保长度是8的倍数，AVX512每次处理8个double
	if len(a)%8 != 0 {
		return -1
	}

	aPtr := (*C.double)(unsafe.Pointer(&a[0]))
	bPtr := (*C.double)(unsafe.Pointer(&b[0]))

	return float64(C.euclidean_distance_squared_avx512(aPtr, bPtr, C.int(len(a))))
}

// findNearestCentroidAVX512 使用 AVX512 指令集查找最近的质心
func findNearestCentroidAVX512(vec []float64, centroids []entity.Point) (int, float64) {
	if len(centroids) == 0 {
		return -1, -1
	}

	// 准备C数组
	vecPtr := (*C.double)(unsafe.Pointer(&vec[0]))

	// 创建质心指针数组
	centroidPtrs := make([]*C.double, len(centroids))
	for i := range centroids {
		centroidPtrs[i] = (*C.double)(unsafe.Pointer(&centroids[i][0]))
	}

	// 调用C函数
	centroidPtrPtr := (**C.double)(unsafe.Pointer(&centroidPtrs[0]))
	nearest := int(C.find_nearest_centroid_avx512(vecPtr, centroidPtrPtr, C.int(len(centroids)), C.int(len(vec))))

	// 计算最小距离（用于返回）
	var minDist float64
	if nearest >= 0 {
		minDist = euclideanDistanceSquaredAVX512(vec, centroids[nearest])
	}

	return nearest, minDist
}

// AVX512CosineSimilarity 使用 AVX512 指令计算余弦相似度
func AVX512CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	n := len(a)
	if n == 0 {
		return 0.0
	}

	// 确保向量长度是 8 的倍数（AVX512 处理 8 个 float64）
	simdLen := (n / 8) * 8

	var dotProduct, normA, normB float64

	if simdLen > 0 {
		// 使用 AVX512 处理对齐部分
		dotProduct += avx512DotProduct(a[:simdLen], b[:simdLen])
		normA += avx512Norm(a[:simdLen])
		normB += avx512Norm(b[:simdLen])
	}

	// 处理剩余元素
	for i := simdLen; i < n; i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// avx512DotProduct AVX512 点积计算（需要汇编实现）
func avx512DotProduct(a, b []float64) float64 {
	// 这里需要汇编实现，简化示例
	var sum float64
	for i := 0; i < len(a); i += 8 {
		// 模拟 AVX512 8-way SIMD 操作
		for j := 0; j < 8 && i+j < len(a); j++ {
			sum += a[i+j] * b[i+j]
		}
	}
	return sum
}

// avx512Norm AVX512 范数计算
func avx512Norm(a []float64) float64 {
	var sum float64
	for i := 0; i < len(a); i += 8 {
		for j := 0; j < 8 && i+j < len(a); j++ {
			sum += a[i+j] * a[i+j]
		}
	}
	return sum
}
