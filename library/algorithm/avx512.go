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
	"VectorSphere/library/entity"
	"errors"
	"math"
	"runtime"
	"unsafe"
)

// alignTo8 returns the largest multiple of 8 less than or equal to n
func alignTo8(n int) int {
	return n / 8 * 8
}

// EuclideanDistanceSquaredAVX512 computes squared Euclidean distance using AVX512. Returns error if input is invalid.
func EuclideanDistanceSquaredAVX512(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("vector length mismatch")
	}
	if len(a) == 0 || len(a)%8 != 0 {
		return 0, errors.New("vector length must be >0 and a multiple of 8")
	}
	aPtr := (*C.double)(unsafe.Pointer(&a[0]))
	bPtr := (*C.double)(unsafe.Pointer(&b[0]))
	res := float64(C.euclidean_distance_squared_avx512(aPtr, bPtr, C.int(len(a))))
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	return res, nil
}

// FindNearestCentroidAVX512 finds the nearest centroid using AVX512. Returns index, distance, error.
func FindNearestCentroidAVX512(vec []float64, centroids []entity.Point) (int, float64, error) {
	if len(centroids) == 0 {
		return -1, 0, errors.New("no centroids")
	}
	if len(vec) == 0 || len(vec)%8 != 0 {
		return -1, 0, errors.New("vector length must be >0 and a multiple of 8")
	}
	for i, c := range centroids {
		if len(c) != len(vec) {
			return -1, 0, errors.New("centroid length mismatch at index "+string(i))
		}
	}
	vecPtr := (*C.double)(unsafe.Pointer(&vec[0]))
	centroidPtrs := make([]*C.double, len(centroids))
	for i := range centroids {
		centroidPtrs[i] = (*C.double)(unsafe.Pointer(&centroids[i][0]))
	}
	centroidPtrPtr := (**C.double)(unsafe.Pointer(&centroidPtrs[0]))
	idx := int(C.find_nearest_centroid_avx512(vecPtr, centroidPtrPtr, C.int(len(centroids)), C.int(len(vec))))
	runtime.KeepAlive(vec)
	runtime.KeepAlive(centroids)
	if idx < 0 {
		return -1, 0, errors.New("no nearest centroid found")
	}
	dist, _ := EuclideanDistanceSquaredAVX512(vec, centroids[idx])
	return idx, dist, nil
}

// AVX512CosineSimilarity computes cosine similarity using AVX512. Returns 0 if input invalid.
func AVX512CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}
	n := len(a)
	simdLen := alignTo8(n)
	var dotProduct, normA, normB float64
	if simdLen > 0 {
		dotProduct += avx512DotOrNorm(a[:simdLen], b[:simdLen], true)
		normA += avx512DotOrNorm(a[:simdLen], a[:simdLen], false)
		normB += avx512DotOrNorm(b[:simdLen], b[:simdLen], false)
	}
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

// avx512DotOrNorm computes dot product if isDot is true, else computes norm (sum of squares)
func avx512DotOrNorm(a, b []float64, isDot bool) float64 {
	var sum float64
	for i := 0; i < len(a); i += 8 {
		for j := 0; j < 8 && i+j < len(a); j++ {
			if isDot {
				sum += a[i+j] * b[i+j]
			} else {
				sum += a[i+j] * a[i+j]
			}
		}
	}
	return sum
}
