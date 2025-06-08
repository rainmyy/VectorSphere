package algorithm

// #cgo CFLAGS: -mavx2 -march=native
// #if defined(__FMA__) || defined(__AVX2__)
// #define USE_FMA 1
// #endif
// #include <immintrin.h>
// #include <stdint.h>
//
// // 使用 AVX2 指令集计算欧氏距离的平方
// double euclidean_distance_squared_avx2(const double* a, const double* b, int len) {
//     __m256d sum = _mm256_setzero_pd(); // 初始化为0
//
//     // 每次处理4个double (256位)
//     for (int i = 0; i < len; i += 4) {
//         __m256d va = _mm256_loadu_pd(a + i); // 加载4个double
//         __m256d vb = _mm256_loadu_pd(b + i); // 加载4个double
//         __m256d diff = _mm256_sub_pd(va, vb); // 计算差值
//         // 使用FMA指令计算平方和累加: sum += diff * diff
//         sum = _mm256_fmadd_pd(diff, diff, sum);
//     }
//
//     // 水平相加得到最终结果
//     __m128d sum128 = _mm_add_pd(_mm256_extractf128_pd(sum, 0), _mm256_extractf128_pd(sum, 1));
//     __m128d sum64 = _mm_add_sd(sum128, _mm_unpackhi_pd(sum128, sum128));
//
//     double result;
//     _mm_store_sd(&result, sum64);
//     return result;
// }
//
// // 使用 AVX2 指令集查找最近的质心
// int find_nearest_centroid_avx2(const double* vec, const double** centroids, int num_centroids, int dim) {
//     double min_dist = 1e100; // 初始化为一个很大的值
//     int nearest_idx = -1;
//
//     for (int i = 0; i < num_centroids; i++) {
//         double dist = euclidean_distance_squared_avx2(vec, centroids[i], dim);
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
	"VectorSphere/src/library/entity"
	"errors"
	"github.com/klauspost/cpuid"
	"runtime"
	"unsafe"
)

// EuclideanDistanceSquaredAVX2 computes squared Euclidean distance using AVX2. Returns error if input is invalid.
func EuclideanDistanceSquaredAVX2(a, b []float64) (float64, error) {
	// 检查AVX2支持
	if !cpuid.CPU.AVX2() {
		return EuclideanDistanceSquaredDefault(a, b), nil
	}

	if len(a) != len(b) {
		return 0, errors.New("vector length mismatch")
	}
	if len(a) == 0 || len(a)%8 != 0 {
		return 0, errors.New("vector length must be >0 and a multiple of 8")
	}
	aPtr := (*C.double)(unsafe.Pointer(&a[0]))
	bPtr := (*C.double)(unsafe.Pointer(&b[0]))
	res := float64(C.euclidean_distance_squared_avx2(aPtr, bPtr, C.int(len(a))))
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	return res, nil
}

// FindNearestCentroidAVX2 finds the nearest centroid using AVX2. Returns index, distance, error.
func FindNearestCentroidAVX2(vec []float64, centroids []entity.Point) (int, float64, error) {
	if len(centroids) == 0 {
		return -1, 0, errors.New("no centroids")
	}
	if len(vec) == 0 || len(vec)%8 != 0 {
		return -1, 0, errors.New("vector length must be >0 and a multiple of 8")
	}
	for i, c := range centroids {
		if len(c) != len(vec) {
			return -1, 0, errors.New("centroid length mismatch at index " + string(i))
		}
	}
	vecPtr := (*C.double)(unsafe.Pointer(&vec[0]))
	centroidPtrs := make([]*C.double, len(centroids))
	for i := range centroids {
		centroidPtrs[i] = (*C.double)(unsafe.Pointer(&centroids[i][0]))
	}
	centroidPtrPtr := (**C.double)(unsafe.Pointer(&centroidPtrs[0]))
	idx := int(C.find_nearest_centroid_avx2(vecPtr, centroidPtrPtr, C.int(len(centroids)), C.int(len(vec))))
	runtime.KeepAlive(vec)
	runtime.KeepAlive(centroids)
	if idx < 0 {
		return -1, 0, errors.New("no nearest centroid found")
	}
	dist, _ := EuclideanDistanceSquaredAVX2(vec, centroids[idx])
	return idx, dist, nil
}
