package util

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
	"seetaSearch/library/entity"
	"unsafe"
)

// euclideanDistanceSquaredAVX2 使用 AVX2 指令集计算两个向量间的欧氏距离平方
func euclideanDistanceSquaredAVX2(a, b []float64) float64 {
	if len(a) != len(b) {
		return -1
	}

	// 确保长度是8的倍数，AVX2每次处理4个double，我们处理两批
	if len(a)%8 != 0 {
		return -1
	}

	aPtr := (*C.double)(unsafe.Pointer(&a[0]))
	bPtr := (*C.double)(unsafe.Pointer(&b[0]))

	return float64(C.euclidean_distance_squared_avx2(aPtr, bPtr, C.int(len(a))))
}

// findNearestCentroidAVX2 使用 AVX2 指令集查找最近的质心
func findNearestCentroidAVX2(vec []float64, centroids []entity.Point) (int, float64) {
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
	nearest := int(C.find_nearest_centroid_avx2(vecPtr, centroidPtrPtr, C.int(len(centroids)), C.int(len(vec))))

	// 计算最小距离（用于返回）
	var minDist float64
	if nearest >= 0 {
		minDist = euclideanDistanceSquaredAVX2(vec, centroids[nearest])
	}

	return nearest, minDist
}
