package main

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"fmt"
	"math/rand"
	"time"
)

func main() {
	fmt.Println("=== RDMA加速器功能测试 ===")

	// 设置随机种子
	rand.Seed(time.Now().UnixNano())

	// 生成测试向量
	vectorDim := 128
	testVector := make([]float64, vectorDim)
	for i := range testVector {
		testVector[i] = rand.Float64()*10 - 5 // 生成-5到5之间的随机数
	}
	fmt.Printf("生成测试向量，维度: %d\n", vectorDim)

	// 生成随机质心
	numCentroids := 10
	centroids := make([]entity.Point, numCentroids)
	for i := range centroids {
		centroids[i] = make(entity.Point, vectorDim)
		for j := range centroids[i] {
			centroids[i][j] = rand.Float64()*10 - 5
		}
	}
	fmt.Printf("生成 %d 个随机质心\n\n", numCentroids)

	// 测试RDMA模拟逻辑查找最近质心
	fmt.Println("--- 测试RDMA模拟逻辑查找最近质心 ---")
	start := time.Now()
	idx, dist, err := acceler.FindNearestCentroidRDMASimulated(testVector, centroids)
	elapsed := time.Since(start)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
		return
	}
	fmt.Printf("最近质心索引: %d\n", idx)
	fmt.Printf("最近距离: %f\n", dist)
	fmt.Printf("计算耗时: %v\n\n", elapsed)

	// 测试RDMA模拟逻辑欧氏距离计算
	fmt.Println("--- 测试RDMA模拟逻辑欧氏距离计算 ---")
	vector2 := make([]float64, vectorDim)
	for i := range vector2 {
		vector2[i] = rand.Float64()*10 - 5
	}

	start = time.Now()
	distance, err := acceler.ComputeEuclideanDistanceRDMASimulated(testVector, vector2)
	elapsed = time.Since(start)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
		return
	}
	fmt.Printf("欧氏距离平方: %f\n", distance)
	fmt.Printf("计算耗时: %v\n\n", elapsed)

	// 性能对比测试
	fmt.Println("--- 性能对比测试 ---")
	iterations := 1000

	// 测试RDMA模拟逻辑性能
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, _, _ = acceler.FindNearestCentroidRDMASimulated(testVector, centroids)
	}
	rdmaElapsed := time.Since(start)
	fmt.Printf("RDMA模拟逻辑 %d 次迭代耗时: %v\n", iterations, rdmaElapsed)

	// 测试标准实现性能（使用简单的线性搜索）
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, _, _ = findNearestCentroidStandard(testVector, centroids)
	}
	standardElapsed := time.Since(start)
	fmt.Printf("标准实现 %d 次迭代耗时: %v\n", iterations, standardElapsed)

	// 计算性能提升比例
	speedup := float64(standardElapsed) / float64(rdmaElapsed)
	fmt.Printf("性能提升比例: %.2fx\n\n", speedup)

	// 准确性验证
	fmt.Println("--- 准确性验证 ---")
	rdmaIdx, rdmaDist, _ := acceler.FindNearestCentroidRDMASimulated(testVector, centroids)
	standardIdx, standardDist, _ := findNearestCentroidStandard(testVector, centroids)

	fmt.Printf("RDMA模拟逻辑结果: 索引=%d, 距离=%f\n", rdmaIdx, rdmaDist)
	fmt.Printf("标准实现结果: 索引=%d, 距离=%f\n", standardIdx, standardDist)

	if rdmaIdx == standardIdx {
		fmt.Println("✓ 准确性验证通过：两种方法找到相同的最近质心")
	} else {
		fmt.Println("✗ 准确性验证失败：两种方法找到不同的最近质心")
	}

	// 验证距离计算准确性
	rdmaDistCalc, _ := acceler.ComputeEuclideanDistanceRDMASimulated(testVector, vector2)
	standardDistCalc := computeEuclideanDistanceStandard(testVector, vector2)

	if abs(rdmaDistCalc-standardDistCalc) < 1e-10 {
		fmt.Println("✓ 距离计算准确性验证通过")
	} else {
		fmt.Printf("✗ 距离计算准确性验证失败: RDMA=%f, 标准=%f\n", rdmaDistCalc, standardDistCalc)
	}

	fmt.Println("\n=== 测试完成 ===")
}

// 标准实现的最近质心查找（用于对比）
func findNearestCentroidStandard(vec []float64, centroids []entity.Point) (int, float64, error) {
	if len(centroids) == 0 {
		return -1, 0, fmt.Errorf("质心列表为空")
	}

	minDist := float64(1e20)
	nearestIdx := -1

	for i, centroid := range centroids {
		dist := 0.0
		for j := 0; j < len(vec); j++ {
			diff := vec[j] - centroid[j]
			dist += diff * diff
		}

		if dist < minDist {
			minDist = dist
			nearestIdx = i
		}
	}

	return nearestIdx, minDist, nil
}

// 标准实现的欧氏距离计算（用于对比）
func computeEuclideanDistanceStandard(v1, v2 []float64) float64 {
	dist := 0.0
	for i := 0; i < len(v1); i++ {
		diff := v1[i] - v2[i]
		dist += diff * diff
	}
	return dist
}

// 辅助函数：计算绝对值
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}