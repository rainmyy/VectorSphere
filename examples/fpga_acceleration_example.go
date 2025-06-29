package main

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"fmt"
	"math/rand"
	"time"
)

func main() {
	fmt.Println("=== FPGA加速器功能测试 ===")

	// 设置随机种子
	rand.Seed(time.Now().UnixNano())

	// 测试参数
	vectorDim := 128
	numCentroids := 10

	// 生成测试向量
	testVector := generateRandomVector(vectorDim)
	fmt.Printf("生成测试向量，维度: %d\n", len(testVector))

	// 生成测试质心
	centroids := generateRandomCentroids(numCentroids, vectorDim)
	fmt.Printf("生成 %d 个随机质心\n", len(centroids))

	// 测试FPGA模拟逻辑 - 查找最近质心
	fmt.Println("\n--- 测试FPGA模拟逻辑查找最近质心 ---")
	start := time.Now()
	nearestIdx, nearestDist, err := acceler.FindNearestCentroidFPGASimulated(testVector, centroids)
	elapsed := time.Since(start)

	if err != nil {
		fmt.Printf("错误: %v\n", err)
	} else {
		fmt.Printf("最近质心索引: %d\n", nearestIdx)
		fmt.Printf("最近距离: %.6f\n", nearestDist)
		fmt.Printf("计算耗时: %v\n", elapsed)
	}

	// 测试FPGA模拟逻辑 - 欧氏距离计算
	fmt.Println("\n--- 测试FPGA模拟逻辑欧氏距离计算 ---")
	testVector2 := generateRandomVector(vectorDim)
	start = time.Now()
	distance, err := acceler.ComputeEuclideanDistanceFPGASimulated(testVector, testVector2)
	elapsed = time.Since(start)

	if err != nil {
		fmt.Printf("错误: %v\n", err)
	} else {
		fmt.Printf("欧氏距离平方: %.6f\n", distance)
		fmt.Printf("计算耗时: %v\n", elapsed)
	}

	// 性能对比测试
	fmt.Println("\n--- 性能对比测试 ---")
	performanceComparison(testVector, centroids)

	// 准确性验证
	fmt.Println("\n--- 准确性验证 ---")
	accuracyValidation(testVector, centroids)

	fmt.Println("\n=== 测试完成 ===")
}

// generateRandomCentroids 生成随机质心
func generateRandomCentroids(numCentroids, dim int) []entity.Point {
	centroids := make([]entity.Point, numCentroids)
	for i := range centroids {
		centroids[i] = make(entity.Point, dim)
		for j := range centroids[i] {
			centroids[i][j] = rand.Float64()*2 - 1 // 范围 [-1, 1]
		}
	}
	return centroids
}

// performanceComparison 性能对比测试
func performanceComparison(testVector []float64, centroids []entity.Point) {
	numIterations := 1000

	// 测试FPGA模拟逻辑性能
	start := time.Now()
	for i := 0; i < numIterations; i++ {
		_, _, _ = acceler.FindNearestCentroidFPGASimulated(testVector, centroids)
	}
	fpgaElapsed := time.Since(start)

	// 测试标准实现性能
	start = time.Now()
	for i := 0; i < numIterations; i++ {
		_, _ = acceler.FindNearestDefaultCentroid(testVector, centroids)
	}
	standardElapsed := time.Since(start)

	fmt.Printf("FPGA模拟逻辑 %d 次迭代耗时: %v\n", numIterations, fpgaElapsed)
	fmt.Printf("标准实现 %d 次迭代耗时: %v\n", numIterations, standardElapsed)
	fmt.Printf("性能提升比例: %.2fx\n", float64(standardElapsed)/float64(fpgaElapsed))
}

// accuracyValidation 准确性验证
func accuracyValidation(testVector []float64, centroids []entity.Point) {
	// 使用FPGA模拟逻辑
	fpgaIdx, fpgaDist, err1 := acceler.FindNearestCentroidFPGASimulated(testVector, centroids)

	// 使用标准实现
	standardIdx, standardDist := acceler.FindNearestDefaultCentroid(testVector, centroids)

	if err1 != nil {
		fmt.Printf("FPGA模拟逻辑计算错误: %v\n", err1)
		return
	}

	fmt.Printf("FPGA模拟逻辑结果: 索引=%d, 距离=%.6f\n", fpgaIdx, fpgaDist)
	fmt.Printf("标准实现结果: 索引=%d, 距离=%.6f\n", standardIdx, standardDist)

	if fpgaIdx == standardIdx {
		fmt.Println("✓ 准确性验证通过：两种方法找到相同的最近质心")
	} else {
		fmt.Println("✗ 准确性验证失败：两种方法找到不同的最近质心")
	}

	// 验证距离计算的准确性（允许小的浮点误差）
	distDiff := fpgaDist - standardDist
	if distDiff < 0 {
		distDiff = -distDiff
	}
	if distDiff < 1e-10 {
		fmt.Println("✓ 距离计算准确性验证通过")
	} else {
		fmt.Printf("✗ 距离计算准确性验证失败，差异: %.2e\n", distDiff)
	}
}
