package algorithm

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Point 代表一个数据点（向量）
type Point []float64

// EuclideanDistanceSquared 计算两个点之间的欧几里得距离的平方
// 返回平方距离可以避免开方运算，在比较大小时效果相同
func EuclideanDistanceSquared(p1, p2 Point) (float64, error) {
	if len(p1) != len(p2) {
		return 0, fmt.Errorf("向量维度不匹配: %d != %d", len(p1), len(p2))
	}
	sum := 0.0
	for i := range p1 {
		diff := p1[i] - p2[i]
		sum += diff * diff
	}
	return sum, nil // 返回平方距离
}

// calculateMean 计算一组点的均值（新的簇中心）
func calculateMean(points []Point) (Point, error) {
	if len(points) == 0 {
		return nil, fmt.Errorf("不能从空点集计算均值")
	}
	dim := len(points[0])
	mean := make(Point, dim)

	for _, p := range points {
		if len(p) != dim {
			return nil, fmt.Errorf("簇内点维度不一致")
		}
		for i := 0; i < dim; i++ {
			mean[i] += p[i]
		}
	}

	for i := 0; i < dim; i++ {
		mean[i] /= float64(len(points))
	}
	return mean, nil
}

// KMeans 执行 K-Means 聚类算法
// data: 数据集，每个元素是一个 Point
// k: 簇的数量
// maxIterations: 最大迭代次数
// tolerance: 簇中心变化的容忍度，用于提前停止迭代
// 返回值: 簇中心点列表, 每个数据点所属的簇索引列表, 错误
func KMeans(data []Point, k int, maxIterations int, tolerance float64) ([]Point, []int, error) {
	if k <= 0 {
		return nil, nil, fmt.Errorf("k 必须是正整数")
	}
	if len(data) < k {
		return nil, nil, fmt.Errorf("数据点数量 (%d) 少于簇数量 (%d)", len(data), k)
	}
	if len(data) == 0 {
		return nil, nil, fmt.Errorf("数据集不能为空")
	}

	dim := len(data[0]) // 假设所有数据点维度相同

	// 1. 初始化簇中心 (随机选择 K 个数据点)
	rand.Seed(time.Now().UnixNano())
	centroids := make([]Point, k)
	perm := rand.Perm(len(data))
	for i := 0; i < k; i++ {
		centroids[i] = make(Point, dim)
		copy(centroids[i], data[perm[i]])
	}

	assignments := make([]int, len(data)) // 存储每个点分配到的簇索引
	converged := false

	for iter := 0; iter < maxIterations && !converged; iter++ {
		fmt.Printf("K-Means 迭代: %d\n", iter+1)
		centroidsMoved := false

		// 2. 分配步骤：将每个点分配到最近的簇中心
		for i, point := range data {
			minDistSq := math.MaxFloat64
			assignedCluster := -1
			for j, centroid := range centroids {
				distSq, err := EuclideanDistanceSquared(point, centroid)
				if err != nil {
					return nil, nil, fmt.Errorf("计算距离失败 (点 %d, 质心 %d): %w", i, j, err)
				}
				if distSq < minDistSq {
					minDistSq = distSq
					assignedCluster = j
				}
			}
			assignments[i] = assignedCluster
		}

		// 3. 更新步骤：重新计算簇中心
		newCentroids := make([]Point, k)
		//clusterCounts := make([]int, k)
		clusterPoints := make([][]Point, k)
		for i := 0; i < k; i++ {
			clusterPoints[i] = make([]Point, 0)
		}

		for i, point := range data {
			clusterIndex := assignments[i]
			clusterPoints[clusterIndex] = append(clusterPoints[clusterIndex], point)
		}

		for j := 0; j < k; j++ {
			if len(clusterPoints[j]) == 0 {
				// 处理空簇：可以重新随机选择一个点作为该簇的中心，或保持不变
				// 这里简单地保持不变，但在实际应用中可能需要更好的策略
				fmt.Printf("警告: 簇 %d 为空，保持其质心不变。\n", j)
				newCentroids[j] = make(Point, dim)
				copy(newCentroids[j], centroids[j]) // 保持旧质心
				continue
			}
			mean, err := calculateMean(clusterPoints[j])
			if err != nil {
				return nil, nil, fmt.Errorf("计算簇 %d 的均值失败: %w", j, err)
			}
			newCentroids[j] = mean
		}

		// 检查收敛性：比较新旧簇中心的移动距离
		maxCentroidShiftSq := 0.0
		for j := 0; j < k; j++ {
			shiftSq, err := EuclideanDistanceSquared(centroids[j], newCentroids[j])
			if err != nil {
				// 维度应该一致，理论上这里不应出错
				return nil, nil, fmt.Errorf("计算质心移动距离失败: %w", err)
			}
			if shiftSq > maxCentroidShiftSq {
				maxCentroidShiftSq = shiftSq
			}
			centroids[j] = newCentroids[j] // 更新质心
		}

		if maxCentroidShiftSq <= tolerance*tolerance { // 比较平方值
			fmt.Printf("K-Means 已收敛，最大质心移动平方: %f\n", maxCentroidShiftSq)
			converged = true
		} else {
			centroidsMoved = true
		}

		// 如果没有质心移动，也认为收敛 (一种简单检查)
		if !centroidsMoved && iter > 0 { // 避免第一次迭代就错误地认为收敛
			fmt.Println("K-Means 已收敛，质心未移动。")
			// converged = true // 这个条件可能过于严格或不准确，依赖 tolerance 更好
		}
	}

	if !converged {
		fmt.Printf("K-Means 在 %d 次迭代后未完全收敛。\n", maxIterations)
	}

	return centroids, assignments, nil
}

// 示例用法
/*
func main() {
	// 示例数据 (二维点)
	data := []Point{
		{1.0, 1.0},
		{1.5, 2.0},
		{3.0, 4.0},
		{5.0, 7.0},
		{3.5, 5.0},
		{4.5, 5.0},
		{3.5, 4.5},
		{10.0, 10.0},
		{10.5, 11.0},
		{12.0, 9.0},
	}

	k := 2
	maxIter := 100
	tolerance := 0.001 // 质心移动距离的容忍度

	centroids, assignments, err := KMeans(data, k, maxIter, tolerance)
	if err != nil {
		fmt.Println("K-Means 错误:", err)
		return
	}

	fmt.Println("最终簇中心:")
	for i, c := range centroids {
		fmt.Printf("簇 %d: %v\n", i, c)
	}

	fmt.Println("数据点分配:")
	for i, point := range data {
		fmt.Printf("点 %v -> 簇 %d\n", point, assignments[i])
	}
}
*/
