package algorithm

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"VectorSphere/src/library/entity"
)

// EuclideanDistanceSquared 计算两个点之间的欧几里得距离的平方
// 返回平方距离可以避免开方运算，在比较大小时效果相同
func EuclideanDistanceSquared(p1, p2 entity.Point) (float64, error) {
	if len(p1) != len(p2) {
		return 0, fmt.Errorf("点的维度不匹配: %d vs %d", len(p1), len(p2))
	}

	sum := 0.0
	for i := 0; i < len(p1); i++ {
		diff := p1[i] - p2[i]
		sum += diff * diff
	}
	return sum, nil
}

// calculateMean 计算一组点的均值
func calculateMean(points []entity.Point) (entity.Point, error) {
	if len(points) == 0 {
		return nil, fmt.Errorf("点集合不能为空")
	}

	dim := len(points[0])
	mean := make(entity.Point, dim)

	for _, point := range points {
		for i := 0; i < dim; i++ {
			mean[i] += point[i]
		}
	}

	for i := 0; i < dim; i++ {
		mean[i] /= float64(len(points))
	}

	return mean, nil
}

// KMeans 执行K-Means聚类算法
// data: 输入数据点
// k: 簇的数量
// maxIterations: 最大迭代次数
// tolerance: 簇中心变化的容忍度（欧几里得距离），用于提前停止迭代
// 返回值: 簇中心点列表, 每个数据点所属的簇索引列表, 错误
func KMeans(data []entity.Point, k int, maxIterations int, tolerance float64) ([]entity.Point, []int, error) {
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

	// 优化1: 对于大数据集，使用采样的K-Means++初始化
	rand.New(rand.NewSource(time.Now().UnixNano()))
	var centroids []entity.Point
	var err error

	if len(data) > 10000 {
		// 对大数据集使用采样优化的初始化
		centroids, err = optimizedKMeansPlusPlusInit(data, k)
	} else {
		// 对小数据集使用标准K-Means++初始化
		centroids, err = standardKMeansPlusPlusInit(data, k)
	}

	if err != nil {
		return nil, nil, fmt.Errorf("初始化质心失败: %w", err)
	}

	assignments := make([]int, len(data)) // 存储每个点分配到的簇索引
	converged := false
	prevAssignments := make([]int, len(data)) // 用于检测分配变化

	for iter := 0; iter < maxIterations && !converged; iter++ {
		// 优化2: 并行分配步骤
		assignmentChanged := assignPointsToClusters(data, centroids, assignments)

		// 优化3: 早期停止 - 如果分配没有变化
		if iter > 0 && !assignmentChanged {
			converged = true
			break
		}

		copy(prevAssignments, assignments)

		// 优化4: 并行更新质心
		newCentroids := updateCentroids(data, assignments, k, dim)

		// 处理空簇
		for j := 0; j < k; j++ {
			if newCentroids[j] == nil {
				// 空簇处理：选择距离当前质心最远的点
				maxDistSq := 0.0
				bestIdx := 0
				for i, point := range data {
					minDistToCentroid := math.MaxFloat64
					for _, centroid := range centroids {
						if distSq, _ := EuclideanDistanceSquared(point, centroid); distSq < minDistToCentroid {
							minDistToCentroid = distSq
						}
					}
					if minDistToCentroid > maxDistSq {
						maxDistSq = minDistToCentroid
						bestIdx = i
					}
				}
				newCentroid := make(entity.Point, dim)
				copy(newCentroid, data[bestIdx])
				newCentroids[j] = newCentroid
			}
		}

		// 优化5: 检查收敛性
		maxCentroidShiftSq := 0.0
		for j := 0; j < k; j++ {
			if newCentroids[j] != nil {
				shiftSq, err := EuclideanDistanceSquared(centroids[j], newCentroids[j])
				if err != nil {
					return nil, nil, fmt.Errorf("计算质心移动距离失败: %w", err)
				}
				if shiftSq > maxCentroidShiftSq {
					maxCentroidShiftSq = shiftSq
				}
				centroids[j] = newCentroids[j] // 更新质心
			}
		}

		// 收敛检查
		if maxCentroidShiftSq <= tolerance*tolerance {
			converged = true
		}
	}

	return centroids, assignments, nil
}

// standardKMeansPlusPlusInit 标准K-Means++初始化
func standardKMeansPlusPlusInit(data []entity.Point, k int) ([]entity.Point, error) {
	if len(data) < k {
		return nil, fmt.Errorf("数据点数量少于簇数量")
	}

	dim := len(data[0])
	centroids := make([]entity.Point, 0, k)

	// 随机选择第一个质心
	firstIdx := rand.Intn(len(data))
	firstCentroid := make(entity.Point, dim)
	copy(firstCentroid, data[firstIdx])
	centroids = append(centroids, firstCentroid)

	// 选择后续质心
	for i := 1; i < k; i++ {
		distSqSum := 0.0
		distSqList := make([]float64, len(data))

		// 计算每个点到最近已选质心的距离平方
		for pIdx, point := range data {
			minDistSq := math.MaxFloat64
			for _, c := range centroids {
				distSq, _ := EuclideanDistanceSquared(point, c)
				if distSq < minDistSq {
					minDistSq = distSq
				}
			}
			distSqList[pIdx] = minDistSq
			distSqSum += minDistSq
		}

		// 根据距离概率选择下一个质心
		r := rand.Float64() * distSqSum
		accumulator := 0.0
		for pIdx, distSq := range distSqList {
			accumulator += distSq
			if accumulator >= r {
				newCentroid := make(entity.Point, dim)
				copy(newCentroid, data[pIdx])
				centroids = append(centroids, newCentroid)
				break
			}
		}
	}

	return centroids, nil
}

// optimizedKMeansPlusPlusInit 优化的K-Means++初始化（用于大数据集）
func optimizedKMeansPlusPlusInit(data []entity.Point, k int) ([]entity.Point, error) {
	if len(data) < k {
		return nil, fmt.Errorf("数据点数量少于簇数量")
	}

	dim := len(data[0])
	centroids := make([]entity.Point, 0, k)

	// 随机选择第一个质心
	firstIdx := rand.Intn(len(data))
	firstCentroid := make(entity.Point, dim)
	copy(firstCentroid, data[firstIdx])
	centroids = append(centroids, firstCentroid)

	// 使用采样优化K-Means++，对大数据集更高效
	for i := 1; i < k; i++ {
		// 对于大数据集，可以只采样部分点计算距离
		sampleSize := min(len(data), 1000)
		sampledIndices := randomSample(len(data), sampleSize)

		distSqSum := 0.0
		distSqList := make([]float64, sampleSize)

		// 计算采样点到最近质心的距离
		for j, idx := range sampledIndices {
			point := data[idx]
			minDistSq := math.MaxFloat64

			for _, c := range centroids {
				distSq, _ := EuclideanDistanceSquared(point, c)
				if distSq < minDistSq {
					minDistSq = distSq
				}
			}

			distSqList[j] = minDistSq
			distSqSum += minDistSq
		}

		// 根据距离概率选择下一个质心
		r := rand.Float64() * distSqSum
		accumulator := 0.0

		for j, distSq := range distSqList {
			accumulator += distSq
			if accumulator >= r {
				idx := sampledIndices[j]
				newCentroid := make(entity.Point, dim)
				copy(newCentroid, data[idx])
				centroids = append(centroids, newCentroid)
				break
			}
		}
	}

	return centroids, nil
}

// assignPointsToClusters 并行分配点到聚类
func assignPointsToClusters(data []entity.Point, centroids []entity.Point, assignments []int) bool {
	changed := false
	for i, point := range data {
		minDistSq := math.MaxFloat64
		assignedCluster := -1
		for j, centroid := range centroids {
			distSq, _ := EuclideanDistanceSquared(point, centroid)
			if distSq < minDistSq {
				minDistSq = distSq
				assignedCluster = j
			}
		}
		if assignments[i] != assignedCluster {
			changed = true
			assignments[i] = assignedCluster
		}
	}
	return changed
}

// updateCentroids 更新聚类质心
func updateCentroids(data []entity.Point, assignments []int, k int, dim int) []entity.Point {
	clusterPoints := make([][]entity.Point, k)
	for i := 0; i < k; i++ {
		clusterPoints[i] = make([]entity.Point, 0)
	}

	for i, point := range data {
		clusterIndex := assignments[i]
		clusterPoints[clusterIndex] = append(clusterPoints[clusterIndex], point)
	}

	newCentroids := make([]entity.Point, k)
	for j := 0; j < k; j++ {
		if len(clusterPoints[j]) == 0 {
			newCentroids[j] = nil // 标记为空簇
			continue
		}
		mean, err := calculateMean(clusterPoints[j])
		if err != nil {
			newCentroids[j] = nil
			continue
		}
		newCentroids[j] = mean
	}
	return newCentroids
}

// 随机采样函数
func randomSample(populationSize, sampleSize int) []int {
	if sampleSize >= populationSize {
		// 返回全部索引
		indices := make([]int, populationSize)
		for i := range indices {
			indices[i] = i
		}
		return indices
	}

	// 使用Fisher-Yates洗牌算法进行采样
	indices := make([]int, populationSize)
	for i := range indices {
		indices[i] = i
	}

	for i := 0; i < sampleSize; i++ {
		j := i + rand.Intn(populationSize-i)
		indices[i], indices[j] = indices[j], indices[i]
	}

	return indices[:sampleSize]
}

// ConvertToPoints 将 [][]float64 转换为 []entity.Point
func ConvertToPoints(data [][]float64) []entity.Point {
	if data == nil {
		return nil
	}
	points := make([]entity.Point, len(data))
	for i, vec := range data {
		points[i] = entity.Point(vec)
	}
	return points
}

// ConvertToFloat64Slice 将 []entity.Point 转换为 [][]float64
func ConvertToFloat64Slice(points []entity.Point) [][]float64 {
	if points == nil {
		return nil
	}
	data := make([][]float64, len(points))
	for i, p := range points {
		data[i] = []float64(p)
	}
	return data
}
