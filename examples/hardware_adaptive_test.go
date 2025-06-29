package main

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/vector"
	"fmt"
	"math/rand"
	"time"
)

// 硬件自适应功能测试
func main() {
	logger.Info("开始硬件自适应功能测试")

	// 1. 创建硬件管理器
	hardwareManager := acceler.NewHardwareManager()
	logger.Info("硬件管理器创建完成")

	// 2. 使用硬件管理器创建向量数据库
	db, err := vector.NewVectorDBWithHardwareManager("test_adaptive.db", 10, hardwareManager)
	if err != nil {
		logger.Error("创建向量数据库失败: %v", err)
		return
	}
	defer db.Close()

	// 3. 生成测试数据
	vectorDim := 128
	numVectors := 1000
	logger.Info("生成 %d 个 %d 维测试向量", numVectors, vectorDim)

	rand.Seed(time.Now().UnixNano())
	for i := 0; i < numVectors; i++ {
		vector := make([]float64, vectorDim)
		for j := 0; j < vectorDim; j++ {
			vector[j] = rand.Float64()*2 - 1 // [-1, 1] 范围内的随机数
		}
		db.Add(fmt.Sprintf("vec_%d", i), vector)
	}

	// 4. 构建索引
	logger.Info("开始构建索引")
	if err := db.BuildIndex(100, 0.001); err != nil {
		logger.Error("构建索引失败: %v", err)
		return
	}

	// 5. 测试硬件自适应搜索
	logger.Info("开始硬件自适应搜索测试")
	queryVector := make([]float64, vectorDim)
	for i := 0; i < vectorDim; i++ {
		queryVector[i] = rand.Float64()*2 - 1
	}

	// 测试不同的搜索策略
	strategies := []string{"BruteForce", "IVF", "HNSW"}
	for _, strategy := range strategies {
		logger.Info("测试策略: %s", strategy)
		start := time.Now()
		
		results, err := db.FindNearest(queryVector, 10, 5)
		if err != nil {
			logger.Error("搜索失败 (%s): %v", strategy, err)
			continue
		}
		
		elapsed := time.Since(start)
		logger.Info("策略 %s: 找到 %d 个结果，耗时 %v", strategy, len(results), elapsed)
		
		// 显示前3个结果
		for i, result := range results {
			if i >= 3 {
				break
			}
			logger.Info("  结果 %d: ID=%s, 距离=%.6f", i+1, result.ID, result.Distance)
		}
	}

	// 6. 测试硬件能力检测
	hardwareInfo := db.GetHardwareInfo()
	logger.Info("硬件能力检测结果:")
	logger.Info("  AVX2支持: %v", hardwareInfo.HasAVX2)
	logger.Info("  AVX512支持: %v", hardwareInfo.HasAVX512)
	logger.Info("  GPU支持: %v", hardwareInfo.HasGPU)
	logger.Info("  CPU核心数: %d", hardwareInfo.CPUCores)
	logger.Info("  GPU设备数: %d", hardwareInfo.GPUDevices)

	// 7. 测试计算策略选择
	currentStrategy := db.GetCurrentStrategy()
	selectedStrategy := db.GetSelectStrategy()
	logger.Info("当前计算策略: %v", currentStrategy)
	logger.Info("选择的计算策略: %v", selectedStrategy)

	// 8. 测试自适应距离计算
	logger.Info("测试自适应距离计算")
	vec1 := make([]float64, vectorDim)
	vec2 := make([]float64, vectorDim)
	for i := 0; i < vectorDim; i++ {
		vec1[i] = rand.Float64()
		vec2[i] = rand.Float64()
	}

	start := time.Now()
	dist, err := acceler.AdaptiveEuclideanDistanceSquaredWithHardware(vec1, vec2, selectedStrategy, hardwareManager)
	if err != nil {
		logger.Error("自适应距离计算失败: %v", err)
	} else {
		elapsed := time.Since(start)
		logger.Info("自适应距离计算结果: %.6f，耗时: %v", dist, elapsed)
	}

	// 9. 测试自适应质心查找
	logger.Info("测试自适应质心查找")
	centroids := make([]entity.Point, 10)
	for i := 0; i < 10; i++ {
		centroids[i] = entity.Point{
			Coordinates: make([]float64, vectorDim),
		}
		for j := 0; j < vectorDim; j++ {
			centroids[i].Coordinates[j] = rand.Float64()
		}
	}

	start = time.Now()
	nearestIdx, nearestDist := acceler.AdaptiveFindNearestCentroidWithHardware(queryVector, centroids, selectedStrategy, hardwareManager)
	elapsed = time.Since(start)
	logger.Info("自适应质心查找结果: 索引=%d，距离=%.6f，耗时=%v", nearestIdx, nearestDist, elapsed)

	logger.Info("硬件自适应功能测试完成")
}