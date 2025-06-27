package vector

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
)

func (db *VectorDB) WarmupIndex() {
	logger.Info("开始预热索引...")

	// 检查数据库是否已索引
	if !db.IsIndexed() {
		logger.Warning("数据库未索引，跳过预热")
		return
	}

	// 尝试获取向量维度
	var vectorDim int
	db.mu.RLock()
	if len(db.vectors) > 0 {
		for _, vec := range db.vectors {
			vectorDim = len(vec)
			break
		}
	}
	db.mu.RUnlock()

	if vectorDim == 0 {
		logger.Warning("无法确定向量维度，跳过预热")
		return
	}

	// 生成不同维度的样本查询向量
	sampleQueries := db.generateSampleQueries(vectorDim)

	// 预热不同的索引类型和计算策略
	warmupStrategies := []struct {
		name     string
		strategy IndexStrategy
	}{
		{"BruteForce", StrategyBruteForce},
		{"IVF", StrategyIVF},
		{"HNSW", StrategyHNSW},
		{"PQ", StrategyPQ},
		{"Hybrid", StrategyHybrid},
		{"EnhancedIVF", StrategyEnhancedIVF}, // 新增：增强IVF索引
		{"EnhancedLSH", StrategyEnhancedLSH}, // 新增：增强LSH索引
	}

	// 获取硬件能力
	hwCaps := db.strategyComputeSelector.GetHardwareCapabilities()

	// 预热每种索引类型
	for _, ws := range warmupStrategies {
		// 检查索引类型是否可用
		canUse := true
		switch ws.strategy {
		case StrategyBruteForce:
			canUse = len(db.vectors) > 0
		case StrategyHNSW:
			canUse = db.useHNSWIndex && db.indexed && db.hnsw != nil
		case StrategyPQ:
			canUse = db.usePQCompression && db.pqCodebook != nil
		case StrategyIVF:
			canUse = db.indexed && len(db.clusters) > 0
		case StrategyHybrid:
			gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
			canUse = hwCaps.HasGPU && gpuAccelerator != nil
		case StrategyEnhancedIVF:
			// 检查增强IVF索引是否可用
			canUse = db.ivfIndex != nil && db.ivfIndex.Enable && db.ivfConfig != nil
		case StrategyEnhancedLSH:
			// 检查增强LSH索引是否可用
			canUse = db.LshIndex != nil && db.LshIndex.Enable && db.LshConfig != nil
		case StrategyIVFHNSW:
			// 检查IVF-HNSW混合索引是否可用
			canUse = db.ivfHnswIndex != nil && db.ivfHnswIndex.Enable
		default:
			logger.Warning("未知的索引策略: %v", ws.strategy)
			canUse = false
		}

		if !canUse {
			logger.Trace("跳过预热索引类型 %s (不可用)", ws.name)
			continue
		}

		logger.Info("开始预热索引类型: %s", ws.name)

		// 对每个样本查询执行搜索
		for i, query := range sampleQueries {
			// 创建搜索上下文
			ctx := SearchContext{
				QueryVector:  query,
				K:            5,
				Nprobe:       10,
				Timeout:      0,
				QualityLevel: 0.8,
			}

			// 执行搜索但不使用结果
			var err error
			var results []entity.Result

			switch ws.strategy {
			case StrategyBruteForce:
				results, err = db.bruteForceSearch(query, 5)
			case StrategyIVF:
				results, err = db.ivfSearchWithScores(query, 5, 10, db.GetOptimalStrategy(query))
			case StrategyHNSW:
				results, err = db.hnswSearchWithScores(query, 5)
			case StrategyPQ:
				results, err = db.pqSearchWithScores(query, 5)
			case StrategyHybrid:
				results, err = db.hybridSearchWithScores(query, 5, ctx)
			case StrategyEnhancedIVF:
				// 预热增强IVF索引
				results, err = db.EnhancedIVFSearch(query, 5, 10)
				if err != nil {
					logger.Warning("增强IVF索引预热失败，尝试预热IVF组件: %v", err)
					// 预热IVF索引的各个组件
					//db.warmupEnhancedIVFComponents(query, ctx)
				}
			case StrategyEnhancedLSH:
				// 预热增强LSH索引
				results, err = db.EnhancedLSHSearch(query, 5)
				if err != nil {
					logger.Warning("增强LSH索引预热失败，尝试预热LSH组件: %v", err)
					// 预热LSH索引的各个组件
					//db.warmupEnhancedLSHComponents(query, ctx)
				}
			case StrategyIVFHNSW:
				// 预热IVF-HNSW混合索引
				results, err = db.ivfHnswSearchWithScores(query, 5, ctx)
				if err != nil {
					logger.Warning("IVF-HNSW索引预热失败: %v", err)
				}
			default:
				logger.Warning("未知的索引策略，跳过预热: %v", ws.strategy)
				continue
			}

			if err != nil {
				logger.Warning("预热索引类型 %s 查询 %d 失败: %v", ws.name, i, err)
			} else {
				logger.Trace("预热索引类型 %s 查询 %d 成功，返回 %d 个结果", ws.name, i, len(results))
			}
		}

		// 针对增强索引进行额外的预热操作
		switch ws.strategy {
		case StrategyEnhancedIVF:
			db.warmupEnhancedIVFAdvanced(sampleQueries)
		case StrategyEnhancedLSH:
			db.warmupEnhancedLSHAdvanced(sampleQueries)
		case StrategyIVFHNSW:
			// IVF-HNSW索引的额外预热操作
			logger.Trace("IVF-HNSW索引预热完成")
		default:
			// 其他策略不需要额外的预热操作
			continue
		}
	}

	// 预热不同的计算策略
	if hwCaps.HasGPU || hwCaps.HasAVX512 || hwCaps.HasAVX2 {
		logger.Info("开始预热计算策略...")

		// 预热每种计算策略
		strategies := []struct {
			name      string
			strategy  acceler.ComputeStrategy
			available bool
		}{
			{"Standard", acceler.StrategyStandard, true},
			{"AVX2", acceler.StrategyAVX2, hwCaps.HasAVX2},
			{"AVX512", acceler.StrategyAVX512, hwCaps.HasAVX512},
			{"GPU", acceler.StrategyGPU, hwCaps.HasGPU},
		}

		for _, s := range strategies {
			if !s.available {
				logger.Trace("跳过预热计算策略 %s (不可用)", s.name)
				continue
			}

			logger.Info("开始预热计算策略: %s", s.name)

			// 对每个样本查询执行余弦相似度计算
			for i, query := range sampleQueries {
				// 随机选择一些向量进行计算
				targets := db.getRandomVectors(10)
				if len(targets) == 0 {
					continue
				}

				// 执行计算
				for _, target := range targets {
					_ = acceler.AdaptiveCosineSimilarity(query, target, s.strategy)
				}

				logger.Trace("预热计算策略 %s 查询 %d 完成", s.name, i)
			}
		}
	}

	// 预热GPU加速器（如果可用）
	gpuAccelerator := db.hardwareManager.GetGPUAccelerator()
	if hwCaps.HasGPU && gpuAccelerator != nil {
		logger.Info("开始预热GPU加速器...")

		// 随机选择一些向量进行批量计算
		targets := db.getRandomVectors(100)
		if len(targets) > 0 {
			_, err := db.gpuBatchCosineSimilarity(sampleQueries, targets)
			if err != nil {
				logger.Warning("预热GPU加速器失败: %v", err)
			} else {
				logger.Info("GPU加速器预热成功")
			}
		}
	}

	logger.Info("索引预热完成")
}

// warmupEnhancedIVFComponents 预热增强IVF索引组件
func (db *VectorDB) warmupEnhancedIVFComponents(query []float64, ctx SearchContext) {
	if db.ivfIndex == nil || !db.ivfIndex.Enable || db.ivfConfig == nil {
		logger.Trace("增强IVF索引或配置不可用，跳过组件预热")
		return
	}

	logger.Trace("开始预热增强IVF索引组件")

	// 1. 预热聚类中心计算/访问
	if db.ivfIndex.Clusters != nil && len(db.ivfIndex.Clusters) > 0 {
		logger.Trace("预热IVF聚类中心...")
		// 模拟计算查询向量与部分聚类中心的距离
		numClustersToWarmup := min(5, len(db.ivfIndex.Clusters)) // 限制预热的聚类数量
		for i := 0; i < numClustersToWarmup; i++ {
			cluster := db.ivfIndex.Clusters[i]
			if len(cluster.Centroid) > 0 {
				_ = acceler.AdaptiveCosineSimilarity(query, cluster.Centroid, db.GetOptimalStrategy(query)) // 使用自适应策略
			}
		}
	}

	// 2. 预热倒排列表访问（使用聚类中的VectorIDs）
	if db.ivfIndex.Clusters != nil {
		logger.Trace("预热IVF倒排列表...")
		// 模拟访问几个聚类中的向量
		numClustersToWarmup := min(3, len(db.ivfIndex.Clusters)) // 限制预热的聚类数量
		for i := 0; i < numClustersToWarmup; i++ {
			cluster := db.ivfIndex.Clusters[i]
			if len(cluster.VectorIDs) > 0 {
				// 访问聚类中的前几个向量
				numVectorsToAccess := min(2, len(cluster.VectorIDs))
				for j := 0; j < numVectorsToAccess; j++ {
					if vec, exists := db.vectors[cluster.VectorIDs[j]]; exists {
						_ = acceler.AdaptiveCosineSimilarity(query, vec, db.GetOptimalStrategy(query))
					}
				}
			}
		}
	}

	// 3. 预热PQ编码器 (如果使用)
	if db.ivfConfig.UsePQCompression && db.pqCodebook != nil {
		logger.Trace("预热PQ编码器...")
		// 使用现有的PQ压缩函数进行预热
		_, err := acceler.OptimizedCompressByPQ(query, db.pqCodebook, db.numSubVectors, db.numCentroidsPerSubVector)
		if err != nil {
			logger.Trace("PQ编码器预热失败: %v", err)
		}
		// 预热PQ解码 (如果适用，通常在搜索时)
		if db.ivfPQIndex != nil && len(db.ivfPQIndex.PQCodes) > 0 {
			for vectorID, pqCodes := range db.ivfPQIndex.PQCodes {
				if pqCodes != nil {
					// 模拟PQ解码过程
					_ = vectorID // 使用vectorID避免未使用变量警告
					break        // 只预热一个
				}
			}
		}
	}

	// 4. 预热IVF量化器 (概念性预热)
	if db.ivfConfig.UsePQCompression {
		logger.Trace("预热IVF量化器...")
		// 使用PQ压缩进行量化预热
		if db.pqCodebook != nil {
			_, err := acceler.OptimizedCompressByPQ(query, db.pqCodebook, db.numSubVectors, db.numCentroidsPerSubVector)
			if err != nil {
				logger.Trace("IVF量化器预热失败: %v", err)
			}
		}
	}

	// 5. 预热距离计算器 (使用通用距离计算)
	if db.ivfIndex != nil {
		logger.Trace("预热IVF距离计算器...")
		if len(db.vectors) > 0 {
			var firstVector []float64
			db.mu.RLock()
			for _, v := range db.vectors {
				firstVector = v
				break
			}
			db.mu.RUnlock()
			if firstVector != nil {
				// 使用通用的余弦相似度计算进行预热
				_ = acceler.AdaptiveCosineSimilarity(query, firstVector, db.GetOptimalStrategy(query))
			}
		}
	}

	// 6. 预热自适应搜索参数调整逻辑 (概念性预热)
	logger.Trace("预热IVF自适应搜索调谐器...")
	// 模拟参数调整逻辑，实际实现可能在其他地方
	_ = ctx.QualityLevel
	_ = len(db.vectors)

	// 7. 预热索引统计信息访问
	logger.Trace("预热IVF索引统计信息访问...")
	// 访问基本的索引信息
	_ = db.ivfIndex.TotalVectors
	_ = len(db.ivfIndex.Clusters)

	// 8. 预热缓存系统 (使用数据库级别的缓存)
	if db.MultiCache != nil {
		logger.Trace("预热IVF缓存系统...")
		// 模拟缓存的Get/Put操作
		_, _ = db.MultiCache.Get("sample_key")
		db.MultiCache.Put("sample_key", []byte("sample_data"))
	}

	// 9. 预热并行处理组件 (如果IVF搜索涉及特定的并行模式)
	// 例如，如果倒排列表的扫描是并行化的，可以尝试触发这个逻辑
	// 这部分比较抽象，具体实现依赖于IVF的并行策略

	// 10. 预热内存池/分配器 (如果IVF使用自定义内存管理)
	// 模拟分配和释放操作

	logger.Trace("增强IVF索引组件预热完成")
}

// warmupEnhancedLSHComponents 预热增强LSH索引组件
func (db *VectorDB) warmupEnhancedLSHComponents(query []float64, ctx SearchContext) {
	if db.LshIndex == nil || !db.LshIndex.Enable {
		return
	}

	logger.Trace("开始预热增强LSH索引组件")

	// 1. 预热LSH哈希函数
	if db.LshIndex.Tables != nil {
		for i, table := range db.LshIndex.Tables {
			if i >= 5 { // 限制预热的哈希表数量
				break
			}
			if table.HashFunctions != nil {
				// 对查询向量进行哈希计算
				for j, hashFunc := range table.HashFunctions {
					if j >= 3 { // 限制预热的哈希函数数量
						break
					}
					_, err := hashFunc.Hash(query)
					if err != nil {
						logger.Trace("LSH哈希函数预热失败: %v", err)
					}
				}
			}
		}
	}

	// 2. 预热LSH桶访问
	if db.LshIndex.Tables != nil {
		for i, table := range db.LshIndex.Tables {
			if i >= 3 { // 限制预热的哈希表数量
				break
			}
			if table.Buckets != nil {
				// 访问几个桶
				bucketCount := 0
				for _, bucket := range table.Buckets {
					if bucketCount >= 5 { // 限制预热的桶数量
						break
					}
					if len(bucket.VectorIDs) > 0 {
						// 访问桶中的前几个向量
						for j := 0; j < min(2, len(bucket.VectorIDs)); j++ {
							if vec, exists := db.vectors[bucket.VectorIDs[j]]; exists {
								_ = acceler.AdaptiveCosineSimilarity(query, vec, db.GetOptimalStrategy(query))
							}
						}
					}
					bucketCount++
				}
			}
		}
	}

	// 3. 预热LSH族
	if db.LshFamilies != nil {
		for familyName, family := range db.LshFamilies {
			if family != nil && family.HashFunctions != nil {
				logger.Trace("预热LSH族: %s", familyName)
				// 对查询向量进行LSH族哈希计算
				for i, hashFunc := range family.HashFunctions {
					if i >= 3 { // 限制预热的哈希函数数量
						break
					}
					_, err := hashFunc.Hash(query)
					if err != nil {
						logger.Trace("LSH族哈希函数预热失败: %v", err)
					}
				}
			}
		}
	}

	// 4. 预热自适应LSH
	if db.AdaptiveLSH != nil {
		// 模拟自适应参数调整
		// 访问自适应LSH的配置信息
		_ = db.AdaptiveLSH.OptimalParams
		_ = db.AdaptiveLSH.LastTuning
	}

	logger.Trace("增强LSH索引组件预热完成")
}

// warmupAdvancedIndexHelper 封装了高级预热中的参数和查询循环
func (db *VectorDB) warmupAdvancedIndexHelper(
	indexName string,
	sampleQueries [][]float64,
	paramIterator func(yield func(params map[string]interface{})),
	searchFunc func(query []float64, params map[string]interface{}) error,
) {
	logger.Info("开始 %s 高级预热", indexName)

	paramIterator(func(params map[string]interface{}) {
		for i, query := range sampleQueries {
			if i >= 2 { // 限制查询数量
				break
			}
			err := searchFunc(query, params)
			if err != nil {
				logger.Trace("%s 参数 %v 预热失败: %v", indexName, params, err)
			}
		}
	})

	logger.Info("%s 高级预热完成", indexName)
}

// warmupEnhancedIVFAdvanced 增强IVF索引高级预热
func (db *VectorDB) warmupEnhancedIVFAdvanced(sampleQueries [][]float64) {
	if db.ivfIndex == nil || !db.ivfIndex.Enable {
		return
	}

	paramIterator := func(yield func(params map[string]interface{})) {
		nprobeValues := []int{1, 5, 10, 20}
		for _, nprobe := range nprobeValues {
			if nprobe > len(db.ivfIndex.Clusters) {
				continue
			}
			yield(map[string]interface{}{"nprobe": nprobe})
		}
	}

	searchFunc := func(query []float64, params map[string]interface{}) error {
		nprobe := params["nprobe"].(int)
		_, err := db.EnhancedIVFSearch(query, 5, nprobe)
		return err
	}

	db.warmupAdvancedIndexHelper("增强IVF索引", sampleQueries, paramIterator, searchFunc)

	// 预热一个较大的k值
	if len(sampleQueries) > 0 {
		_, err := db.EnhancedIVFSearch(sampleQueries[0], 20, 10) // k=20
		if err != nil {
			logger.Trace("大K值搜索预热失败: %v", err)
		}
	}
}

// warmupEnhancedLSHAdvanced 增强LSH索引高级预热
func (db *VectorDB) warmupEnhancedLSHAdvanced(sampleQueries [][]float64) {
	if db.LshIndex == nil || !db.LshIndex.Enable || db.LshConfig == nil {
		return
	}

	originalR := db.LshConfig.R
	originalW := db.LshConfig.W
	defer func() { // 恢复原始参数
		db.LshConfig.R = originalR
		db.LshConfig.W = originalW
	}()

	paramIterator := func(yield func(params map[string]interface{})) {
		paramCombos := []struct{ R, W float64 }{
			{R: originalR, W: originalW},
			{R: originalR / 2, W: originalW},
			{R: originalR, W: originalW / 2},
		}
		for _, combo := range paramCombos {
			if combo.R <= 0 || combo.W <= 0 {
				continue
			}
			yield(map[string]interface{}{"R": combo.R, "W": combo.W})
		}
	}

	searchFunc := func(query []float64, params map[string]interface{}) error {
		db.LshConfig.R = params["R"].(float64)
		db.LshConfig.W = params["W"].(float64)
		_, err := db.EnhancedLSHSearch(query, 5)
		return err
	}

	db.warmupAdvancedIndexHelper("增强LSH索引", sampleQueries, paramIterator, searchFunc)
}
