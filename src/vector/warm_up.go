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
		case StrategyHNSW:
			canUse = db.useHNSWIndex && db.indexed && db.hnsw != nil
		case StrategyPQ:
			canUse = db.usePQCompression && db.pqCodebook != nil
		case StrategyIVF:
			canUse = db.indexed && len(db.clusters) > 0
		case StrategyHybrid:
			canUse = hwCaps.HasGPU && db.gpuAccelerator != nil
		case StrategyEnhancedIVF:
			// 检查增强IVF索引是否可用
			canUse = db.ivfIndex != nil && db.ivfIndex.Enable && db.ivfConfig != nil
		case StrategyEnhancedLSH:
			// 检查增强LSH索引是否可用
			canUse = db.LshIndex != nil && db.LshIndex.Enable && db.LshConfig != nil
		default:
			panic("unhandled default case")
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
			}

			if err != nil {
				logger.Warning("预热索引类型 %s 查询 %d 失败: %v", ws.name, i, err)
			} else {
				logger.Trace("预热索引类型 %s 查询 %d 成功，返回 %d 个结果", ws.name, i, len(results))
			}
		}

		// 针对增强索引进行额外的预热操作
		//switch ws.strategy {
		//case EnhancedIVF:
		//	db.warmupEnhancedIVFAdvanced(sampleQueries)
		//case EnhancedLSH:
		//	db.warmupEnhancedLSHAdvanced(sampleQueries)
		//}
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
	if hwCaps.HasGPU && db.gpuAccelerator != nil {
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
//func (db *VectorDB) warmupEnhancedIVFComponents(query []float64, ctx SearchContext) {
//	if db.ivfIndex == nil || !db.ivfIndex.Enable || db.ivfConfig == nil {
//		log.Trace("增强IVF索引或配置不可用，跳过组件预热")
//		return
//	}
//
//	log.Trace("开始预热增强IVF索引组件")
//
//	// 1. 预热聚类中心计算/访问
//	if db.ivfIndex.Clusters != nil && len(db.ivfIndex.Clusters) > 0 {
//		log.Trace("预热IVF聚类中心...")
//		// 模拟计算查询向量与部分聚类中心的距离
//		numClustersToWarmup := min(5, len(db.ivfIndex.Clusters)) // 限制预热的聚类数量
//		for i := 0; i < numClustersToWarmup; i++ {
//			cluster := db.ivfIndex.Clusters[i]
//			if cluster != nil && len(cluster.Centroid) > 0 {
//				_ = acceler.AdaptiveCosineSimilarity(query, cluster.Centroid, db.GetOptimalStrategy(query)) // 使用自适应策略
//			}
//		}
//	}
//
//	// 2. 预热倒排列表访问
//	if db.ivfIndex.InvertedLists != nil {
//		log.Trace("预热IVF倒排列表...")
//		// 模拟访问几个倒排列表中的向量
//		numListsToWarmup := 0
//		for _, list := range db.ivfIndex.InvertedLists {
//			if numListsToWarmup >= 3 { // 限制预热的倒排列表数量
//				break
//			}
//			if len(list) > 0 {
//				// 访问列表中的前几个向量
//				numVectorsToAccess := min(2, len(list))
//				for i := 0; i < numVectorsToAccess; i++ {
//					if vec, exists := db.vectors[list[i]]; exists {
//						_ = acceler.AdaptiveCosineSimilarity(query, vec, db.GetOptimalStrategy(query))
//					}
//				}
//			}
//			numListsToWarmup++
//		}
//	}
//
//	// 3. 预热PQ编码器 (如果使用)
//	if db.ivfConfig.EnablePQ && db.pqCodebook != nil && db.pqCodebook.Encoder != nil {
//		log.Trace("预热PQ编码器...")
//		_, err := db.pqCodebook.Encoder.Encode(query)
//		if err != nil {
//			log.Trace("PQ编码器预热失败: %v", err)
//		}
//		// 预热PQ解码 (如果适用，通常在搜索时)
//		if len(db.ivfIndex.InvertedLists) > 0 {
//			for _, list := range db.ivfIndex.InvertedLists {
//				if len(list) > 0 {
//					vectorID := list[0]
//					db.mu.RLock()
//					pqCodes, pqExists := db.pqCodes[vectorID]
//					db.mu.RUnlock()
//					if pqExists && pqCodes != nil {
//						_, err = db.pqCodebook.Encoder.Decode(pqCodes)
//						if err != nil {
//							log.Trace("PQ解码器预热失败: %v", err)
//						}
//					}
//					break // 只预热一个
//				}
//			}
//		}
//	}
//
//	// 4. 预热IVF量化器 (如果独立于PQ)
//	if db.ivfConfig.EnableQuantization && db.quantizer != nil {
//		log.Trace("预热IVF量化器...")
//		_, err := db.quantizer.Quantize(query)
//		if err != nil {
//			log.Trace("IVF量化器预热失败: %v", err)
//		}
//	}
//
//	// 5. 预热距离计算器 (特定于IVF的优化版本，如果存在)
//	// 假设有一个优化的距离计算器，这里只是概念性预热
//	if db.ivfIndex != nil {
//		log.Trace("预热IVF距离计算器...")
//		if len(db.vectors) > 0 {
//			var firstVector []float64
//			db.mu.RLock()
//			for _, v := range db.vectors {
//				firstVector = v
//				break
//			}
//			db.mu.RUnlock()
//			if firstVector != nil {
//				_, err := db.ivfIndex.DistanceCalculator.ComputeDistance(query, firstVector)
//				if err != nil {
//					log.Trace("IVF距离计算器预热失败: %v", err)
//				}
//			}
//		}
//	}
//
//	// 6. 预热自适应搜索参数调整逻辑 (如果IVF有此机制)
//	if db.ivfIndex.AdaptiveSearchTuner != nil {
//		log.Trace("预热IVF自适应搜索调谐器...")
//		// 模拟参数更新
//		db.ivfIndex.AdaptiveSearchTuner.AdjustParameters(ctx.QualityLevel, len(db.vectors))
//	}
//
//	// 7. 预热索引统计信息访问
//	if db.ivfIndex.Stats != nil {
//		log.Trace("预热IVF索引统计信息访问...")
//		_ = db.ivfIndex.Stats.GetTotalVectors()
//		_ = db.ivfIndex.Stats.GetAvgVectorsPerList()
//	}
//
//	// 8. 预热缓存系统 (如果IVF有特定缓存，如最近访问的倒排列表)
//	if db.ivfIndex.Cache != nil {
//		log.Trace("预热IVF缓存系统...")
//		// 模拟缓存的Get/Put操作
//		_ = db.ivfIndex.Cache.Get("sample_key")
//		db.ivfIndex.Cache.Put("sample_key", []byte("sample_data"))
//	}
//
//	// 9. 预热并行处理组件 (如果IVF搜索涉及特定的并行模式)
//	// 例如，如果倒排列表的扫描是并行化的，可以尝试触发这个逻辑
//	// 这部分比较抽象，具体实现依赖于IVF的并行策略
//
//	// 10. 预热内存池/分配器 (如果IVF使用自定义内存管理)
//	// 模拟分配和释放操作
//
//	log.Trace("增强IVF索引组件预热完成")
//}
//
//// warmupEnhancedLSHComponents 预热增强LSH索引组件
//func (db *VectorDB) warmupEnhancedLSHComponents(query []float64, ctx SearchContext) {
//	if db.LshIndex == nil || !db.LshIndex.Enable {
//		return
//	}
//
//	log.Trace("开始预热增强LSH索引组件")
//
//	// 1. 预热LSH哈希函数
//	if db.LshIndex.Tables != nil {
//		for i, table := range db.LshIndex.Tables {
//			if i >= 5 { // 限制预热的哈希表数量
//				break
//			}
//			if table.HashFunctions != nil {
//				// 对查询向量进行哈希计算
//				for j, hashFunc := range table.HashFunctions {
//					if j >= 3 { // 限制预热的哈希函数数量
//						break
//					}
//					_, err := hashFunc.Hash(query)
//					if err != nil {
//						log.Trace("LSH哈希函数预热失败: %v", err)
//					}
//				}
//			}
//		}
//	}
//
//	// 2. 预热LSH桶访问
//	if db.LshIndex.Tables != nil {
//		for i, table := range db.LshIndex.Tables {
//			if i >= 3 { // 限制预热的哈希表数量
//				break
//			}
//			if table.Buckets != nil {
//				// 访问几个桶
//				bucketCount := 0
//				for _, bucket := range table.Buckets {
//					if bucketCount >= 5 { // 限制预热的桶数量
//						break
//					}
//					if len(bucket.VectorIDs) > 0 {
//						// 访问桶中的前几个向量
//						for j := 0; j < min(2, len(bucket.VectorIDs)); j++ {
//							if vec, exists := db.vectors[bucket.VectorIDs[j]]; exists {
//								_ = acceler.AdaptiveCosineSimilarity(query, vec, db.GetOptimalStrategy(query))
//							}
//						}
//					}
//					bucketCount++
//				}
//			}
//		}
//	}
//
//	// 3. 预热LSH族
//	if db.LshFamilies != nil {
//		for familyName, family := range db.LshFamilies {
//			if family != nil && family.HashFunctions != nil {
//				log.Trace("预热LSH族: %s", familyName)
//				// 对查询向量进行LSH族哈希计算
//				for i, hashFunc := range family.HashFunctions {
//					if i >= 3 { // 限制预热的哈希函数数量
//						break
//					}
//					_, err := hashFunc.Hash(query)
//					if err != nil {
//						log.Trace("LSH族哈希函数预热失败: %v", err)
//					}
//				}
//			}
//		}
//	}
//
//	// 4. 预热自适应LSH
//	if db.AdaptiveLSH != nil {
//		// 模拟自适应参数调整
//		db.AdaptiveLSH.UpdateParameters(len(db.vectors), len(query), ctx.QualityLevel)
//	}
//
//	log.Trace("增强LSH索引组件预热完成")
//}
//
//// warmupEnhancedIVFAdvanced 增强IVF索引高级预热
//func (db *VectorDB) warmupEnhancedIVFAdvanced(sampleQueries [][]float64) {
//	if db.ivfIndex == nil || !db.ivfIndex.Enable {
//		return
//	}
//
//	log.Info("开始增强IVF索引高级预热")
//
//	// 1. 预热不同的nprobe值
//	nprobeValues := []int{1, 5, 10, 20, 50}
//	for _, nprobe := range nprobeValues {
//		if nprobe > len(db.ivfIndex.Clusters) {
//			continue
//		}
//		for i, query := range sampleQueries {
//			if i >= 2 { // 限制查询数量
//				break
//			}
//			_, err := db.EnhancedIVFSearch(query, 5, nprobe)
//			if err != nil {
//				log.Trace("nprobe=%d 预热失败: %v", nprobe, err)
//			}
//		}
//	}
//
//	// 2. 预热批量搜索
//	if len(sampleQueries) >= 3 {
//		batchQueries := sampleQueries[:3]
//		_, err := db.EnhancedIVFBatchSearch(batchQueries, 5, 10)
//		if err != nil {
//			log.Trace("批量搜索预热失败: %v", err)
//		}
//	}
//
//	// 3. 预热范围搜索（如果支持）
//	if len(sampleQueries) > 0 {
//		_, err := db.EnhancedIVFRangeSearch(sampleQueries[0], 0.8, 10)
//		if err != nil {
//			log.Trace("范围搜索预热失败: %v", err)
//		}
//	}
//
//	log.Info("增强IVF索引高级预热完成")
//}
//
//// warmupEnhancedLSHAdvanced 增强LSH索引高级预热
//func (db *VectorDB) warmupEnhancedLSHAdvanced(sampleQueries [][]float64) {
//	if db.LshIndex == nil || !db.LshIndex.Enable {
//		return
//	}
//
//	log.Info("开始增强LSH索引高级预热")
//
//	// 1. 预热不同的LSH参数组合
//	if db.LshConfig != nil {
//		// 保存原始参数
//		originalR := db.LshConfig.R
//		originalW := db.LshConfig.W
//
//		// 尝试不同的参数组合
//		paramCombos := []struct {
//			R float64
//			W float64
//		}{
//			{R: originalR, W: originalW},
//			{R: originalR / 2, W: originalW},
//			{R: originalR, W: originalW / 2},
//		}
//
//		for _, combo := range paramCombos {
//			if combo.R <= 0 || combo.W <= 0 {
//				continue
//			}
//			// 临时调整参数
//			db.LshConfig.R = float64(combo.R)
//			db.LshConfig.W = float64(combo.W)
//
//			for i, query := range sampleQueries {
//				if i >= 2 { // 限制查询数量
//					break
//				}
//				_, err := db.EnhancedLSHSearch(query, 5)
//				if err != nil {
//					log.Trace("LSH参数L=%d,K=%d 预热失败: %v", combo.R, combo.W, err)
//				}
//			}
//		}
//
//		// 恢复原始参数
//		db.LshConfig.R = originalR
//		db.LshConfig.W = originalW
//	}
//
//	// 2. 预热多表查询
//	if len(sampleQueries) > 0 {
//		_, err := db.EnhancedLSHMultiTableSearch(sampleQueries[0], 5)
//		if err != nil {
//			log.Trace("多表查询预热失败: %v", err)
//		}
//	}
//
//	// 3. 预热近似搜索
//	if len(sampleQueries) > 0 {
//		_, err := db.EnhancedLSHApproximateSearch(sampleQueries[0], 5, 0.8)
//		if err != nil {
//			log.Trace("近似搜索预热失败: %v", err)
//		}
//	}
//
//	log.Info("增强LSH索引高级预热完成")
//}
