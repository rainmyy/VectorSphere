package test

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"testing"
	"time"
)

// TestRDMAAcceleratorCreation 测试RDMA加速器创建
func TestRDMAAcceleratorCreation(t *testing.T) {
	t.Run("创建RDMA加速器", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable:   true,
			DeviceID: 0,
			PortNum:  8080,
			ClusterNodes: []string{"192.168.1.100:8080"},
		}
		
		rdma := acceler.NewRDMAAccelerator(0, 1, config)
		if rdma == nil {
			t.Fatal("RDMA加速器创建失败")
		}
		
		if rdma.GetType() != "RDMA" {
			t.Errorf("期望类型为RDMA，实际为%s", rdma.GetType())
		}
	})
	
	t.Run("RDMA加速器可用性检查", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		rdma := acceler.NewRDMAAccelerator(0, 1, config)
		
		// 注意：在没有RDMA的环境中，IsAvailable可能返回false
		available := rdma.IsAvailable()
		t.Logf("RDMA可用性: %v", available)
	})
	
	t.Run("无效配置", func(t *testing.T) {
		// 测试nil配置
		rdma := acceler.NewRDMAAccelerator(0, 1, nil)
		if rdma == nil {
			t.Error("即使配置为nil，RDMA加速器也应该能创建")
		}
	})
	
	t.Run("无效节点数量", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		// 测试0个节点
		rdma := acceler.NewRDMAAccelerator(0, 0, config)
		if rdma == nil {
			t.Error("即使节点数为0，RDMA加速器也应该能创建")
		}
	})
}

// TestRDMAAcceleratorInitialization 测试RDMA加速器初始化
func TestRDMAAcceleratorInitialization(t *testing.T) {
	t.Run("RDMA初始化", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable:   true,
			DeviceID: 0,
			PortNum:  8080,
			ClusterNodes: []string{"192.168.1.100:8080"},
		}
		rdma := acceler.NewRDMAAccelerator(0, 2, config)
		
		err := rdma.Initialize()
		if err != nil {
			// 在没有RDMA的环境中，初始化失败是正常的
			t.Logf("RDMA初始化失败（可能是因为没有RDMA设备）: %v", err)
			return
		}
		
		defer rdma.Shutdown()
		
		if !rdma.IsAvailable() {
			t.Error("初始化后RDMA应该可用")
		}
	})
	
	t.Run("重复初始化", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		rdma := acceler.NewRDMAAccelerator(0, 1, config)
		
		err1 := rdma.Initialize()
		if err1 != nil {
			t.Logf("RDMA初始化失败: %v", err1)
			return
		}
		
		defer rdma.Shutdown()
		
		// 重复初始化应该成功
		err2 := rdma.Initialize()
		if err2 != nil {
			t.Errorf("重复初始化失败: %v", err2)
		}
	})
	
	t.Run("启动和停止", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		rdma := acceler.NewRDMAAccelerator(0, 1, config)
		
		err := rdma.Start()
		if err != nil {
			t.Logf("RDMA启动失败: %v", err)
			return
		}
		
		err = rdma.Stop()
		if err != nil {
			t.Errorf("RDMA停止失败: %v", err)
		}
	})
	
	t.Run("集群连接管理", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable:   true,
			DeviceID: 0,
			PortNum:  8080,
			ClusterNodes: []string{"192.168.1.100:8080"},
		}
		rdma := acceler.NewRDMAAccelerator(0, 3, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过集群测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		// 测试集群连接
		err = rdma.Start()
		if err != nil {
			t.Logf("RDMA集群启动失败: %v", err)
			return
		}
		
		defer rdma.Stop()
		
		// 验证连接状态
		if !rdma.IsAvailable() {
			t.Error("启动后RDMA应该可用")
		}
	})
}

// TestRDMAAcceleratorComputation 测试RDMA计算功能
func TestRDMAAcceleratorComputation(t *testing.T) {
	t.Run("分布式距离计算", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		rdma := acceler.NewRDMAAccelerator(0, 2, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过计算测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
			{0.0, 0.0, 1.0},
			{1.0, 1.0, 0.0},
		}
		
		distances, err := rdma.ComputeDistance(query, vectors)
		if err != nil {
			t.Fatalf("RDMA距离计算失败: %v", err)
		}
		
		if len(distances) != len(vectors) {
			t.Errorf("期望距离数量为%d，实际为%d", len(vectors), len(distances))
		}
		
		// 验证第一个向量的距离应该最小（相同向量）
		if len(distances) > 0 && distances[0] > 0.1 {
			t.Errorf("相同向量的距离应该接近0，实际为%f", distances[0])
		}
	})
	
	t.Run("分布式批量搜索", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		rdma := acceler.NewRDMAAccelerator(0, 2, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过批量搜索测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		query := []float64{1.0, 0.0, 0.0}
		database := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
			{0.0, 0.0, 1.0},
			{1.0, 1.0, 0.0},
			{0.5, 0.5, 0.0},
		}
		
		options := entity.SearchOptions{
			K:         3,
			Threshold: 0.8,
		}
		
		results, err := rdma.BatchSearch([][]float64{query}, database, options.K)
		if err != nil {
			t.Fatalf("RDMA批量搜索失败: %v", err)
		}
		
		if len(results) != 1 {
			t.Errorf("期望结果数量为1，实际为%d", len(results))
		}
		
		if len(results) > 0 && len(results[0]) > options.K {
			t.Errorf("返回结果数量不应超过K=%d，实际为%d", options.K, len(results[0]))
		}
	})
	
	t.Run("分布式数据传输", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable:   true,
			DeviceID: 0,
			PortNum:  8080,
		}
		rdma := acceler.NewRDMAAccelerator(0, 2, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过数据传输测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		// 创建大量数据进行传输测试
		dimension := 128
		numVectors := 1000
		
		query := make([]float64, dimension)
		for i := range query {
			query[i] = float64(i) / float64(dimension)
		}
		
		vectors := make([][]float64, numVectors)
		for i := range vectors {
			vectors[i] = make([]float64, dimension)
			for j := range vectors[i] {
				vectors[i][j] = float64((i+j)%100) / 100.0
			}
		}
		
		start := time.Now()
		distances, err := rdma.ComputeDistance(query, vectors)
		duration := time.Since(start)
		
		if err != nil {
			t.Fatalf("RDMA数据传输失败: %v", err)
		}
		
		if len(distances) != numVectors {
			t.Errorf("期望距离数量为%d，实际为%d", numVectors, len(distances))
		}
		
		t.Logf("RDMA处理%d个%d维向量耗时: %v", numVectors, dimension, duration)
	})
}

// TestRDMAAcceleratorPerformance 测试RDMA性能
func TestRDMAAcceleratorPerformance(t *testing.T) {
	t.Run("网络性能指标", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		rdma := acceler.NewRDMAAccelerator(0, 2, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过性能测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		// 执行一些操作来生成性能数据
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
		}
		
		_, err = rdma.ComputeDistance(query, vectors)
		if err != nil {
			t.Fatalf("计算失败: %v", err)
		}
		
		// 检查性能指标
		metrics := rdma.GetPerformanceMetrics()
		if metrics.LatencyCurrent < 0 {
			t.Error("性能指标应该有效")
		}
		
		// 检查网络统计信息
		stats := rdma.GetStats()
		if stats.TotalOperations < 0 {
			t.Error("统计信息应该有效")
		}
	})
	
	t.Run("带宽利用率", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable:   true,
			DeviceID: 0,
			PortNum:  8080,
		}
		rdma := acceler.NewRDMAAccelerator(0, 2, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过带宽测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		// 获取网络统计信息
		netStats := rdma.GetNetworkStats()
		
		// 验证统计信息的合理性
		if netStats != nil {
			if bytesSent, ok := netStats["bytes_sent"]; ok {
				if sent, ok := bytesSent.(int64); ok && sent < 0 {
					t.Errorf("发送字节数应该非负，实际为%d", sent)
				}
			}
			if bytesReceived, ok := netStats["bytes_received"]; ok {
				if received, ok := bytesReceived.(int64); ok && received < 0 {
					t.Errorf("接收字节数应该非负，实际为%d", received)
				}
			}
		}
	})
	
	t.Run("多节点负载均衡", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		// 创建多节点RDMA集群
		rdma := acceler.NewRDMAAccelerator(0, 4, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过负载均衡测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		// 创建大规模测试数据
		dimension := 256
		numQueries := 100
		numVectors := 5000
		
		queries := make([][]float64, numQueries)
		for i := range queries {
			queries[i] = make([]float64, dimension)
			for j := range queries[i] {
				queries[i][j] = float64((i+j)%100) / 100.0
			}
		}
		
		vectors := make([][]float64, numVectors)
		for i := range vectors {
			vectors[i] = make([]float64, dimension)
			for j := range vectors[i] {
				vectors[i][j] = float64((i+j)%100) / 100.0
			}
		}
		
		options := entity.SearchOptions{
			K:         10,
			Threshold: 0.5,
		}
		
		start := time.Now()
		results, err := rdma.BatchSearch(queries, vectors, options.K)
		duration := time.Since(start)
		
		if err != nil {
			t.Fatalf("多节点批量搜索失败: %v", err)
		}
		
		if len(results) != numQueries {
			t.Errorf("期望结果数量为%d，实际为%d", numQueries, len(results))
		}
		
		t.Logf("RDMA多节点处理%d个查询，%d个向量，耗时: %v", numQueries, numVectors, duration)
	})
}

// TestRDMAAcceleratorErrorHandling 测试RDMA错误处理
func TestRDMAAcceleratorErrorHandling(t *testing.T) {
	t.Run("无效服务器地址", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable:   true,
			DeviceID: 0,
			PortNum:  8080,
			ClusterNodes: []string{"invalid_address"},
		}
		rdma := acceler.NewRDMAAccelerator(0, 1, config)
		
		err := rdma.Initialize()
		if err == nil {
			t.Error("无效服务器地址应该导致初始化失败")
		}
	})
	
	t.Run("网络连接失败", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable:   true,
			DeviceID: 0,
			PortNum:  9999,
			ClusterNodes: []string{"192.168.255.255:9999"}, // 不存在的地址
		}
		rdma := acceler.NewRDMAAccelerator(0, 1, config)
		
		err := rdma.Initialize()
		if err == nil {
			t.Error("无法连接的地址应该导致初始化失败")
		}
	})
	
	t.Run("未初始化操作", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		rdma := acceler.NewRDMAAccelerator(0, 1, config)
		// 不调用Initialize
		
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		
		_, err := rdma.ComputeDistance(query, vectors)
		if err == nil {
			t.Error("未初始化的RDMA应该返回错误")
		}
	})
	
	t.Run("空数据处理", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		rdma := acceler.NewRDMAAccelerator(0, 1, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过空数据测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		// 测试空查询向量
		_, err = rdma.ComputeDistance(nil, [][]float64{{1.0, 0.0}})
		if err == nil {
			t.Error("空查询向量应该返回错误")
		}
		
		// 测试空数据库向量
		_, err = rdma.ComputeDistance([]float64{1.0, 0.0}, nil)
		if err == nil {
			t.Error("空数据库向量应该返回错误")
		}
	})
	
	t.Run("节点故障处理", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		rdma := acceler.NewRDMAAccelerator(0, 3, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过节点故障测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		// 模拟节点故障后的计算
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		
		_, err = rdma.ComputeDistance(query, vectors)
		// 在节点故障的情况下，应该能够容错处理
		if err != nil {
			t.Logf("节点故障时计算失败（可能是预期的）: %v", err)
		}
	})
}

// TestRDMAAcceleratorConfiguration 测试RDMA配置
func TestRDMAAcceleratorConfiguration(t *testing.T) {
	t.Run("配置验证", func(t *testing.T) {
		// 有效配置
		validConfig := &acceler.RDMAConfig{
			Enable:   true,
			DeviceID: 0,
			PortNum:  8080,
			ClusterNodes: []string{"192.168.1.100:8080"},
		}
		
		rdma := acceler.NewRDMAAccelerator(0, 2, validConfig)
		if rdma == nil {
			t.Error("有效配置应该能创建RDMA加速器")
		}
		
		// 禁用配置
		disabledConfig := &acceler.RDMAConfig{
			Enable: false,
		}
		
		rdma2 := acceler.NewRDMAAccelerator(0, 1, disabledConfig)
		if rdma2 == nil {
			t.Error("即使禁用，也应该能创建RDMA加速器")
		}
	})
	
	t.Run("端口配置", func(t *testing.T) {
		// 测试不同的端口配置
		ports := []int{8080, 8081, 8082}
		
		for _, port := range ports {
			config := &acceler.RDMAConfig{
				Enable:   true,
				DeviceID: 0,
				PortNum:  port,
			}
			
			rdma := acceler.NewRDMAAccelerator(0, 1, config)
			if rdma == nil {
				t.Errorf("端口%d应该能创建RDMA加速器", port)
			}
		}
	})
	
	t.Run("队列大小限制", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable:   true,
			DeviceID: 0,
			PortNum:  8080,
			QueueSize: 1, // 限制队列大小
		}
		
		rdma := acceler.NewRDMAAccelerator(0, 5, config) // 5个节点但只允许1个连接
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("队列大小限制导致初始化失败（可能是预期的）: %v", err)
		}
	})
}

// TestRDMAAcceleratorConcurrency 测试RDMA并发安全性
func TestRDMAAcceleratorConcurrency(t *testing.T) {
	config := &acceler.RDMAConfig{
		Enable: true,
	}
	rdma := acceler.NewRDMAAccelerator(0, 2, config)
	
	err := rdma.Initialize()
	if err != nil {
		t.Logf("RDMA初始化失败，跳过并发测试: %v", err)
		return
	}
	defer rdma.Shutdown()
	
	// 并发执行多个计算任务
	const numGoroutines = 10
	done := make(chan bool, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			query := []float64{float64(id), 0.0, 0.0}
			vectors := [][]float64{
				{1.0, 0.0, 0.0},
				{0.0, 1.0, 0.0},
			}
			
			_, err := rdma.ComputeDistance(query, vectors)
			if err != nil {
				t.Errorf("Goroutine %d 计算失败: %v", id, err)
			}
			
			done <- true
		}(i)
	}
	
	// 等待所有goroutine完成
	for i := 0; i < numGoroutines; i++ {
		<-done
	}
	
	stats := rdma.GetStats()
	if stats.TotalOperations < numGoroutines {
		t.Errorf("期望至少%d次操作，实际为%d", numGoroutines, stats.TotalOperations)
	}
}

// TestRDMAAcceleratorClusterOperations 测试RDMA集群操作
func TestRDMAAcceleratorClusterOperations(t *testing.T) {
	t.Run("集群扩展", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		
		// 从小集群开始
		rdma := acceler.NewRDMAAccelerator(0, 1, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过集群扩展测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		// 测试基本功能
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		
		_, err = rdma.ComputeDistance(query, vectors)
		if err != nil {
			t.Fatalf("单节点计算失败: %v", err)
		}
		
		t.Log("单节点RDMA集群测试通过")
	})
	
	t.Run("集群故障恢复", func(t *testing.T) {
		config := &acceler.RDMAConfig{
			Enable: true,
		}
		rdma := acceler.NewRDMAAccelerator(0, 3, config)
		
		err := rdma.Initialize()
		if err != nil {
			t.Logf("RDMA初始化失败，跳过故障恢复测试: %v", err)
			return
		}
		defer rdma.Shutdown()
		
		// 模拟故障前的正常操作
		query := []float64{1.0, 0.0, 0.0}
		vectors := [][]float64{{1.0, 0.0, 0.0}}
		
		_, err = rdma.ComputeDistance(query, vectors)
		if err != nil {
			t.Fatalf("故障前计算失败: %v", err)
		}
		
		// 模拟重启
		err = rdma.Stop()
		if err != nil {
			t.Logf("停止RDMA失败: %v", err)
		}
		
		err = rdma.Start()
		if err != nil {
			t.Logf("重启RDMA失败: %v", err)
			return
		}
		
		// 故障恢复后的操作
		_, err = rdma.ComputeDistance(query, vectors)
		if err != nil {
			t.Logf("故障恢复后计算失败: %v", err)
		}
		
		t.Log("RDMA集群故障恢复测试完成")
	})
}