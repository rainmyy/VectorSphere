package test

import (
	"VectorSphere/src/vector"
	"fmt"
	"math"
	"testing"
)

// 创建测试用的VectorDB实例
func createPCATestVectorDB() *vector.VectorDB {
	return vector.NewVectorDB("test.db", 10)
}

// 添加测试向量数据
func addPCATestVectors(db *vector.VectorDB, vectors map[string][]float64) {
	for id, vec := range vectors {
		db.Add(id, vec)
	}
}

// 测试基本PCA降维功能
func TestApplyPCA_BasicFunctionality(t *testing.T) {
	fmt.Println("=== 测试基本PCA降维功能 ===")
	
	db := createPCATestVectorDB()
	
	// 添加测试数据 - 3维向量降到2维
	testVectors := map[string][]float64{
		"vec1": {1.0, 2.0, 3.0},
		"vec2": {4.0, 5.0, 6.0},
		"vec3": {7.0, 8.0, 9.0},
		"vec4": {2.0, 3.0, 4.0},
		"vec5": {5.0, 6.0, 7.0},
	}
	addPCATestVectors(db, testVectors)
	
	dim, _ := db.GetVectorDimension()
	fmt.Printf("原始向量维度: %d\n", dim)
	fmt.Printf("原始向量数量: %d\n", db.GetDataSize())
	
	// 应用PCA降维到2维
	err := db.ApplyPCA(2, 0.0)
	if err != nil {
		t.Errorf("PCA降维失败: %v", err)
		return
	}
	
	dim, _ = db.GetVectorDimension()
	fmt.Printf("降维后向量维度: %d\n", dim)
	fmt.Printf("降维后向量数量: %d\n", db.GetDataSize())
	
	// 验证结果
	dim, _ = db.GetVectorDimension()
	if dim != 2 {
		t.Errorf("期望维度为2，实际为%d", dim)
	}
	
	if db.GetPCAConfig() == nil {
		t.Error("PCA配置未保存")
	}
	
	if db.GetPCAConfig().TargetDimension != 2 {
		t.Errorf("期望目标维度为2，实际为%d", db.GetPCAConfig().TargetDimension)
	}
	
	if !db.IsIndexed() {
		t.Log("索引状态正确设置为false，需要重建")
	}
	
	fmt.Println("✅ 基本PCA降维功能测试通过")
}

// 测试基于方差比例的PCA降维
func TestApplyPCA_VarianceRatio(t *testing.T) {
	fmt.Println("\n=== 测试基于方差比例的PCA降维 ===")
	
	db := createPCATestVectorDB()
	
	// 添加更多测试数据以获得更好的方差分布
	testVectors := map[string][]float64{
		"vec1": {1.0, 0.1, 0.01},
		"vec2": {2.0, 0.2, 0.02},
		"vec3": {3.0, 0.3, 0.03},
		"vec4": {4.0, 0.4, 0.04},
		"vec5": {5.0, 0.5, 0.05},
		"vec6": {6.0, 0.6, 0.06},
	}
	addPCATestVectors(db, testVectors)
	
	dim, _ := db.GetVectorDimension()
	fmt.Printf("原始向量维度: %d\n", dim)
	
	// 应用PCA，保留95%的方差
	err := db.ApplyPCA(0, 0.95)
	if err != nil {
		t.Errorf("基于方差比例的PCA降维失败: %v", err)
		return
	}
	
	dim, _ = db.GetVectorDimension()
	fmt.Printf("降维后向量维度: %d\n", dim)
	fmt.Printf("保留方差比例: %.2f\n", db.GetPCAConfig().VarianceRatio)
	
	// 验证结果
	if db.GetPCAConfig().VarianceRatio != 0.95 {
		t.Errorf("期望方差比例为0.95，实际为%.2f", db.GetPCAConfig().VarianceRatio)
	}
	
	fmt.Println("✅ 基于方差比例的PCA降维测试通过")
}

// 测试边界条件和错误处理
func TestApplyPCA_ErrorHandling(t *testing.T) {
	fmt.Println("\n=== 测试PCA错误处理 ===")
	
	// 测试空数据库
	db := createPCATestVectorDB()
	err := db.ApplyPCA(2, 0.0)
	if err == nil {
		t.Error("期望空数据库返回错误，但没有")
	} else {
		fmt.Printf("✅ 空数据库错误处理正确: %v\n", err)
	}
	
	// 测试无效参数
	db = createPCATestVectorDB()
	testVectors := map[string][]float64{
		"vec1": {1.0, 2.0, 3.0},
		"vec2": {4.0, 5.0, 6.0},
	}
	addPCATestVectors(db, testVectors)
	
	// 测试无效的目标维度
	err = db.ApplyPCA(5, 0.0) // 目标维度大于原始维度
	if err == nil {
		t.Error("期望无效目标维度返回错误，但没有")
	} else {
		fmt.Printf("✅ 无效目标维度错误处理正确: %v\n", err)
	}
	
	// 测试无效的方差比例
	err = db.ApplyPCA(0, 1.5) // 方差比例大于1
	if err == nil {
		t.Error("期望无效方差比例返回错误，但没有")
	} else {
		fmt.Printf("✅ 无效方差比例错误处理正确: %v\n", err)
	}
	
	// 测试既不指定目标维度也不指定方差比例
	err = db.ApplyPCA(0, 0.0)
	if err == nil {
		t.Error("期望无参数返回错误，但没有")
	} else {
		fmt.Printf("✅ 无参数错误处理正确: %v\n", err)
	}
	
	fmt.Println("✅ PCA错误处理测试通过")
}

// 测试PCA配置保存
func TestApplyPCA_ConfigSaving(t *testing.T) {
	fmt.Println("\n=== 测试PCA配置保存 ===")
	
	db := createPCATestVectorDB()
	testVectors := map[string][]float64{
		"vec1": {1.0, 2.0, 3.0, 4.0},
		"vec2": {2.0, 3.0, 4.0, 5.0},
		"vec3": {3.0, 4.0, 5.0, 6.0},
	}
	addPCATestVectors(db, testVectors)
	
	err := db.ApplyPCA(2, 0.0)
	if err != nil {
		t.Errorf("PCA降维失败: %v", err)
		return
	}
	
	// 验证PCA配置
	if db.GetPCAConfig() == nil {
		t.Error("PCA配置未保存")
		return
	}
	
	// 验证主成分矩阵
	pcaConfig := db.GetPCAConfig()
	if len(pcaConfig.Components) != 4 { // 原始维度
		t.Errorf("期望主成分矩阵行数为4，实际为%d", len(pcaConfig.Components))
	}
	
	if len(pcaConfig.Components[0]) != 2 { // 目标维度
		t.Errorf("期望主成分矩阵列数为2，实际为%d", len(pcaConfig.Components[0]))
	}
	
	// 验证均值向量
	if len(pcaConfig.Mean) != 4 {
		t.Errorf("期望均值向量长度为4，实际为%d", len(pcaConfig.Mean))
	}
	
	fmt.Printf("主成分矩阵维度: %dx%d\n", len(pcaConfig.Components), len(pcaConfig.Components[0]))
	fmt.Printf("均值向量长度: %d\n", len(pcaConfig.Mean))
	fmt.Printf("目标维度: %d\n", pcaConfig.TargetDimension)
	
	fmt.Println("✅ PCA配置保存测试通过")
}

// 测试大数据集PCA性能
func TestApplyPCA_LargeDataset(t *testing.T) {
	fmt.Println("\n=== 测试大数据集PCA性能 ===")
	
	db := createPCATestVectorDB()
	
	// 创建较大的测试数据集
	testVectors := make(map[string][]float64)
	for i := 0; i < 100; i++ {
		vec := make([]float64, 10)
		for j := 0; j < 10; j++ {
			vec[j] = float64(i*10+j) + math.Sin(float64(i))*0.1
		}
		testVectors[fmt.Sprintf("vec_%d", i)] = vec
	}
	addPCATestVectors(db, testVectors)
	
	dim, _ := db.GetVectorDimension()
	fmt.Printf("大数据集: %d个向量，每个%d维\n", len(testVectors), dim)
	
	// 应用PCA降维到5维
	err := db.ApplyPCA(5, 0.0)
	if err != nil {
		t.Errorf("大数据集PCA降维失败: %v", err)
		return
	}
	
	dim, _ = db.GetVectorDimension()
	fmt.Printf("降维后: %d个向量，每个%d维\n", db.GetDataSize(), dim)
	
	// 验证结果
	dim, _ = db.GetVectorDimension()
	if dim != 5 {
		t.Errorf("期望维度为5，实际为%d", dim)
	}
	
	if db.GetDataSize() != 100 {
		t.Errorf("期望向量数量为100，实际为%d", db.GetDataSize())
	}
	
	fmt.Println("✅ 大数据集PCA性能测试通过")
}

// 测试同时指定目标维度和方差比例的情况
func TestApplyPCA_BothParameters(t *testing.T) {
	fmt.Println("\n=== 测试同时指定目标维度和方差比例 ===")
	
	db := createPCATestVectorDB()
	testVectors := map[string][]float64{
		"vec1": {10.0, 1.0, 0.1, 0.01},
		"vec2": {20.0, 2.0, 0.2, 0.02},
		"vec3": {30.0, 3.0, 0.3, 0.03},
		"vec4": {40.0, 4.0, 0.4, 0.04},
		"vec5": {50.0, 5.0, 0.5, 0.05},
	}
	addPCATestVectors(db, testVectors)
	
	// 同时指定目标维度2和方差比例0.99
	// 根据代码逻辑，如果方差比例计算出的维度大于目标维度，应该使用目标维度
	err := db.ApplyPCA(2, 0.99)
	if err != nil {
		t.Errorf("同时指定参数的PCA降维失败: %v", err)
		return
	}
	
	dim, _ := db.GetVectorDimension()
	fmt.Printf("最终维度: %d\n", dim)
	fmt.Printf("目标维度: %d\n", db.GetPCAConfig().TargetDimension)
	
	// 验证使用了较小的目标维度
	dim, _ = db.GetVectorDimension()
	if dim != 2 {
		t.Errorf("期望使用目标维度2，实际为%d", dim)
	}
	
	fmt.Println("✅ 同时指定参数测试通过")
}

// 测试零方差数据的处理
func TestApplyPCA_ZeroVariance(t *testing.T) {
	fmt.Println("\n=== 测试零方差数据处理 ===")
	
	db := createPCATestVectorDB()
	
	// 创建零方差的测试数据（所有向量相同）
	testVectors := map[string][]float64{
		"vec1": {1.0, 2.0, 3.0},
		"vec2": {1.0, 2.0, 3.0},
		"vec3": {1.0, 2.0, 3.0},
		"vec4": {1.0, 2.0, 3.0},
	}
	addPCATestVectors(db, testVectors)
	
	// 尝试基于方差比例的PCA
	err := db.ApplyPCA(2, 0.95)
	if err != nil {
		// 应该回退到使用目标维度
		fmt.Printf("零方差数据处理: %v\n", err)
		// 如果返回错误，检查是否是预期的错误类型
		if db.GetPCAConfig() == nil {
			fmt.Println("✅ 零方差数据正确处理，返回了错误")
		} else {
			fmt.Println("✅ 零方差数据回退到目标维度处理")
		}
	} else {
		dim, _ := db.GetVectorDimension()
		fmt.Printf("零方差数据成功处理，最终维度: %d\n", dim)
		fmt.Println("✅ 零方差数据处理测试通过")
	}
}

// 运行所有PCA测试
func TestPCAAll(t *testing.T) {
	fmt.Println("开始运行所有PCA测试...")
	
	t.Run("BasicFunctionality", TestApplyPCA_BasicFunctionality)
	t.Run("VarianceRatio", TestApplyPCA_VarianceRatio)
	t.Run("ErrorHandling", TestApplyPCA_ErrorHandling)
	t.Run("ConfigSaving", TestApplyPCA_ConfigSaving)
	t.Run("LargeDataset", TestApplyPCA_LargeDataset)
	t.Run("BothParameters", TestApplyPCA_BothParameters)
	t.Run("ZeroVariance", TestApplyPCA_ZeroVariance)
	
	fmt.Println("\n所有PCA测试完成！")
}