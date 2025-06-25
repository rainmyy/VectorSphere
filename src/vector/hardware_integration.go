package vector

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/logger"
)

// NewVectorDBWithHardwareManager 使用硬件管理器创建向量数据库
func NewVectorDBWithHardwareManager(filePath string, numClusters int, hardwareManager *acceler.HardwareManager) (*VectorDB, error) {
	// 创建基本的VectorDB实例
	db := NewVectorDB(filePath, numClusters)
	
	// 检查参数
	if hardwareManager == nil {
		return db, nil // 如果硬件管理器为空，返回默认实例
	}
	
	// 应用硬件管理器
	err := db.ApplyHardwareManager(hardwareManager)
	if err != nil {
		logger.Warning("应用硬件管理器失败: %v", err)
		// 即使应用失败，仍然返回数据库实例，只是没有硬件加速
	}
	
	return db, nil
}

// NewVectorDBFromConfigFile 从配置文件创建向量数据库
func NewVectorDBFromConfigFile(filePath string, numClusters int, hardwareConfigPath string) (*VectorDB, error) {
	// 检查参数
	if hardwareConfigPath == "" {
		// 如果没有提供配置文件路径，使用默认配置
		return NewVectorDB(filePath, numClusters), nil
	}
	
	// 从配置文件创建硬件管理器
	hardwareManager, err := acceler.NewHardwareManagerFromFile(hardwareConfigPath)
	if err != nil {
		logger.Warning("从配置文件加载硬件管理器失败: %v", err)
		// 如果加载失败，使用默认配置
		return NewVectorDB(filePath, numClusters), nil
	}
	
	// 使用硬件管理器创建向量数据库
	return NewVectorDBWithHardwareManager(filePath, numClusters, hardwareManager)
}