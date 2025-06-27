package main

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/vector"
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	// 设置日志级别
	logger.SetLogLevel(logger.INFO)

	// 示例1：使用默认配置创建向量数据库
	db1 := vector.NewVectorDB("test_vectors.json", 16)
	fmt.Println("使用默认配置创建的向量数据库：", db1 != nil)

	// 示例2：从配置文件创建向量数据库
	hardwareConfigPath := filepath.Join("configs", "hardware_config.json")
	db2, err := vector.NewVectorDBFromConfigFile("test_vectors.json", 16, hardwareConfigPath)
	if err != nil {
		fmt.Printf("从配置文件创建向量数据库失败: %v\n", err)
	} else {
		fmt.Println("从配置文件创建的向量数据库：", db2 != nil)
	}

	// 示例3：手动创建硬件管理器并应用到向量数据库
	// 创建硬件配置
	hardwareConfig := &acceler.HardwareConfig{
		GPU: acceler.GPUConfig{
			Enable:  true,
			Devices: []int{0}, // 使用第一个GPU设备
			CUDA: acceler.CUDAConfig{
				Enable:            true,
				MemoryLimit:       4096, // 4GB
				BatchSize:         128,
				ComputeCapability: "7.5",
			},
		},
		CPU: acceler.CPUConfig{
			Enable:       true,
			Threads:      8,
			EnableAVX:    true,
			EnableAVX512: true,
		},
	}

	// 创建硬件管理器
	hardwareManager := acceler.NewHardwareManager(hardwareConfig)

	// 使用硬件管理器创建向量数据库
	db3, err := vector.NewVectorDBWithHardwareManager("test_vectors.json", 16, hardwareManager)
	if err != nil {
		fmt.Printf("使用硬件管理器创建向量数据库失败: %v\n", err)
	} else {
		fmt.Println("使用硬件管理器创建的向量数据库：", db3 != nil)
	}

	// 示例4：保存和加载硬件配置
	configFilePath := filepath.Join("configs", "custom_hardware_config.json")

	// 确保目录存在
	os.MkdirAll(filepath.Dir(configFilePath), 0755)

	// 保存配置到文件
	err = hardwareManager.SaveConfigToFile(configFilePath)
	if err != nil {
		fmt.Printf("保存硬件配置失败: %v\n", err)
	} else {
		fmt.Println("硬件配置已保存到:", configFilePath)
	}

	// 从文件加载配置
	newManager, err := acceler.NewHardwareManagerFromFile(configFilePath)
	if err != nil {
		fmt.Printf("从文件加载硬件管理器失败: %v\n", err)
	} else {
		fmt.Println("从文件加载的硬件管理器：", newManager != nil)
	}
}
