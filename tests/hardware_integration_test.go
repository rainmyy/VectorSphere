package tests

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/vector"
	"os"
	"path/filepath"
	"testing"
)

// 测试从配置文件创建向量数据库
func TestNewVectorDBFromConfigFile(t *testing.T) {
	// 创建临时配置文件
	tmpDir, err := os.MkdirTemp("", "vectorsphere-test")
	if err != nil {
		t.Fatalf("无法创建临时目录: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// 创建配置文件
	configPath := filepath.Join(tmpDir, "hardware_config.json")
	configContent := `{
		"GPU": {
			"Enable": true,
			"Devices": [0],
			"CUDA": {
				"Enable": true,
				"MemoryLimit": 4096,
				"BatchSize": 128
			}
		},
		"CPU": {
			"Enable": true,
			"NumThreads": 8,
			"EnableAVX": true,
			"EnableAVX512": true
		}
	}`

	err = os.WriteFile(configPath, []byte(configContent), 0644)
	if err != nil {
		t.Fatalf("无法写入配置文件: %v", err)
	}

	// 从配置文件创建向量数据库
	db, err := vector.NewVectorDBFromConfigFile("test_vectors.json", 16, configPath)
	if err != nil {
		t.Fatalf("从配置文件创建向量数据库失败: %v", err)
	}

	// 验证数据库是否正确创建
	if db == nil {
		t.Fatal("创建的数据库为空")
	}
}

// 测试使用硬件管理器创建向量数据库
func TestNewVectorDBWithHardwareManager(t *testing.T) {
	// 创建硬件配置
	hardwareConfig := &acceler.HardwareConfig{
		GPU: acceler.GPUConfig{
			Enable:  true,
			Devices: []int{0},
			CUDA: acceler.CUDAConfig{
				Enable:      true,
				MemoryLimit: 4096,
				BatchSize:   128,
			},
		},
		CPU: acceler.CPUConfig{
			Enable:       true,
			NumThreads:   8,
			EnableAVX:    true,
			EnableAVX512: true,
		},
	}

	// 创建硬件管理器
	hardwareManager, err := acceler.NewHardwareManager(hardwareConfig)
	if err != nil {
		t.Fatalf("创建硬件管理器失败: %v", err)
	}

	// 使用硬件管理器创建向量数据库
	db, err := vector.NewVectorDBWithHardwareManager("test_vectors.json", 16, hardwareManager)
	if err != nil {
		t.Fatalf("使用硬件管理器创建向量数据库失败: %v", err)
	}

	// 验证数据库是否正确创建
	if db == nil {
		t.Fatal("创建的数据库为空")
	}

	// 验证硬件能力是否正确设置
	if !db.HardwareCaps.HasGPU && hardwareConfig.GPU.Enable {
		t.Error("GPU 配置已启用，但数据库的 HasGPU 为 false")
	}
}

// 测试应用硬件管理器到现有向量数据库
func TestApplyHardwareManager(t *testing.T) {
	// 创建向量数据库
	db := vector.NewVectorDB("test_vectors.json", 16)

	// 创建硬件配置
	hardwareConfig := &acceler.HardwareConfig{
		GPU: acceler.GPUConfig{
			Enable:  true,
			Devices: []int{0},
		},
		CPU: acceler.CPUConfig{
			Enable:     true,
			NumThreads: 8,
			EnableAVX:  true,
		},
	}

	// 创建硬件管理器
	hardwareManager, err := acceler.NewHardwareManager(hardwareConfig)
	if err != nil {
		t.Fatalf("创建硬件管理器失败: %v", err)
	}

	// 应用硬件管理器
	err = db.ApplyHardwareManager(hardwareManager)
	if err != nil {
		t.Fatalf("应用硬件管理器失败: %v", err)
	}

	// 验证硬件能力是否正确设置
	if !db.HardwareCaps.HasGPU && hardwareConfig.GPU.Enable {
		t.Error("GPU 配置已启用，但数据库的 HasGPU 为 false")
	}
}

// 测试硬件配置的保存和加载
func TestHardwareConfigSaveAndLoad(t *testing.T) {
	// 创建临时目录
	tmpDir, err := os.MkdirTemp("", "vectorsphere-test")
	if err != nil {
		t.Fatalf("无法创建临时目录: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// 创建硬件配置
	originalConfig := &acceler.HardwareConfig{
		GPU: acceler.GPUConfig{
			Enable:  true,
			Devices: []int{0, 1},
			CUDA: acceler.CUDAConfig{
				Enable:      true,
				MemoryLimit: 8192,
				BatchSize:   256,
			},
		},
		CPU: acceler.CPUConfig{
			Enable:       true,
			NumThreads:   16,
			EnableAVX:    true,
			EnableAVX512: true,
		},
	}

	// 创建硬件管理器
	originalManager, err := acceler.NewHardwareManager(originalConfig)
	if err != nil {
		t.Fatalf("创建硬件管理器失败: %v", err)
	}

	// 保存配置到文件
	configPath := filepath.Join(tmpDir, "hardware_config.json")
	err = originalManager.SaveConfigToFile(configPath)
	if err != nil {
		t.Fatalf("保存硬件配置失败: %v", err)
	}

	// 从文件加载配置
	loadedManager, err := acceler.NewHardwareManagerFromFile(configPath)
	if err != nil {
		t.Fatalf("从文件加载硬件管理器失败: %v", err)
	}

	// 验证加载的配置是否与原始配置一致
	loadedConfig := loadedManager.GetConfig()
	if loadedConfig.GPU.Enable != originalConfig.GPU.Enable {
		t.Errorf("GPU.Enable 不匹配: 原始=%v, 加载=%v", 
			originalConfig.GPU.Enable, loadedConfig.GPU.Enable)
	}

	if loadedConfig.CPU.NumThreads != originalConfig.CPU.NumThreads {
		t.Errorf("CPU.NumThreads 不匹配: 原始=%v, 加载=%v", 
			originalConfig.CPU.NumThreads, loadedConfig.CPU.NumThreads)
	}
}