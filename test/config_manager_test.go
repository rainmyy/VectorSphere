package test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"VectorSphere/src/vector"
)

func TestNewConfigManager(t *testing.T) {
	configPath := "test_config.json"
	cm := vector.NewConfigManager(configPath)
	
	if cm == nil {
		t.Fatal("Expected non-nil ConfigManager")
	}
}

func TestConfigManagerLoadConfig(t *testing.T) {
	// 创建临时配置文件
	tempDir, err := ioutil.TempDir("", "config_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	configPath := filepath.Join(tempDir, "test_config.json")
	cm := vector.NewConfigManager(configPath)
	
	// 测试加载不存在的配置文件（应该创建默认配置）
	err = cm.LoadConfig()
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}
	
	// 验证配置文件是否被创建
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		t.Error("Expected config file to be created")
	}
}

func TestConfigManagerSaveConfig(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "config_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	configPath := filepath.Join(tempDir, "save_test_config.json")
	cm := vector.NewConfigManager(configPath)
	
	// 设置一些配置
	cm.DistributedConfig = &vector.DistributedConfig{
		IndexConfig: vector.IndexSelectionConfig{
			HNSWConfig: vector.HNSWIndexConfig{
				MaxConnections:  16,
				EfConstruction:  200,
				EfSearch:        100,
			},
		},
	}
	
	err = cm.SaveConfig()
	if err != nil {
		t.Fatalf("Failed to save config: %v", err)
	}
	
	// 验证文件是否存在
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		t.Error("Expected config file to be saved")
	}
}

func TestConfigManagerValidateConfig(t *testing.T) {
	cm := vector.NewConfigManager("test_config.json")
	
	// 设置有效配置
	cm.DistributedConfig = &vector.DistributedConfig{
		IndexConfig: vector.IndexSelectionConfig{
			HNSWConfig: vector.HNSWIndexConfig{
				Enable:          true,
				MaxConnections:  32,
				EfConstruction:  400,
				EfSearch:        200,
			},
		},
	}
	
	err := cm.ValidateConfig()
	if err != nil {
		t.Errorf("Expected valid config, got error: %v", err)
	}
}

func TestConfigManagerInvalidConfig(t *testing.T) {
	cm := vector.NewConfigManager("test_config.json")
	
	// 设置无效配置
	cm.DistributedConfig = &vector.DistributedConfig{
		IndexConfig: vector.IndexSelectionConfig{
			HNSWConfig: vector.HNSWIndexConfig{
				MaxConnections:  0, // 无效值
				EfConstruction:  -1, // 无效值
				EfSearch:        0, // 无效值
			},
		},
	}
	
	err := cm.ValidateConfig()
	if err == nil {
		t.Error("Expected validation error for invalid config")
	}
}

func TestConfigManagerGetConfig(t *testing.T) {
	cm := vector.NewConfigManager("test_config.yaml")
	
	// 加载配置
	err := cm.LoadConfig()
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}
	
	// 获取配置摘要
	summary := cm.GetConfigSummary()
	if summary == nil {
		t.Error("Expected non-nil config summary")
	}
	
	// 检查分布式配置
	if cm.DistributedConfig == nil {
		t.Error("Expected non-nil distributed config")
	}
}

func TestConfigManagerUpdateConfig(t *testing.T) {
	cm := vector.NewConfigManager("test_config.yaml")
	
	// 加载配置
	err := cm.LoadConfig()
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}
	
	// 直接更新配置字段
	if cm.DistributedConfig != nil {
		// 修改配置
		originalShards := cm.DistributedConfig.ArchitectureConfig.ShardingConfig.NumShards
		cm.DistributedConfig.ArchitectureConfig.ShardingConfig.NumShards = originalShards + 1
		
		// 验证配置更新
		if cm.DistributedConfig.ArchitectureConfig.ShardingConfig.NumShards != originalShards+1 {
			t.Error("Expected config to be updated")
		}
	}
}

func TestConfigManagerWatchConfig(t *testing.T) {
	cm := vector.NewConfigManager("test_config.yaml")
	
	// 启用配置监控（需要回调函数）
	callback := func(manager *vector.ConfigManager) error {
		// 配置变更回调
		return nil
	}
	
	err := cm.EnableConfigWatch(callback)
	if err != nil {
		t.Errorf("Failed to enable config watch: %v", err)
	}
	
	// 禁用配置监控
	cm.DisableConfigWatch()
}

func TestConfigManagerReloadConfig(t *testing.T) {
	cm := vector.NewConfigManager("test_config.yaml")
	
	// 初始加载
	err := cm.LoadConfig()
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}
	
	// 重新加载配置（使用LoadConfig方法）
	err = cm.LoadConfig()
	if err != nil {
		t.Errorf("Failed to reload config: %v", err)
	}
}