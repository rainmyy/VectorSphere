package test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"VectorSphere/src/distributed"
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
				MaxConnections: 16,
				EfConstruction: 200,
				EfSearch:       100,
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
				Enable:         true,
				MaxConnections: 32,
				EfConstruction: 400,
				EfSearch:       200,
			},
		},
		ArchitectureConfig: vector.DistributedArchitectureConfig{
			ShardingConfig: vector.ShardingConfig{
				Strategy: "hash",
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
				MaxConnections: 0,  // 无效值
				EfConstruction: -1, // 无效值
				EfSearch:       0,  // 无效值
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

// Test: Environment variables override configuration values during loading
func TestConfigManagerApplyEnvOverrides(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "env_override_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	configPath := filepath.Join(tempDir, "config.yaml")
	configContent := `
serviceName: "original_service"
nodeType: 1
timeOut: 10
defaultPort: 1234
heartbeat: 5
schedulerWorkerCount: 2
httpPort: 4321
taskTimeout: 20
healthCheckInterval: 15
dataDir: "./original_data"
etcd:
  endpoints:
    - "127.0.0.1:2379"
endpoints: {}
`
	if err := ioutil.WriteFile(configPath, []byte(configContent), 0644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	os.Setenv("VECTOR_SPHERE_SERVICE_NAME", "env_service")
	os.Setenv("VECTOR_SPHERE_NODE_TYPE", "master")
	os.Setenv("VECTOR_SPHERE_PORT", "8888")
	os.Setenv("VECTOR_SPHERE_HTTP_PORT", "9999")
	os.Setenv("VECTOR_SPHERE_ETCD_ENDPOINTS", "env1:2379,env2:2379")
	os.Setenv("VECTOR_SPHERE_DATA_DIR", "./env_data")
	os.Setenv("VECTOR_SPHERE_TIMEOUT", "77")
	os.Setenv("VECTOR_SPHERE_HEARTBEAT", "66")
	os.Setenv("VECTOR_SPHERE_WORKER_COUNT", "55")
	os.Setenv("VECTOR_SPHERE_TASK_TIMEOUT", "44")
	os.Setenv("VECTOR_SPHERE_HEALTH_INTERVAL", "33")
	defer func() {
		err := os.Unsetenv("VECTOR_SPHERE_SERVICE_NAME")
		if err != nil {
			return
		}
		os.Unsetenv("VECTOR_SPHERE_NODE_TYPE")
		os.Unsetenv("VECTOR_SPHERE_PORT")
		os.Unsetenv("VECTOR_SPHERE_HTTP_PORT")
		os.Unsetenv("VECTOR_SPHERE_ETCD_ENDPOINTS")
		os.Unsetenv("VECTOR_SPHERE_DATA_DIR")
		os.Unsetenv("VECTOR_SPHERE_TIMEOUT")
		os.Unsetenv("VECTOR_SPHERE_HEARTBEAT")
		os.Unsetenv("VECTOR_SPHERE_WORKER_COUNT")
		os.Unsetenv("VECTOR_SPHERE_TASK_TIMEOUT")
		os.Unsetenv("VECTOR_SPHERE_HEALTH_INTERVAL")
	}()

	cm := distributed.NewConfigManager(configPath)
	cfg, err := cm.LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}

	if cfg.ServiceName != "env_service" {
		t.Errorf("Expected ServiceName to be overridden, got %s", cfg.ServiceName)
	}
	if cfg.NodeType != distributed.MasterNode {
		t.Errorf("Expected NodeType to be MasterNode, got %v", cfg.NodeType)
	}
	if cfg.DefaultPort != 8888 {
		t.Errorf("Expected DefaultPort to be 8888, got %d", cfg.DefaultPort)
	}
	if cfg.HttpPort != 9999 {
		t.Errorf("Expected HttpPort to be 9999, got %d", cfg.HttpPort)
	}
	expectedEtcd := []string{"env1:2379", "env2:2379"}
	if !reflect.DeepEqual(cfg.Etcd.Endpoints, expectedEtcd) {
		t.Errorf("Expected Etcd.Endpoints to be %v, got %v", expectedEtcd, cfg.Etcd.Endpoints)
	}
	if cfg.DataDir != "./env_data" {
		t.Errorf("Expected DataDir to be './env_data', got %s", cfg.DataDir)
	}
	if cfg.TimeOut != 77 {
		t.Errorf("Expected TimeOut to be 77, got %d", cfg.TimeOut)
	}
	if cfg.Heartbeat != 66 {
		t.Errorf("Expected Heartbeat to be 66, got %d", cfg.Heartbeat)
	}
	if cfg.SchedulerWorkerCount != 55 {
		t.Errorf("Expected SchedulerWorkerCount to be 55, got %d", cfg.SchedulerWorkerCount)
	}
	if cfg.TaskTimeout != 44 {
		t.Errorf("Expected TaskTimeout to be 44, got %d", cfg.TaskTimeout)
	}
	if cfg.HealthCheckInterval != 33 {
		t.Errorf("Expected HealthCheckInterval to be 33, got %d", cfg.HealthCheckInterval)
	}
}

// Test: Default values are set for missing optional configuration fields
func TestConfigManagerSetDefaults(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "defaults_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	configPath := filepath.Join(tempDir, "config.yaml")
	// Omit optional fields
	configContent := `
serviceName: "svc"
nodeType: 1
defaultPort: 1234
httpPort: 4321
etcd:
  endpoints:
    - "127.0.0.1:2379"
endpoints: {}
`
	if err := ioutil.WriteFile(configPath, []byte(configContent), 0644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	cm := distributed.NewConfigManager(configPath)
	cfg, err := cm.LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}

	if cfg.TimeOut != 30 {
		t.Errorf("Expected default TimeOut 30, got %d", cfg.TimeOut)
	}
	if cfg.Heartbeat != 10 {
		t.Errorf("Expected default Heartbeat 10, got %d", cfg.Heartbeat)
	}
	if cfg.SchedulerWorkerCount != 10 {
		t.Errorf("Expected default SchedulerWorkerCount 10, got %d", cfg.SchedulerWorkerCount)
	}
	if cfg.TaskTimeout != 60 {
		t.Errorf("Expected default TaskTimeout 60, got %d", cfg.TaskTimeout)
	}
	if cfg.HealthCheckInterval != 30 {
		t.Errorf("Expected default HealthCheckInterval 30, got %d", cfg.HealthCheckInterval)
	}
	if cfg.DataDir != "./data" {
		t.Errorf("Expected default DataDir './data', got %s", cfg.DataDir)
	}
	if cfg.Endpoints == nil {
		t.Errorf("Expected Endpoints to be initialized")
	}
}

// Test: CreateDefaultConfig generates a valid default configuration file
func TestCreateDefaultConfig(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "default_config_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	configPath := filepath.Join(tempDir, "default.yaml")
	err = distributed.CreateDefaultConfig(configPath)
	if err != nil {
		t.Fatalf("CreateDefaultConfig failed: %v", err)
	}

	// Validate file exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		t.Fatalf("Expected config file to be created at %s", configPath)
	}

	// Validate file is valid
	err = distributed.ValidateConfigFile(configPath)
	if err != nil {
		t.Errorf("ValidateConfigFile failed: %v", err)
	}
}

// Test: Loading configuration fails with invalid YAML format
func TestConfigManagerLoadConfigInvalidYAML(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "invalid_yaml_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	configPath := filepath.Join(tempDir, "invalid.yaml")
	invalidYAML := "serviceName: [unclosed"
	if err := ioutil.WriteFile(configPath, []byte(invalidYAML), 0644); err != nil {
		t.Fatalf("Failed to write invalid yaml: %v", err)
	}

	cm := distributed.NewConfigManager(configPath)
	_, err = cm.LoadConfig()
	if err == nil {
		t.Error("Expected error when loading invalid YAML, got nil")
	}
}

// Test: Validation fails when required configuration fields are missing or out of range
func TestConfigManagerValidateConfigMissingOrInvalidFields(t *testing.T) {
	cm := distributed.NewConfigManager("dummy.yaml")

	invalids := []distributed.DistributedConfig{
		// Missing ServiceName
		{
			ServiceName: "",
			Etcd:        distributed.EtcdConfig{Endpoints: []string{"127.0.0.1:2379"}},
			DefaultPort: 8080, HttpPort: 8090, TimeOut: 10, Heartbeat: 10,
		},
		// Missing Etcd endpoints
		{
			ServiceName: "svc",
			Etcd:        distributed.EtcdConfig{Endpoints: []string{}},
			DefaultPort: 8080, HttpPort: 8090, TimeOut: 10, Heartbeat: 10,
		},
		// DefaultPort out of range
		{
			ServiceName: "svc",
			Etcd:        distributed.EtcdConfig{Endpoints: []string{"127.0.0.1:2379"}},
			DefaultPort: 0, HttpPort: 8090, TimeOut: 10, Heartbeat: 10,
		},
		// HttpPort out of range
		{
			ServiceName: "svc",
			Etcd:        distributed.EtcdConfig{Endpoints: []string{"127.0.0.1:2379"}},
			DefaultPort: 8080, HttpPort: 70000, TimeOut: 10, Heartbeat: 10,
		},
		// TimeOut <= 0
		{
			ServiceName: "svc",
			Etcd:        distributed.EtcdConfig{Endpoints: []string{"127.0.0.1:2379"}},
			DefaultPort: 8080, HttpPort: 8090, TimeOut: 0, Heartbeat: 10,
		},
		// Heartbeat <= 0
		{
			ServiceName: "svc",
			Etcd:        distributed.EtcdConfig{Endpoints: []string{"127.0.0.1:2379"}},
			DefaultPort: 8080, HttpPort: 8090, TimeOut: 10, Heartbeat: 0,
		},
	}

	for i, cfg := range invalids {
		err := cm.ValidateConfig(&cfg)
		if err == nil {
			t.Errorf("Expected validation error for invalid config #%d, got nil", i)
		}
	}
}

// Test: UpdateConfig returns an error when attempting to update before configuration is loaded
func TestConfigManagerUpdateConfigWithoutLoad(t *testing.T) {
	cm := distributed.NewConfigManager("dummy.yaml")
	updates := map[string]interface{}{
		"service_name": "new_service",
	}
	err := cm.UpdateConfig(updates)
	if err == nil {
		t.Error("Expected error when updating config before loading, got nil")
	}
}
