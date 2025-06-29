package vector

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"encoding/json"
	"fmt"
	"gopkg.in/yaml.v2"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// AdaptiveConfig 自适应配置结构
type AdaptiveConfig struct {
	// 配置文件相关
	ConfigFilePath       string        `json:"config_file_path" yaml:"config_file_path"`             // 配置文件路径
	EnableConfigReload   bool          `json:"enable_config_reload" yaml:"enable_config_reload"`     // 是否启用配置文件热重载
	EnableHotReload      bool          `json:"enable_hot_reload" yaml:"enable_hot_reload"`           // 是否启用热重载
	ConfigReloadInterval time.Duration `json:"config_reload_interval" yaml:"config_reload_interval"` // 配置重载间隔

	// 自适应开关
	EnableAdaptive         bool `json:"enable_adaptive" yaml:"enable_adaptive"`                   // 是否启用自适应功能
	EnableHardwareAdaptive bool `json:"enable_hardware_adaptive" yaml:"enable_hardware_adaptive"` // 是否启用硬件自适应
	EnableIndexAdaptive    bool `json:"enable_index_adaptive" yaml:"enable_index_adaptive"`       // 是否启用索引自适应
	EnableStrategyAdaptive bool `json:"enable_strategy_adaptive" yaml:"enable_strategy_adaptive"` // 是否启用策略自适应

	// 索引参数
	NumClusters           int     `json:"num_clusters" yaml:"num_clusters"`                       // 簇数量
	IndexRebuildThreshold float64 `json:"index_rebuild_threshold" yaml:"index_rebuild_threshold"` // 更新比例阈值，超过此值重建索引

	// 查询参数
	DefaultNprobe int           `json:"default_nprobe" yaml:"default_nprobe"` // 默认探测簇数量
	CacheTimeout  time.Duration `json:"cache_timeout" yaml:"cache_timeout"`   // 缓存超时时间

	// 系统参数
	MaxWorkers         int  `json:"max_workers" yaml:"max_workers"`                     // 最大工作协程数
	VectorCompression  bool `json:"vector_compression" yaml:"vector_compression"`       // 是否启用向量压缩
	UseMultiLevelIndex bool `json:"use_multi_level_index" yaml:"use_multi_level_index"` // 是否使用多级索引

	// 自适应 nprobe 参数
	MinNprobe    int     `json:"min_nprobe" yaml:"min_nprobe"`       // 最小探测簇数
	MaxNprobe    int     `json:"max_nprobe" yaml:"max_nprobe"`       // 最大探测簇数
	RecallTarget float64 `json:"recall_target" yaml:"recall_target"` // 目标召回率

	// HNSW 自适应参数
	MinEfConstruction float64 `json:"min_ef_construction" yaml:"min_ef_construction"` // 最小构建参数
	MaxEfConstruction float64 `json:"max_ef_construction" yaml:"max_ef_construction"` // 最大构建参数
	QualityThreshold  float64 `json:"quality_threshold" yaml:"quality_threshold"`     // 质量阈值

	// 硬件自适应参数
	HardwareDetectionInterval time.Duration `json:"hardware_detection_interval" yaml:"hardware_detection_interval"` // 硬件检测间隔
	GPUMemoryThreshold        float64       `json:"gpu_memory_threshold" yaml:"gpu_memory_threshold"`               // GPU内存使用阈值
	CPUUsageThreshold         float64       `json:"cpu_usage_threshold" yaml:"cpu_usage_threshold"`                 // CPU使用率阈值
	AutoFallbackEnabled       bool          `json:"auto_fallback_enabled" yaml:"auto_fallback_enabled"`             // 是否启用自动回退

	// 性能监控参数
	WindowSize           int           `json:"window_size" yaml:"window_size"`                     // 性能监控窗口大小
	OptimizationInterval time.Duration `json:"optimization_interval" yaml:"optimization_interval"` // 优化间隔
}

// AdjustConfig 自适应配置调整
func (db *VectorDB) AdjustConfig() {
	db.mu.RLock()
	vectorCount := len(db.vectors)
	db.mu.RUnlock()

	config := db.config

	// 根据向量数量调整簇数量
	if vectorCount > 1000000 {
		config.NumClusters = 1000
	} else if vectorCount > 100000 {
		config.NumClusters = 100
	} else if vectorCount >= 10000 {
		config.NumClusters = 50
	} else {
		config.NumClusters = 10
	}

	// 根据系统资源调整工作协程数
	config.MaxWorkers = runtime.NumCPU()

	// 根据内存使用情况决定是否启用向量压缩
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	if m.Alloc > 1024*1024*1024 { // 如果内存使用超过1GB
		config.VectorCompression = true
	}

	db.mu.Lock()
	db.config = config
	db.mu.Unlock()
}

// AdaptiveNprobeSearch 自适应 nprobe 搜索
func (db *VectorDB) AdaptiveNprobeSearch(query []float64, k int) ([]entity.Result, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	dataSize := len(db.vectors)

	// 根据数据规模自适应调整 nprobe
	var nprobe int
	switch {
	case dataSize < 10000:
		nprobe = int(math.Max(1, float64(db.numClusters)/4))
	case dataSize < 100000:
		nprobe = int(math.Max(2, float64(db.numClusters)/3))
	case dataSize < 1000000:
		nprobe = int(math.Max(3, float64(db.numClusters)/2))
	default:
		nprobe = int(math.Max(5, float64(db.numClusters)*2/3))
	}

	// 确保 nprobe 在合理范围内
	if nprobe > db.numClusters {
		nprobe = db.numClusters
	}

	return db.ivfSearch(query, k, nprobe)
}

// AdaptiveHNSWConfig HNSW 自适应配置
func (db *VectorDB) AdaptiveHNSWConfig() {
	db.mu.Lock()
	defer db.mu.Unlock()

	dataSize := len(db.vectors)

	// 根据数据规模调整 efConstruction
	switch {
	case dataSize < 10000:
		db.efConstruction = 100.0
	case dataSize < 100000:
		db.efConstruction = 200.0
	case dataSize < 1000000:
		db.efConstruction = 400.0
	default:
		db.efConstruction = 800.0
	}

	// 根据向量维度调整连接数
	if db.vectorDim > 0 {
		db.maxConnections = int(math.Min(64, math.Max(16, float64(db.vectorDim)/10)))
	}
}

// LoadConfigFromFile 从文件加载配置
func (db *VectorDB) LoadConfigFromFile(configPath string) error {
	if configPath == "" {
		return fmt.Errorf("配置文件路径为空")
	}

	// 检查文件是否存在
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return fmt.Errorf("配置文件不存在: %s", configPath)
	}

	// 读取文件内容
	data, err := ioutil.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("读取配置文件失败: %v", err)
	}

	// 根据文件扩展名选择解析方式
	ext := filepath.Ext(configPath)
	var config AdaptiveConfig

	switch ext {
	case ".json":
		if err := json.Unmarshal(data, &config); err != nil {
			return fmt.Errorf("解析JSON配置文件失败: %v", err)
		}
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, &config); err != nil {
			return fmt.Errorf("解析YAML配置文件失败: %v", err)
		}
	default:
		return fmt.Errorf("不支持的配置文件格式: %s", ext)
	}

	// 应用配置
	db.mu.Lock()
	db.config = config
	db.config.ConfigFilePath = configPath
	db.mu.Unlock()

	logger.Info("成功加载配置文件: %s", configPath)
	return nil
}

// SaveConfigToFile 保存配置到文件
func (db *VectorDB) SaveConfigToFile(configPath string) error {
	if configPath == "" {
		return fmt.Errorf("配置文件路径为空")
	}

	db.mu.RLock()
	config := db.config
	db.mu.RUnlock()

	// 根据文件扩展名选择保存方式
	ext := filepath.Ext(configPath)
	var data []byte
	var err error

	switch ext {
	case ".json":
		data, err = json.MarshalIndent(config, "", "  ")
	case ".yaml", ".yml":
		data, err = yaml.Marshal(config)
	default:
		return fmt.Errorf("不支持的配置文件格式: %s", ext)
	}

	if err != nil {
		return fmt.Errorf("序列化配置失败: %v", err)
	}

	// 确保目录存在
	dir := filepath.Dir(configPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("创建配置目录失败: %v", err)
	}

	// 写入文件
	if err := ioutil.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("写入配置文件失败: %v", err)
	}

	logger.Info("成功保存配置文件: %s", configPath)
	return nil
}

// StartConfigReloader 启动配置文件热重载
func (db *VectorDB) StartConfigReloader() {
	db.mu.RLock()
	configPath := db.config.ConfigFilePath
	enableReload := db.config.EnableConfigReload
	reloadInterval := db.config.ConfigReloadInterval
	db.mu.RUnlock()

	if !enableReload || configPath == "" {
		return
	}

	if reloadInterval <= 0 {
		reloadInterval = 30 * time.Second // 默认30秒检查一次
	}

	logger.Info("启动配置文件热重载，路径: %s，间隔: %v", configPath, reloadInterval)

	go func() {
		ticker := time.NewTicker(reloadInterval)
		defer ticker.Stop()

		var lastModTime time.Time
		if stat, err := os.Stat(configPath); err == nil {
			lastModTime = stat.ModTime()
		}

		for {
			select {
			case <-ticker.C:
				if stat, err := os.Stat(configPath); err == nil {
					if stat.ModTime().After(lastModTime) {
						logger.Info("检测到配置文件变更，重新加载: %s", configPath)
						if err := db.LoadConfigFromFile(configPath); err != nil {
							logger.Error("重新加载配置文件失败: %v", err)
						} else {
							logger.Info("配置文件重新加载成功")
							// 应用新配置
							db.ApplyAdaptiveConfig()
						}
						lastModTime = stat.ModTime()
					}
				}
			case <-db.stopCh:
				logger.Info("停止配置文件热重载")
				return
			}
		}
	}()
}

// ApplyAdaptiveConfig 应用自适应配置
func (db *VectorDB) ApplyAdaptiveConfig() {
	db.mu.RLock()
	config := db.config
	db.mu.RUnlock()

	if !config.EnableAdaptive {
		logger.Info("自适应功能已禁用")
		return
	}

	logger.Info("应用自适应配置...")

	// 应用硬件自适应配置
	if config.EnableHardwareAdaptive && db.hardwareManager != nil {
		db.applyHardwareAdaptiveConfig(config)
	}

	// 应用索引自适应配置
	if config.EnableIndexAdaptive {
		db.applyIndexAdaptiveConfig(config)
	}

	// 应用策略自适应配置
	if config.EnableStrategyAdaptive {
		db.applyStrategyAdaptiveConfig(config)
	}

	logger.Info("自适应配置应用完成")
}

// applyHardwareAdaptiveConfig 应用硬件自适应配置
func (db *VectorDB) applyHardwareAdaptiveConfig(config AdaptiveConfig) {
	if db.hardwareManager == nil {
		return
	}

	// 更新硬件检测间隔
	if config.HardwareDetectionInterval > 0 {
		db.hardwareManager.SetDetectionInterval(config.HardwareDetectionInterval)
	}

	// 设置GPU内存阈值
	if config.GPUMemoryThreshold > 0 {
		db.hardwareManager.SetGPUMemoryThreshold(config.GPUMemoryThreshold)
	}

	// 设置CPU使用率阈值
	if config.CPUUsageThreshold > 0 {
		db.hardwareManager.SetCPUUsageThreshold(config.CPUUsageThreshold)
	}

	// 设置自动回退
	db.hardwareManager.SetAutoFallback(config.AutoFallbackEnabled)

	logger.Info("硬件自适应配置已应用")
}

// applyIndexAdaptiveConfig 应用索引自适应配置
func (db *VectorDB) applyIndexAdaptiveConfig(config AdaptiveConfig) {
	// 更新索引重建阈值
	if config.IndexRebuildThreshold > 0 {
		db.mu.Lock()
		db.config.IndexRebuildThreshold = config.IndexRebuildThreshold
		db.mu.Unlock()
	}

	// 应用自适应配置调整
	db.AdjustConfig()

	logger.Info("索引自适应配置已应用")
}

// applyStrategyAdaptiveConfig 应用策略自适应配置
func (db *VectorDB) applyStrategyAdaptiveConfig(config AdaptiveConfig) {
	if db.adaptiveSelector != nil {
		// 更新自适应选择器的配置
		db.adaptiveSelector.UpdateConfig(&config)
	}

	logger.Info("策略自适应配置已应用")
}

// GetDefaultAdaptiveConfig 获取默认自适应配置
func GetDefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		// 配置文件相关
		ConfigFilePath:       "",
		EnableConfigReload:   false,
		ConfigReloadInterval: 30 * time.Second,

		// 自适应开关
		EnableAdaptive:         true,
		EnableHardwareAdaptive: true,
		EnableIndexAdaptive:    true,
		EnableStrategyAdaptive: true,

		// 索引参数
		NumClusters:           100,
		IndexRebuildThreshold: 0.3,

		// 查询参数
		DefaultNprobe: 10,
		CacheTimeout:  5 * time.Minute,

		// 系统参数
		MaxWorkers:         runtime.NumCPU(),
		VectorCompression:  false,
		UseMultiLevelIndex: false,

		// 自适应 nprobe 参数
		MinNprobe:    1,
		MaxNprobe:    50,
		RecallTarget: 0.9,

		// HNSW 自适应参数
		MinEfConstruction: 100.0,
		MaxEfConstruction: 800.0,
		QualityThreshold:  0.8,

		// 硬件自适应参数
		HardwareDetectionInterval: 60 * time.Second,
		GPUMemoryThreshold:        0.8,
		CPUUsageThreshold:         0.8,
		AutoFallbackEnabled:       true,
	}
}
