package scheduler

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/conf"
	"VectorSphere/src/library/log"
	"VectorSphere/src/vector"
	"fmt"
	"sync"
	"time"
)

type PQTrainingConfig struct {
	TaskName                      string  `json:"task_name" yaml:"task_name"`
	Enable                        bool    `json:"enable" yaml:"enable"`
	CronSpec                      string  `json:"cron_spec" yaml:"cron_spec"`
	NumSubVectors                 int     `json:"num_sub_vector" yaml:"num_sub_vector"`
	NumbCentroidsPerSubVector     int     `json:"num_centroids_per_sub_vector" yaml:"num_centroids_per_sub_vector"`
	MaxKMeansIterations           int     `json:"max_k_means_iterations" yaml:"max_k_means_iterations"`
	KMeansTolerance               float64 `json:"k_means_tolerance" yaml:"k_means_tolerance"`
	SampleRateForSubspaceTraining float64 `json:"sample_rate_for_subspace_training" yaml:"sample_rate_for_subspace_training"`
	MaxVectorsForTraining         int     `json:"max_vectors_for_training" yaml:"max_vectors_for_training"`
	CodebookFilePath              string  `json:"codebook_file_path" yaml:"codebook_file_path"`

	VersionControl bool `json:"version_control" yaml:"version_control"`
	MaxVersions    int  `json:"max_versions" yaml:"max_versions"`
}

// PQTrainingScheduler 结构体用于管理PQ训练任务
type PQTrainingScheduler struct {
	vectorDB         *vector.VectorDB // 或者是一个能获取 VectorDB 实例的接口/方法
	lastTrainingTime time.Time
	trainingStatus   string
	trainingErrors   []string
	mutex            sync.Mutex
	config           *PQTrainingConfig
	taskName         string
	taskID           string
	taskTarget       string
}

// LoadPQTrainingConfig 从YAML文件加载PQ训练配置
func LoadPQTrainingConfig(configPath string) (*PQTrainingConfig, error) {
	var config PQTrainingConfig

	// 设置默认值
	config = PQTrainingConfig{
		TaskName:                      "PQTrainingTask",
		Enable:                        true,
		CronSpec:                      "0 0 2 * * *", // 每天凌晨2点执行
		NumSubVectors:                 8,
		NumbCentroidsPerSubVector:     256,
		MaxKMeansIterations:           100,
		KMeansTolerance:               0.001,
		SampleRateForSubspaceTraining: 0.1,
		MaxVectorsForTraining:         10000,
		CodebookFilePath:              "./data/codebook.bin",
		VersionControl:                true,
		MaxVersions:                   5,
	}

	// 如果配置路径为空，使用默认配置
	if configPath == "" {
		log.Warning("未指定PQ训练配置文件路径，使用默认配置")
		return &config, nil
	}

	// 从YAML文件读取配置
	err := conf.ReadYAML(configPath, &config)
	if err != nil {
		log.Error("读取PQ训练配置文件失败: %v，将使用默认配置", err)
		return &config, err
	}

	log.Info("成功从 %s 加载PQ训练配置", configPath)
	return &config, nil
}

// NewPQTrainingScheduler 创建一个新的 PQTrainingScheduler
func NewPQTrainingScheduler(vectorDB *vector.VectorDB) (*PQTrainingScheduler, error) {
	if vectorDB == nil {
		return nil, fmt.Errorf("VectorDB 实例不能为空")
	}

	// 加载配置
	config, err := LoadPQTrainingConfig("")
	if err != nil {
		log.Warning("加载PQ训练配置文件失败: %v，将使用默认配置", err)
	}
	s := &PQTrainingScheduler{
		vectorDB: vectorDB,
		config:   config,
		taskName: config.TaskName,
	}
	return s, nil
}

// GetTrainingStatus 获取训练状态
func (s *PQTrainingScheduler) GetTrainingStatus() map[string]interface{} {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	return map[string]interface{}{
		"lastTrainingTime": s.lastTrainingTime,
		"status":           s.trainingStatus,
		"errors":           s.trainingErrors,
	}
}

// Run 实现 ScheduledTask 接口的 Run 方法，执行 PQ 训练任务
func (s *PQTrainingScheduler) Run() error {
	log.Info("开始执行PQ码本定时训练任务...")

	s.mutex.Lock()
	s.trainingStatus = "running"
	s.mutex.Unlock()

	// 确保 VectorDB 实例是有效的 TrainingDataSource
	var dataSource acceler.TrainingDataSource = s.vectorDB

	trainingErr := acceler.TrainPQCodebook(
		dataSource,
		s.config.NumSubVectors,
		s.config.NumbCentroidsPerSubVector,
		s.config.MaxKMeansIterations,
		s.config.KMeansTolerance,
		s.config.SampleRateForSubspaceTraining,
		s.config.MaxVectorsForTraining,
		s.config.CodebookFilePath,
	)

	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.lastTrainingTime = time.Now()

	if trainingErr != nil {
		log.Error("PQ码本训练失败: %v", trainingErr)
		s.trainingStatus = "failed"
		s.trainingErrors = append(s.trainingErrors, trainingErr.Error())
		// 保持错误记录在合理范围内
		if len(s.trainingErrors) > 10 {
			s.trainingErrors = s.trainingErrors[len(s.trainingErrors)-10:]
		}
		return trainingErr // 训练失败，返回错误
	}

	log.Info("PQ码本训练成功完成。")
	s.trainingStatus = "completed"

	// 码本热加载/更新机制
	if s.vectorDB.IsPQCompressionEnabled() {
		log.Info("尝试热加载新的PQ码本: %s", s.config.CodebookFilePath)
		if loadErr := s.vectorDB.LoadPQCodebookFromFile(s.config.CodebookFilePath); loadErr != nil {
			log.Error("热加载新码本失败: %v", loadErr)
			s.trainingErrors = append(s.trainingErrors, fmt.Sprintf("热加载失败: %v", loadErr))
			return loadErr
		} else {
			log.Info("新码本热加载成功。")
		}
	} else {
		log.Info("PQ压缩未在VectorDB中启用，跳过热加载码本。")
	}

	return nil
}

// GetCronSpec 实现 ScheduledTask 接口的 GetCronSpec 方法
func (s *PQTrainingScheduler) GetCronSpec() string {
	return s.config.CronSpec
}

// GetName 实现 ScheduledTask 接口的 GetName 方法
func (s *PQTrainingScheduler) GetName() string {
	return s.taskName
}

// Init 实现 ScheduledTask 接口的 Init 方法
func (s *PQTrainingScheduler) Init() error {
	log.Info("初始化 PQ 训练任务: %s", s.taskName)

	// 验证配置
	if !s.config.Enable {
		log.Info("PQ 定时训练未启用，任务将不会执行。")
		return nil
	}

	if s.config.CronSpec == "" {
		return fmt.Errorf("PQ训练的cron表达式未配置")
	}

	if s.config.CodebookFilePath == "" {
		return fmt.Errorf("PQ训练的码本保存路径未配置")
	}

	log.Info("PQ训练任务初始化成功，Cron表达式: '%s'", s.config.CronSpec)
	return nil
}

// Stop 实现 ScheduledTask 接口的 Stop 方法
func (s *PQTrainingScheduler) Stop() error {
	log.Info("停止 PQ 训练任务: %s", s.taskName)
	return nil
}

// Name 实现 ScheduledTask 接口的 Name 方法
func (s *PQTrainingScheduler) Name() string {
	return s.taskName
}

// Params 实现 ScheduledTask 接口的 Params 方法
func (s *PQTrainingScheduler) Params() map[string]interface{} {
	return map[string]interface{}{
		"enable":                            s.config.Enable,
		"cron_spec":                         s.config.CronSpec,
		"num_subvectors":                    s.config.NumSubVectors,
		"num_centroids_per_subvector":       s.config.NumbCentroidsPerSubVector,
		"max_k_means_iterations":            s.config.MaxKMeansIterations,
		"k_means_tolerance":                 s.config.KMeansTolerance,
		"sample_rate_for_subspace_training": s.config.SampleRateForSubspaceTraining,
		"max_vectors_for_training":          s.config.MaxVectorsForTraining,
		"codebook_file_path":                s.config.CodebookFilePath,
		"version_control":                   s.config.VersionControl,
		"max_versions":                      s.config.MaxVersions,
	}
}

// Timeout 实现 ScheduledTask 接口的 Timeout 方法
func (s *PQTrainingScheduler) Timeout() time.Duration {
	// 设置任务超时时间，根据实际情况调整
	return 30 * time.Minute
}

// Clone 实现 ScheduledTask 接口的 Clone 方法
func (s *PQTrainingScheduler) Clone() ScheduledTask {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	cloned := &PQTrainingScheduler{
		vectorDB:       s.vectorDB,
		config:         s.config,
		taskName:       s.taskName,
		taskID:         s.taskID,
		taskTarget:     s.taskTarget,
		trainingStatus: s.trainingStatus,
	}

	// 复制错误列表
	if len(s.trainingErrors) > 0 {
		cloned.trainingErrors = make([]string, len(s.trainingErrors))
		copy(cloned.trainingErrors, s.trainingErrors)
	}

	return cloned
}

// SetID 实现 ScheduledTask 接口的 SetID 方法
func (s *PQTrainingScheduler) SetID(id string) {
	s.taskID = id
}

// SetTarget 实现 ScheduledTask 接口的 SetTarget 方法
func (s *PQTrainingScheduler) SetTarget(target string) {
	s.taskTarget = target
}
