package scheduler

import (
	"fmt"
	"github.com/robfig/cron/v3"
	"seetaSearch/db"
	"seetaSearch/library/log"
	"seetaSearch/library/util"
)

type PQTrainingConfig struct {
	Enable                        bool    `json:"enable" yaml:"enable"`
	CronSpec                      string  `json:"cron_spec" yaml:"cron_spec"`
	NumSubvectors                 int     `json:"num_subvectors" yaml:"num_subvectors"`
	NumCentroidsPerSubvector      int     `json:"num_centroids_per_subvector" yaml:"num_centroids_per_subvector"`
	MaxKMeansIterations           int     `json:"max_k_means_iterations" yaml:"max_k_means_iterations"`
	KMeansTolerance               float64 `json:"k_means_tolerance" yaml:"k_means_tolerance"`
	SampleRateForSubspaceTraining float64 `json:"sample_rate_for_subspace_training" yaml:"sample_rate_for_subspace_training"`
	MaxVectorsForTraining         int     `json:"max_vectors_for_training" yaml:"max_vectors_for_training"`
	CodebookFilePath              string  `json:"codebook_file_path" yaml:"codebook_file_path"`
}

// PQTrainingScheduler 结构体用于管理PQ训练任务
type PQTrainingScheduler struct {
	cronScheduler *cron.Cron
	vectorDB      *db.VectorDB // 或者是一个能获取 VectorDB 实例的接口/方法
	// 可以添加其他依赖，如配置对象
}

// NewPQTrainingScheduler 创建一个新的 PQTrainingScheduler
// vectorDBProvider 是一个函数，用于获取 VectorDB 实例。您需要根据您的架构调整此部分。
func NewPQTrainingScheduler(vectorDB *db.VectorDB) (*PQTrainingScheduler, error) {
	if vectorDB == nil {
		return nil, fmt.Errorf("VectorDB 实例不能为空")
	}
	s := &PQTrainingScheduler{
		cronScheduler: cron.New(cron.WithSeconds()), // 支持秒级精度，如果不需要可以去掉 WithSeconds()
		vectorDB:      vectorDB,
	}
	return s, nil
}

// Start 启动定时训练任务
func (s *PQTrainingScheduler) Start() error {
	// 从配置加载PQ训练参数 (示例，请根据您的配置系统调整)
	pqConfig := PQTrainingConfig{}
	if !pqConfig.Enable {
		log.Info("PQ 定时训练未启用，跳过启动调度器。")
		return nil
	}

	if pqConfig.CronSpec == "" {
		return fmt.Errorf("PQ训练的cron表达式 (pq.train.cron_spec) 未配置")
	}
	if pqConfig.CodebookFilePath == "" {
		return fmt.Errorf("PQ训练的码本保存路径 (pq.train.codebook_file_path) 未配置")
	}

	log.Info("准备启动PQ码本定时训练任务，Cron表达式: '%s'", pqConfig.CronSpec)

	entryID, err := s.cronScheduler.AddFunc(pqConfig.CronSpec, func() { // 保存 entryID 以便将来可能移除或管理任务
		log.Info("开始执行PQ码本定时训练任务...")

		// 确保 VectorDB 实例是有效的 TrainingDataSource
		var dataSource util.TrainingDataSource = s.vectorDB

		trainingErr := util.TrainPQCodebook(
			dataSource,
			pqConfig.NumSubvectors,
			pqConfig.NumCentroidsPerSubvector,
			pqConfig.MaxKMeansIterations,
			pqConfig.KMeansTolerance,
			pqConfig.SampleRateForSubspaceTraining,
			pqConfig.MaxVectorsForTraining,
			pqConfig.CodebookFilePath,
		)

		if trainingErr != nil {
			log.Error("PQ码本训练失败: %v", trainingErr)
			return // 训练失败，不尝试热加载
		}
		log.Info("PQ码本训练成功完成。")

		// 码本热加载/更新机制
		// 假设 VectorDB 实例中已有 IsPQCompressionEnabled 和 LoadPQCodebookFromFile 方法
		if s.vectorDB.IsPQCompressionEnabled() {
			log.Info("尝试热加载新的PQ码本: %s", pqConfig.CodebookFilePath)
			if loadErr := s.vectorDB.LoadPQCodebookFromFile(pqConfig.CodebookFilePath); loadErr != nil {
				log.Error("热加载新码本失败: %v", loadErr)
			} else {
				log.Info("新码本热加载成功。")
			}
		} else {
			log.Info("PQ压缩未在VectorDB中启用，跳过热加载码本。")
		}
	})

	if err != nil {
		return fmt.Errorf("添加PQ训练定时任务失败: %w", err)
	}
	log.Info("PQ训练定时任务已添加，EntryID: %d", entryID) // 记录一下任务ID

	s.cronScheduler.Start()
	log.Info("PQ训练定时任务调度器已启动。")
	return nil
}

// Stop 停止定时训练任务
func (s *PQTrainingScheduler) Stop() {
	if s.cronScheduler != nil {
		log.Info("正在停止PQ训练定时任务调度器...")
		s.cronScheduler.Stop() // Stop会等待已在执行的任务完成
		log.Info("PQ训练定时任务调度器已停止。")
	}
}
