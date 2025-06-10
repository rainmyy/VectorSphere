package vector

// TwoStageSearchConfig 两阶段搜索配置
type TwoStageSearchConfig struct {
	CoarseK      int     // 粗筛阶段返回的候选数量
	CoarseNprobe int     // 粗筛阶段的 nprobe
	FineK        int     // 精排阶段最终返回数量
	UseGPU       bool    // 是否在精排阶段使用 GPU
	Threshold    float64 // 相似度阈值
}
