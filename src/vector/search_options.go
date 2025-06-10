package vector

import "time"

// SearchOptions 搜索选项结构
type SearchOptions struct {
	NumHashTables int  // LSH哈希表数量
	UseANN        bool // 是否使用近似最近邻

	Nprobe        int           `json:"nprobe"`         // IVF搜索的探测簇数量
	SearchTimeout time.Duration `json:"search_timeout"` // 搜索超时时间
	QualityLevel  float64       `json:"quality_level"`  // 质量要求等级 0.0-1.0
	UseCache      bool          `json:"use_cache"`      // 是否使用缓存
	MaxCandidates int           `json:"max_candidates"` // 最大候选数量
}
