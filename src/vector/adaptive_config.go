package vector

import (
	"VectorSphere/src/library/entity"
	"runtime"
	"time"
)

// AdaptiveConfig 自适应配置结构
type AdaptiveConfig struct {
	// 索引参数
	NumClusters           int     // 簇数量
	IndexRebuildThreshold float64 // 更新比例阈值，超过此值重建索引

	// 查询参数
	DefaultNprobe int           // 默认探测簇数量
	CacheTimeout  time.Duration // 缓存超时时间

	// 系统参数
	MaxWorkers         int  // 最大工作协程数
	VectorCompression  bool // 是否启用向量压缩
	UseMultiLevelIndex bool // 是否使用多级索引

	// 自适应 nprobe 参数
	MinNprobe    int     // 最小探测簇数
	MaxNprobe    int     // 最大探测簇数
	RecallTarget float64 // 目标召回率

	// HNSW 自适应参数
	MinEfConstruction float64 // 最小构建参数
	MaxEfConstruction float64 // 最大构建参数
	QualityThreshold  float64 // 质量阈值
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
	} else if vectorCount > 10000 {
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
		nprobe = max(1, db.numClusters/4)
	case dataSize < 100000:
		nprobe = max(2, db.numClusters/3)
	case dataSize < 1000000:
		nprobe = max(3, db.numClusters/2)
	default:
		nprobe = max(5, db.numClusters*2/3)
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
		db.maxConnections = min(64, max(16, db.vectorDim/10))
	}
}
