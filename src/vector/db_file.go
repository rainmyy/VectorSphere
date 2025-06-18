package vector

import (
	"VectorSphere/src/library/acceler"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/graph"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/library/storage"
	"bytes"
	"encoding/gob"
	"fmt"
	"os"
)

// dataToSave 结构用于 gob 编码，包含所有需要持久化的字段
type dataToSave struct {
	Vectors                  map[string][]float64
	Clusters                 []Cluster
	NumClusters              int
	Indexed                  bool
	InvertedIndex            map[string][]string
	VectorDim                int
	NormalizedVectors        map[string][]float64
	CompressedVectors        map[string]entity.CompressedVector
	PQCodebook               []float64
	NumSubVectors            int
	NumCentroidsPerSubVector int
	UsePQCompression         bool
}

// SaveToFileWithMmap 使用 mmap 优化的保存方法
func (db *VectorDB) SaveToFileWithMmap(filePath string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.backupPath == "" {
		return fmt.Errorf("文件路径未设置，无法保存数据库")
	}

	// 创建 mmap 文件
	mmapFile, err := storage.NewMmap(db.backupPath, storage.MODE_CREATE)
	if err != nil {
		logger.Warning("mmap 创建失败，回退到标准方式: %v", err)
		return db.saveToFileStandard(filePath) // 回退到原方法
	}
	defer func(mmapFile *storage.Mmap) {
		err := mmapFile.Unmap()
		if err != nil {
			logger.Error("unmap file has error:%v", err.Error())
		}
	}(mmapFile)

	// 序列化数据到内存缓冲区
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)

	data := struct {
		Vectors                  map[string][]float64
		Clusters                 []Cluster
		NumClusters              int
		Indexed                  bool
		InvertedIndex            map[string][]string
		VectorDim                int
		VectorizedType           int
		NormalizedVectors        map[string][]float64
		CompressedVectors        map[string]entity.CompressedVector
		UseCompression           bool
		PqCodebookFilePath       string
		NumSubVectors            int
		NumCentroidsPerSubVector int
		UsePQCompression         bool
		UseHNSWIndex             bool
		MaxConnections           int
		EfConstruction           float64
		EfSearch                 float64
	}{
		Vectors:                  db.vectors,
		Clusters:                 db.clusters,
		NumClusters:              db.numClusters,
		Indexed:                  db.indexed,
		InvertedIndex:            db.invertedIndex,
		VectorDim:                db.vectorDim,
		VectorizedType:           db.vectorizedType,
		NormalizedVectors:        db.normalizedVectors,
		CompressedVectors:        db.compressedVectors,
		UseCompression:           db.useCompression,
		PqCodebookFilePath:       db.pqCodebookFilePath,
		NumSubVectors:            db.numSubVectors,
		NumCentroidsPerSubVector: db.numCentroidsPerSubVector,
		UsePQCompression:         db.usePQCompression,
		UseHNSWIndex:             db.useHNSWIndex,
		MaxConnections:           db.maxConnections,
		EfConstruction:           db.efConstruction,
		EfSearch:                 db.efSearch,
	}

	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("序列化数据失败: %v", err)
	}

	// 将序列化数据写入 mmap
	serializedData := buf.Bytes()
	if err := mmapFile.WriteBytes(0, serializedData); err != nil {
		return fmt.Errorf("写入 mmap 失败: %v", err)
	}

	// 同步到磁盘
	if err := mmapFile.Sync(); err != nil {
		return fmt.Errorf("同步 mmap 到磁盘失败: %v", err)
	}

	// 保存 HNSW 索引
	if db.useHNSWIndex && db.hnsw != nil {
		hnswFilePath := filePath + ".hnsw"
		err := db.hnsw.SaveToFile(hnswFilePath)
		if err != nil {
			return fmt.Errorf("保存 HNSW 图结构失败: %w", err)
		}
	}

	logger.Info("VectorDB 数据成功通过 mmap 保存到 %s", db.backupPath)
	return nil
}

// SaveToFile 保存数据库到文件（智能选择是否使用 mmap）
func (db *VectorDB) SaveToFile(filePath string) error {
	// 根据数据大小决定是否使用 mmap
	estimatedSize := db.estimateDataSize()

	// 大于 10MB 的文件使用 mmap 优化
	if estimatedSize > 10*1024*1024 {
		if err := db.SaveToFileWithMmap(filePath); err != nil {
			logger.Warning("mmap 保存失败，回退到标准方式: %v", err)
			return db.saveToFileStandard(filePath)
		}
		return nil
	}

	return db.saveToFileStandard(filePath)
}

// SaveToFile 将当前数据库状态（包括索引）保存到其配置的文件中。
func (db *VectorDB) saveToFileStandard(filePath string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.backupPath == "" {
		return fmt.Errorf("文件路径未设置，无法保存数据库")
	}

	file, err := os.Create(db.backupPath)
	if err != nil {
		return fmt.Errorf("创建数据库文件 %s 失败: %v", db.backupPath, err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			logger.Error("close file failed: %v", err)
		}
	}(file)

	encoder := gob.NewEncoder(file)

	// 序列化 VectorDB 的核心数据
	// 为了向前兼容和模块化，可以考虑为每个主要部分创建独立的结构进行序列化
	data := struct {
		Vectors           map[string][]float64
		Clusters          []Cluster
		NumClusters       int
		Indexed           bool
		InvertedIndex     map[string][]string
		VectorDim         int
		VectorizedType    int
		NormalizedVectors map[string][]float64
		CompressedVectors map[string]entity.CompressedVector
		UseCompression    bool
		// PQ 相关字段也需要保存，以便下次加载时能正确恢复状态
		PqCodebookFilePath       string // 保存码本路径，而不是码本本身，码本由外部文件管理
		NumSubvectors            int
		NumCentroidsPerSubvector int
		UsePQCompression         bool
		MultiIndex               *MultiLevelIndex // 如果 MultiLevelIndex 可序列化
		Config                   AdaptiveConfig   // 如果 AdaptiveConfig 可序列化
		UseHNSWIndex             bool
		MaxConnections           int
		EfConstruction           float64
		EfSearch                 float64
	}{
		Vectors:                  db.vectors,
		Clusters:                 db.clusters,
		NumClusters:              db.numClusters,
		Indexed:                  db.indexed,
		InvertedIndex:            db.invertedIndex,
		VectorDim:                db.vectorDim,
		VectorizedType:           db.vectorizedType,
		NormalizedVectors:        db.normalizedVectors,
		CompressedVectors:        db.compressedVectors,
		UseCompression:           db.useCompression,
		PqCodebookFilePath:       db.pqCodebookFilePath, // 保存码本文件路径
		NumSubvectors:            db.numSubVectors,
		NumCentroidsPerSubvector: db.numCentroidsPerSubVector,
		UsePQCompression:         db.usePQCompression,
		MultiIndex:               db.multiIndex,
		Config:                   db.config,
		UseHNSWIndex:             db.useHNSWIndex,
		MaxConnections:           db.maxConnections,
		EfConstruction:           db.efConstruction,
		EfSearch:                 db.efSearch,
	}

	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("序列化数据库到 %s 失败: %v", db.filePath, err)
	}
	// 如果启用了 HNSW 索引，保存 HNSW 图结构
	if db.useHNSWIndex && db.hnsw != nil {
		hnswFilePath := filePath + ".hnsw"
		err := db.hnsw.SaveToFile(hnswFilePath)
		if err != nil {
			return fmt.Errorf("保存 HNSW 图结构失败: %w", err)
		}
	}
	logger.Info("VectorDB 数据成功保存到 %s", db.filePath)

	return nil
}

// LoadFromFileWithMmap 使用 mmap 优化的加载方法
func (db *VectorDB) LoadFromFileWithMmap(filePath string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.filePath == "" {
		return fmt.Errorf("文件路径未设置，无法加载数据库")
	}

	// 检查文件是否存在
	if _, err := os.Stat(db.filePath); os.IsNotExist(err) {
		logger.Info("数据库文件 %s 不存在，将创建一个新的空数据库。", db.filePath)
		db.initializeEmptyDB()
		return nil
	}

	// 创建 mmap 文件映射
	mmapFile, err := storage.NewMmap(db.filePath, storage.MODE_APPEND)
	if err != nil {
		logger.Warning("mmap 打开失败，回退到标准方式: %v", err)
		return db.LoadFromFile(filePath) // 回退到原方法
	}
	defer func(mmapFile *storage.Mmap) {
		err := mmapFile.Unmap()
		if err != nil {
			logger.Error("unmap file has error:%v", err.Error())
		}
	}(mmapFile)

	// 从 mmap 读取所有数据
	fileSize := mmapFile.FileLen
	if fileSize == 0 {
		db.initializeEmptyDB()
		return nil
	}

	serializedData := mmapFile.Read(0, fileSize)
	if len(serializedData) == 0 {
		return fmt.Errorf("从 mmap 读取数据为空")
	}

	// 反序列化数据
	buf := bytes.NewReader(serializedData)
	decoder := gob.NewDecoder(buf)

	data := struct {
		Vectors                  map[string][]float64
		Clusters                 []Cluster
		NumClusters              int
		Indexed                  bool
		InvertedIndex            map[string][]string
		VectorDim                int
		VectorizedType           int
		NormalizedVectors        map[string][]float64
		CompressedVectors        map[string]entity.CompressedVector
		UseCompression           bool
		PqCodebookFilePath       string
		NumSubVectors            int
		NumCentroidsPerSubVector int
		UsePQCompression         bool
		UseHNSWIndex             bool
		MaxConnections           int
		EfConstruction           float64
		EfSearch                 float64
	}{}

	if err := decoder.Decode(&data); err != nil {
		logger.Error("从 mmap 反序列化数据库失败: %v。将使用空数据库启动。", err)
		db.initializeEmptyDB()
		return nil
	}

	// 恢复数据
	db.restoreDataFromStruct(data)

	// 加载 HNSW 索引
	if db.useHNSWIndex {
		hnswFilePath := filePath + ".hnsw"
		if _, err := os.Stat(hnswFilePath); err == nil {
			db.hnsw = graph.NewHNSWGraph(db.maxConnections, db.efConstruction, db.efSearch)
			if err := db.hnsw.LoadFromFile(hnswFilePath); err != nil {
				logger.Warning("加载 HNSW 图结构失败: %v", err)
				db.useHNSWIndex = false
				db.hnsw = nil
			}
		}
	}

	logger.Info("VectorDB 数据成功通过 mmap 从 %s 加载，向量数量: %d", db.filePath, len(db.vectors))
	return nil
}

// LoadFromFile 从文件加载数据库（智能选择是否使用 mmap）
func (db *VectorDB) LoadFromFile(filePath string) error {
	// 检查文件大小
	if fileInfo, err := os.Stat(db.filePath); err == nil {
		fileSize := fileInfo.Size()

		// 大于 10MB 的文件使用 mmap 优化
		if fileSize > 10*1024*1024 {
			if err := db.LoadFromFileWithMmap(filePath); err != nil {
				logger.Warning("mmap 加载失败，回退到标准方式: %v", err)
				return db.loadFromFileStandard(filePath)
			}
			return nil
		}
	}

	return db.loadFromFileStandard(filePath)
}

// LoadFromFile 从其配置的文件中加载数据库状态（包括索引）。
func (db *VectorDB) loadFromFileStandard(filePath string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.filePath == "" {
		return fmt.Errorf("文件路径未设置，无法加载数据库")
	}

	file, err := os.Open(db.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			logger.Info("数据库文件 %s 不存在，将创建一个新的空数据库。", db.filePath)
			// 初始化为空数据库状态，确保所有 map 都已创建
			db.vectors = make(map[string][]float64)
			db.clusters = make([]Cluster, 0)
			db.indexed = false
			db.invertedIndex = make(map[string][]string)
			db.normalizedVectors = make(map[string][]float64)
			db.compressedVectors = make(map[string]entity.CompressedVector)
			db.pqCodebook = nil
			return nil // 文件不存在不是错误，是正常启动流程
		}
		return fmt.Errorf("打开数据库文件 %s 失败: %v", db.filePath, err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			logger.Error("close file has error: %v", err.Error())
		}
	}(file)

	decoder := gob.NewDecoder(file)

	data := struct {
		Vectors                  map[string][]float64
		Clusters                 []Cluster
		NumClusters              int
		Indexed                  bool
		InvertedIndex            map[string][]string
		VectorDim                int
		VectorizedType           int
		NormalizedVectors        map[string][]float64
		CompressedVectors        map[string]entity.CompressedVector
		UseCompression           bool
		PqCodebookFilePath       string
		NumSubvectors            int
		NumCentroidsPerSubvector int
		UsePQCompression         bool

		UseHNSWIndex   bool
		MaxConnections int
		EfConstruction float64
		EfSearch       float64
	}{}

	if err := decoder.Decode(&data); err != nil {
		// 如果解码失败，可能是文件损坏或格式不兼容
		// 记录错误，并以空数据库启动，避免程序崩溃
		logger.Error("从 %s 反序列化数据库失败: %v。将使用空数据库启动。", db.filePath, err)
		db.vectors = make(map[string][]float64)
		db.clusters = make([]Cluster, 0)
		db.indexed = false
		db.invertedIndex = make(map[string][]string)
		db.normalizedVectors = make(map[string][]float64)
		db.compressedVectors = make(map[string]entity.CompressedVector)
		db.pqCodebook = nil
		return nil // 即使加载失败，也返回nil，让程序继续运行
	}

	// 恢复数据
	db.vectors = data.Vectors
	db.clusters = data.Clusters
	db.numClusters = data.NumClusters
	db.indexed = data.Indexed
	db.invertedIndex = data.InvertedIndex
	db.vectorDim = data.VectorDim
	db.vectorizedType = data.VectorizedType
	db.normalizedVectors = data.NormalizedVectors
	db.compressedVectors = data.CompressedVectors
	db.useCompression = data.UseCompression
	db.pqCodebookFilePath = data.PqCodebookFilePath
	db.numSubVectors = data.NumSubvectors
	db.numCentroidsPerSubVector = data.NumCentroidsPerSubvector
	db.usePQCompression = data.UsePQCompression
	// 从加载的数据中恢复 HNSW 相关字段
	db.useHNSWIndex = data.UseHNSWIndex
	db.maxConnections = data.MaxConnections
	db.efConstruction = data.EfConstruction
	db.efSearch = data.EfSearch

	// 确保 map 在 nil 的情况下被初始化
	if db.vectors == nil {
		db.vectors = make(map[string][]float64)
	}
	if db.invertedIndex == nil {
		db.invertedIndex = make(map[string][]string)
	}
	if db.normalizedVectors == nil {
		db.normalizedVectors = make(map[string][]float64)
	}
	if db.compressedVectors == nil {
		db.compressedVectors = make(map[string]entity.CompressedVector)
	}

	// 如果启用了 PQ 压缩且有码本路径，则尝试加载码本
	if db.usePQCompression && db.pqCodebookFilePath != "" {
		// 这里使用临时变量，避免在 LoadPQCodebookFromFile 中发生死锁
		tempPath := db.pqCodebookFilePath
		db.pqCodebookFilePath = "" // 暂时清除，避免 LoadPQCodebookFromFile 内部逻辑冲突
		db.usePQCompression = false

		db.mu.Unlock() // 解锁以便 LoadPQCodebookFromFile 可以获取锁
		errLoadCodebook := db.LoadPQCodebookFromFile(tempPath)
		db.mu.Lock() // 重新获取锁

		if errLoadCodebook != nil {
			logger.Error("从 %s 加载数据库后，尝试加载 PQ 码本 %s 失败: %v。PQ 压缩将禁用。", db.filePath, tempPath, errLoadCodebook)
			db.usePQCompression = false
			db.pqCodebook = nil
		} else {
			// LoadPQCodebookFromFile 会更新 db.pqCodebook, db.numSubVectors, db.numCentroidsPerSubVector
			// 它也会在成功加载码本后设置 db.usePQCompression = true (如果码本非空)
			// 所以这里我们只需要确保 db.usePQCompression 反映了加载结果
			if db.pqCodebook == nil || len(db.pqCodebook) == 0 {
				db.usePQCompression = false
			} else {
				db.usePQCompression = true // 确保与加载的码本状态一致
			}
		}
		// 恢复原始配置的 usePQCompression 状态，如果码本加载失败，则它会被设为 false
		// 如果码本加载成功，LoadPQCodebookFromFile 内部会处理
		// 实际上，我们应该信任 LoadPQCodebookFromFile 设置的 usePQCompression
		// 所以，如果 tempUsePQ 为 true 但加载失败，usePQCompression 会是 false，这是正确的
	} else if db.usePQCompression && db.pqCodebookFilePath == "" {
		logger.Warning("数据库配置为使用 PQ 压缩，但未指定码本文件路径。PQ 压缩将禁用。")
		db.usePQCompression = false
		db.pqCodebook = nil
	}
	// 如果启用了 HNSW 索引，加载 HNSW 图结构
	if db.useHNSWIndex {
		hnswFilePath := filePath + ".hnsw"
		db.hnsw = graph.NewHNSWGraph(db.maxConnections, db.efConstruction, db.efSearch)
		// 尝试加载 HNSW 图结构
		err := db.hnsw.LoadFromFile(hnswFilePath)

		// 设置距离函数,。在加载后，需要使用 SetDistanceFunc 方法重新设置距离函数
		db.hnsw.SetDistanceFunc(func(a, b []float64) (float64, error) {
			// 使用余弦距离（1 - 余弦相似度）
			sim := acceler.CosineSimilarity(a, b)
			return 1.0 - sim, nil
		})

		if err != nil {
			logger.Warning("加载 HNSW 图结构失败: %v，将重新构建索引。", err)
			db.indexed = false
		}
	}
	// 设置备份路径
	db.backupPath = filePath + ".bat"
	logger.Info("VectorDB 数据成功从 %s 加载。向量数: %d, 是否已索引: %t, PQ压缩: %t", db.filePath, len(db.vectors), db.indexed, db.usePQCompression)
	return nil
}

// GetFilePath 获取数据库文件路径
func (db *VectorDB) GetFilePath() string {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.filePath
}

// SetFilePath 设置数据库文件路径
func (db *VectorDB) SetFilePath(path string) {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.filePath = path
}

// LoadPQCodebookFromFile 从文件加载 PQ 码本
func (db *VectorDB) LoadPQCodebookFromFile(filePath string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if filePath == "" {
		logger.Warning("PQ 码本文件路径为空，跳过加载。")
		db.pqCodebook = nil
		db.usePQCompression = false // 如果码本路径为空，则禁用PQ
		return nil
	}

	file, err := os.Open(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			logger.Warning("PQ 码本文件 %s 不存在，PQ 压缩将不可用。", filePath)
			db.pqCodebook = nil
			db.usePQCompression = false
			return nil // 文件不存在不是致命错误，只是PQ不可用
		}
		return fmt.Errorf("打开 PQ 码本文件 %s 失败: %v", filePath, err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			logger.Error("close file has error:%v", err.Error())
		}
	}(file)

	decoder := gob.NewDecoder(file)
	var codebook [][]entity.Point
	if err := decoder.Decode(&codebook); err != nil {
		return fmt.Errorf("解码 PQ 码本文件 %s 失败: %v", filePath, err)
	}

	db.pqCodebook = codebook
	db.pqCodebookFilePath = filePath // 存储路径以备将来热更新检查
	// 可以在这里根据码本结构验证 numSubVectors 和 numCentroidsPerSubVector
	if len(codebook) > 0 {
		db.numSubVectors = len(codebook)
		if len(codebook[0]) > 0 {
			db.numCentroidsPerSubVector = len(codebook[0])
		} else {
			logger.Warning("加载的 PQ 码本子空间为空，PQ 参数可能不正确。")
			db.numCentroidsPerSubVector = 0
		}
	} else {
		logger.Warning("加载的 PQ 码本为空，PQ 参数可能不正确。")
		db.numSubVectors = 0
		db.numCentroidsPerSubVector = 0
	}

	logger.Info("成功从 %s 加载 PQ 码本。子空间数: %d, 每子空间质心数: %d", filePath, db.numSubVectors, db.numCentroidsPerSubVector)
	return nil
}
