package db

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"seetaSearch/library/algorithm"
	"sort"
	"sync"
)

// Cluster 代表一个向量簇
type Cluster struct {
	Centroid  algorithm.Point // 簇的中心点
	VectorIDs []string        // 属于该簇的向量ID列表
}
type VectorDB struct {
	vectors     map[string][]float64
	mu          sync.RWMutex
	filePath    string    // 数据库文件的存储路径
	clusters    []Cluster // 存储簇信息，用于IVF索引
	numClusters int       // K-Means中的K值，即簇的数量
	indexed     bool      // 标记数据库是否已建立索引
}

// NewVectorDB 创建一个新的 VectorDB 实例。
// 如果 filePath 非空且文件存在，则尝试从中加载数据。
// numClusters 指定了用于索引的簇数量，如果 <=0，则不启用索引功能。
func NewVectorDB(filePath string, numClusters int) *VectorDB {
	db := &VectorDB{
		vectors:     make(map[string][]float64),
		filePath:    filePath,
		numClusters: numClusters,
		clusters:    make([]Cluster, 0),
		indexed:     false,
	}
	if filePath != "" {
		if err := db.LoadFromFile(); err != nil {
			fmt.Printf("警告: 从 %s 加载向量数据库时出错: %v。将使用空数据库启动。\n", filePath, err)
			db.vectors = make(map[string][]float64)
			db.clusters = make([]Cluster, 0)
			db.indexed = false
		}
	}
	return db
}

// Add 添加向量。如果已建立索引，则将索引标记为失效。
func (db *VectorDB) Add(id string, vector []float64) {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.vectors[id] = vector
	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		fmt.Println("提示: 添加新向量后，索引已失效，请重新调用 BuildIndex()。")
	}
}

// Get 获取向量
func (db *VectorDB) Get(id string) ([]float64, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()
	vec, exists := db.vectors[id]
	return vec, exists
}

// Update 更新向量。如果已建立索引，则将索引标记为失效。
func (db *VectorDB) Update(id string, vector []float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if _, exists := db.vectors[id]; !exists {
		return fmt.Errorf("未找到ID为 '%s' 的向量", id)
	}
	db.vectors[id] = vector
	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		fmt.Println("提示: 更新向量后，索引已失效，请重新调用 BuildIndex()。")
	}
	return nil
}

// Delete 删除向量。如果已建立索引，则将索引标记为失效。
func (db *VectorDB) Delete(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if _, exists := db.vectors[id]; !exists {
		return fmt.Errorf("未找到ID为 '%s' 的向量", id)
	}
	delete(db.vectors, id)
	if db.indexed {
		db.indexed = false // 索引失效，需要重建
		fmt.Println("提示: 删除向量后，索引已失效，请重新调用 BuildIndex()。")
	}
	return nil
}

// BuildIndex 使用K-Means算法为数据库中的向量构建索引。
// maxIterations: K-Means的最大迭代次数。
// tolerance: K-Means的收敛容忍度。
func (db *VectorDB) BuildIndex(maxIterations int, tolerance float64) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.numClusters <= 0 {
		return fmt.Errorf("未配置有效的簇数量 (numClusters: %d)，无法构建索引", db.numClusters)
	}

	if len(db.vectors) < db.numClusters {
		db.indexed = false
		return fmt.Errorf("向量数量 (%d) 少于簇数量 (%d)，无法构建有效索引", len(db.vectors), db.numClusters)
	}

	fmt.Println("开始构建索引...")
	// 1. 收集所有向量及其ID
	var allVectorsData []algorithm.Point
	var vectorIDs []string // 保持与allVectorsData顺序一致的ID
	for id, vec := range db.vectors {
		allVectorsData = append(allVectorsData, algorithm.Point(vec))
		vectorIDs = append(vectorIDs, id)
	}

	// 2. 调用KMeans算法
	centroids, assignments, err := algorithm.KMeans(allVectorsData, db.numClusters, maxIterations, tolerance)
	if err != nil {
		return fmt.Errorf("KMeans聚类失败: %w", err)
	}

	// 3. 根据KMeans结果填充db.clusters
	db.clusters = make([]Cluster, db.numClusters)
	for i := 0; i < db.numClusters; i++ {
		db.clusters[i] = Cluster{
			Centroid:  centroids[i],
			VectorIDs: make([]string, 0),
		}
	}

	for i, clusterIndex := range assignments {
		if clusterIndex >= 0 && clusterIndex < db.numClusters { // 确保索引有效
			db.clusters[clusterIndex].VectorIDs = append(db.clusters[clusterIndex].VectorIDs, vectorIDs[i])
		} else {
			fmt.Printf("警告: 向量 %s 被分配到无效的簇索引 %d\n", vectorIDs[i], clusterIndex)
		}
	}

	db.indexed = true
	fmt.Printf("索引构建完成，共 %d 个簇。\n", db.numClusters)
	return nil
}

// dataToSave 结构用于 gob 编码，包含所有需要持久化的字段
type dataToSave struct {
	Vectors     map[string][]float64
	Clusters    []Cluster
	NumClusters int
	Indexed     bool
}

// SaveToFile 将当前数据库状态（包括索引）保存到其配置的文件中。
func (db *VectorDB) SaveToFile() error {
	if db.filePath == "" {
		return fmt.Errorf("此 VectorDB 实例未配置 filePath")
	}
	db.mu.RLock()
	defer db.mu.RUnlock()

	file, err := os.Create(db.filePath)
	if err != nil {
		return fmt.Errorf("创建文件 %s 失败: %w", db.filePath, err)
	}
	defer file.Close()

	data := dataToSave{
		Vectors:     db.vectors,
		Clusters:    db.clusters,
		NumClusters: db.numClusters,
		Indexed:     db.indexed,
	}

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("将数据编码到文件 %s 失败: %w", db.filePath, err)
	}
	return nil
}

// LoadFromFile 从其配置的文件中加载数据库状态（包括索引）。
func (db *VectorDB) LoadFromFile() error {
	if db.filePath == "" {
		return fmt.Errorf("此 VectorDB 实例未配置 filePath")
	}
	db.mu.Lock()
	defer db.mu.Unlock()

	file, err := os.Open(db.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // 文件未找到对于初始加载不是错误
		}
		return fmt.Errorf("打开文件 %s 失败: %w", db.filePath, err)
	}
	defer file.Close()

	var loadedData dataToSave
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&loadedData); err != nil {
		return fmt.Errorf("从文件 %s 解码数据失败: %w", db.filePath, err)
	}

	db.vectors = loadedData.Vectors
	db.clusters = loadedData.Clusters
	db.numClusters = loadedData.NumClusters
	db.indexed = loadedData.Indexed

	// 确保 map 和 slice 在加载后不是 nil，即使它们是空的
	if db.vectors == nil {
		db.vectors = make(map[string][]float64)
	}
	if db.clusters == nil {
		db.clusters = make([]Cluster, 0)
	}

	fmt.Printf("从 %s 加载数据库成功。索引状态: %t, 簇数量: %d\n", db.filePath, db.indexed, db.numClusters)
	return nil
}

// 计算欧几里得距离 (与kmeans.go中的保持一致，或者直接使用kmeans.go中的，这里为了独立性保留)
func euclideanDistance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector dimensions mismatch: %d != %d", len(a), len(b))
	}
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum), nil // KMeans 中用的是平方距离，这里为了FindNearest保持开方
}

// FindNearest 查找最相似的向量。
// 如果索引已构建，则使用IVF方法；否则，使用暴力搜索。
// nprobe: 当使用索引时，指定要搜索的最近簇的数量 (默认为1)。增加nprobe可以提高召回率但会增加搜索时间。
func (db *VectorDB) FindNearest(query []float64, k int, nprobe int) ([]string, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if k <= 0 {
		return nil, fmt.Errorf("k 必须是正整数")
	}

	if len(db.vectors) == 0 {
		return []string{}, nil
	}

	if db.indexed && len(db.clusters) > 0 && db.numClusters > 0 {
		// --- 使用IVF索引进行搜索 ---
		fmt.Println("使用IVF索引进行搜索...")
		if nprobe <= 0 {
			nprobe = 1 // 默认搜索最近的一个簇
		}
		if nprobe > db.numClusters {
			nprobe = db.numClusters // 不能超过总簇数
		}

		type centroidDist struct {
			index    int
			distance float64
		}
		var centroidDists []centroidDist

		// 1. 找到查询向量最近的 nprobe 个簇中心
		for i, cluster := range db.clusters {
			distSq, err := algorithm.EuclideanDistanceSquared(query, cluster.Centroid) // 使用KMeans包的距离函数
			if err != nil {
				// 理论上维度应该匹配，如果KMeans构建成功
				fmt.Printf("警告: 计算到簇 %d 中心的距离时出错: %v\n", i, err)
				continue
			}
			centroidDists = append(centroidDists, centroidDist{i, distSq})
		}

		sort.Slice(centroidDists, func(i, j int) bool {
			return centroidDists[i].distance < centroidDists[j].distance
		})

		type result struct {
			id       string
			distance float64
		}
		var candidateResults []result

		// 2. 在这 nprobe 个簇中搜索向量
		numProbedClusters := 0
		for i := 0; i < len(centroidDists) && numProbedClusters < nprobe; i++ {
			clusterIndex := centroidDists[i].index
			selectedCluster := db.clusters[clusterIndex]
			numProbedClusters++

			for _, vecID := range selectedCluster.VectorIDs {
				vec, exists := db.vectors[vecID] // 从主存储获取向量
				if !exists {
					fmt.Printf("警告: 索引中存在但在主存储中未找到的向量ID: %s\n", vecID)
					continue
				}
				dist, err := euclideanDistance(query, vec) // 使用原始距离进行精确排序
				if err != nil {
					fmt.Printf("警告: 计算与向量 %s 的距离时出错: %v\n", vecID, err)
					continue
				}
				candidateResults = append(candidateResults, result{vecID, dist})
			}
		}

		// 3. 对候选结果进行排序并返回top-k
		sort.Slice(candidateResults, func(i, j int) bool {
			return candidateResults[i].distance < candidateResults[j].distance
		})

		var ids []string
		count := k
		if len(candidateResults) < k {
			count = len(candidateResults)
		}
		for i := 0; i < count; i++ {
			ids = append(ids, candidateResults[i].id)
		}
		return ids, nil

	} else {
		// --- 回退到暴力搜索 ---
		fmt.Println("索引未构建或无效，执行暴力搜索...")
		type result struct {
			id       string
			distance float64
		}
		var results []result

		for id, vec := range db.vectors {
			dist, err := euclideanDistance(query, vec)
			if err != nil {
				fmt.Printf("警告: 由于维度不匹配或其他错误，跳过向量 %s: %v\n", id, err)
				continue
			}
			results = append(results, result{id, dist})
		}

		sort.Slice(results, func(i, j int) bool {
			return results[i].distance < results[j].distance
		})

		var ids []string
		count := k
		if len(results) < k {
			count = len(results)
		}
		for i := 0; i < count; i++ {
			ids = append(ids, results[i].id)
		}
		return ids, nil
	}
}

func normalizeVector(vec []float64) []float64 {
	sum := 0.0
	for _, v := range vec {
		sum += v * v
	}
	norm := math.Sqrt(sum)

	if norm == 0 {
		return vec
	}

	normalized := make([]float64, len(vec))
	for i, v := range vec {
		normalized[i] = v / norm
	}

	return normalized
}
