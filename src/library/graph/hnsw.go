package graph

import (
	"VectorSphere/src/library/algorithm"
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/storage"
	"container/heap"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
)

// HNSWNode 表示 HNSW 图中的一个节点
type HNSWNode struct {
	ID        string       // 向量ID
	Vector    []float64    // 原始向量
	Neighbors [][]string   // 每层的邻居节点ID列表
	mu        sync.RWMutex // 读写锁，保护并发访问
}

// HNSWGraph 实现 Hierarchical Navigable Small World 图结构
type HNSWGraph struct {
	nodes          map[string]*HNSWNode                        // 所有节点
	maxLevel       int                                         // 最大层数
	maxConnections int                                         // 每个节点的最大连接数
	efConstruction float64                                     // 构建时的扩展因子
	EfSearch       float64                                     // 搜索时的扩展因子
	mu             sync.RWMutex                                // 读写锁，保护并发访问
	entryPointID   string                                      // 入口点ID
	levelMult      float64                                     // 层级乘数
	distanceFunc   func([]float64, []float64) (float64, error) // 距离计算函数
}

// NewHNSWGraph 创建一个新的 HNSW 图
func NewHNSWGraph(maxConnections int, efConstruction, efSearch float64) *HNSWGraph {
	return &HNSWGraph{
		nodes:          make(map[string]*HNSWNode),
		maxLevel:       0,
		maxConnections: maxConnections,
		efConstruction: efConstruction,
		EfSearch:       efSearch,
		levelMult:      1.0 / math.Log(1.0*float64(maxConnections)),
		distanceFunc: func(v1, v2 []float64) (float64, error) {
			return algorithm.EuclideanDistanceSquared(v1, v2)
		},
	}
}
func (g *HNSWGraph) ParallelAddNodes(ids []string, vectors [][]float64, numWorkers int) error {
	if len(ids) != len(vectors) {
		return fmt.Errorf("ID数量与向量数量不匹配: %d != %d", len(ids), len(vectors))
	}

	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	// 创建工作通道和错误通道
	type nodeData struct {
		id     string
		vector []float64
	}
	workChan := make(chan nodeData, len(ids))
	errChan := make(chan error, len(ids))

	// 启动工作协程
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for data := range workChan {
				if err := g.AddNode(data.id, data.vector); err != nil {
					errChan <- fmt.Errorf("添加节点 %s 失败: %w", data.id, err)
					return
				}
			}
		}()
	}

	// 发送任务到工作通道
	for i := range ids {
		workChan <- nodeData{id: ids[i], vector: vectors[i]}
	}
	close(workChan)

	// 等待所有工作完成
	wg.Wait()
	close(errChan)

	// 检查是否有错误
	for err := range errChan {
		return err // 返回第一个遇到的错误
	}

	return nil
}

// BatchSearch 批量搜索多个查询向量
func (g *HNSWGraph) BatchSearch(queryVectors [][]float64, k int, numWorkers int) ([][]entity.Result, error) {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	results := make([][]entity.Result, len(queryVectors))
	errChan := make(chan error, len(queryVectors))

	// 使用信号量限制并发数
	sem := make(chan struct{}, numWorkers)
	var wg sync.WaitGroup

	for i, queryVector := range queryVectors {
		wg.Add(1)
		go func(idx int, query []float64) {
			defer wg.Done()

			// 获取信号量
			sem <- struct{}{}
			defer func() { <-sem }()

			// 执行搜索
			res, err := g.Search(query, k)
			if err != nil {
				errChan <- fmt.Errorf("搜索查询向量 %d 失败: %w", idx, err)
				return
			}
			results[idx] = res
		}(i, queryVector)
	}

	wg.Wait()
	close(errChan)

	// 检查是否有错误
	select {
	case err := <-errChan:
		return nil, err
	default:
		return results, nil
	}
}

// SetDistanceFunc 设置距离计算函数
func (g *HNSWGraph) SetDistanceFunc(distFunc func([]float64, []float64) (float64, error)) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.distanceFunc = distFunc
}

// randomLevel 随机生成节点的层级
func (g *HNSWGraph) randomLevel() int {
	r := rand.Float64() * rand.Float64() * rand.Float64() * rand.Float64()
	// 防止r为0导致log(0)无穷大
	if r <= 0 {
		r = 1e-10
	}
	level := int(-math.Log(r) * g.levelMult)
	// 限制最大层级
	if level > 32 {
		level = 32
	}
	if level < 0 {
		level = 0
	}
	return level
}

// distanceHeapItem 用于优先队列的距离堆项
type distanceHeapItem struct {
	id       string
	distance float64
}

// distanceHeap 实现优先队列接口
type distanceHeap []distanceHeapItem

func (h distanceHeap) Len() int           { return len(h) }
func (h distanceHeap) Less(i, j int) bool { return h[i].distance < h[j].distance }
func (h distanceHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *distanceHeap) Push(x interface{}) {
	*h = append(*h, x.(distanceHeapItem))
}

func (h *distanceHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// AddNode 向图中添加节点
func (g *HNSWGraph) AddNode(id string, vector []float64) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// 检查节点是否已存在
	if _, exists := g.nodes[id]; exists {
		return fmt.Errorf("节点 %s 已存在", id)
	}

	// 随机生成节点层级
	level := g.randomLevel()

	// 创建新节点
	node := &HNSWNode{
		ID:        id,
		Vector:    vector,
		Neighbors: make([][]string, level+1),
	}

	// 初始化每层的邻居列表
	for i := 0; i <= level; i++ {
		node.Neighbors[i] = make([]string, 0, g.maxConnections)
	}

	// 如果是第一个节点，设为入口点
	if len(g.nodes) == 0 {
		g.entryPointID = id
		g.nodes[id] = node
		g.maxLevel = level
		return nil
	}

	// 获取当前入口点
	entryPoint := g.nodes[g.entryPointID]

	// 从最高层开始搜索
	currentLevel := g.maxLevel
	currentBestID := g.entryPointID
	currentBestDistance, err := g.distanceFunc(vector, entryPoint.Vector)
	if err != nil {
		return fmt.Errorf("计算距离失败: %w", err)
	}

	// 逐层搜索最近邻
	for currentLevel > level {
		changed := true
		for changed {
			changed = false

			// 检查当前最佳点的邻居
			currentNode := g.nodes[currentBestID]
			currentNode.mu.RLock()
			neighbors := currentNode.Neighbors[currentLevel]
			currentNode.mu.RUnlock()

			for _, neighborID := range neighbors {
				neighborNode := g.nodes[neighborID]
				dist, err := g.distanceFunc(vector, neighborNode.Vector)
				if err != nil {
					return fmt.Errorf("计算距离失败: %w", err)
				}

				if dist < currentBestDistance {
					currentBestDistance = dist
					currentBestID = neighborID
					changed = true
				}
			}
		}
		currentLevel--
	}

	// 对每一层执行插入
	for l := min(level, g.maxLevel); l >= 0; l-- {
		// 搜索当前层的最近邻
		candidates := g.searchLayer(vector, currentBestID, int(g.efConstruction), l)

		// 选择并连接最近的 maxConnections 个邻居
		selectedNeighbors := g.selectNeighbors(vector, candidates, g.maxConnections)

		// 添加双向连接
		for _, neighborID := range selectedNeighbors {
			neighborNode := g.nodes[neighborID]

			// 添加从新节点到邻居的连接
			node.Neighbors[l] = append(node.Neighbors[l], neighborID)

			// 添加从邻居到新节点的连接
			neighborNode.mu.Lock()
			neighborNode.Neighbors[l] = append(neighborNode.Neighbors[l], id)

			// 如果邻居的连接数超过最大值，进行修剪
			if len(neighborNode.Neighbors[l]) > g.maxConnections {
				g.pruneConnections(neighborNode, l, vector)
			}
			neighborNode.mu.Unlock()
		}

		// 更新当前最佳点
		if len(candidates) > 0 {
			currentBestID = candidates[0].id
		}
	}

	// 存储新节点
	g.nodes[id] = node

	// 如果新节点的层级高于当前最高层级，更新入口点
	if level > g.maxLevel {
		g.maxLevel = level
		g.entryPointID = id
	}

	return nil
}

// searchLayer 在指定层搜索最近的ef个邻居
func (g *HNSWGraph) searchLayer(queryVector []float64, entryPointID string, ef int, level int) []distanceHeapItem {
	// 已访问节点集合
	visited := make(map[string]bool)
	visited[entryPointID] = true

	// 候选节点优先队列（距离小的优先）
	candidates := &distanceHeap{}
	heap.Init(candidates)

	// 结果集优先队列（距离大的优先，用于维护ef个最近邻）
	results := &distanceHeap{}
	heap.Init(results)

	// 计算入口点距离
	entryNode := g.nodes[entryPointID]
	entryDist, _ := g.distanceFunc(queryVector, entryNode.Vector)

	// 添加入口点到候选集和结果集
	heap.Push(candidates, distanceHeapItem{id: entryPointID, distance: entryDist})
	heap.Push(results, distanceHeapItem{id: entryPointID, distance: -entryDist}) // 注意负号，使其成为最大堆

	// 当候选集不为空时继续搜索
	for candidates.Len() > 0 {
		// 取出距离最近的候选节点
		closest := heap.Pop(candidates).(distanceHeapItem)

		// 如果结果集中最远的节点比当前候选节点更近，则停止搜索
		if results.Len() >= ef && closest.distance > -(*results)[0].distance {
			break
		}

		// 检查当前节点的邻居
		currentNode := g.nodes[closest.id]
		currentNode.mu.RLock()
		neighbors := currentNode.Neighbors[level]
		currentNode.mu.RUnlock()

		for _, neighborID := range neighbors {
			if !visited[neighborID] {
				visited[neighborID] = true

				neighborNode := g.nodes[neighborID]
				dist, _ := g.distanceFunc(queryVector, neighborNode.Vector)

				// 如果结果集未满或当前节点比结果集中最远的节点更近
				if results.Len() < ef || dist < -(*results)[0].distance {
					heap.Push(candidates, distanceHeapItem{id: neighborID, distance: dist})
					heap.Push(results, distanceHeapItem{id: neighborID, distance: -dist})

					// 如果结果集超过ef，移除最远的节点
					if results.Len() > ef {
						heap.Pop(results)
					}
				}
			}
		}
	}

	// 转换结果为切片并按距离排序
	resultList := make([]distanceHeapItem, results.Len())
	for i := 0; i < len(resultList); i++ {
		item := heap.Pop(results).(distanceHeapItem)
		item.distance = -item.distance // 恢复正确的距离值
		resultList[len(resultList)-i-1] = item
	}

	return resultList
}

// selectNeighbors 从候选集中选择最近的k个邻居
func (g *HNSWGraph) selectNeighbors(queryVector []float64, candidates []distanceHeapItem, k int) []string {
	if len(candidates) <= k {
		// 如果候选数量不超过k，直接返回所有候选
		result := make([]string, len(candidates))
		for i, item := range candidates {
			result[i] = item.id
		}
		return result
	}

	// 按距离排序（应该已经排序，但为了安全起见）
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance < candidates[j].distance
	})

	// 选择前k个
	result := make([]string, k)
	for i := 0; i < k; i++ {
		result[i] = candidates[i].id
	}

	return result
}

// pruneConnections 修剪节点的连接，保留最近的maxConnections个邻居
func (g *HNSWGraph) pruneConnections(node *HNSWNode, level int, queryVector []float64) {
	// 收集所有邻居及其距离
	neighborsWithDist := make([]distanceHeapItem, 0, len(node.Neighbors[level]))

	for _, neighborID := range node.Neighbors[level] {
		neighborNode := g.nodes[neighborID]
		dist, _ := g.distanceFunc(queryVector, neighborNode.Vector)
		neighborsWithDist = append(neighborsWithDist, distanceHeapItem{id: neighborID, distance: dist})
	}

	// 按距离排序
	sort.Slice(neighborsWithDist, func(i, j int) bool {
		return neighborsWithDist[i].distance < neighborsWithDist[j].distance
	})

	// 保留最近的maxConnections个邻居
	count := min(g.maxConnections, len(neighborsWithDist))
	newNeighbors := make([]string, count)
	for i := 0; i < count; i++ {
		newNeighbors[i] = neighborsWithDist[i].id
	}

	node.Neighbors[level] = newNeighbors
}

// Search 在图中搜索与查询向量最近的k个节点
func (g *HNSWGraph) Search(queryVector []float64, k int) ([]entity.Result, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// 如果图为空，返回空结果
	if len(g.nodes) == 0 {
		return []entity.Result{}, nil
	}

	// 从入口点开始搜索
	entryPointID := g.entryPointID
	entryNode := g.nodes[entryPointID]
	currentBestID := entryPointID
	currentBestDistance, err := g.distanceFunc(queryVector, entryNode.Vector)
	if err != nil {
		return nil, fmt.Errorf("计算距离失败: %w", err)
	}

	// 从最高层开始搜索
	for l := g.maxLevel; l > 0; l-- {
		changed := true
		for changed {
			changed = false

			// 检查当前最佳点的邻居
			currentNode := g.nodes[currentBestID]
			currentNode.mu.RLock()
			neighbors := currentNode.Neighbors[l]
			currentNode.mu.RUnlock()

			for _, neighborID := range neighbors {
				neighborNode := g.nodes[neighborID]
				dist, err := g.distanceFunc(queryVector, neighborNode.Vector)
				if err != nil {
					return nil, fmt.Errorf("计算距离失败: %w", err)
				}

				if dist < currentBestDistance {
					currentBestDistance = dist
					currentBestID = neighborID
					changed = true
				}
			}
		}
	}

	// 在最底层执行精确搜索
	candidates := g.searchLayer(queryVector, currentBestID, max(k, int(g.EfSearch)), 0)

	// 转换结果为entity.Result格式
	results := make([]entity.Result, min(k, len(candidates)))
	for i := 0; i < len(results); i++ {
		// 将距离转换为相似度（1 - 归一化距离）
		// 注意：这里假设距离已经是平方欧氏距离，可能需要根据实际情况调整
		similarity := 1.0 / (1.0 + candidates[i].distance)
		results[i] = entity.Result{
			Id:         candidates[i].id,
			Similarity: similarity,
		}
	}

	return results, nil
}

// SearchWithFilter 在图中搜索与查询向量最近的k个节点，并应用过滤器
func (g *HNSWGraph) SearchWithFilter(queryVector []float64, k int, filter func(string) bool) ([]entity.Result, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// 如果图为空，返回空结果
	if len(g.nodes) == 0 {
		return []entity.Result{}, nil
	}

	// 从入口点开始搜索
	entryPointID := g.entryPointID
	entryNode := g.nodes[entryPointID]
	currentBestID := entryPointID
	currentBestDistance, err := g.distanceFunc(queryVector, entryNode.Vector)
	if err != nil {
		return nil, fmt.Errorf("计算距离失败: %w", err)
	}

	// 从最高层开始搜索
	for l := g.maxLevel; l > 0; l-- {
		changed := true
		for changed {
			changed = false

			// 检查当前最佳点的邻居
			currentNode := g.nodes[currentBestID]
			currentNode.mu.RLock()
			neighbors := currentNode.Neighbors[l]
			currentNode.mu.RUnlock()

			for _, neighborID := range neighbors {
				// 应用过滤器
				if !filter(neighborID) {
					continue
				}

				neighborNode := g.nodes[neighborID]
				dist, err := g.distanceFunc(queryVector, neighborNode.Vector)
				if err != nil {
					return nil, fmt.Errorf("计算距离失败: %w", err)
				}

				if dist < currentBestDistance {
					currentBestDistance = dist
					currentBestID = neighborID
					changed = true
				}
			}
		}
	}

	// 在最底层执行精确搜索，带过滤器
	// 已访问节点集合
	visited := make(map[string]bool)
	visited[currentBestID] = true

	// 候选节点优先队列（距离小的优先）
	candidates := &distanceHeap{}
	heap.Init(candidates)

	// 结果集优先队列（距离大的优先，用于维护ef个最近邻）
	results := &distanceHeap{}
	heap.Init(results)

	// 计算入口点距离
	currentNode := g.nodes[currentBestID]
	currentDist, _ := g.distanceFunc(queryVector, currentNode.Vector)

	// 应用过滤器
	if filter(currentBestID) {
		// 添加入口点到候选集和结果集
		heap.Push(candidates, distanceHeapItem{id: currentBestID, distance: currentDist})
		heap.Push(results, distanceHeapItem{id: currentBestID, distance: -currentDist}) // 注意负号，使其成为最大堆
	}

	ef := max(k, int(g.EfSearch))

	// 当候选集不为空时继续搜索
	for candidates.Len() > 0 {
		// 取出距离最近的候选节点
		closest := heap.Pop(candidates).(distanceHeapItem)

		// 如果结果集中最远的节点比当前候选节点更近，则停止搜索
		if results.Len() >= ef && closest.distance > -(*results)[0].distance {
			break
		}

		// 检查当前节点的邻居
		currentNode := g.nodes[closest.id]
		currentNode.mu.RLock()
		neighbors := currentNode.Neighbors[0] // 最底层
		currentNode.mu.RUnlock()

		for _, neighborID := range neighbors {
			if !visited[neighborID] {
				visited[neighborID] = true

				// 应用过滤器
				if !filter(neighborID) {
					continue
				}

				neighborNode := g.nodes[neighborID]
				dist, _ := g.distanceFunc(queryVector, neighborNode.Vector)

				// 如果结果集未满或当前节点比结果集中最远的节点更近
				if results.Len() < ef || dist < -(*results)[0].distance {
					heap.Push(candidates, distanceHeapItem{id: neighborID, distance: dist})
					heap.Push(results, distanceHeapItem{id: neighborID, distance: -dist})

					// 如果结果集超过ef，移除最远的节点
					if results.Len() > ef {
						heap.Pop(results)
					}
				}
			}
		}
	}

	// 转换结果为切片并按距离排序
	resultList := make([]distanceHeapItem, results.Len())
	for i := 0; i < len(resultList); i++ {
		item := heap.Pop(results).(distanceHeapItem)
		item.distance = -item.distance // 恢复正确的距离值
		resultList[len(resultList)-i-1] = item
	}

	// 转换结果为entity.Result格式
	finalResults := make([]entity.Result, min(k, len(resultList)))
	for i := 0; i < len(finalResults); i++ {
		// 将距离转换为相似度（1 - 归一化距离）
		similarity := 1.0 / (1.0 + resultList[i].distance)
		finalResults[i] = entity.Result{
			Id:         resultList[i].id,
			Similarity: similarity,
		}
	}

	return finalResults, nil
}

// DeleteNode 从图中删除节点
func (g *HNSWGraph) DeleteNode(id string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// 检查节点是否存在
	node, exists := g.nodes[id]
	if !exists {
		return fmt.Errorf("节点 %s 不存在", id)
	}

	// 从所有邻居的连接列表中移除该节点
	for level := 0; level < len(node.Neighbors); level++ {
		for _, neighborID := range node.Neighbors[level] {
			neighborNode, exists := g.nodes[neighborID]
			if !exists {
				continue
			}

			neighborNode.mu.Lock()
			// 移除对该节点的引用
			newNeighbors := make([]string, 0, len(neighborNode.Neighbors[level]))
			for _, nID := range neighborNode.Neighbors[level] {
				if nID != id {
					newNeighbors = append(newNeighbors, nID)
				}
			}
			neighborNode.Neighbors[level] = newNeighbors
			neighborNode.mu.Unlock()
		}
	}

	// 如果删除的是入口点，需要选择新的入口点
	if id == g.entryPointID {
		if len(g.nodes) > 1 {
			// 选择任意其他节点作为新的入口点
			for newEntryID := range g.nodes {
				if newEntryID != id {
					g.entryPointID = newEntryID
					break
				}
			}
		} else {
			// 如果没有其他节点，清空入口点
			g.entryPointID = ""
			g.maxLevel = 0
		}
	}

	// 删除节点
	delete(g.nodes, id)

	// 更新最大层级（如果需要）
	if len(node.Neighbors) == g.maxLevel+1 {
		// 查找新的最大层级
		newMaxLevel := 0
		for _, n := range g.nodes {
			if len(n.Neighbors) > newMaxLevel {
				newMaxLevel = len(n.Neighbors) - 1
			}
		}
		g.maxLevel = newMaxLevel
	}

	return nil
}

// GetNode 获取指定ID的节点
func (g *HNSWGraph) GetNode(id string) (*HNSWNode, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	node, exists := g.nodes[id]
	return node, exists
}

// GetAllNodeIDs 获取图中所有节点的ID
func (g *HNSWGraph) GetAllNodeIDs() []string {
	g.mu.RLock()
	defer g.mu.RUnlock()

	ids := make([]string, 0, len(g.nodes))
	for id := range g.nodes {
		ids = append(ids, id)
	}

	return ids
}

// Size 返回图中节点的数量
func (g *HNSWGraph) Size() int {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return len(g.nodes)
}

// SaveToFileWithMmap 使用mmap优化的保存方法
func (g *HNSWGraph) SaveToFileWithMmap(filePath string) error {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// 创建mmap文件
	mmap, err := storage.NewMmap(filePath, storage.MODE_CREATE)
	if err != nil {
		return fmt.Errorf("创建mmap文件 %s 失败: %v", filePath, err)
	}
	defer mmap.Unmap()

	// 写入图的基本信息
	if err := mmap.AppendInt64(int64(g.maxLevel)); err != nil {
		return fmt.Errorf("写入maxLevel失败: %v", err)
	}
	if err := mmap.AppendInt64(int64(g.maxConnections)); err != nil {
		return fmt.Errorf("写入maxConnections失败: %v", err)
	}
	if err := mmap.AppendInt64(int64(math.Float64bits(g.efConstruction))); err != nil {
		return fmt.Errorf("写入efConstruction失败: %v", err)
	}
	if err := mmap.AppendInt64(int64(math.Float64bits(g.EfSearch))); err != nil {
		return fmt.Errorf("写入EfSearch失败: %v", err)
	}
	if err := mmap.AppendInt64(int64(math.Float64bits(g.levelMult))); err != nil {
		return fmt.Errorf("写入levelMult失败: %v", err)
	}
	if err := mmap.AppendStringWithLen(g.entryPointID); err != nil {
		return fmt.Errorf("写入entryPointID失败: %v", err)
	}

	// 写入节点数量
	if err := mmap.AppendInt64(int64(len(g.nodes))); err != nil {
		return fmt.Errorf("写入节点数量失败: %v", err)
	}

	// 写入每个节点的数据
	for id, node := range g.nodes {
		node.mu.RLock()

		// 写入节点ID
		if err := mmap.AppendStringWithLen(id); err != nil {
			node.mu.RUnlock()
			return fmt.Errorf("写入节点ID %s 失败: %v", id, err)
		}

		// 写入向量维度和数据
		if err := mmap.AppendInt64(int64(len(node.Vector))); err != nil {
			node.mu.RUnlock()
			return fmt.Errorf("写入向量维度失败: %v", err)
		}
		for _, val := range node.Vector {
			if err := mmap.AppendInt64(int64(math.Float64bits(val))); err != nil {
				node.mu.RUnlock()
				return fmt.Errorf("写入向量数据失败: %v", err)
			}
		}

		// 写入邻居层数
		if err := mmap.AppendInt64(int64(len(node.Neighbors))); err != nil {
			node.mu.RUnlock()
			return fmt.Errorf("写入邻居层数失败: %v", err)
		}

		// 写入每层的邻居
		for _, neighbors := range node.Neighbors {
			if err := mmap.AppendInt64(int64(len(neighbors))); err != nil {
				node.mu.RUnlock()
				return fmt.Errorf("写入邻居数量失败: %v", err)
			}
			for _, neighborID := range neighbors {
				if err := mmap.AppendStringWithLen(neighborID); err != nil {
					node.mu.RUnlock()
					return fmt.Errorf("写入邻居ID失败: %v", err)
				}
			}
		}

		node.mu.RUnlock()
	}

	// 同步数据到磁盘
	if err := mmap.Sync(); err != nil {
		return fmt.Errorf("同步数据到磁盘失败: %v", err)
	}

	return nil
}

// SaveToFile 修改原有的SaveToFile方法，智能选择使用mmap或标准方法
func (g *HNSWGraph) SaveToFile(filePath string) error {
	// 估算数据大小，如果较大则使用mmap优化
	estimatedSize := g.estimateDataSize()
	if estimatedSize > 10*1024*1024 { // 大于10MB使用mmap
		if err := g.SaveToFileWithMmap(filePath); err != nil {
			// mmap失败时回退到标准方法
			return g.saveToFileStandard(filePath)
		}
		return nil
	}
	return g.saveToFileStandard(filePath)
}

// SaveToFile 将图结构保存到文件
func (g *HNSWGraph) saveToFileStandard(filePath string) error {
	g.mu.RLock()
	defer g.mu.RUnlock()

	// 创建文件
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("创建文件 %s 失败: %v", filePath, err)
	}
	defer file.Close()

	// 创建 gob 编码器
	encoder := gob.NewEncoder(file)

	// 创建一个可序列化的结构体，包含图的所有必要信息
	type nodeData struct {
		ID        string
		Vector    []float64
		Neighbors [][]string
	}

	// 收集所有节点数据
	nodes := make(map[string]nodeData)
	for id, node := range g.nodes {
		node.mu.RLock()
		nodes[id] = nodeData{
			ID:        node.ID,
			Vector:    node.Vector,
			Neighbors: node.Neighbors,
		}
		node.mu.RUnlock()
	}

	// 创建要序列化的数据结构
	data := struct {
		Nodes          map[string]nodeData
		MaxLevel       int
		MaxConnections int
		EfConstruction float64
		EfSearch       float64
		EntryPointID   string
		LevelMult      float64
	}{
		Nodes:          nodes,
		MaxLevel:       g.maxLevel,
		MaxConnections: g.maxConnections,
		EfConstruction: g.efConstruction,
		EfSearch:       g.EfSearch,
		EntryPointID:   g.entryPointID,
		LevelMult:      g.levelMult,
	}

	// 序列化数据
	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("序列化图结构到 %s 失败: %v", filePath, err)
	}

	return nil
}

// LoadFromFileWithMmap 使用mmap优化的加载方法
func (g *HNSWGraph) LoadFromFileWithMmap(filePath string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// 打开mmap文件
	mmap, err := storage.NewMmap(filePath, storage.MODE_APPEND)
	if err != nil {
		return fmt.Errorf("打开mmap文件 %s 失败: %v", filePath, err)
	}
	defer mmap.Unmap()

	pointer := int64(0)

	// 读取图的基本信息
	g.maxLevel = int(mmap.ReadInt64(pointer))
	pointer += 8
	g.maxConnections = int(mmap.ReadInt64(pointer))
	pointer += 8
	g.efConstruction = math.Float64frombits(uint64(mmap.ReadInt64(pointer)))
	pointer += 8
	g.EfSearch = math.Float64frombits(uint64(mmap.ReadInt64(pointer)))
	pointer += 8
	g.levelMult = math.Float64frombits(uint64(mmap.ReadInt64(pointer)))
	pointer += 8

	// 读取entryPointID
	entryPointIDLen := mmap.ReadInt64(pointer)
	pointer += 8
	g.entryPointID = mmap.ReadString(pointer, entryPointIDLen)
	pointer += entryPointIDLen

	// 读取节点数量
	nodeCount := int(mmap.ReadInt64(pointer))
	pointer += 8

	// 重置节点映射
	g.nodes = make(map[string]*HNSWNode)

	// 读取每个节点的数据
	for i := 0; i < nodeCount; i++ {
		// 读取节点ID
		idLen := mmap.ReadInt64(pointer)
		pointer += 8
		id := mmap.ReadString(pointer, idLen)
		pointer += idLen

		// 读取向量维度
		vectorDim := int(mmap.ReadInt64(pointer))
		pointer += 8

		// 读取向量数据
		vector := make([]float64, vectorDim)
		for j := 0; j < vectorDim; j++ {
			vector[j] = math.Float64frombits(uint64(mmap.ReadInt64(pointer)))
			pointer += 8
		}

		// 读取邻居层数
		layerCount := int(mmap.ReadInt64(pointer))
		pointer += 8

		// 读取每层的邻居
		neighbors := make([][]string, layerCount)
		for layer := 0; layer < layerCount; layer++ {
			neighborCount := int(mmap.ReadInt64(pointer))
			pointer += 8

			neighbors[layer] = make([]string, neighborCount)
			for k := 0; k < neighborCount; k++ {
				neighborIDLen := mmap.ReadInt64(pointer)
				pointer += 8
				neighbors[layer][k] = mmap.ReadString(pointer, neighborIDLen)
				pointer += neighborIDLen
			}
		}

		// 创建节点
		node := &HNSWNode{
			ID:        id,
			Vector:    vector,
			Neighbors: neighbors,
		}
		g.nodes[id] = node
	}

	return nil
}

// LoadFromFile 修改原有的LoadFromFile方法，智能选择使用mmap或标准方法
func (g *HNSWGraph) LoadFromFile(filePath string) error {
	// 检查文件大小
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return fmt.Errorf("获取文件信息失败: %v", err)
	}

	if fileInfo.Size() > 10*1024*1024 { // 大于10MB使用mmap
		if err := g.LoadFromFileWithMmap(filePath); err != nil {
			// mmap失败时回退到标准方法
			return g.loadFromFileStandard(filePath)
		}
		return nil
	}
	return g.loadFromFileStandard(filePath)
}

// LoadFromFile 从文件加载图结构
func (g *HNSWGraph) loadFromFileStandard(filePath string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("打开文件 %s 失败: %v", filePath, err)
	}
	defer file.Close()

	// 创建 gob 解码器
	decoder := gob.NewDecoder(file)

	// 定义与 SaveToFile 中相同的节点数据结构
	type nodeData struct {
		ID        string
		Vector    []float64
		Neighbors [][]string
	}

	// 定义要解码的数据结构
	data := struct {
		Nodes          map[string]nodeData
		MaxLevel       int
		MaxConnections int
		EfConstruction float64
		EfSearch       float64
		EntryPointID   string
		LevelMult      float64
	}{}

	// 解码数据
	if err := decoder.Decode(&data); err != nil {
		return fmt.Errorf("从 %s 反序列化图结构失败: %v", filePath, err)
	}

	// 重置图结构
	g.nodes = make(map[string]*HNSWNode)
	g.maxLevel = data.MaxLevel
	g.maxConnections = data.MaxConnections
	g.efConstruction = data.EfConstruction
	g.EfSearch = data.EfSearch
	g.entryPointID = data.EntryPointID
	g.levelMult = data.LevelMult

	// 重建节点
	for id, nodeData := range data.Nodes {
		node := &HNSWNode{
			ID:        nodeData.ID,
			Vector:    nodeData.Vector,
			Neighbors: nodeData.Neighbors,
		}
		g.nodes[id] = node
	}

	return nil
}

// estimateDataSize 估算序列化后的数据大小
func (g *HNSWGraph) estimateDataSize() int64 {
	size := int64(0)

	// 基本字段大小
	size += 8 * 5                          // maxLevel, maxConnections, efConstruction, EfSearch, levelMult
	size += int64(len(g.entryPointID)) + 8 // entryPointID with length
	size += 8                              // node count

	// 估算节点数据大小
	for id, node := range g.nodes {
		node.mu.RLock()
		size += int64(len(id)) + 8            // node ID with length
		size += int64(len(node.Vector))*8 + 8 // vector data + dimension
		size += 8                             // layer count

		// 估算邻居数据大小
		for _, neighbors := range node.Neighbors {
			size += 8 // neighbor count
			for _, neighborID := range neighbors {
				size += int64(len(neighborID)) + 8 // neighbor ID with length
			}
		}
		node.mu.RUnlock()
	}

	return size
}
