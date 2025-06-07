package tree

import (
	"VectorSphere/src/library/algorithm"
	"VectorSphere/src/library/entity"
	"container/heap"
	"math"
)

// KDTreeNode KD树节点结构
type KDTreeNode struct {
	Point []float64 // 节点对应的向量
	ID    string    // 向量ID
	Axis  int       // 分割维度
	Left  *KDTreeNode
	Right *KDTreeNode
}

// KDTree KD树结构
type KDTree struct {
	Root      *KDTreeNode
	Dimension int // 向量维度
}

// NewKDTree 创建新的KD树
func NewKDTree(dimension int) *KDTree {
	return &KDTree{
		Root:      nil,
		Dimension: dimension,
	}
}

// Insert 向KD树中插入节点
func (tree *KDTree) Insert(point []float64, id string) {
	if tree.Root == nil {
		tree.Root = &KDTreeNode{
			Point: point,
			ID:    id,
			Axis:  0,
		}
		return
	}

	tree.insertRecursive(tree.Root, point, id, 0)
}

// insertRecursive 递归插入节点
func (tree *KDTree) insertRecursive(node *KDTreeNode, point []float64, id string, depth int) {
	// 计算当前分割轴
	axis := depth % tree.Dimension

	// 根据分割轴比较决定向左还是向右
	if point[axis] < node.Point[axis] {
		if node.Left == nil {
			node.Left = &KDTreeNode{
				Point: point,
				ID:    id,
				Axis:  axis,
			}
		} else {
			tree.insertRecursive(node.Left, point, id, depth+1)
		}
	} else {
		if node.Right == nil {
			node.Right = &KDTreeNode{
				Point: point,
				ID:    id,
				Axis:  axis,
			}
		} else {
			tree.insertRecursive(node.Right, point, id, depth+1)
		}
	}
}

// FindNearest 在KD树中查找最近的k个点
func (tree *KDTree) FindNearest(query []float64, k int) []string {
	if tree.Root == nil {
		return []string{}
	}

	// 使用优先队列存储最近的k个点
	resultHeap := make(entity.ResultHeap, 0, k)
	heap.Init(&resultHeap)

	// 递归搜索
	tree.searchNearest(tree.Root, query, k, &resultHeap, 0)

	// 从堆中提取结果
	results := make([]string, 0, resultHeap.Len())
	for resultHeap.Len() > 0 {
		results = append([]string{heap.Pop(&resultHeap).(entity.Result).Id}, results...)
	}

	return results
}

// searchNearest 递归搜索最近的k个点
func (tree *KDTree) searchNearest(node *KDTreeNode, query []float64, k int, resultHeap *entity.ResultHeap, depth int) {
	if node == nil {
		return
	}

	// 计算当前节点与查询点的相似度
	sim := algorithm.CosineSimilarity(query, node.Point)

	// 更新结果堆
	if resultHeap.Len() < k {
		heap.Push(resultHeap, entity.Result{Id: node.ID, Similarity: sim})
	} else if sim > (*resultHeap)[0].Similarity {
		heap.Pop(resultHeap)
		heap.Push(resultHeap, entity.Result{Id: node.ID, Similarity: sim})
	}

	// 计算当前分割轴
	axis := depth % tree.Dimension

	// 决定先搜索哪个子树
	firstChild, secondChild := node.Left, node.Right
	if query[axis] > node.Point[axis] {
		firstChild, secondChild = node.Right, node.Left
	}

	// 先搜索更可能包含近邻的子树
	tree.searchNearest(firstChild, query, k, resultHeap, depth+1)

	// 检查是否需要搜索另一个子树
	// 如果查询点到分割超平面的距离小于当前最大距离，则需要搜索另一个子树
	if resultHeap.Len() < k || math.Abs(query[axis]-node.Point[axis]) < 1.0-(*resultHeap)[0].Similarity {
		tree.searchNearest(secondChild, query, k, resultHeap, depth+1)
	}
}
