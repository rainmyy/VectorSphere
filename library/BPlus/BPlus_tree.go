package bplus

import "sync"

const (
	DefaultOrder = 4 // 默认阶数
)

// Key 类型
type Key int

// Value 类型，可以根据需要修改
type Value interface{}

// Node 表示 B+ 树的节点
type Node struct {
	isLeaf   bool
	keys     []Key
	children []interface{} // 非叶子节点存储子节点指针，叶子节点存储值
	next     *Node         // 叶子节点的下一个节点(用于范围查询)
	parent   *Node
	mu       sync.RWMutex
}

// BPlusTree B+ 树结构
type BPlusTree struct {
	root  *Node
	order int // 阶数
}

// NewBPlusTree 创建新的 B+ 树
func NewBPlusTree(order int) *BPlusTree {
	if order < 2 {
		order = DefaultOrder
	}
	return &BPlusTree{
		order: order,
		root:  newLeafNode(order),
	}
}

// newLeafNode 创建新的叶子节点
func newLeafNode(order int) *Node {
	return &Node{
		isLeaf:   true,
		keys:     make([]Key, 0, order),
		children: make([]interface{}, 0, order),
	}
}

// newInternalNode 创建新的内部节点
func newInternalNode(order int) *Node {
	return &Node{
		isLeaf:   false,
		keys:     make([]Key, 0, order-1),
		children: make([]interface{}, 0, order),
	}
}

func (t *BPlusTree) Insert(key Key, value Value) {
	// 查找应该插入的叶子节点
	leaf := t.findLeaf(key)
	// 检查是否已经存在该键
	for i, k := range leaf.keys {
		if k == key {
			// 键已存在，更新值
			leaf.children[i] = value
			return
		}
	}

	// 插入键值对到叶子节点
	t.insertIntoLeaf(leaf, key, value)

	// 如果叶子节点溢出，则分裂
	if len(leaf.keys) > t.order-1 {
		t.splitLeaf(leaf)
	}
}

// findLeaf 查找包含指定键的叶子节点
func (t *BPlusTree) findLeaf(key Key) *Node {
	current := t.root
	current.mu.RLock()
	defer current.mu.RUnlock()

	for !current.isLeaf {
		i := 0
		for i < len(current.keys) && key >= current.keys[i] {
			i++
		}

		next := current.children[i].(*Node)
		next.mu.RLock()
		current.mu.RUnlock()
		current = next
	}
	return current
}

// insertIntoLeaf 将键值对插入到叶子节点
func (t *BPlusTree) insertIntoLeaf(leaf *Node, key Key, value Value) {
	leaf.mu.Lock()
	defer leaf.mu.Unlock()

	i := 0
	for i < len(leaf.keys) && leaf.keys[i] < key {
		i++
	}

	// 插入键和值
	leaf.keys = append(leaf.keys[:i], append([]Key{key}, leaf.keys[i:]...)...)
	leaf.children = append(leaf.children[:i], append([]interface{}{value}, leaf.children[i:]...)...)
}

// splitLeaf 分裂叶子节点
func (t *BPlusTree) splitLeaf(leaf *Node) {
	// 创建新的叶子节点
	newLeaf := newLeafNode(t.order)

	// 分裂点
	split := (t.order + 1) / 2

	// 移动后半部分键值到新节点
	newLeaf.keys = append(newLeaf.keys, leaf.keys[split:]...)
	newLeaf.children = append(newLeaf.children, leaf.children[split:]...)

	// 更新原节点
	leaf.keys = leaf.keys[:split]
	leaf.children = leaf.children[:split]

	// 设置叶子节点链表
	newLeaf.next = leaf.next
	leaf.next = newLeaf
	newLeaf.parent = leaf.parent

	// 将新节点的第一个键插入到父节点
	t.insertIntoParent(leaf, newLeaf.keys[0], newLeaf)
}

// insertIntoParent 将分裂后的节点插入到父节点
func (t *BPlusTree) insertIntoParent(left *Node, key Key, right *Node) {
	parent := left.parent

	// 如果没有父节点(根节点分裂)，创建新的根节点
	if parent == nil {
		t.root = newInternalNode(t.order)
		t.root.keys = append(t.root.keys, key)
		t.root.children = append(t.root.children, left, right)
		left.parent = t.root
		right.parent = t.root
		return
	}

	// 找到插入位置
	i := 0
	for i < len(parent.keys) && key >= parent.keys[i] {
		i++
	}

	// 插入键和子节点指针
	parent.keys = append(parent.keys[:i], append([]Key{key}, parent.keys[i:]...)...)
	parent.children = append(parent.children[:i+1], append([]interface{}{right}, parent.children[i+1:]...)...)

	// 设置父节点
	right.parent = parent

	// 检查是否需要分裂父节点
	if len(parent.keys) > t.order-1 {
		t.splitInternal(parent)
	}
}

// splitInternal 分裂内部节点
func (t *BPlusTree) splitInternal(node *Node) {
	// 创建新的内部节点
	newNode := newInternalNode(t.order)

	// 分裂点
	split := t.order / 2
	medianKey := node.keys[split]

	// 移动后半部分键和子节点到新节点
	newNode.keys = append(newNode.keys, node.keys[split+1:]...)
	newNode.children = append(newNode.children, node.children[split+1:]...)

	// 更新原节点
	node.keys = node.keys[:split]
	node.children = node.children[:split+1]

	// 更新子节点的父指针
	for _, child := range newNode.children {
		child.(*Node).parent = newNode
	}

	newNode.parent = node.parent

	// 将中间键插入到父节点
	t.insertIntoParent(node, medianKey, newNode)
}

func (t *BPlusTree) Get(key Key) (Value, bool) {
	leaf := t.findLeaf(key)
	for i, k := range leaf.keys {
		if k == key {
			return leaf.children[i].(Value), true
		}
	}
	return "", false
}

// RangeQuery 范围查询
func (t *BPlusTree) RangeQuery(start, end Key) []Value {
	var results []Value
	leaf := t.findLeaf(start)

	for leaf != nil {
		for i, key := range leaf.keys {
			if key >= start && key <= end {
				results = append(results, leaf.children[i].(Value))
			} else if key > end {
				return results
			}
		}
		leaf = leaf.next
	}

	return results
}

// Delete 删除指定键
func (t *BPlusTree) Delete(key Key) bool {
	leaf := t.findLeaf(key)
	var index int
	var found bool

	// 查找键在叶子节点中的位置
	for i, k := range leaf.keys {
		if k == key {
			index = i
			found = true
			break
		}
	}

	if !found {
		return false
	}

	// 从叶子节点中删除键值对
	leaf.keys = append(leaf.keys[:index], leaf.keys[index+1:]...)
	leaf.children = append(leaf.children[:index], leaf.children[index+1:]...)

	// 如果叶子节点键数过少，进行合并或重分配
	if len(leaf.keys) < (t.order-1)/2 {
		t.handleLeafUnderflow(leaf)
	}

	return true
}

// handleLeafUnderflow 处理叶子节点下溢
func (t *BPlusTree) handleLeafUnderflow(leaf *Node) {
	// 尝试从左兄弟借一个键
	leftSibling, leftIndex := t.getLeftSibling(leaf)
	if leftSibling != nil && len(leftSibling.keys) > (t.order-1)/2 {
		// 从左兄弟借最后一个键值对
		borrowedKey := leftSibling.keys[len(leftSibling.keys)-1]
		borrowedValue := leftSibling.children[len(leftSibling.children)-1]

		// 从左兄弟删除
		leftSibling.keys = leftSibling.keys[:len(leftSibling.keys)-1]
		leftSibling.children = leftSibling.children[:len(leftSibling.children)-1]

		// 插入到当前节点
		leaf.keys = append([]Key{borrowedKey}, leaf.keys...)
		leaf.children = append([]interface{}{borrowedValue}, leaf.children...)

		// 更新父节点的键
		leaf.parent.keys[leftIndex] = leaf.keys[0]
		return
	}

	// 尝试从右兄弟借一个键
	rightSibling, rightIndex := t.getRightSibling(leaf)
	if rightSibling != nil && len(rightSibling.keys) > (t.order-1)/2 {
		// 从右兄弟借第一个键值对
		borrowedKey := rightSibling.keys[0]
		borrowedValue := rightSibling.children[0]

		// 从右兄弟删除
		rightSibling.keys = rightSibling.keys[1:]
		rightSibling.children = rightSibling.children[1:]

		// 插入到当前节点
		leaf.keys = append(leaf.keys, borrowedKey)
		leaf.children = append(leaf.children, borrowedValue)

		// 更新父节点的键
		leaf.parent.keys[rightIndex-1] = rightSibling.keys[0]
		return
	}

	// 无法借键，需要合并
	if leftSibling != nil {
		// 与左兄弟合并
		t.mergeLeafNodes(leftSibling, leaf)
		t.deleteInternalKey(leaf.parent, leftIndex)
	} else if rightSibling != nil {
		// 与右兄弟合并
		t.mergeLeafNodes(leaf, rightSibling)
		t.deleteInternalKey(leaf.parent, rightIndex-1)
	}

	// 如果根节点只剩下一个子节点，降低树高
	if t.root == leaf.parent && len(t.root.keys) == 0 {
		t.root = leaf.parent.children[0].(*Node)
		t.root.parent = nil
	}
}

// mergeLeafNodes 合并两个叶子节点
func (t *BPlusTree) mergeLeafNodes(left, right *Node) {
	left.keys = append(left.keys, right.keys...)
	left.children = append(left.children, right.children...)
	left.next = right.next
}

// deleteInternalKey 从内部节点删除键
func (t *BPlusTree) deleteInternalKey(node *Node, index int) {
	node.keys = append(node.keys[:index], node.keys[index+1:]...)
	node.children = append(node.children[:index+1], node.children[index+2:]...)
	// 检查是否需要处理内部节点下溢
	if node.parent != nil && len(node.keys) < (t.order-1)/2 {
		t.handleInternalUnderflow(node)
	}
	// 根节点降级
	if node == t.root && len(node.keys) == 0 {
		if len(node.children) > 0 {
			t.root = node.children[0].(*Node)
			t.root.parent = nil
		} else {
			t.root = nil
		}
	}
}

func (t *BPlusTree) handleInternalUnderflow(node *Node) {
	leftSibling, leftIndex := t.getLeftSibling(node)
	if leftSibling != nil && len(leftSibling.keys) > (t.order-1)/2 {
		// 从左兄弟借
		borrowedKey := leftSibling.keys[len(leftSibling.keys)-1]
		borrowedChild := leftSibling.children[len(leftSibling.children)-1]
		leftSibling.keys = leftSibling.keys[:len(leftSibling.keys)-1]
		leftSibling.children = leftSibling.children[:len(leftSibling.children)-1]
		node.keys = append([]Key{node.parent.keys[leftIndex]}, node.keys...)
		node.children = append([]interface{}{borrowedChild}, node.children...)
		node.parent.keys[leftIndex] = borrowedKey
		if childNode, ok := borrowedChild.(*Node); ok {
			childNode.parent = node
		}
		return
	}
	rightSibling, rightIndex := t.getRightSibling(node)
	if rightSibling != nil && len(rightSibling.keys) > (t.order-1)/2 {
		// 从右兄弟借
		borrowedKey := rightSibling.keys[0]
		borrowedChild := rightSibling.children[0]
		rightSibling.keys = rightSibling.keys[1:]
		rightSibling.children = rightSibling.children[1:]
		node.keys = append(node.keys, node.parent.keys[rightIndex-1])
		node.children = append(node.children, borrowedChild)
		node.parent.keys[rightIndex-1] = borrowedKey
		if childNode, ok := borrowedChild.(*Node); ok {
			childNode.parent = node
		}
		return
	}
	// 合并
	if leftSibling != nil {
		t.mergeInternalNodes(leftSibling, node, leftIndex)
	} else if rightSibling != nil {
		t.mergeInternalNodes(node, rightSibling, rightIndex-1)
	}
}

// mergeInternalNodes 合并两个内部节点
func (t *BPlusTree) mergeInternalNodes(left, right *Node, parentKeyIndex int) {
	left.keys = append(left.keys, left.parent.keys[parentKeyIndex])
	left.keys = append(left.keys, right.keys...)
	left.children = append(left.children, right.children...)
	for _, child := range right.children {
		if childNode, ok := child.(*Node); ok {
			childNode.parent = left
		}
	}
	t.deleteInternalKey(left.parent, parentKeyIndex)
}

// getLeftSibling 查找节点的左兄弟及其在父节点中的索引
func (t *BPlusTree) getLeftSibling(node *Node) (*Node, int) {
	if node.parent == nil {
		return nil, -1
	}
	for i, child := range node.parent.children {
		if child == node {
			if i > 0 {
				return node.parent.children[i-1].(*Node), i - 1
			}
			break
		}
	}
	return nil, -1
}

// getRightSibling 查找节点的右兄弟及其在父节点中的索引
func (t *BPlusTree) getRightSibling(node *Node) (*Node, int) {
	if node.parent == nil {
		return nil, -1
	}
	for i, child := range node.parent.children {
		if child == node {
			if i < len(node.parent.children)-1 {
				return node.parent.children[i+1].(*Node), i + 1
			}
			break
		}
	}
	return nil, -1
}
