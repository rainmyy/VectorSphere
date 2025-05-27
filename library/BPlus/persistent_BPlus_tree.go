package bplus

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"os"
	"sync"
)

const (
	nodeSize = 4096 // 假设每个节点占4KB
)

type PersistentNode struct {
	mu           sync.Mutex
	isLeaf       bool
	keys         []Key
	children     []interface{} // 对于内部节点是 []int64 (offsets), 对于叶子节点是 []Value
	nextOffset   int64         // 叶子节点间的链表，指向下一个叶子节点的磁盘偏移量
	parentOffset int64         // 父节点的磁盘偏移量
	selfOffset   int64         //资深节点的偏移量
	// 可能还有其他字段，如 pageID/offsetInFile 自身，方便缓存管理
}

type key interface{}
type value interface{}

// DiskManager 管理磁盘存储
type DiskManager struct {
	file     *os.File
	freeList []int64 // 空闲块列表
	mu       sync.Mutex
}

func newPersistentLeafNode(order int) *PersistentNode {
	return &PersistentNode{
		isLeaf:   true,
		keys:     make([]Key, 0, order),
		children: make([]interface{}, 0, order),
	}
}
func newPersistentInternalNode(order int) *PersistentNode {
	return &PersistentNode{
		isLeaf:   false,
		keys:     make([]Key, 0, order-1),
		children: make([]interface{}, 0, order),
	}
}

// NewDiskManager 创建磁盘管理器
func NewDiskManager(filename string) (*DiskManager, error) {
	file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		return nil, err
	}
	return &DiskManager{file: file}, nil
}

// Allocate 分配磁盘块
func (dm *DiskManager) Allocate() (int64, error) {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	if len(dm.freeList) > 0 {
		offset := dm.freeList[len(dm.freeList)-1]
		dm.freeList = dm.freeList[:len(dm.freeList)-1]
		return offset, nil
	}

	// 获取文件大小
	stat, err := dm.file.Stat()
	if err != nil {
		return 0, err
	}
	return stat.Size(), nil
}

// Free 释放磁盘块
func (dm *DiskManager) Free(offset int64) {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	dm.freeList = append(dm.freeList, offset)
}

// ReadNode 从磁盘读取节点
func (dm *DiskManager) ReadNode(offset int64) (*PersistentNode, error) {
	data := make([]byte, nodeSize)
	_, err := dm.file.ReadAt(data, offset)
	if err != nil {
		return nil, err
	}
	return deserializeNode(data)
}

// WriteNode 将节点写入磁盘
func (dm *DiskManager) WriteNode(node *PersistentNode) (int64, error) {
	data, err := serializeNode(node)
	if err != nil {
		return 0, err
	}

	offset, err := dm.Allocate()
	if err != nil {
		return 0, err
	}

	_, err = dm.file.WriteAt(data, offset)
	if err != nil {
		dm.Free(offset)
		return 0, err
	}

	return offset, nil
}

// 序列化和反序列化方法
func serializeNode(node *PersistentNode) ([]byte, error) {
	buffer := new(bytes.Buffer)
	encoder := gob.NewEncoder(buffer)
	err := encoder.Encode(node)
	if err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}

func deserializeNode(data []byte) (*PersistentNode, error) {
	buffer := bytes.NewBuffer(data)
	decoder := gob.NewDecoder(buffer)
	var node PersistentNode
	err := decoder.Decode(&node)
	if err != nil {
		return nil, err
	}
	return &node, nil
}

type PersistentBPlusTree struct {
	order      int
	rootOffset int64 // 根节点磁盘偏移量
	disk       *DiskManager
	cache      *NodeCache // 节点缓存
	mu         sync.RWMutex
}

// NewPersistentBPlusTree 创建持久化B+树
func NewPersistentBPlusTree(order int, filename string) (*PersistentBPlusTree, error) {
	disk, err := NewDiskManager(filename)
	if err != nil {
		return nil, err
	}

	// 初始化缓存
	cache := NewNodeCache(1000) // 假设缓存1000个节点

	// 如果文件为空，创建根节点
	stat, err := disk.file.Stat()
	if err != nil {
		return nil, err
	}

	var rootOffset int64
	if stat.Size() == 0 {
		root := newPersistentLeafNode(order)
		rootOffset, err = disk.WriteNode(root)
		if err != nil {
			return nil, err
		}
	} else {
		// 从文件读取根节点偏移量(可以存储在文件开头)
		// 这里简化处理，假设根节点在偏移量0处
		rootOffset = 0
	}

	return &PersistentBPlusTree{
		order:      order,
		rootOffset: rootOffset,
		disk:       disk,
		cache:      cache,
	}, nil
}

// 带缓存的节点获取
func (t *PersistentBPlusTree) getNode(offset int64) (*PersistentNode, error) {
	// 先查缓存
	if node, ok := t.cache.Get(offset); ok {
		return node, nil
	}

	// 从磁盘读取
	node, err := t.disk.ReadNode(offset)
	if err != nil {
		return nil, err
	}

	// 存入缓存
	t.cache.Put(offset, node)
	return node, nil
}

// 修改后的查找叶子节点方法
func (t *PersistentBPlusTree) findLeaf(key Key) (*PersistentNode, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	current, err := t.getNode(t.rootOffset)
	if err != nil {
		return nil, err
	}

	for !current.isLeaf {
		i := 0
		for i < len(current.keys) && key >= current.keys[i] {
			i++
		}

		childOffset, ok := current.children[i].(int64)
		if !ok {
			return nil, fmt.Errorf("internal node child is not an offset: %T", current.children[i])
		}
		current, err = t.getNode(childOffset)
		if err != nil {
			return nil, err
		}
	}

	return current, nil
}

// Insert 修改后的插入方法
func (t *PersistentBPlusTree) Insert(key Key, value Value) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	leaf, err := t.findLeaf(key)
	if err != nil {
		return err
	}

	// 检查键是否已存在
	for i, k := range leaf.keys {
		if k == key {
			// 更新值
			leaf.children[i] = value
			_, err := t.disk.WriteNode(leaf)
			return err
		}
	}

	// 插入键值对
	t.insertIntoLeaf(leaf, key, value)

	// 如果节点溢出，分裂
	if len(leaf.keys) > t.order-1 {
		return t.splitLeaf(leaf)
	}

	// 写回磁盘
	_, err = t.disk.WriteNode(leaf)
	return err
}

// 修改后的分裂方法需要处理磁盘偏移量
func (t *PersistentBPlusTree) splitLeaf(leaf *PersistentNode) error {
	newLeaf := newPersistentLeafNode(t.order)

	split := (t.order + 1) / 2
	newLeaf.keys = append(newLeaf.keys, leaf.keys[split:]...)
	newLeaf.children = append(newLeaf.children, leaf.children[split:]...)

	leaf.keys = leaf.keys[:split]
	leaf.children = leaf.children[:split]

	// 写入磁盘
	newLeafOffset, err := t.disk.WriteNode(newLeaf)
	if err != nil {
		return err
	}
	newLeaf.SetOffset(newLeafOffset)
	t.cache.Put(newLeafOffset, newLeaf)
	updatedLeafOffset, err := t.disk.WriteNode(leaf)
	if err != nil {
		// 这里需要考虑回滚 newLeaf 的写入，或者标记 newLeafOffset 为空闲
		t.disk.Free(newLeafOffset) // 尝试释放
		t.cache.Delete(newLeafOffset)
		return err
	}
	leaf.SetOffset(updatedLeafOffset)    // 假设 Node 有 SetOffset 方法
	t.cache.Put(updatedLeafOffset, leaf) // 更新缓存

	// 更新链表
	newLeaf.nextOffset = leaf.nextOffset
	leaf.nextOffset = newLeafOffset

	// 更新父节点
	return t.insertIntoParent(updatedLeafOffset, newLeaf.keys[0], newLeafOffset)
}

//	func (t *PersistentBPlusTree) insertIntoParent1(leftOffset int64, key Key, rightOffset int64) error {
//		leftNode, err := t.getNode(leftOffset)
//		if err != nil {
//			return err
//		}
//		parentOffset := leftNode.parentOffset
//
//		// 没有父节点，创建新根
//		if parentOffset == 0 || parentOffset == -1 {
//			newRoot := newPersistentInternalNode(t.order)
//			newRoot.keys = append(newRoot.keys, key)
//			newRoot.children = append(newRoot.children, leftOffset, rightOffset)
//			newRootOffset, err := t.disk.WriteNode(newRoot)
//			if err != nil {
//				return err
//			}
//			leftNode.parentOffset = newRootOffset
//			rightNode, err := t.getNode(rightOffset)
//			if err != nil {
//				return err
//			}
//			rightNode.parentOffset = newRootOffset
//			_, err = t.disk.WriteNode(leftNode)
//			if err != nil {
//				return err
//			}
//			_, err = t.disk.WriteNode(rightNode)
//			if err != nil {
//				return err
//			}
//			t.rootOffset = newRootOffset
//			return nil
//		}
//
//		// 有父节点，插入key和rightOffset
//		parent, err := t.getNode(parentOffset)
//		if err != nil {
//			return err
//		}
//		i := 0
//		for i < len(parent.keys) && key > parent.keys[i] {
//			i++
//		}
//		parent.keys = append(parent.keys[:i], append([]Key{key}, parent.keys[i:]...)...)
//		parent.children = append(parent.children[:i+1], append([]interface{}{rightOffset}, parent.children[i+1:]...)...)
//
//		// 更新rightNode的父指针
//		rightNode, err := t.getNode(rightOffset)
//		if err != nil {
//			return err
//		}
//		rightNode.parentOffset = parentOffset
//		_, err = t.disk.WriteNode(rightNode)
//		if err != nil {
//			return err
//		}
//
//		// 父节点溢出递归分裂
//		if len(parent.keys) > t.order-1 {
//			return t.splitInternal(parent)
//		}
//		_, err = t.disk.WriteNode(parent)
//		return err
//	}
func (t *PersistentBPlusTree) insertIntoParent(leftOffset int64, key Key, rightOffset int64) error {
	leftNode, err := t.getNode(leftOffset)
	if err != nil {
		return err
	}
	parentOffset := leftNode.parentOffset

	// 没有父节点，创建新根
	if parentOffset == 0 || parentOffset == -1 { // -1 通常表示无效或未设置的偏移量
		newRoot := newPersistentLeafNode(t.order)
		newRoot.keys = append(newRoot.keys, key)
		// 内部节点的 children 存储子节点的磁盘偏移量 (int64)
		newRoot.children = append(newRoot.children, leftOffset, rightOffset)

		newRootOffset, err := t.disk.WriteNode(newRoot)
		if err != nil {
			return err
		}
		newRoot.SetOffset(newRootOffset) // 假设 Node 有 SetOffset
		t.cache.Put(newRootOffset, newRoot)

		leftNode.parentOffset = newRootOffset
		// _, err = t.disk.WriteNode(leftNode) // 需要写回 leftNode
		// if err != nil { return err }
		// t.cache.Put(leftOffset, leftNode) // leftOffset 可能也变了，如果 WriteNode 分配新块
		// 简化：假设 getNode 返回的 leftNode 修改后，其 offset 不变，后续 WriteNode 会处理
		// 但这不安全。正确的做法是，修改后写回，并获取新 offset
		// 假设 leftNode 的 offset 是 leftOffset，修改 parentOffset 后需要写回
		updatedLeftOffset, err := t.disk.WriteNode(leftNode)
		if err != nil { /* 回滚 newRoot? */
			return err
		}
		if updatedLeftOffset != leftOffset { /* 父节点(newRoot)中存的leftOffset需要更新! */
			// 这是非常复杂的一点，节点的 offset 改变会级联影响
			// 实际系统中，WriteNode 通常会尝试原地更新，或者返回新 offset，然后调用者负责更新引用
		}
		t.cache.Put(updatedLeftOffset, leftNode)

		rightNode, err := t.getNode(rightOffset)
		if err != nil {
			return err
		}
		rightNode.parentOffset = newRootOffset
		// _, err = t.disk.WriteNode(rightNode)
		// if err != nil { return err }
		// t.cache.Put(rightOffset, rightNode)
		updatedRightOffset, err := t.disk.WriteNode(rightNode)
		if err != nil {
			return err
		}
		t.cache.Put(updatedRightOffset, rightNode)

		// 更新树的根偏移量
		t.rootOffset = newRootOffset
		// !!! 重要：需要持久化新的 rootOffset 到文件的超级块中
		return nil
	}

	// 有父节点，插入key和rightOffset
	parent, err := t.getNode(parentOffset)
	if err != nil {
		return err
	}
	i := 0

	for i < len(parent.keys) && key > parent.keys[i] { // key > parent.keys[i]
		i++
	}
	// 插入 key
	parent.keys = append(parent.keys[:i], append([]Key{key}, parent.keys[i:]...)...)
	// 插入 rightOffset (作为 interface{})
	parent.children = append(parent.children[:i+1], append([]interface{}{rightOffset}, parent.children[i+1:]...)...)

	// 更新rightNode的父指针
	rightNode, err := t.getNode(rightOffset)
	if err != nil {
		return err
	}
	rightNode.parentOffset = parentOffset // 父节点是 parent，其 offset 是 parentOffset
	// _, err = t.disk.WriteNode(rightNode) // 写回 rightNode
	// if err != nil { return err }
	updatedRightNodeOffset, err := t.disk.WriteNode(rightNode)
	if err != nil {
		return err
	}
	t.cache.Put(updatedRightNodeOffset, rightNode)
	if updatedRightNodeOffset != rightOffset {
		// 如果 rightNode 的 offset 变了，父节点中对它的引用 (刚插入的 rightOffset) 也需要更新！
		// parent.children[i+1] = updatedRightNodeOffset
	}

	// 父节点溢出递归分裂
	if len(parent.keys) > t.order-1 { // 假设 order 是阶数
		// return t.splitInternal(parent) // parent 的 offset 是 parentOffset
		// 需要传递 parent 节点本身及其当前的 offset
		return t.splitInternal(parent)
	}
	// _, err = t.disk.WriteNode(parent) // 写回 parent
	updatedParentOffset, err := t.disk.WriteNode(parent)
	if err != nil {
		return err
	}
	t.cache.Put(updatedParentOffset, parent)
	if updatedParentOffset != parentOffset {
		// 如果 parent 的 offset 变了，它的父节点中对它的引用也需要更新！
		// 这就是为什么通常节点会携带自己的 offset，或者 WriteNode 尝试原地更新
	}
	return err
}

// 分裂内部节点
//
//	func (t *PersistentBPlusTree) splitInternal1(node *PersistentNode) error {
//		newNode := newPersistentInternalNode(t.order)
//		split := (t.order + 1) / 2
//
//		// 分裂key和children
//		upKey := node.keys[split]
//		newNode.keys = append(newNode.keys, node.keys[split+1:]...)
//		newNode.children = append(newNode.children, node.children[split+1:]...)
//
//		// 更新原节点
//		node.keys = node.keys[:split]
//		node.children = node.children[:split+1]
//
//		// 写入新节点
//		newNodeOffset, err := t.disk.WriteNode(newNode)
//		if err != nil {
//			return err
//		}
//
//		// 更新新节点所有子节点的父指针
//		for _, child := range newNode.children {
//			childOffset, ok := child.(int64)
//			if !ok {
//				continue
//			}
//			childNode, err := t.getNode(childOffset)
//			if err != nil {
//				return err
//			}
//			childNode.parentOffset = newNodeOffset
//			_, err = t.disk.WriteNode(childNode)
//			if err != nil {
//				return err
//			}
//		}
//
//		// 写回原节点
//		nodeOffset, err := t.disk.WriteNode(node)
//		if err != nil {
//			return err
//		}
//
//		// 插入到父节点
//		return t.insertIntoParent(nodeOffset, upKey, newNodeOffset)
//	}

func (t *PersistentBPlusTree) splitInternal(node *PersistentNode) error {
	newNode := newPersistentLeafNode(t.order)
	// newNode.parentOffset = node.parentOffset // 初始时父节点相同

	// split := (t.order + 1) / 2 // 对于内部节点，通常是 ceil(order/2)-1 个键给左边
	// 或者简单地取中间的键作为提升的键
	split := len(node.keys) / 2

	// 分裂key和children
	upKey := node.keys[split] // 这个键被提升到父节点
	newNode.keys = append(newNode.keys, node.keys[split+1:]...)
	newNode.children = append(newNode.children, node.children[split+1:]...)

	// 更新原节点
	node.keys = node.keys[:split]
	node.children = node.children[:split+1] // 保留指向 upKey 左侧子树的指针

	// 写入新节点
	newNodeOffset, err := t.disk.WriteNode(newNode)
	if err != nil {
		return err
	}
	newNode.SetOffset(newNodeOffset)
	t.cache.Put(newNodeOffset, newNode)

	// 更新新节点所有子节点的父指针
	for _, childPtr := range newNode.children {
		childOffset, ok := childPtr.(int64)
		if !ok {
			return fmt.Errorf("internal node child in newNode is not an offset: %T", childPtr)
		}
		childNode, err := t.getNode(childOffset)
		if err != nil {
			return err
		}
		childNode.parentOffset = newNodeOffset // 指向新分裂出来的内部节点
		_, err = t.disk.WriteNode(childNode)   // 写回子节点
		if err != nil {
			return err
		}
		updatedChildOffset, err := t.disk.WriteNode(childNode)
		if err != nil {
			return err
		}
		t.cache.Put(updatedChildOffset, childNode)
	}

	updatedNodeOffset, err := t.disk.WriteNode(node)
	if err != nil {
		// 回滚 newNode 的写入？
		t.disk.Free(newNodeOffset)
		t.cache.Delete(newNodeOffset)
		return err
	}
	t.cache.Put(updatedNodeOffset, node)

	// 插入到父节点
	// 传递的是更新后的原节点 offset (updatedNodeOffset) 和新节点的 offset (newNodeOffset)
	return t.insertIntoParent(updatedNodeOffset, upKey, newNodeOffset)
}

// insertIntoLeaf 将键值对插入到叶子节点
func (t *PersistentBPlusTree) insertIntoLeaf(leaf *PersistentNode, key Key, value Value) {
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

type NodeCache struct {
	size  int
	cache map[int64]*PersistentNode
	mu    sync.RWMutex
}

func NewNodeCache(size int) *NodeCache {
	return &NodeCache{
		size:  size,
		cache: make(map[int64]*PersistentNode),
	}
}

func (nc *NodeCache) Get(offset int64) (*PersistentNode, bool) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	node, ok := nc.cache[offset]
	return node, ok
}

func (nc *NodeCache) Put(offset int64, node *PersistentNode) {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	if len(nc.cache) >= nc.size {
		// 简单的LRU淘汰策略
		for k := range nc.cache {
			delete(nc.cache, k)
			break
		}
	}
	nc.cache[offset] = node
}

func (nc *NodeCache) Delete(offset int64) {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	delete(nc.cache, offset)
}

func (n *PersistentNode) SetOffset(offset int64) {
	n.selfOffset = offset
}
