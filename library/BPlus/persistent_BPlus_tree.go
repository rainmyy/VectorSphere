package bplus

import (
	"os"
	"sync"
)

const (
	nodeSize = 4096 // 假设每个节点占4KB
)

// DiskManager 管理磁盘存储
type DiskManager struct {
	file     *os.File
	freeList []int64 // 空闲块列表
	mu       sync.Mutex
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
func (dm *DiskManager) ReadNode(offset int64) (*Node, error) {
	data := make([]byte, nodeSize)
	_, err := dm.file.ReadAt(data, offset)
	if err != nil {
		return nil, err
	}
	return deserializeNode(data)
}

// WriteNode 将节点写入磁盘
func (dm *DiskManager) WriteNode(node *Node) (int64, error) {
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
func serializeNode(node *Node) ([]byte, error) {
	// 实现节点序列化逻辑
	// 将节点结构转换为字节数组
}

func deserializeNode(data []byte) (*Node, error) {
	// 实现节点反序列化逻辑
	// 从字节数组重建节点结构
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
		root := newLeafNode(order)
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
func (t *PersistentBPlusTree) getNode(offset int64) (*Node, error) {
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
func (t *PersistentBPlusTree) findLeaf(key Key) (*Node, error) {
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

		childOffset := current.children[i].(int64) // 子节点存储的是磁盘偏移量
		current, err = t.getNode(childOffset)
		if err != nil {
			return nil, err
		}
	}

	return current, nil
}

// 修改后的插入方法
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
func (t *PersistentBPlusTree) splitLeaf(leaf *Node) error {
	newLeaf := newLeafNode(t.order)

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

	leafOffset, err := t.disk.WriteNode(leaf)
	if err != nil {
		return err
	}

	// 更新链表
	newLeaf.nextOffset = leaf.nextOffset
	leaf.nextOffset = newLeafOffset

	// 更新父节点
	return t.insertIntoParent(leafOffset, newLeaf.keys[0], newLeafOffset)
}

// insertIntoLeaf 将键值对插入到叶子节点
func (t *PersistentBPlusTree) insertIntoLeaf(leaf *Node, key Key, value Value) {
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
	cache map[int64]*Node
	mu    sync.RWMutex
}

func NewNodeCache(size int) *NodeCache {
	return &NodeCache{
		size:  size,
		cache: make(map[int64]*Node),
	}
}

func (nc *NodeCache) Get(offset int64) (*Node, bool) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	node, ok := nc.cache[offset]
	return node, ok
}

func (nc *NodeCache) Put(offset int64, node *Node) {
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
