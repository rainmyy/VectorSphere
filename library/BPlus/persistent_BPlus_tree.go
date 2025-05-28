package bplus

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"os"
	"seetaSearch/library/strategy"
	"sync"
)

const (
	nodeSize       = 4096 // 假设每个节点占4KB
	invalidOffset  = int64(-1)
	superBlockSize = 4096     // For storing rootOffset, etc.
	rootOffsetPtr  = int64(0) // Location within superblock where root offset is stored
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

func init() {
	gob.Register((*PersistentNode)(nil))
	gob.Register(int64(0))
}

// DiskManager 管理磁盘存储
type DiskManager struct {
	file     *os.File
	freeList []int64 // 空闲块列表
	mu       sync.Mutex
}

// Sync 添加DiskManager的Sync方法
func (dm *DiskManager) Sync() error {
	return dm.file.Sync()
}

func newPersistentLeafNode(order int) *PersistentNode {
	return &PersistentNode{
		isLeaf:       true,
		keys:         make([]Key, 0, order),
		children:     make([]interface{}, 0, order),
		parentOffset: invalidOffset,
		selfOffset:   invalidOffset,
		nextOffset:   invalidOffset,
	}
}
func newPersistentInternalNode(order int) *PersistentNode {
	return &PersistentNode{
		isLeaf:       false,
		keys:         make([]Key, 0, order-1),
		children:     make([]interface{}, 0, order), // Will store int64 offsets
		parentOffset: invalidOffset,
		selfOffset:   invalidOffset,
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
	// Ensure allocation happens after any potential superblock
	if stat.Size() < superBlockSize {
		return superBlockSize, nil
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
	if offset == invalidOffset || offset < superBlockSize { // Basic check for validity
		return nil, fmt.Errorf("cannot read node from invalid offset: %d", offset)
	}
	data := make([]byte, nodeSize)
	_, err := dm.file.ReadAt(data, offset)
	if err != nil {
		return nil, err
	}
	return deserializeNode(data)
}

// WriteNode 将节点写入磁盘。如果node.selfOffset有效，则尝试原地更新，否则分配新块。
// 返回写入的偏移量。
func (dm *DiskManager) WriteNode(node *PersistentNode) (int64, error) {
	data, err := serializeNode(node)
	if err != nil {
		return invalidOffset, err
	}
	if len(data) > nodeSize {
		return invalidOffset, fmt.Errorf("serialized node size (%d) exceeds max nodeSize (%d)", len(data), nodeSize)
	}

	paddedData := make([]byte, nodeSize)
	copy(paddedData, data)

	offset := node.selfOffset
	// If node.selfOffset is invalid, or we decide to always allocate for simplicity for now
	// A more advanced WriteNode might try to overwrite if offset is valid and space is sufficient.
	// For now, we always allocate if selfOffset is invalid, or re-use selfOffset if valid.
	// This simplified version still allocates new space if selfOffset was not set,
	// or overwrites if selfOffset was set.
	// A true persistent B+tree might need more sophisticated block management (e.g. COW or specific update-in-place)

	// If the node has a valid offset, we assume we are updating it in place.
	// If not, we allocate a new one.
	if offset == invalidOffset || offset < superBlockSize { // Ensure offset is valid and not in superblock area
		offset, err = dm.Allocate()
		if err != nil {
			return invalidOffset, err
		}
	}

	_, err = dm.file.WriteAt(paddedData, offset)
	if err != nil {
		// If allocation was new and write failed, we might want to free it.
		// However, if selfOffset was valid, freeing it might be wrong if it was an existing block.
		// This part needs careful consideration based on allocation strategy.
		// For now, if it was a new allocation (node.selfOffset was invalid), we can free it.
		if node.selfOffset == invalidOffset { // only free if it was a fresh allocation attempt
			dm.Free(offset)
		}
		return invalidOffset, err
	}
	node.selfOffset = offset // Update node's own record of its offset
	return offset, nil
}

// writeSuperBlock (conceptual)
func (dm *DiskManager) writeSuperBlock(rootOffset int64) error {
	// In a real implementation, you'd serialize a SuperBlock struct
	// For simplicity, just writing the rootOffset as bytes.
	buf := new(bytes.Buffer)
	encoder := gob.NewEncoder(buf)
	if err := encoder.Encode(&rootOffset); err != nil {
		return fmt.Errorf("failed to encode root offset: %w", err)
	}
	if int64(buf.Len()) > superBlockSize {
		return fmt.Errorf("superblock data too large")
	}
	paddedData := make([]byte, superBlockSize)
	copy(paddedData, buf.Bytes())

	_, err := dm.file.WriteAt(paddedData, 0) // Write at the beginning of the file
	return err
}

// readSuperBlock (conceptual)
func (dm *DiskManager) readSuperBlock() (int64, error) {
	data := make([]byte, superBlockSize)
	_, err := dm.file.ReadAt(data, 0)
	if err != nil {
		// If it's EOF and file size is 0, it's a new DB, that's okay.
		// os.IsNotExist(err) or io.EOF might be checked here depending on OS and file state.
		// For simplicity, if read fails, assume new DB or corrupted, try to init.
		return invalidOffset, fmt.Errorf("failed to read superblock: %w", err)
	}
	buf := bytes.NewBuffer(data)
	decoder := gob.NewDecoder(buf)
	var rootOffset int64
	if err := decoder.Decode(&rootOffset); err != nil {
		return invalidOffset, fmt.Errorf("failed to decode root offset: %w", err)
	}
	return rootOffset, nil
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

// PersistentBPlusTree 首先需要扩展PersistentBPlusTree结构，添加事务相关字段
type PersistentBPlusTree struct {
	order      int
	rootOffset int64 // 根节点磁盘偏移量
	disk       *DiskManager
	cache      *NodeCache // 节点缓存
	mu         sync.RWMutex
	// 添加事务支持
	txMgr   *TransactionManager
	lockMgr *LockManager
	wal     *WALManager
}

// NewPersistentBPlusTreeWithTx 更新构造函数以支持事务
func NewPersistentBPlusTreeWithTx(order int, filename string, txMgr *TransactionManager, lockMgr *LockManager, wal *WALManager) (*PersistentBPlusTree, error) {
	tree, err := NewPersistentBPlusTree(order, filename)
	if err != nil {
		return nil, err
	}

	tree.txMgr = txMgr
	tree.lockMgr = lockMgr
	tree.wal = wal

	return tree, nil
}

// NewPersistentBPlusTree 创建持久化B+树
func NewPersistentBPlusTree(order int, filename string) (*PersistentBPlusTree, error) {
	disk, err := NewDiskManager(filename)
	if err != nil {
		return nil, err
	}

	cache := NewNodeCache(1000) // 假设缓存1000个节点

	var rootOffset int64
	stat, err := disk.file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to stat file %s: %w", filename, err)
	}

	if stat.Size() < superBlockSize { // The File is too small to even contain a superblock, treat as new
		fmt.Println("Initializing new B+ tree database...")
		root := newPersistentLeafNode(order)
		rootOffset, err = disk.WriteNode(root) // This will set root.selfOffset
		if err != nil {
			return nil, fmt.Errorf("failed to write initial root node: %w", err)
		}
		if err := disk.writeSuperBlock(rootOffset); err != nil {
			// Attempt to free the allocated root node if superblock write fails
			disk.Free(rootOffset)
			return nil, fmt.Errorf("failed to write superblock: %w", err)
		}
		fmt.Printf("New B+ tree initialized. Root offset: %d\n", rootOffset)
	} else {
		fmt.Println("Loading B+ tree from existing file...")
		rootOffset, err = disk.readSuperBlock()
		if err != nil || rootOffset == invalidOffset {
			return nil, fmt.Errorf("failed to read or invalid root offset from superblock: %w", err)
		}
		fmt.Printf("B+ tree loaded. Root offset: %d\n", rootOffset)
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

func (t *PersistentBPlusTree) findLeaf(key Key) (*PersistentNode, error) {
	t.mu.RLock()
	// No defer t.mu.RUnlock() here if we are calling other tree methods that might lock

	if t.rootOffset == invalidOffset {
		t.mu.RUnlock()
		return nil, fmt.Errorf("tree has an invalid root offset")
	}

	current, err := t.getNode(t.rootOffset)
	if err != nil {
		t.mu.RUnlock()
		return nil, err
	}
	t.mu.RUnlock() // Unlock after initial root fetch, subsequent getNode calls are internally safe

	for !current.isLeaf {
		i := 0
		// Ensure key comparison is correct for your Key type
		// For Key int, this is fine.
		for i < len(current.keys) && !key.Less(current.keys[i]) {
			i++
		}

		childOffsetRaw, ok := current.children[i].(int64)
		if !ok {
			return nil, fmt.Errorf("internal node child is not an offset: child type %T at index %d, node offset %d", current.children[i], i, current.selfOffset)
		}
		if childOffsetRaw == invalidOffset {
			return nil, fmt.Errorf("internal node child has invalid offset at index %d, node offset %d", i, current.selfOffset)
		}

		current, err = t.getNode(childOffsetRaw)
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

	if t.rootOffset == invalidOffset {
		// This case should ideally be handled by NewPersistentBPlusTree creating an initial root.
		// If we reach here, it means the tree is in an inconsistent state or wasn't initialized.
		// For robustness, we could re-initialize the root here, but it's better to ensure NewPersistentBPlusTree handles it.
		// Let's assume NewPersistentBPlusTree ensures a valid rootOffset.
		// If not, create one.
		fmt.Println("Insert called on tree with invalid root. Initializing root.")
		root := newPersistentLeafNode(t.order)
		rootOffset, err := t.disk.WriteNode(root)
		if err != nil {
			return fmt.Errorf("failed to create initial root during insert: %w", err)
		}
		if err := t.disk.writeSuperBlock(rootOffset); err != nil {
			t.disk.Free(rootOffset)
			return fmt.Errorf("failed to write superblock for new root during insert: %w", err)
		}
		t.rootOffset = rootOffset
		t.cache.Put(rootOffset, root) // Add to cache
	}

	leaf, err := t.findLeaf(key) // findLeaf should be called without the tree lock or handle it carefully
	if err != nil {
		return err
	}

	leaf.mu.Lock() // Lock the specific leaf node
	// 检查键是否已存在
	for i, k := range leaf.keys {
		if k == key {
			// 更新值
			leaf.children[i] = value
			// Write node back to disk and update cache
			updatedOffset, writeErr := t.disk.WriteNode(leaf) // leaf.selfOffset should be set by WriteNode
			leaf.mu.Unlock()
			if writeErr != nil {
				return writeErr
			}
			t.cache.Put(updatedOffset, leaf) // Update cache with potentially new offset
			return nil
		}
	}

	// 插入键值对
	t.insertIntoLeafWithoutLock(leaf, key, value) // Assumes leaf is already locked

	// 如果节点溢出，分裂
	if len(leaf.keys) > t.order-1 {
		leaf.mu.Unlock()         // Unlock before split, splitLeaf will handle its own locking
		return t.splitLeaf(leaf) // Pass key for context if needed, or just leaf
	}

	// 写回磁盘
	updatedOffset, writeErr := t.disk.WriteNode(leaf)
	leaf.mu.Unlock()
	if writeErr != nil {
		return writeErr
	}
	t.cache.Put(updatedOffset, leaf)
	return nil
}

// splitLeaf 修改后的分裂方法需要处理磁盘偏移量
func (t *PersistentBPlusTree) splitLeaf(leaf *PersistentNode) error {
	// Node-level lock should be acquired by the caller or here if necessary.
	// For simplicity, assuming caller (Insert) released lock on 'leaf' if it was held at tree-level.
	// If splitLeaf is called internally, it needs to manage locks carefully.
	// Let's assume 'leaf' is not locked by a higher-level tree lock here, but we should lock it for modification.
	leaf.mu.Lock()

	newLeaf := newPersistentLeafNode(t.order)
	newLeaf.parentOffset = leaf.parentOffset // New leaf initially has same parent

	split := (t.order) / 2 // Ensure order is B+ tree order (max children/pointers for internal, max keys for leaf)
	// For leaf nodes; order often means max number of key-value pairs.
	// If order is max keys, then split is (order)/2.

	// Copy keys and children to newLeaf
	newLeaf.keys = append(newLeaf.keys, leaf.keys[split:]...)
	newLeaf.children = append(newLeaf.children, leaf.children[split:]...)

	// Truncate original leaf
	leaf.keys = leaf.keys[:split]
	leaf.children = leaf.children[:split]

	// Write newLeaf to disk first
	newLeafOffset, err := t.disk.WriteNode(newLeaf) // This sets newLeaf.selfOffset
	if err != nil {
		leaf.mu.Unlock()
		return err
	}
	t.cache.Put(newLeafOffset, newLeaf)

	// Update original leaf's next pointer and write it to disk
	newLeaf.nextOffset = leaf.nextOffset // newLeaf takes over old leaf's next
	leaf.nextOffset = newLeafOffset      // old leaf points to newLeaf

	// Write updated original leaf to disk
	updatedLeafOffset, err := t.disk.WriteNode(leaf) // This sets/updates leaf.selfOffset
	leaf.mu.Unlock()                                 // Unlock leaf after modifications and before calling insertIntoParent
	if err != nil {
		// Attempt to rollback: free newLeaf and remove from cache
		t.disk.Free(newLeafOffset)
		t.cache.Delete(newLeafOffset)
		// Potentially restore original leaf's state if possible, or mark tree as inconsistent
		return err
	}
	t.cache.Put(updatedLeafOffset, leaf) // Update cache for original leaf

	// Update parent node
	// The key to be inserted into parent is the first key of the newLeaf
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
func (t *PersistentBPlusTree) insertIntoParent(leftNodeOffset int64, key Key, rightNodeOffset int64) error {
	// This function is critical and can be complex due to potential recursive splits and root changes.
	// Tree-level lock should be held by the caller (e.g. Insert after a split).

	leftNode, err := t.getNode(leftNodeOffset)
	if err != nil {
		return fmt.Errorf("insertIntoParent: failed to get left node (offset %d): %w", leftNodeOffset, err)
	}
	// rightNode, err := t.getNode(rightNodeOffset) // Not strictly needed immediately, but good for parentOffset update
	// if err != nil { return fmt.Errorf("insertIntoParent: failed to get right node (offset %d): %w", rightNodeOffset, err) }

	parentOffset := leftNode.parentOffset

	// Case 1: No parent, leftNode was the root. Create a new root.
	if parentOffset == invalidOffset {
		newRoot := newPersistentInternalNode(t.order) // New root is an internal node
		newRoot.keys = append(newRoot.keys, key)
		newRoot.children = append(newRoot.children, leftNodeOffset, rightNodeOffset)

		newRootDiskOffset, err := t.disk.WriteNode(newRoot) // This sets newRoot.selfOffset
		if err != nil {
			return fmt.Errorf("failed to write new root: %w", err)
		}
		t.cache.Put(newRootDiskOffset, newRoot)

		// Update children's parent pointers
		leftNode.parentOffset = newRootDiskOffset
		updatedLeftOffset, err := t.disk.WriteNode(leftNode)
		if err != nil { /* TODO: Handle error, maybe try to free newRootDiskOffset */
			return err
		}
		t.cache.Put(updatedLeftOffset, leftNode)

		// Need to fetch rightNode again if not already, or ensure it's the correct one from cache
		rightNodeForParentUpdate, err := t.getNode(rightNodeOffset)
		if err != nil {
			return err
		}
		rightNodeForParentUpdate.parentOffset = newRootDiskOffset
		updatedRightOffset, err := t.disk.WriteNode(rightNodeForParentUpdate)
		if err != nil { /* TODO: Handle error */
			return err
		}
		t.cache.Put(updatedRightOffset, rightNodeForParentUpdate)

		// Update tree's root offset and persist it
		t.rootOffset = newRootDiskOffset
		return t.disk.writeSuperBlock(t.rootOffset) // Persist the new root offset
	}

	// Case 2: Parent exists. Insert key and right child pointer into parent.
	parent, err := t.getNode(parentOffset)
	if err != nil {
		return fmt.Errorf("insertIntoParent: failed to get parent node (offset %d): %w", parentOffset, err)
	}

	parent.mu.Lock()
	// Find position to insert a key and right child pointer
	i := 0
	for i < len(parent.keys) && !key.Less(parent.keys[i]) { // key >= parent.keys[i] for B+ tree property
		i++
	}

	// Insert key
	tempKeys := make([]Key, len(parent.keys)+1)
	copy(tempKeys, parent.keys[:i])
	tempKeys[i] = key
	copy(tempKeys[i+1:], parent.keys[i:])
	parent.keys = tempKeys

	// Insert right child pointer (rightNodeOffset)
	tempChildren := make([]interface{}, len(parent.children)+1)
	copy(tempChildren, parent.children[:i+1])
	tempChildren[i+1] = rightNodeOffset // This must be int64
	copy(tempChildren[i+2:], parent.children[i+1:])
	parent.children = tempChildren

	// Update parentOffset of the newly added right child (rightNodeOffset)
	// This is crucial: the rightNode's parent is now 'parent'
	rightNodeForParentUpdate, err := t.getNode(rightNodeOffset)
	if err != nil {
		parent.mu.Unlock()
		return fmt.Errorf("failed to get right node (%d) for parent update: %w", rightNodeOffset, err)
	}
	rightNodeForParentUpdate.parentOffset = parent.selfOffset // parent.selfOffset should be valid
	updatedRightChildOffset, err := t.disk.WriteNode(rightNodeForParentUpdate)
	if err != nil {
		parent.mu.Unlock()
		return fmt.Errorf("failed to write updated right child (%d): %w", rightNodeOffset, err)
	}
	t.cache.Put(updatedRightChildOffset, rightNodeForParentUpdate)
	// If rightNodeOffset changed, we need to update parent.children[i+1] - this is complex if WriteNode always allocates new.
	// Assuming WriteNode with valid selfOffset updates in place or returns same offset if no change.
	// If updatedRightChildOffset != rightNodeOffset, then parent.children[i+1] needs to be updatedRightChildOffset.
	// This highlights the complexity of non-in-place updates.
	// For now, we assume if rightNodeForParentUpdate.selfOffset was valid, it's updated in place.
	// If it was invalid, WriteNode assigned it, and that's what we stored in parent.children.

	// If parent overflows, split parent
	if len(parent.keys) > t.order-1 { // order is max keys for internal node
		parent.mu.Unlock()                  // Unlock before recursive call
		return t.splitInternal(parent, key) // Pass parent and the key that caused its split
	}

	// Write parent back to disk
	updatedParentOffset, err := t.disk.WriteNode(parent)
	parent.mu.Unlock()
	if err != nil {
		return err
	}
	t.cache.Put(updatedParentOffset, parent)
	return nil
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

// splitInternal splits an internal node.
// The 'triggerKey' is the key that was just inserted into 'node' causing it to split.
func (t *PersistentBPlusTree) splitInternal(node *PersistentNode, triggerKey Key) error {
	// Similar to splitLeaf, assume tree-level lock is held by caller.
	// Lock the node being split.
	node.mu.Lock()
	defer node.mu.Unlock()

	newNode := newPersistentInternalNode(t.order) // Splitting internal node creates another internal node
	newNode.parentOffset = node.parentOffset      // Initially same parent

	// Determine split point and the key to be pushed up to the parent
	// For internal nodes, order is often max number of children (degree).
	// The Number of keys is order - 1.
	// Split point: median key goes up.
	split := (t.order - 1) / 2 // Index of the key to be pushed up

	upKey := node.keys[split]

	// Copy keys and children to newNode (keys and children to the right of upKey)
	newNode.keys = append(newNode.keys, node.keys[split+1:]...)
	newNode.children = append(newNode.children, node.children[split+1:]...)

	// Truncate the original node (keys and children to the left of upKey, upKey is removed)
	node.keys = node.keys[:split]
	node.children = node.children[:split+1] // Children up to and including the one for upKey's left side

	// Write newNode to disk
	newNodeOffset, err := t.disk.WriteNode(newNode) // Sets newNode.selfOffset
	if err != nil {
		return err
	}
	t.cache.Put(newNodeOffset, newNode)

	// Update parent pointers of children that moved to newNode
	for _, childOffsetRaw := range newNode.children {
		childOffset, ok := childOffsetRaw.(int64)
		if !ok {
			// This should not happen if children are always int64 for internal nodes
			node.mu.Unlock()
			t.disk.Free(newNodeOffset) // Attempt cleanup
			t.cache.Delete(newNodeOffset)
			return fmt.Errorf("splitInternal: child in newNode is not an offset: %T", childOffsetRaw)
		}
		childNode, err := t.getNode(childOffset)
		if err != nil {
			// More complex cleanup might be needed here
			return err
		}
		childNode.parentOffset = newNodeOffset // Child now belongs to newNode
		updatedChildOffset, err := t.disk.WriteNode(childNode)
		if err != nil {
			return err
		}
		t.cache.Put(updatedChildOffset, childNode)
	}

	// Write updated original node to disk
	updatedNodeOffset, err := t.disk.WriteNode(node) // Sets/updates node.selfOffset
	node.mu.Unlock()                                 // Unlock node before calling insertIntoParent
	if err != nil {
		// Attempt to rollback: free newNode and remove from cache
		t.disk.Free(newNodeOffset)
		t.cache.Delete(newNodeOffset)
		return err
	}
	t.cache.Put(updatedNodeOffset, node)

	// Insert upKey into parent node
	// Pass original node's (now potentially new) offset and new node's offset
	return t.insertIntoParent(updatedNodeOffset, upKey, newNodeOffset)
}

// insertIntoLeaf 将键值对插入到叶子节点
func (t *PersistentBPlusTree) insertIntoLeaf(leaf *PersistentNode, key Key, value Value) {
	leaf.mu.Lock()
	defer leaf.mu.Unlock()

	i := 0
	for i < len(leaf.keys) && leaf.keys[i].Less(key) {
		i++
	}

	// 插入键和值
	leaf.keys = append(leaf.keys[:i], append([]Key{key}, leaf.keys[i:]...)...)
	leaf.children = append(leaf.children[:i], append([]interface{}{value}, leaf.children[i:]...)...)
}

// insertIntoLeafWithoutLock inserts a key-value pair into a leaf node.
// Assumes the caller already locks the leaf node.
func (t *PersistentBPlusTree) insertIntoLeafWithoutLock(leaf *PersistentNode, key Key, value Value) {
	i := 0
	// Find position to insert key
	// For Key int; this comparison is fine.
	for i < len(leaf.keys) && leaf.keys[i].Less(key) { // Corrected: key > leaf.keys[i] for sorted order
		i++
	}

	// Insert key
	tempKeys := make([]Key, len(leaf.keys)+1)
	copy(tempKeys, leaf.keys[:i])
	tempKeys[i] = key
	copy(tempKeys[i+1:], leaf.keys[i:])
	leaf.keys = tempKeys

	// Insert value
	tempChildren := make([]interface{}, len(leaf.children)+1)
	copy(tempChildren, leaf.children[:i])
	tempChildren[i] = value // Value is interface{}
	copy(tempChildren[i+1:], leaf.children[i:])
	leaf.children = tempChildren
}

type NodeCache struct {
	lruList *strategy.LRUCache
	mu      sync.RWMutex
}

func NewNodeCache(size int) *NodeCache {
	return &NodeCache{
		lruList: strategy.NewLRUCache(size),
	}
}

func (nc *NodeCache) Get(offset int64) (*PersistentNode, bool) {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	elem, ok := nc.lruList.Get(offset)
	if !ok {
		return nil, false
	}
	return elem.(*PersistentNode), false
}

func (nc *NodeCache) Put(offset int64, node *PersistentNode) {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	nc.lruList.Put(offset, node)
}

func (nc *NodeCache) Delete(offset int64) {
	nc.mu.Lock()
	defer nc.mu.Unlock()

	nc.lruList.Delete(offset)
}

func (n *PersistentNode) SetOffset(offset int64) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.selfOffset = offset
}
