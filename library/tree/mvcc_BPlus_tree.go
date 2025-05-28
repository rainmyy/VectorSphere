package tree

import (
	"errors"
	"fmt"
	"seetaSearch/library/log"
	"sync"
	"time"
)

// Version 表示数据的一个版本
type Version struct {
	value   Value
	txID    uint64   // 创建该版本的事务ID
	beginTS uint64   // 版本开始时间戳
	endTS   uint64   // 版本结束时间戳(0表示当前版本)
	prev    *Version // 前一个版本(版本链)
}

// MVCCNode 支持MVCC的节点
type MVCCNode struct {
	key      Key
	versions *Version // 版本链头
	mu       sync.RWMutex

	// 添加B+树节点必要的属性
	isLeaf   bool          // 是否为叶子节点
	keys     []Key         // 键数组
	children []interface{} // 子节点数组（可能是MVCCNode或其他类型）
	next     *MVCCNode     // 叶子节点链表的下一个节点
}

// GetValue 根据事务获取合适的版本
func (n *MVCCNode) GetValue(tx *Transaction) (Value, bool) {
	n.mu.RLock()
	defer n.mu.RUnlock()

	for v := n.versions; v != nil; v = v.prev {
		if tx.isVisible(v) {
			return v.value, true
		}
	}
	return nil, false
}

func (n *MVCCNode) IsLeaf() bool {
	return n.isLeaf
}

func (n *MVCCNode) GetChildren() []interface{} {
	return n.children
}

func (n *MVCCNode) GetKeys() []Key {
	return n.keys
}

func (n *MVCCNode) GetNext() *MVCCNode {
	return n.next
}

// AddVersion 添加新版本
func (n *MVCCNode) AddVersion(value Value, tx *Transaction) {
	n.mu.Lock()
	defer n.mu.Unlock()

	newVersion := &Version{
		value:   value,
		txID:    tx.txID,
		beginTS: tx.startTS,
		prev:    n.versions,
		endTS:   0,
	}

	if n.versions != nil {
		n.versions.endTS = tx.startTS
	}

	n.versions = newVersion
}

// 事务可见性判断
func (tx *Transaction) isVisible(v *Version) bool {
	// 规则1: 事务总是能看到自己创建的版本
	if v.txID == tx.txID {
		return true
	}

	// 规则2: 版本必须在事务开始前已提交
	//    v.beginTS < tx.startTS
	//    并且版本创建事务已经提交 (这里简化，假设 txID < tx.txID 意味着已提交且在前)
	//    实际需要查询 TransactionManager 获取 v.txID 的状态

	// 规则3: 版本未被晚于事务开始时间戳的其他事务覆盖
	//    v.endTS == 0 (当前版本) 或 v.endTS >= tx.startTS

	switch tx.isolation {
	case ReadUncommitted:
		// 可以读取任何版本，即使是未提交事务所创建的最新版本
		// 但通常只读取最新的那个（v.endTS == 0）
		return v.endTS == 0 // 简化：只看最新的，不管提交状态
	case ReadCommitted:
		// 版本创建事务已提交 (v.txID != tx.txID 且 v.txID 已提交)
		// 且版本在事务开始时是有效的 (v.beginTS < tx.startTS 且 (v.endTS == 0 || v.endTS >= tx.startTS))
		// 简化：如果不是自己的，则必须是已提交的最新版本
		isCommitted := tx.snapshot.txMgr.IsCommitted(v.txID) // 需要 TransactionManager 支持

		return isCommitted && (v.endTS == 0 || v.endTS >= tx.startTS) && v.beginTS < tx.startTS

	case RepeatableRead, Serializable:
		// 版本对事务可见的条件：
		// 1. 版本由当前事务自己创建 (已在开头处理)。
		// 2. 版本在事务开始时 (tx.startTS) 就存在且已提交：
		//    v.beginTS < tx.startTS
		//    创建该版本的事务 (v.txID) 在 tx.startTS 时已提交。
		//    并且，在 tx.startTS 时，该版本没有被其他已提交事务所覆盖 (v.endTS == 0 或 v.endTS >= tx.startTS)。
		//    或者，如果 v.endTS < tx.startTS，则意味着在事务开始时此版本已被覆盖，不可见。

		// 另外，创建该版本的事务 (v.txID) 不能是当前事务开始时还活跃的事务 (不在 tx.readView 中)。
		_, isActiveInReadView := tx.readView[v.txID]
		if isActiveInReadView {
			return false
		}
		isCommittedAtTxStart := tx.snapshot.txMgr.IsCommittedBefore(v.txID, tx.startTS) // 需要 TransactionManager 支持

		return isCommittedAtTxStart && v.beginTS < tx.startTS && (v.endTS == 0 || v.endTS >= tx.startTS)
	}
	return false
}

type MVCCBPlusTree struct {
	root      *MVCCNode // B+树的根节点
	order     int       // B+树的阶数
	txMgr     *TransactionManager
	lockMgr   *LockManager
	wal       *WALManager
	versionTS uint64       // 全局版本时间戳 (可能由TSO服务管理)
	mu        sync.RWMutex //保护树结构修改，如分裂合并，根节点变更
}

func NewMVCCBPlusTree(order int, txMgr *TransactionManager, lockMgr *LockManager, wal *WALManager) *MVCCBPlusTree {
	// 初始化空的B+树，根节点是一个叶子节点
	rootNode := &MVCCNode{
		isLeaf:   true,                   // 设置为叶子节点
		keys:     make([]Key, 0),         // 初始化空的键数组
		children: make([]interface{}, 0), // 初始化空的子节点数组
	}
	return &MVCCBPlusTree{
		root:    rootNode,
		order:   order,
		txMgr:   txMgr,
		lockMgr: lockMgr,
		wal:     wal,
	}
}

func (t *MVCCBPlusTree) GetRoot() *MVCCNode {
	return t.root
}

// Get 带事务的读取
func (t *MVCCBPlusTree) Get(tx *Transaction, key Key) (Value, bool) {
	// 注意：这里的锁获取和释放在MVCC下可能需要调整，
	// 例如ReadCommitted通常不需要S锁，RR和Serializable可能在事务开始时获取范围锁或在读取时获取。
	switch tx.isolation {
	case ReadUncommitted:
		// 1. 读取未提交：直接获取最新版本
		t.mu.RLock()
		defer t.mu.RUnlock()
		_, mvccNode := t.findMVCCNodeInLeaf(key)
		if mvccNode == nil {
			return nil, false
		}
		return mvccNode.GetValue(tx)
	case ReadCommitted:
		// 2. 读取已提交：获取事务开始时已提交的最新版本
		t.lockMgr.Acquire(tx.txID, key, LockShared)
		defer t.lockMgr.Release(tx.txID, key)

		t.mu.RLock()
		defer t.mu.RUnlock()
		return t.getVersionVisibleAt(key, tx)
	case RepeatableRead:
		// 3. 可重复读：基于快照读取
		if tx.snapshot != nil {
			return tx.snapshot.Get(tx, key)
		}

		// 获取快照锁并创建读取视图
		t.lockMgr.Acquire(tx.txID, key, LockShared)
		defer t.lockMgr.Release(tx.txID, key)

		t.mu.RLock()
		defer t.mu.RUnlock()
		return t.getVersionVisibleAt(key, tx)
	case Serializable:
		// 4. 序列化：使用谓词锁防止幻读
		if _, err := t.lockMgr.Acquire(tx.txID, key, LockShared); err != nil {
			return nil, false
		}
		defer t.lockMgr.Release(tx.txID, key)

		t.mu.RLock()
		defer t.mu.RUnlock()
		return t.getVersionVisibleAt(key, tx)
	default:
		return nil, false
	}
}

//// Put 带事务的写入
//func (t *MVCCBPlusTree) Put(tx *Transaction, key Key, value Value) error {
//	t.mu.Lock()
//	defer t.mu.Unlock()
//
//	// 在MVCC中，Put/Delete通常是创建新版本，而不是原地修改。
//	// 写锁通常在事务提交阶段的两阶段锁协议中获取，或者在操作时获取并持有到事务结束。
//	if _, err := t.lockMgr.Acquire(tx.txID, key, LockExclusive); err != nil { // X锁
//		return err
//	}
//	// Release 应该在事务结束 (commit/abort) 时，这里用 defer 只是简化
//	defer t.lockMgr.Release(tx.txID, key)
//
//	t.mu.RLock() // 读取树结构时加读锁
//	leaf, mvccNode := t.findMVCCNodeInLeaf(key)
//	t.mu.RUnlock()
//
//	if leaf == nil { // 意味着树是空的或者key的路径不存在，理论上findLeaf应该能处理空树
//		// 如果树是空的，需要创建根，然后插入
//		// 这里简化，假设findMVCCNodeInLeaf能找到或指示在哪里创建
//		return errors.New("failed to find or create leaf path for key")
//	}
//
//	if mvccNode == nil {
//		mvccNode = &MVCCNode{key: key}
//		// 将新的 MVCCNode 插入到 B+ 树的叶子节点中
//		// 这需要写锁保护树结构 t.mu.Lock()
//		t.mu.Lock()
//		newLeaf, newMVCCNode := t.insertIntoLeafAndCreateMVCCNode(key, mvccNode) // 此方法需要处理分裂等
//		t.mu.Unlock()
//		if newLeaf == nil || newMVCCNode == nil {
//			return errors.New("failed to insert new MVCCNode into B+ tree")
//		}
//		mvccNode = newMVCCNode // 使用实际插入或找到的节点
//	}
//
//	// 添加新版本到MVCCNode
//	mvccNode.AddVersion(value, tx)
//
//	// 记录写操作到事务的write set
//	tx.recordWrite(key, value)
//	return nil
//}

// Put 带事务的写入
func (t *MVCCBPlusTree) Put(tx *Transaction, key Key, value Value) error {
	// 1. 获取树结构的写锁
	t.mu.Lock()
	defer t.mu.Unlock()

	// 2. 获取排他锁并注册到事务的写集合
	if _, err := t.lockMgr.Acquire(tx.txID, key, LockExclusive); err != nil {
		return fmt.Errorf("acquire X lock failed: %w", err)
	}
	tx.recordWrite(key, value)

	// 3. WAL日志记录
	if t.wal != nil {
		if err := t.wal.LogWrite(tx.txID, key, value); err != nil {
			return fmt.Errorf("WAL log write failed: %w", err)
		}
	}

	// 4. 查找或创建MVCC节点
	leaf, mvccNode := t.findMVCCNodeInLeaf(key)
	if mvccNode == nil {
		var err error
		leaf, mvccNode, err = t.createNewMVCCNode(key)
		if err != nil {
			return fmt.Errorf("create MVCC node failed: %w", err)
		}
	}

	// 5. 添加新版本到MVCC节点
	mvccNode.mu.Lock()
	defer mvccNode.mu.Unlock()

	newVersion := &Version{
		value:   value,
		txID:    tx.txID,
		beginTS: tx.startTS,
		prev:    mvccNode.versions,
		endTS:   0, // 新版本默认为当前版本
	}

	// 更新旧版本的endTS
	if mvccNode.versions != nil {
		mvccNode.versions.endTS = tx.startTS
	}

	mvccNode.versions = newVersion

	// 6. 处理节点分裂
	if len(leaf.keys) >= t.order {
		t.handleLeafSplit(leaf)
	}

	return nil
}

// 新增辅助方法
func (t *MVCCBPlusTree) createNewMVCCNode(key Key) (*MVCCNode, *MVCCNode, error) {
	newNode := &MVCCNode{
		key:      key,
		versions: nil,
		isLeaf:   true,
		keys:     []Key{key},
		children: make([]interface{}, 0),
	}

	// 插入到B+树结构中
	if err := t.insertNode(newNode); err != nil {
		return nil, nil, err
	}
	return newNode, newNode, nil
}

func (t *MVCCBPlusTree) handleLeafSplit(leaf *MVCCNode) {
	// 具体分裂逻辑（需要保持树结构一致性）
	mid := len(leaf.keys) / 2
	rightNode := &MVCCNode{
		isLeaf:   true,
		keys:     leaf.keys[mid:],
		children: leaf.children[mid:],
		next:     leaf.next,
	}

	// 更新原节点
	leaf.keys = leaf.keys[:mid]
	leaf.children = leaf.children[:mid]
	leaf.next = rightNode

	// 更新父节点（修复参数错误：promotedKey 应为 rightNode 的第一个键）
	t.updateParent(leaf, rightNode, rightNode.keys[0])
}

// insertNode 将新节点插入到B+树中（处理叶子和内部节点插入）
func (t *MVCCBPlusTree) insertNode(newNode *MVCCNode) error {
	// 从根节点开始查找插入位置
	current := t.root
	for {
		if current.isLeaf {
			// 叶子节点插入逻辑
			return t.insertIntoLeaf(current, newNode)
		}
		// 内部节点：找到子节点指针
		idx := t.findInsertIndex(current.keys, newNode.keys[0])
		child, ok := current.children[idx].(*MVCCNode)
		if !ok {
			return fmt.Errorf("invalid child type in internal node")
		}
		current = child
	}
}

// insertIntoLeaf 在叶子节点中插入新节点（处理键和子节点的插入）
func (t *MVCCBPlusTree) insertIntoLeaf(leaf *MVCCNode, newNode *MVCCNode) error {
	// 找到插入位置
	idx := t.findInsertIndex(leaf.keys, newNode.keys[0])

	// 插入键和子节点
	leaf.keys = append(leaf.keys[:idx], append([]Key{newNode.keys[0]}, leaf.keys[idx:]...)...)
	leaf.children = append(leaf.children[:idx], append([]interface{}{newNode}, leaf.children[idx:]...)...)

	// 检查是否需要分裂
	if len(leaf.keys) > t.order-1 {
		t.splitLeaf(leaf)
	}
	return nil
}

// findInsertIndex 找到键的插入位置
func (t *MVCCBPlusTree) findInsertIndex(keys []Key, target Key) int {
	idx := 0
	for idx < len(keys) && compareKeys(target, keys[idx]) > 0 {
		idx++
	}
	return idx
}

// updateParent 更新父节点的键和子节点指针（处理分裂后的父节点同步）
func (t *MVCCBPlusTree) updateParent(oldChild, newChild *MVCCNode, promotedKey Key) {
	parent := t.findParent(t.root, oldChild)
	if parent == nil {
		// 旧子节点是根节点，创建新根
		newRoot := &MVCCNode{
			isLeaf:   false,
			keys:     []Key{promotedKey},
			children: []interface{}{oldChild, newChild},
		}
		t.root = newRoot
		return
	}

	// 找到旧子节点在父节点中的位置
	idx := t.findChildIndex(parent, oldChild)
	if idx == -1 {
		return // 未找到父节点中的子节点（异常情况）
	}

	// 插入提升的键和新子节点
	parent.keys = append(parent.keys[:idx], append([]Key{promotedKey}, parent.keys[idx:]...)...)
	parent.children = append(parent.children[:idx+1], parent.children[idx:]...)
	parent.children[idx+1] = newChild

	// 检查父节点是否需要分裂
	if len(parent.keys) > t.order-1 {
		t.splitInternalNode(parent)
	}
}

// findChildIndex 找到子节点在父节点中的索引
func (t *MVCCBPlusTree) findChildIndex(parent *MVCCNode, child *MVCCNode) int {
	for i, c := range parent.children {
		if c == child {
			return i
		}
	}
	return -1
}

// Delete 带事务的删除 (通过插入一个nil值的版本来实现，即墓碑)
func (t *MVCCBPlusTree) Delete(tx *Transaction, key Key) error {
	return t.Put(tx, key, nil) // 插入一个value为nil的版本作为删除标记
}

// getLatestVersionFromTree 获取指定key的最新版本 (不管提交状态，用于ReadUncommitted)
func (t *MVCCBPlusTree) getLatestVersionFromTree(key Key, tx *Transaction) (Value, bool) {
	t.mu.RLock() // 保护树的遍历
	defer t.mu.RUnlock()

	_, mvccNode := t.findMVCCNodeInLeaf(key)
	if mvccNode == nil {
		return nil, false
	}

	mvccNode.mu.RLock()
	defer mvccNode.mu.RUnlock()

	// 优先返回自己事务中未提交的版本
	for v := mvccNode.versions; v != nil; v = v.prev {
		if v.txID == tx.txID {
			if v.value == nil {
				return nil, false
			} // 自己的删除操作
			return v.value, true
		}
	}
	// 否则返回最新的版本 (endTS == 0)
	if mvccNode.versions != nil && mvccNode.versions.endTS == 0 {
		if mvccNode.versions.value == nil {
			return nil, false
		} // 最新的是删除标记
		return mvccNode.versions.value, true
	}
	return nil, false
}

// getVersionVisibleAt 获取在事务tx看来可见的版本
func (t *MVCCBPlusTree) getVersionVisibleAt(key Key, tx *Transaction) (Value, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	_, mvccNode := t.findMVCCNodeInLeaf(key)
	if mvccNode == nil {
		return nil, false
	}
	return mvccNode.GetValue(tx) // MVCCNode.GetValue 内部处理可见性逻辑
}

// findMVCCNodeInLeaf 在B+树中查找key对应的叶子节点和MVCCNode
// 返回叶子节点和MVCCNode。如果MVCCNode不存在，则返回nil
func (t *MVCCBPlusTree) findMVCCNodeInLeaf(key Key) (*MVCCNode, *MVCCNode) {
	leaf := t.findLeaf(t.root, key) // findLeaf 是标准的B+树查找叶子节点操作
	if leaf == nil {
		return nil, nil
	}
	// 在叶子节点中查找MVCCNode
	for i, k := range leaf.keys {
		// Key需要能比较，这里假设 k == key
		if compareKeys(k, key) == 0 { // compareKeys(k1, k2) -> -1 if k1<k2, 0 if k1==k2, 1 if k1>k2
			if mvccNode, ok := leaf.children[i].(*MVCCNode); ok {
				return leaf, mvccNode
			}
			// 如果children[i]不是*MVCCNode，说明数据结构有问题或未初始化
			return leaf, nil // 或者panic
		}
	}
	return leaf, nil // Key不在叶子节点中
}

// findLeaf 查找包含key的叶子节点 (标准B+树操作)
func (t *MVCCBPlusTree) findLeaf(node *MVCCNode, key Key) *MVCCNode {
	if node == nil {
		return nil
	}
	currentNode := node
	for !currentNode.isLeaf {
		// 找到合适的子节点指针
		i := 0
		for i < len(currentNode.keys) && compareKeys(key, currentNode.keys[i]) >= 0 {
			i++
		}
		child, ok := currentNode.children[i].(*MVCCNode)
		if !ok || child == nil {
			return nil // 树结构错误
		}
		currentNode = child
	}
	return currentNode
}

// insertIntoLeafAndCreateMVCCNode 将新的MVCCNode插入到B+树的叶子节点
// 如果key已存在于某个MVCCNode中，则返回该节点；否则创建并插入新的MVCCNode。
// 返回实际的叶子节点（可能因分裂而改变）和对应的MVCCNode。
// 这个方法需要处理B+树的插入逻辑，包括节点分裂、父节点更新等，且必须是线程安全的。
func (t *MVCCBPlusTree) insertIntoLeafAndCreateMVCCNode(key Key, newNodeData *MVCCNode) (*MVCCNode, *MVCCNode) {
	// 实际的B+树插入逻辑会复杂得多，这里只是一个高度简化的示意
	// 需要 t.mu.Lock() 保护
	leaf := t.findLeaf(t.root, key)

	// 检查key是否已存在
	for i, k := range leaf.keys {
		if compareKeys(k, key) == 0 {
			if existingMVCCNode, ok := leaf.children[i].(*MVCCNode); ok {
				return leaf, existingMVCCNode // Key已存在，返回现有的MVCCNode
			}
			// 类型错误，叶子节点的children应该是MVCCNode
			panic("B+ tree leaf child is not MVCCNode")
		}
	}

	// Key不存在，插入新的MVCCNode (newNodeData)
	// 这部分需要完整的B+树插入逻辑：insert_into_leaf, insert_into_parent, split等
	// ... (B+树插入和分裂逻辑) ...
	// 假设插入成功，并且newNodeData被放入了某个叶子节点 leaf.children[idx]
	// 如果发生分裂，leaf 和 root 可能改变

	// 简化：直接在找到的leaf中插入（不处理分裂）
	idxToInsert := 0
	for idxToInsert < len(leaf.keys) && compareKeys(key, leaf.keys[idxToInsert]) > 0 {
		idxToInsert++
	}
	// 插入key
	leaf.keys = append(leaf.keys[:idxToInsert], append([]Key{key}, leaf.keys[idxToInsert:]...)...)
	// 插入MVCCNode
	leaf.children = append(leaf.children[:idxToInsert], append([]interface{}{newNodeData}, leaf.children[idxToInsert:]...)...)

	// 如果叶子节点满了，需要分裂
	if len(leaf.keys) > t.order-1 {
		t.splitLeaf(leaf)
	}

	return leaf, newNodeData // 返回修改后的叶子和新插入的MVCCNode
}

// splitLeaf 分裂叶子节点
func (t *MVCCBPlusTree) splitLeaf(leaf *MVCCNode) {
	// 确保已经持有树的写锁

	// 计算分裂点
	midIdx := t.order / 2

	// 创建新的右侧叶子节点
	rightLeaf := &MVCCNode{
		isLeaf:   true,
		keys:     append([]Key{}, leaf.keys[midIdx:]...),
		children: append([]interface{}{}, leaf.children[midIdx:]...),
		next:     leaf.next,
	}

	// 更新原叶子节点
	leaf.keys = leaf.keys[:midIdx]
	leaf.children = leaf.children[:midIdx]
	leaf.next = rightLeaf
	// 要插入到父节点的键
	newKey := rightLeaf.keys[0]

	// 将新节点的第一个键插入到父节点
	// 如果没有父节点（即当前节点是根节点），则创建新的根节点
	if leaf == t.root {
		newRoot := &MVCCNode{
			isLeaf:   false,
			keys:     []Key{newKey},
			children: []interface{}{leaf, rightLeaf},
		}
		t.root = newRoot
	} else {
		// 找到父节点并插入新键和子节点指针
		parent := t.findParent(t.root, leaf)
		if parent == nil {
			// 如果找不到父节点，说明树结构有问题
			panic("Cannot find parent node in B+ tree")
		}
		// 在父节点中插入新键和右侧子节点
		t.insertIntoParent(parent, leaf, newKey, rightLeaf)
	}
}

// insertIntoParent 将新键和子节点插入到父节点中，必要时递归分裂父节点
func (t *MVCCBPlusTree) insertIntoParent(parent, leftChild *MVCCNode, newKey Key, rightChild *MVCCNode) {
	if parent == nil {
		return
	}

	// 找到插入位置
	insertIndex := 0
	for insertIndex < len(parent.keys) && compareKeys(newKey, parent.keys[insertIndex]) > 0 {
		insertIndex++
	}

	// 插入新键
	parent.keys = append(parent.keys[:insertIndex], append([]Key{newKey}, parent.keys[insertIndex:]...)...)
	// 插入新的子节点指针
	parent.children = append(parent.children[:insertIndex+1], parent.children[insertIndex:]...)
	parent.children[insertIndex+1] = rightChild // 原代码这里逻辑有误，应改为正确插入新节点

	// 检查父节点是否需要分裂
	if len(parent.keys) > t.order-1 {
		t.splitInternalNode(parent)
	}
}

func (t *MVCCBPlusTree) RangeQuery(tx *Transaction, start, end Key) []Value {
	var result []Value

	// 找到起始键所在的叶子节点
	startLeaf := t.findLeaf(t.root, start)
	if startLeaf == nil {
		return result
	}

	// 定位起始键在叶子节点中的位置
	startIndex := 0
	for startIndex < len(startLeaf.keys) && compareKeys(startLeaf.keys[startIndex], start) < 0 {
		startIndex++
	}

	currentLeaf := startLeaf
	currentIndex := startIndex

	for currentLeaf != nil {
		for i := currentIndex; i < len(currentLeaf.keys); i++ {
			key := currentLeaf.keys[i]
			// 检查是否超过结束键
			if compareKeys(key, end) > 0 {
				return result
			}

			// 获取 MVCCNode
			mvccNode, ok := currentLeaf.children[i].(*MVCCNode)
			if !ok {
				continue
			}

			// 获取对事务可见的值
			value, ok := mvccNode.GetValue(tx)
			if ok {
				result = append(result, value)
			}
		}

		// 移动到下一个叶子节点
		currentLeaf = currentLeaf.next
		currentIndex = 0
	}

	return result
}

// splitInternalNode 分裂内部节点
func (t *MVCCBPlusTree) splitInternalNode(node *MVCCNode) {
	// 计算分裂点
	midIdx := t.order / 2

	// 创建新的右侧内部节点
	rightNode := &MVCCNode{
		isLeaf:   false,
		keys:     append([]Key{}, node.keys[midIdx+1:]...),
		children: append([]interface{}{}, node.children[midIdx+1:]...),
	}

	// 提升的键
	promotedKey := node.keys[midIdx]

	// 更新原节点
	node.keys = node.keys[:midIdx]
	node.children = node.children[:midIdx+1]

	if node == t.root {
		// 如果当前节点是根节点，创建新的根节点
		newRoot := &MVCCNode{
			isLeaf:   false,
			keys:     []Key{promotedKey},
			children: []interface{}{node, rightNode},
		}
		t.root = newRoot
	} else {
		// 找到父节点并插入提升的键和新的子节点指针
		parent := t.findParent(t.root, node)
		t.insertIntoParent(parent, node, promotedKey, rightNode)
	}
}

// findParent 查找指定节点的父节点
func (t *MVCCBPlusTree) findParent(current, child *MVCCNode) *MVCCNode {
	if current == nil || current.isLeaf {
		return nil
	}

	for _, c := range current.children {
		childNode := c.(*MVCCNode)
		if childNode == child {
			return current
		}
		if !childNode.isLeaf {
			if parent := t.findParent(childNode, child); parent != nil {
				return parent
			}
		}
	}
	return nil
}

// cloneNode 递归深拷贝节点
func cloneNode(node *MVCCNode) *MVCCNode {
	if node == nil {
		return nil
	}
	newNode := &MVCCNode{
		isLeaf:   node.isLeaf,
		keys:     append([]Key{}, node.keys...),
		children: make([]interface{}, len(node.children)),
		next:     node.next, // 叶子节点的 next 指针保持不变
	}
	for i, child := range node.children {
		if node.isLeaf {
			newNode.children[i] = child
		} else {
			newNode.children[i] = cloneNode(child.(*MVCCNode))
		}
	}
	return newNode
}

// commitTransaction 实现事务提交逻辑
func (t *PersistentBPlusTree) commitTransaction(tx *Transaction) error {
	// 阶段1: 准备阶段 (Prepare Phase)
	// 1.1 验证事务状态
	if tx.status != TxActive {
		return errors.New("transaction is not active")
	}

	// 1.2 验证可串行化隔离级别下的读写冲突
	if tx.isolation == Serializable {
		if err := t.txMgr.validateSerializable(tx); err != nil {
			return err
		}
	}

	// 1.3 获取树锁，确保事务提交期间树结构不变
	t.mu.Lock()
	defer t.mu.Unlock()

	// 1.4 写入Prepare记录到WAL
	if t.wal != nil {
		if err := t.wal.Prepare(tx); err != nil {
			return fmt.Errorf("WAL Prepare failed: %w", err)
		}

		// 强制刷盘确保Prepare记录持久化
		if err := t.wal.Sync(); err != nil {
			return fmt.Errorf("WAL Prepare sync failed: %w", err)
		}
	}

	// 阶段2: 提交阶段 (Commit Phase)
	// 2.1 写入Commit记录到WAL
	if t.wal != nil {
		if err := t.wal.Commit(tx.txID); err != nil {
			return fmt.Errorf("WAL Commit failed: %w", err)
		}
	}

	// 2.2 更新事务状态为已提交
	tx.status = TxCommitted

	// 2.3 确保所有修改已写入磁盘
	if err := t.disk.Sync(); err != nil {
		return fmt.Errorf("disk sync failed: %w", err)
	}

	// 2.4 释放事务持有的所有锁
	if t.lockMgr != nil {
		for _, writeOp := range tx.writes {
			t.lockMgr.Release(tx.txID, writeOp.Key)
		}
	}

	// 2.5 从TransactionManager中标记事务为已提交
	if t.txMgr != nil {
		if err := t.txMgr.MarkCommitted(tx.txID); err != nil {
			return fmt.Errorf("failed to mark transaction as committed: %w", err)
		}
	}

	return nil
}

// makeVersionsVisible 使事务的修改对其他事务可见
func (t *MVCCBPlusTree) makeVersionsVisible(tx *Transaction) {
	// 遍历事务的写操作集合
	for _, writeOp := range tx.writes {
		// 查找对应的 MVCCNode
		_, mvccNode := t.findMVCCNodeInLeaf(writeOp.Key)
		if mvccNode == nil {
			continue
		}

		mvccNode.mu.Lock()
		// 遍历版本链，找到该事务创建的版本
		for v := mvccNode.versions; v != nil; v = v.prev {
			if v.txID == tx.txID {
				// 确保版本的开始时间戳正确设置
				if v.beginTS == 0 {
					v.beginTS = tx.startTS
				}
				// 版本的 endTS 保持为 0，表示当前有效版本
				v.endTS = 0
			}
		}
		mvccNode.mu.Unlock()
	}

	// 触发版本链的垃圾回收
	go t.scheduleVersionGC(tx)
}

// scheduleVersionGC 调度版本垃圾回收
func (t *MVCCBPlusTree) scheduleVersionGC(tx *Transaction) {
	// 异步清理不再需要的旧版本
	// 这是一个后台任务，可以延迟执行
	time.Sleep(100 * time.Millisecond) // 简单延迟

	// 获取所有叶子节点
	leaves := t.getAllLeafNodes()

	// 遍历所有叶子节点
	for _, leaf := range leaves {
		for _, child := range leaf.children {
			if mvccNode, ok := child.(*MVCCNode); ok {
				mvccNode.mu.Lock()
				// 清理不可见的旧版本
				mvccNode.versions = t.cleanInvisibleVersions(mvccNode.versions, tx)
				mvccNode.mu.Unlock()
			}
		}
	}
}

// getAllLeafNodes 获取所有叶子节点
func (t *MVCCBPlusTree) getAllLeafNodes() []*MVCCNode {
	var leaves []*MVCCNode
	if t.root == nil {
		return leaves
	}

	current := t.root
	for !current.isLeaf {
		current = current.children[0].(*MVCCNode)
	}

	for current != nil {
		leaves = append(leaves, current)
		current = current.next
	}

	return leaves
}

// cleanInvisibleVersions 清理不可见的旧版本
func (t *MVCCBPlusTree) cleanInvisibleVersions(version *Version, tx *Transaction) *Version {
	// 找到第一个可见的版本
	for version != nil && !tx.isVisible(version) {
		version = version.prev
	}

	// 从第一个可见的版本开始，清理后续不可见的版本
	if version != nil {
		prev := version
		current := version.prev
		for current != nil {
			if tx.isVisible(current) {
				prev = current
			} else {
				prev.prev = current.prev
			}
			current = current.prev
		}
	}

	return version
}

// rollbackTransaction 实现事务回滚
func (t *MVCCBPlusTree) rollbackTransaction(tx *Transaction) error {
	// 对于MVCC，回滚主要是标记事务为已中止，并清理其影响。
	// 已经创建的版本由于其txID是中止事务的ID，所以对其他事务不可见（除了ReadUncommitted）。
	// 可能需要将这些"无效"版本从版本链中移除或标记为无效，以供后续清理（垃圾回收）。

	// 记录Abort到WAL
	if t.wal != nil {
		if err := t.wal.Abort(tx.txID); err != nil {
			return fmt.Errorf("WAL Abort failed: %w", err)
		}
	}

	// 释放锁
	if t.lockMgr != nil {
		for _, writeOp := range tx.writes {
			t.lockMgr.Release(tx.txID, writeOp.Key)
		}
	}

	// 从TransactionManager中移除或标记事务为已中止
	if t.txMgr != nil {
		if err := t.txMgr.MarkAborted(tx.txID); err != nil {
			return fmt.Errorf("failed to mark transaction as aborted: %w", err)
		}
	}

	// 清理该事务产生的版本
	t.mu.RLock() // 读锁保护树结构
	defer t.mu.RUnlock()

	// 遍历tx.writes，找到对应的MVCCNode，然后移除txID为此事务ID的版本
	for _, writeOp := range tx.writes {
		_, mvccNode := t.findMVCCNodeInLeaf(writeOp.Key)
		if mvccNode == nil {
			continue
		}

		mvccNode.mu.Lock()
		// 处理版本链
		if mvccNode.versions != nil {
			// 如果头部版本是当前事务创建的，直接移除
			if mvccNode.versions.txID == tx.txID {
				mvccNode.versions = mvccNode.versions.prev
			} else {
				// 否则遍历版本链查找
				current := mvccNode.versions
				for current != nil && current.prev != nil {
					if current.prev.txID == tx.txID {
						// 跳过当前事务创建的版本
						current.prev = current.prev.prev
						break
					}
					current = current.prev
				}
			}
		}
		mvccNode.mu.Unlock()
	}

	// 记录回滚日志
	log.Info("Transaction %d rolled back\n", tx.txID)
	return nil
}

// validateConsistency 校验数据一致性
func (t *MVCCBPlusTree) validateConsistency() error {
	// 获取树的读锁，确保验证过程中树结构不变
	t.mu.RLock()
	defer t.mu.RUnlock()

	// 验证B+树结构
	if err := t.validateTreeStructure(t.root); err != nil {
		return fmt.Errorf("tree structure validation failed: %w", err)
	}

	// 验证版本链
	if err := t.validateVersionChains(); err != nil {
		return fmt.Errorf("version chains validation failed: %w", err)
	}

	// 验证事务状态一致性
	if t.txMgr != nil {
		if err := t.validateTransactionConsistency(); err != nil {
			return fmt.Errorf("transaction consistency validation failed: %w", err)
		}
	}

	return nil
}

func (t *MVCCBPlusTree) validateTreeStructure(node *MVCCNode) error {
	if node == nil {
		return nil
	}

	// 检查节点键值对数量是否合法
	// 根节点可以有较少的键，非根内部节点必须至少半满
	isRoot := node == t.root
	if !isRoot && !node.isLeaf && len(node.keys) < (t.order/2) {
		return fmt.Errorf("internal node underfilled: has %d keys, minimum is %d",
			len(node.keys), t.order/2)
	}

	// 检查键是否有序
	for i := 1; i < len(node.keys); i++ {
		if compareKeys(node.keys[i-1], node.keys[i]) >= 0 {
			return fmt.Errorf("keys not in order at index %d: %v >= %v",
				i, node.keys[i-1], node.keys[i])
		}
	}

	// 检查子节点数量是否正确
	if !node.isLeaf {
		if len(node.children) != len(node.keys)+1 {
			return fmt.Errorf("internal node has %d keys but %d children",
				len(node.keys), len(node.children))
		}

		// 递归检查所有子节点
		childDepths := make([]int, len(node.children))
		for i, child := range node.children {
			mvccChild, ok := child.(*MVCCNode)
			if !ok || mvccChild == nil {
				return fmt.Errorf("child at index %d is not a valid MVCCNode", i)
			}

			// 递归验证子节点
			if err := t.validateTreeStructure(mvccChild); err != nil {
				return fmt.Errorf("child at index %d invalid: %w", i, err)
			}

			// 计算子树深度（用于后续验证所有叶子节点是否在同一层）
			if mvccChild.isLeaf {
				childDepths[i] = 1
			} else {
				// 这里简化处理，实际实现可能需要额外的深度计算函数
				childDepths[i] = 2 // 占位，实际应该计算真实深度
			}
		}

		// 验证所有子树深度相同（所有叶子节点在同一层）
		for i := 1; i < len(childDepths); i++ {
			if childDepths[i] != childDepths[0] {
				return fmt.Errorf("leaf nodes not at same level: depths %v", childDepths)
			}
		}
	} else {
		// 叶子节点的children数组应该与keys数组长度相同
		if len(node.children) != len(node.keys) {
			return fmt.Errorf("leaf node has %d keys but %d values",
				len(node.keys), len(node.children))
		}

		// 检查叶子节点的每个值是否是有效的MVCCNode
		for i, child := range node.children {
			mvccNode, ok := child.(*MVCCNode)
			if !ok || mvccNode == nil {
				return fmt.Errorf("value at index %d is not a valid MVCCNode", i)
			}
		}
	}

	return nil
}

// 辅助函数：找到最左侧的叶子节点
func (t *MVCCBPlusTree) findLeftmostLeaf(node *MVCCNode) *MVCCNode {
	if node == nil {
		return nil
	}

	current := node
	for !current.isLeaf {
		if len(current.children) == 0 {
			return nil // 内部节点没有子节点，树结构有问题
		}

		child, ok := current.children[0].(*MVCCNode)
		if !ok || child == nil {
			return nil // 子节点类型错误
		}

		current = child
	}

	return current
}

// 验证事务状态一致性
func (t *MVCCBPlusTree) validateTransactionConsistency() error {
	if t.txMgr == nil {
		return nil
	}

	// 获取事务管理器的读锁，确保在检查期间事务状态不变
	t.txMgr.mu.RLock()
	defer t.txMgr.mu.RUnlock()

	// 检查活跃事务的写集合是否与树中的版本一致
	for _, tx := range t.txMgr.GetActiveTxs() {
		// 只检查活跃状态的事务
		if tx.status != TxActive {
			continue
		}

		for _, writeOp := range tx.writes {
			_, mvccNode := t.findMVCCNodeInLeaf(writeOp.Key)
			if mvccNode == nil {
				return fmt.Errorf("transaction %d has write for key %v but key not found in tree",
					tx.txID, writeOp.Key)
			}

			// 检查事务的写操作是否反映在版本链中
			mvccNode.mu.RLock()
			found := false
			for v := mvccNode.versions; v != nil; v = v.prev {
				if v.txID == tx.txID {
					found = true
					break
				}
			}
			mvccNode.mu.RUnlock()

			if !found {
				return fmt.Errorf("active transaction %d has write for key %v but no corresponding version found",
					tx.txID, writeOp.Key)
			}
		}
	}

	return nil
}

// validateVersionChains 验证所有 MVCCNode 的版本链是否有效
func (t *MVCCBPlusTree) validateVersionChains() error {
	// 获取所有叶子节点
	leaves := t.getAllLeafNodes()

	// 遍历所有叶子节点
	for _, leaf := range leaves {
		for i, child := range leaf.children {
			if mvccNode, ok := child.(*MVCCNode); ok {
				mvccNode.mu.RLock()
				// 验证当前 MVCCNode 的版本链
				if err := t.validateSingleVersionChain(mvccNode.versions); err != nil {
					mvccNode.mu.RUnlock()
					return fmt.Errorf("节点 %v (索引 %d) 的版本链验证失败: %w", mvccNode.key, i, err)
				}

				// 验证版本链中的事务ID是否有效
				if t.txMgr != nil {
					for v := mvccNode.versions; v != nil; v = v.prev {
						// 检查事务ID是否在有效范围内
						if v.txID > t.txMgr.nextTxID {
							mvccNode.mu.RUnlock()
							return fmt.Errorf("节点 %v 的版本 (txID=%d) 使用了未分配的事务ID", mvccNode.key, v.txID)
						}
					}
				}

				// 验证版本链的完整性
				if err := t.validateVersionChainCompleteness(mvccNode.versions); err != nil {
					mvccNode.mu.RUnlock()
					return fmt.Errorf("节点 %v 的版本链完整性验证失败: %w", mvccNode.key, err)
				}

				mvccNode.mu.RUnlock()
			} else {
				return fmt.Errorf("叶子节点的子节点不是有效的MVCCNode类型: 索引 %d", i)
			}
		}
	}

	return nil
}

// validateSingleVersionChain 验证单个 MVCCNode 的版本链是否有效
func (t *MVCCBPlusTree) validateSingleVersionChain(version *Version) error {
	var prevBeginTS uint64 = 0
	var firstIteration bool = true

	for v := version; v != nil; v = v.prev {
		// 验证开始时间戳是否小于结束时间戳
		if v.beginTS >= v.endTS && v.endTS != 0 {
			return fmt.Errorf("版本 %d 的开始时间戳 (%d) 大于或等于结束时间戳 (%d)", v.txID, v.beginTS, v.endTS)
		}

		// 验证版本链按beginTS降序排列
		if !firstIteration {
			if v.beginTS >= prevBeginTS {
				return fmt.Errorf("版本链顺序错误: 版本 %d (beginTS=%d) 应该早于前一个版本 (beginTS=%d)",
					v.txID, v.beginTS, prevBeginTS)
			}
		}

		// 记录当前版本的beginTS用于下一次迭代比较
		prevBeginTS = v.beginTS
		firstIteration = false
	}

	return nil
}

// validateVersionChainCompleteness 验证版本链的完整性
func (t *MVCCBPlusTree) validateVersionChainCompleteness(version *Version) error {
	if version == nil {
		return nil // 空版本链是有效的
	}

	// 检查版本链中是否存在时间戳重叠
	versions := make([]*Version, 0)
	for v := version; v != nil; v = v.prev {
		versions = append(versions, v)
	}

	for i := 0; i < len(versions); i++ {
		for j := i + 1; j < len(versions); j++ {
			vi := versions[i]
			vj := versions[j]

			// 检查时间戳区间是否重叠
			viEnd := vi.endTS
			if viEnd == 0 { // 当前活跃版本
				viEnd = ^uint64(0) // 最大值表示无限
			}

			vjEnd := vj.endTS
			if vjEnd == 0 {
				vjEnd = ^uint64(0)
			}

			// 检查重叠: [vi.beginTS, viEnd) 与 [vj.beginTS, vjEnd) 是否有交集
			if vi.beginTS < vjEnd && vj.beginTS < viEnd {
				return fmt.Errorf("版本时间戳重叠: 版本 %d [%d,%d) 与版本 %d [%d,%d)",
					vi.txID, vi.beginTS, viEnd, vj.txID, vj.beginTS, vjEnd)
			}
		}
	}

	return nil
}
func (t *MVCCBPlusTree) Clone() *MVCCBPlusTree {
	newTree := &MVCCBPlusTree{
		order: t.order,
		root:  cloneNode(t.root),
	}
	return newTree
}

// compareKeys 是一个比较键的函数，你需要根据你的Key类型来实现它
// 返回 -1 (k1 < k2), 0 (k1 == k2), 1 (k1 > k2)
func compareKeys(k1, k2 Key) int {
	if k1.Less(k2) {
		return -1
	}
	if !k1.Less(k2) {
		return 1
	}
	return 0
}
