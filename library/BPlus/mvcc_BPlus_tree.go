package bplus

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

type MccKey interface{}   // 键类型，应具体化并实现比较
type MccValue interface{} // 值类型
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
	keys     []MccKey      // 键数组
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
		keys:     make([]MccKey, 0),      // 初始化空的键数组
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

// Get 带事务的读取
func (t *MVCCBPlusTree) Get(tx *Transaction, key Key) (Value, bool) {
	// 注意：这里的锁获取和释放在MVCC下可能需要调整，
	// 例如ReadCommitted通常不需要S锁，RR和Serializable可能在事务开始时获取范围锁或在读取时获取。
	switch tx.isolation {
	case ReadUncommitted:
		// ReadUncommitted 读取最新的版本，不管是否提交
		return t.getLatestVersionFromTree(key, tx) // 传入tx是为了能看到自己的未提交修改
	case ReadCommitted:
		// ReadCommitted 读取事务开始时已提交的最新版本
		t.lockMgr.Acquire(tx.txID, key, LockShared) // RC通常不加读锁
		defer t.lockMgr.Release(tx.txID, key)
		return t.getVersionVisibleAt(key, tx)
	case RepeatableRead:
		// RepeatableRead 基于事务开始时的快照读取
		// 如果实现了快照机制，则从快照读取
		if tx.snapshot != nil {
			// return tx.snapshot.Get(tx, key) // 快照Get也需要事务上下文
			// 简化：直接读取对当前事务可见的版本
			return t.getVersionVisibleAt(key, tx)
		}
		// 如果没有快照，则读取事务开始时可见的版本
		t.lockMgr.Acquire(tx.txID, key, LockShared) // 可能需要
		defer t.lockMgr.Release(tx.txID, key)
		return t.getVersionVisibleAt(key, tx)
	case Serializable:
		// Serializable 类似RepeatableRead，但有更强的锁机制防止幻读
		// 通常在扫描范围时使用谓词锁或范围锁
		t.lockMgr.Acquire(tx.txID, key, LockShared) // S锁
		defer t.lockMgr.Release(tx.txID, key)
		return t.getVersionVisibleAt(key, tx)
	default:
		return nil, false
	}
}

// Put 带事务的写入
func (t *MVCCBPlusTree) Put(tx *Transaction, key Key, value Value) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// 在MVCC中，Put/Delete通常是创建新版本，而不是原地修改。
	// 写锁通常在事务提交阶段的两阶段锁协议中获取，或者在操作时获取并持有到事务结束。
	if _, err := t.lockMgr.Acquire(tx.txID, key, LockExclusive); err != nil { // X锁
		return err
	}
	// Release 应该在事务结束 (commit/abort) 时，这里用 defer 只是简化
	defer t.lockMgr.Release(tx.txID, key)

	t.mu.RLock() // 读取树结构时加读锁
	leaf, mvccNode := t.findMVCCNodeInLeaf(key)
	t.mu.RUnlock()

	if leaf == nil { // 意味着树是空的或者key的路径不存在，理论上findLeaf应该能处理空树
		// 如果树是空的，需要创建根，然后插入
		// 这里简化，假设findMVCCNodeInLeaf能找到或指示在哪里创建
		return errors.New("failed to find or create leaf path for key")
	}

	if mvccNode == nil {
		mvccNode = &MVCCNode{key: key}
		// 将新的 MVCCNode 插入到 B+ 树的叶子节点中
		// 这需要写锁保护树结构 t.mu.Lock()
		t.mu.Lock()
		newLeaf, newMVCCNode := t.insertIntoLeafAndCreateMVCCNode(key, mvccNode) // 此方法需要处理分裂等
		t.mu.Unlock()
		if newLeaf == nil || newMVCCNode == nil {
			return errors.New("failed to insert new MVCCNode into B+ tree")
		}
		mvccNode = newMVCCNode // 使用实际插入或找到的节点
	}

	// 添加新版本到MVCCNode
	mvccNode.AddVersion(value, tx)

	// 记录写操作到事务的write set
	tx.recordWrite(key, value)
	return nil
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
	leaf.keys = append(leaf.keys[:idxToInsert], append([]MccKey{key}, leaf.keys[idxToInsert:]...)...)
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
		keys:     append([]MccKey{}, leaf.keys[midIdx:]...),
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
			keys:     []MccKey{rightLeaf.keys[0]},
			children: []interface{}{leaf, rightLeaf},
		}
		t.root = newRoot
	} else {
		// 找到父节点并插入新键和子节点指针
		parent := t.findParent(t.root, leaf)
		t.insertIntoParent(parent, leaf, newKey, rightLeaf)
	}
}

// insertIntoParent 将新键和子节点插入到父节点中，必要时递归分裂父节点
func (t *MVCCBPlusTree) insertIntoParent(parent, leftChild *MVCCNode, newKey MccKey, rightChild *MVCCNode) {
	if parent == nil {
		return
	}

	// 找到插入位置
	insertIndex := 0
	for insertIndex < len(parent.keys) && compareKeys(newKey, parent.keys[insertIndex]) > 0 {
		insertIndex++
	}

	// 插入新键
	parent.keys = append(parent.keys[:insertIndex], append([]MccKey{newKey}, parent.keys[insertIndex:])...)
	// 插入新的子节点指针
	parent.children = append(parent.children[:insertIndex+1], parent.children[insertIndex:]...)
	parent.children[insertIndex+1] = rightChild

	// 检查父节点是否需要分裂
	if len(parent.keys) > t.order-1 {
		t.splitInternalNode(parent)
	}
}

// splitInternalNode 分裂内部节点
func (t *MVCCBPlusTree) splitInternalNode(node *MVCCNode) {
	// 计算分裂点
	midIdx := t.order / 2

	// 创建新的右侧内部节点
	rightNode := &MVCCNode{
		isLeaf:   false,
		keys:     append([]MccKey{}, node.keys[midIdx+1:]...),
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
			keys:     []MccKey{promotedKey},
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
		keys:     append([]MccKey{}, node.keys...),
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

// commitTransaction 实现事务提交 (完整2PC流程)
func (t *MVCCBPlusTree) commitTransaction(tx *Transaction) error {
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

	// 1.3 写入Prepare记录到WAL
	if err := t.wal.Prepare(tx); err != nil {
		// 如果Prepare失败，回滚事务
		if rollbackErr := t.rollbackTransaction(tx); rollbackErr != nil {
			return fmt.Errorf("prepare failed and rollback failed: prepare=%w, rollback=%w", err, rollbackErr)
		}
		return fmt.Errorf("WAL Prepare failed: %w", err)
	}

	// 1.4 强制刷盘确保Prepare记录持久化
	if err := t.wal.Sync(); err != nil {
		if rollbackErr := t.rollbackTransaction(tx); rollbackErr != nil {
			return fmt.Errorf("prepare sync failed and rollback failed: sync=%w, rollback=%w", err, rollbackErr)
		}
		return fmt.Errorf("WAL Prepare sync failed: %w", err)
	}

	// 阶段2: 提交阶段 (Commit Phase)
	// 2.1 写入Commit记录到WAL
	if err := t.wal.Commit(tx.txID); err != nil {
		// Commit失败是严重问题，需要特殊处理
		// 此时Prepare已成功，理论上Commit应该能成功
		// 可以尝试重试或标记为需要恢复的状态
		return fmt.Errorf("WAL Commit failed after successful prepare: %w", err)
	}

	// 2.2 更新事务状态为已提交
	tx.status = TxCommitted

	// 2.3 使事务的修改对其他事务可见
	// 在MVCC中，版本已在Put/Delete时创建，这里主要是状态更新
	t.makeVersionsVisible(tx)

	// 2.4 释放事务持有的所有锁
	for _, writeOp := range tx.writes {
		t.lockMgr.Release(tx.txID, writeOp.Key)
	}

	// 2.5 从TransactionManager中标记事务为已提交
	if err := t.txMgr.MarkCommitted(tx.txID); err != nil {
		return fmt.Errorf("failed to mark transaction as committed: %w", err)
	}

	return nil
}

// makeVersionsVisible 使事务的修改对其他事务可见
func (t *MVCCBPlusTree) makeVersionsVisible(tx *Transaction) {
	// 在MVCC中，版本的可见性主要通过isVisible方法中的逻辑控制
	// 这里可以进行一些优化，比如更新版本的状态标记等
	// 由于我们的实现中版本可见性主要依赖事务状态，这里可以是空实现
	// 或者进行一些性能优化操作

	// 可选：触发版本链的垃圾回收
	go t.scheduleVersionGC(tx)
}

// scheduleVersionGC 调度版本垃圾回收
func (t *MVCCBPlusTree) scheduleVersionGC(tx *Transaction) {
	// 异步清理不再需要的旧版本
	// 这是一个后台任务，可以延迟执行
	time.Sleep(100 * time.Millisecond) // 简单延迟
	// 实际实现中应该有更复杂的GC策略
}

// rollbackTransaction 实现事务回滚
func (t *MVCCBPlusTree) rollbackTransaction(tx *Transaction) error {
	// 对于MVCC，回滚主要是标记事务为已中止，并清理其影响。
	// 已经创建的版本由于其txID是中止事务的ID，所以对其他事务不可见（除了ReadUncommitted）。
	// 可能需要将这些“无效”版本从版本链中移除或标记为无效，以供后续清理（垃圾回收）。

	// 记录Abort到WAL
	if t.wal != nil {
		t.wal.Abort(tx.txID)
	}

	// 释放锁
	for _, writeOp := range tx.writes {
		t.lockMgr.Release(tx.txID, writeOp.Key)
	}

	// 从TransactionManager中移除或标记事务为已中止
	t.txMgr.MarkAborted(tx.txID)

	// 清理该事务产生的版本 (可选，可以由后台GC处理)
	// 遍历tx.writes，找到对应的MVCCNode，然后移除txID为此事务ID的版本
	// 这比较复杂，因为需要修改版本链，且要考虑并发

	fmt.Printf("Transaction %d rolled back\n", tx.txID)
	return nil
}

// validateConsistency 校验数据一致性 (骨架)
func (t *MVCCBPlusTree) validateConsistency() error {
	if err := t.validateTreeStructure(t.root); err != nil {
		return fmt.Errorf("tree structure validation failed: %w", err)
	}
	if err := t.validateVersionChains(); err != nil {
		return fmt.Errorf("version chains validation failed: %w", err)
	}
	return nil
}
func (t *MVCCBPlusTree) validateTreeStructure(node *MVCCNode) error {
	// 实现B+树结构校验逻辑
	// - 检查每个节点的key数量是否在[order/2, order-1]范围内 (根节点除外)
	// - 检查key是否有序
	// - 检查叶子节点是否都在同一层
	// - 检查内部节点的子节点指针是否正确
	return nil
}
func (t *MVCCBPlusTree) validateVersionChains() error {
	// 遍历所有叶子节点的所有MVCCNode，检查其版本链的逻辑一致性
	// - beginTS < endTS (如果endTS非0)
	// - 版本链按beginTS降序排列
	// - 等等
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
func compareKeys(k1, k2 MccKey) int {
	// 假设Key是int类型，你需要替换成你实际的比较逻辑
	k1Int, ok1 := k1.(int)
	k2Int, ok2 := k2.(int)
	if !ok1 || !ok2 {
		panic("Key comparison error: keys are not integers")
	}
	if k1Int < k2Int {
		return -1
	}
	if k1Int > k2Int {
		return 1
	}
	return 0
}
