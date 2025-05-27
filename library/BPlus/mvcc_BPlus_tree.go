package bplus

import (
	"errors"
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
	switch tx.isolation {
	case ReadUncommitted:
		return true
	case ReadCommitted:
		return v.txID == tx.txID || v.endTS == 0 || v.endTS > tx.startTS
	case RepeatableRead, Serializable:
		// 自己的修改总是可见
		if v.txID == tx.txID {
			return true
		}
		// 已提交且不在读视图中的版本可见
		_, active := tx.readView[v.txID]
		return !active && v.endTS == 0
	}
	return false
}

type MVCCBPlusTree struct {
	root      *Node
	order     int
	txMgr     *TransactionManager
	versionTS uint64 // 全局版本时间戳
	mu        sync.RWMutex
}

// Get 带事务的读取
func (t *MVCCBPlusTree) Get(tx *Transaction, key Key) (Value, bool) {
	switch tx.isolation {
	case ReadUncommitted:
		return t.getLatest(key)
	case ReadCommitted:
		t.lockMgr.Acquire(tx.txID, key, LockShared)
		defer t.lockMgr.Release(tx.txID, key)
		return t.getVersion(key, tx.startTS)
	case RepeatableRead:
		if tx.snapshot != nil {
			return tx.snapshot.Get(tx, key)
		}
		return t.getVersion(key, tx.startTS)
	case Serializable:
		t.lockMgr.Acquire(tx.txID, key, LockShared)
		defer t.lockMgr.Release(tx.txID, key)
		return t.getVersion(key, tx.startTS)
	default:
		return "", false
	}
}

// Put 带事务的写入
func (t *MVCCBPlusTree) Put(tx *Transaction, key Key, value Value) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	leaf := t.findLeaf(key)
	if leaf == nil {
		return errors.New("leaf not found")
	}

	// 查找或创建MVCC节点
	var mvccNode *MVCCNode
	for i, k := range leaf.keys {
		if k == key {
			if node, ok := leaf.children[i].(*MVCCNode); ok {
				mvccNode = node
				break
			}
		}
	}

	if mvccNode == nil {
		// 创建新的MVCC节点
		mvccNode = &MVCCNode{
			key: key,
		}
		// 插入到叶子节点(需要处理节点分裂等)
		t.insertIntoLeaf(leaf, key, mvccNode)
	}

	// 添加新版本
	mvccNode.AddVersion(value, tx)

	// 记录写操作到事务
	tx.recordWrite(key)

	return nil
}

// 通过WAL和两阶段提交实现
func (t *MVCCBPlusTree) commitTransaction(tx *Transaction) error {
	// 阶段1: 准备(写入WAL)
	if err := t.wal.Prepare(tx); err != nil {
		return err
	}

	// 阶段2: 提交(更新数据页)
	t.mu.Lock()
	defer t.mu.Unlock()

	// 应用所有修改
	for _, write := range tx.writes {
		if err := t.applyWrite(write); err != nil {
			// 失败时需要回滚(使用WAL中的undo日志)
			t.rollbackTx(tx)
			return err
		}
	}

	// 标记事务为已提交
	if err := t.wal.Commit(tx.txID); err != nil {
		t.rollbackTx(tx)
		return err
	}

	return nil
}

// 通过校验和与约束检查实现
func (t *MVCCBPlusTree) validateConsistency() error {
	// 检查B+树结构完整性
	if err := t.validateTreeStructure(); err != nil {
		return err
	}

	// 检查MVCC版本链完整性
	if err := t.validateVersionChains(); err != nil {
		return err
	}

	// 可选: 检查业务约束
	return nil
}

// 通过WAL和fsync实现
func (w *WALManager) Commit(txID uint64) error {
	// 写入提交记录
	entry := &WALEntry{
		txID:      txID,
		opType:    OpCommit,
		timestamp: time.Now(),
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.Log(entry); err != nil {
		return err
	}

	// 强制刷盘确保持久化
	return w.file.Sync()
}
