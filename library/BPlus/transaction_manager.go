package bplus

import (
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// WriteOp 代表一个写操作
type WriteOp struct {
	Key    Key
	Value  Value
	OpType string
}

// Transaction 表示一个事务
type Transaction struct {
	txID      uint64 // 事务ID
	startTS   uint64
	startTime time.Time           // 开始时间
	readView  map[uint64]struct{} // 读视图(活跃事务ID集合)
	snapshot  *MVCCBPlusTree      // 快照(用于可重复读)
	writes    []WriteOp           // 事务的写操作记录
	isolation IsolationLevel
	status    TxStatus
	mu        sync.RWMutex
}

type TxStatus int

const (
	TxActive TxStatus = iota
	TxCommitted
	TxAborted
)

type IsolationLevel int

const (
	ReadUncommitted IsolationLevel = iota
	ReadCommitted
	RepeatableRead
	Serializable
)

// TransactionManager 事务管理器
type TransactionManager struct {
	nextTxID uint64
	activeTx map[uint64]*Transaction
	mu       sync.RWMutex
}

func NewTransactionManager() *TransactionManager {
	return &TransactionManager{
		nextTxID: 1,
		activeTx: make(map[uint64]*Transaction),
	}
}
func (tx *Transaction) recordWrite(key Key, value Value) {
	tx.writes = append(tx.writes, WriteOp{Key: key, Value: value})
}

// Begin 开始新事务
func (tm *TransactionManager) Begin(isolation IsolationLevel) *Transaction {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	txID := atomic.AddUint64(&tm.nextTxID, 1)
	tx := &Transaction{
		txID:      txID,
		startTime: time.Now(),
		isolation: isolation,
		status:    TxActive,
	}

	// 设置读视图(除ReadUncommitted外)
	if isolation > ReadUncommitted {
		tx.readView = make(map[uint64]struct{})
		for id := range tm.activeTx {
			tx.readView[id] = struct{}{}
		}
	}

	// 创建快照(RepeatableRead及以上)
	if isolation >= RepeatableRead {
		// 这里简化处理，实际需要实现快照机制
		tx.snapshot = createSnapshot(tm)
	}

	tm.activeTx[txID] = tx
	return tx
}

// Commit 提交事务
func (tm *TransactionManager) Commit(tx *Transaction) error {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	if tx.status != TxActive {
		return errors.New("transaction is not active")
	}

	// 验证可串行化隔离级别下的读写冲突
	if tx.isolation == Serializable {
		if err := tm.validateSerializable(tx); err != nil {
			return err
		}
	}

	// 写入WAL日志
	if err := tm.writeCommitLog(tx); err != nil {
		return err
	}

	// 更新事务状态
	tx.status = TxCommitted

	tm.mu.Lock()
	delete(tm.activeTx, tx.txID)
	tm.mu.Unlock()

	return nil
}

func (tm *TransactionManager) GetActiveTxs() []*Transaction {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	txs := make([]*Transaction, 0, len(tm.activeTx))
	for _, tx := range tm.activeTx {
		txs = append(txs, tx)
	}
	return txs
}

// Abort 中止事务
func (tm *TransactionManager) Abort(tx *Transaction) {
	tx.mu.Lock()
	defer tx.mu.Unlock()

	if tx.status != TxActive {
		return
	}

	// 回滚所有修改
	tm.Rollback(tx)

	tx.status = TxAborted

	tm.mu.Lock()
	delete(tm.activeTx, tx.txID)
	tm.mu.Unlock()
}

func createSnapshot(tm *TransactionManager) *MVCCBPlusTree {
	if tm.activeTx == nil || len(tm.activeTx) == 0 {
		return nil
	}
	// 获取当前活跃事务中的一个BPlusTree实例进行克隆
	var tree *MVCCBPlusTree
	for _, tx := range tm.activeTx {
		if tx.snapshot != nil {
			tree = tx.snapshot
			break
		}
	}
	if tree == nil {
		return nil
	}
	return tree.Clone()
}

func (tm *TransactionManager) validateSerializable(tx *Transaction) error {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	// 遍历所有活跃事务，检查是否有读写冲突
	for _, activeTx := range tm.activeTx {
		if activeTx.txID == tx.txID {
			continue
		}

		// 检查当前事务的读视图是否与其他事务的写入有冲突
		for key := range activeTx.readView {
			if _, exists := tx.readView[key]; exists {
				return errors.New("serializable conflict detected")
			}
		}
	}
	return nil
}

func (tm *TransactionManager) writeCommitLog(tx *Transaction) error {
	logFile, err := os.OpenFile("commit_log.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer logFile.Close()

	// 写入事务ID和提交时间
	logEntry := fmt.Sprintf("Transaction ID: %d committed at %s\n", tx.txID, time.Now().Format(time.RFC3339))
	if _, err := logFile.WriteString(logEntry); err != nil {
		return err
	}

	return nil
}

func (tm *TransactionManager) Rollback(tx *Transaction) {
	logFile, err := os.OpenFile("rollback_log.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Error opening rollback log file:", err)
		return
	}
	defer logFile.Close()

	// 写入回滚操作日志
	logEntry := fmt.Sprintf("Transaction ID: %d rolled back at %s\n", tx.txID, time.Now().Format(time.RFC3339))
	if _, err := logFile.WriteString(logEntry); err != nil {
		fmt.Println("Error writing to rollback log file:", err)
	}
}

func (tm *TransactionManager) IsCommitted(txID uint64) bool {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	// 检查事务是否在活跃事务列表中
	if tx, exists := tm.activeTx[txID]; exists {
		return tx.status == TxCommitted
	}

	// 如果不在活跃事务列表中，假设是已提交的历史事务
	// 在实际实现中，这里应该查询持久化的事务状态表
	return true
}

// IsCommittedBefore 检查指定事务在给定时间戳之前是否已提交
func (tm *TransactionManager) IsCommittedBefore(txID uint64, timestamp uint64) bool {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	// 检查事务是否在活跃事务列表中
	if tx, exists := tm.activeTx[txID]; exists {
		// 如果事务仍然活跃，检查其开始时间是否在指定时间戳之前
		if tx.status == TxCommitted && tx.startTS < timestamp {
			return true
		}
		return false
	}

	// 如果不在活跃事务列表中，假设是已提交的历史事务
	// 在实际实现中，这里应该查询持久化的事务状态表和提交时间
	// 简化处理：假设历史事务都是在指定时间戳之前提交的
	return txID < timestamp
}

// MarkCommitted 标记事务为已提交
func (tm *TransactionManager) MarkCommitted(txID uint64) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tx, exists := tm.activeTx[txID]; exists {
		tx.status = TxCommitted
		// 可以选择从活跃事务列表中移除，或保留一段时间供查询
		delete(tm.activeTx, txID)
		return nil
	}
	return fmt.Errorf("transaction %d not found in active transactions", txID)
}

// MarkAborted 标记事务为已中止
func (tm *TransactionManager) MarkAborted(txID uint64) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tx, exists := tm.activeTx[txID]; exists {
		tx.status = TxAborted
		delete(tm.activeTx, txID)
		return nil
	}
	return fmt.Errorf("transaction %d not found in active transactions", txID)
}
