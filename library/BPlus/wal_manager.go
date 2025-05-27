package bplus

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"seetaSearch/library/log"
	"sync"
	"time"
)

// WALEntry 预写日志条目
type WALEntry struct {
	txID      uint64
	opType    OperationType
	key       Key
	oldValue  Value
	value     Value
	timestamp time.Time
}

// OperationType 定义WAL操作类型
type OperationType int

const (
	OpPut OperationType = iota
	OpDelete
	OpCommit
	OpAbort
	OpPrepare // for 2PC
)

// WALManager 预写日志管理器
type WALManager struct {
	file      *os.File
	mu        sync.Mutex
	batchChan chan []*WALEntry
}

func NewWALManager(filename string) (*WALManager, error) {
	file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		return nil, err
	}

	w := &WALManager{
		file:      file,
		batchChan: make(chan []*WALEntry, 1000),
	}

	// 启动后台刷盘协程
	go w.flushBatch()

	return w, nil
}

// Log 记录日志条目
func (w *WALManager) Log(entry *WALEntry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	// 先写长度再写数据
	lenBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(lenBuf, uint32(len(data)))

	if _, err := w.file.Write(lenBuf); err != nil {
		return err
	}
	if _, err := w.file.Write(data); err != nil {
		return err
	}

	// 可选: 同步刷盘(影响性能但更安全)
	// return w.file.Sync()
	return nil
}
func (w *WALManager) Prepare(tx *Transaction) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	// 记录事务所有写操作到WAL，标记为Prepare
	// 这通常是2PC的第一阶段
	for _, write := range tx.writes {
		entry := &WALEntry{
			txID:      tx.txID,
			opType:    OpPrepare, // 或者更具体的 OpPutPrepare
			key:       write.Key,
			value:     write.Value,
			timestamp: time.Now(),
		}
		if err := w.Log(entry); err != nil {
			return err
		}
	}
	return nil
}

func (w *WALManager) Commit(txID uint64) error {
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
	return w.file.Sync()
}

// flushBatch 批量刷盘
func (w *WALManager) flushBatch() {
	for batch := range w.batchChan {
		w.mu.Lock()
		for _, entry := range batch {
			data, err := json.Marshal(entry)
			if err != nil {
				log.Error("Error marshaling WALEntry:", err)
				continue
			}

			lenBuf := make([]byte, 4)
			binary.BigEndian.PutUint32(lenBuf, uint32(len(data)))

			if _, err := w.file.Write(lenBuf); err != nil {
				log.Error("Error writing length to WAL file:", err)
				continue
			}
			if _, err := w.file.Write(data); err != nil {
				log.Error("Error writing data to WAL file:", err)
				continue
			}
		}
		w.file.Sync()
		w.mu.Unlock()
	}
}

// Sync 强制刷盘
func (w *WALManager) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.file.Sync()
}

func (w *WALManager) Abort(txID uint64) error {
	entry := &WALEntry{
		txID:      txID,
		opType:    OpAbort,
		timestamp: time.Now(),
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.Log(entry); err != nil {
		return err
	}
	return w.file.Sync()
}
