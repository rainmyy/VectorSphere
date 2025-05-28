package tree

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"seetaSearch/library/log"
	"sync"
	"time"
)

// WALEntry 预写日志条目
type WALEntry struct {
	TxID      uint64
	OpType    OperationType
	Key       Key
	OldValue  Value
	Value     Value
	Timestamp time.Time
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
			TxID:      tx.txID,
			OpType:    OpPrepare, // 或者更具体的 OpPutPrepare
			Key:       write.Key,
			Value:     write.Value,
			Timestamp: time.Now(),
		}
		if err := w.Log(entry); err != nil {
			return err
		}
	}
	return nil
}

func (w *WALManager) Commit(txID uint64) error {
	entry := &WALEntry{
		TxID:      txID,
		OpType:    OpCommit,
		Timestamp: time.Now(),
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.Log(entry); err != nil {
		return err
	}
	return w.file.Sync()
}

// LogWrite 将写操作记录到WAL
func (w *WALManager) LogWrite(txID uint64, key Key, value Value) error {
	if w.file == nil {
		return errors.New("WAL file not initialized")
	}

	// 序列化日志条目（统一使用json.Marshal）
	entry := WALEntry{
		TxID:      txID,
		OpType:    OpPut,
		Key:       key,
		Value:     value,
		Timestamp: time.Now(),
	}

	// 使用批量刷盘机制（通过batchChan传递）
	select {
	case w.batchChan <- []*WALEntry{&entry}:
		// 条目已加入批量队列，由flushBatch协程处理
	default:
		// 队列满时降级为直接写入（避免阻塞）
		w.mu.Lock()
		defer w.mu.Unlock()

		data, err := json.Marshal(entry)
		if err != nil {
			return fmt.Errorf("serialize WAL entry failed: %w", err)
		}

		// 写入日志长度前缀（4字节大端序）
		lenBuf := make([]byte, 4)
		binary.BigEndian.PutUint32(lenBuf, uint32(len(data)))

		if _, err := w.file.Write(lenBuf); err != nil {
			return fmt.Errorf("write WAL length failed: %w", err)
		}
		if _, err := w.file.Write(data); err != nil {
			return fmt.Errorf("write WAL entry failed: %w", err)
		}
		// 直接写入时强制刷盘保证一致性
		err = w.file.Sync()
		if err != nil {
			return err
		}
	}

	return nil
}

// uint32ToBytes 将uint32转换为大端序字节数组
func uint32ToBytes(n uint32) []byte {
	return []byte{
		byte(n >> 24),
		byte(n >> 16),
		byte(n >> 8),
		byte(n),
	}
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
		err := w.file.Sync()
		if err != nil {
			return
		}
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
		TxID:      txID,
		OpType:    OpAbort,
		Timestamp: time.Now(),
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.Log(entry); err != nil {
		return err
	}
	return w.file.Sync()
}
