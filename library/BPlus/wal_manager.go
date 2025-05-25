package bplus

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"sync"
	"time"
)

// WALEntry 预写日志条目
type WALEntry struct {
	txID      uint64
	opType    OpType
	key       Key
	oldValue  Value
	newValue  Value
	timestamp time.Time
}

type OpType int

const (
	OpInsert OpType = iota
	OpUpdate
	OpDelete
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

// flushBatch 批量刷盘
func (w *WALManager) flushBatch() {
	for batch := range w.batchChan {
		w.mu.Lock()
		for _, entry := range batch {
			// 同上写入逻辑
		}
		w.file.Sync() // 批量同步
		w.mu.Unlock()
	}
}
