//go:build linux

package storage

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sync"
	"syscall"
	"unsafe"
)

const (
	// SharedMemSize 共享内存区域大小
	SharedMemSize = 64 * 1024 * 1024 // 64MB
	// SharedMemName 共享内存名称
	SharedMemName = "/VectorSphereL2Cache"
	// HeaderSize 头部大小
	HeaderSize = 8
	// IPC权限
	ipcPerm = 0666
)

// SharedMemory 共享内存管理结构
type SharedMemory struct {
	memory     []byte           // 内存映射区域
	shmid      int              // 共享内存ID
	mutex      sync.RWMutex     // 互斥锁
	indexTable map[string]int64 // 键到内存位置的索引表
}

// SharedMemoryEntry 共享内存中的条目
type SharedMemoryEntry struct {
	Key       string
	Results   []string
	Timestamp int64
}

// formatError 格式化错误信息
func formatError(message string, err error) error {
	return fmt.Errorf("%s: %v", message, err)
}

// readEntry 从指定偏移量读取并反序列化条目
func (sm *SharedMemory) readEntry(offset int64) (SharedMemoryEntry, uint32, error) {
	var entry SharedMemoryEntry

	// 读取数据长度
	dataLen := binary.LittleEndian.Uint32(sm.memory[offset : offset+4])

	// 读取数据
	data := sm.memory[offset+4 : offset+4+int64(dataLen)]

	// 反序列化
	err := json.Unmarshal(data, &entry)
	return entry, dataLen, err
}

// writeEntry 将条目写入指定偏移量
func (sm *SharedMemory) writeEntry(offset uint64, data []byte) {
	// 写入数据长度
	binary.LittleEndian.PutUint32(sm.memory[offset:offset+4], uint32(len(data)))

	// 写入数据
	copy(sm.memory[offset+4:offset+4+uint64(len(data))], data)
}

// updateOffset 更新共享内存头部的偏移量
func (sm *SharedMemory) updateOffset(offset uint64) {
	binary.LittleEndian.PutUint64(sm.memory[:HeaderSize], offset)
}

// getOffset 获取当前偏移量
func (sm *SharedMemory) getOffset() uint64 {
	return binary.LittleEndian.Uint64(sm.memory[:HeaderSize])
}

// NewSharedMemory 创建新的共享内存管理器
func NewSharedMemory() (*SharedMemory, error) {
	sm := &SharedMemory{
		indexTable: make(map[string]int64),
	}

	// 生成唯一的键
	key, err := syscall.Ftok(SharedMemName, 65)
	if err != nil {
		// 如果无法生成键，使用一个固定值
		key = 0x1234ABCD
	}

	// 尝试创建共享内存
	shmid, err := syscall.ShmGet(key, SharedMemSize, ipcPerm|syscall.IPC_CREAT)
	if err != nil {
		return nil, formatError("无法创建共享内存", err)
	}

	// 附加到共享内存
	addr, err := syscall.ShmAttach(shmid, 0, 0)
	if err != nil {
		return nil, formatError("无法附加到共享内存", err)
	}

	// 转换为字节切片
	mem := (*[SharedMemSize]byte)(unsafe.Pointer(addr))
	sm.memory = mem[:]
	sm.shmid = shmid

	// 初始化头部
	if sm.getOffset() == 0 {
		// 设置初始偏移量为头部大小
		sm.updateOffset(HeaderSize)
	}

	// 加载索引表
	sm.loadIndexTable()

	return sm, nil
}

// Close 关闭共享内存
func (sm *SharedMemory) Close() error {
	if sm.memory != nil {
		// 分离共享内存
		addr := uintptr(unsafe.Pointer(&sm.memory[0]))
		err := syscall.ShmDt(addr)
		if err != nil {
			return formatError("无法分离共享内存", err)
		}
	}

	return nil
}

// Put 将数据存入共享内存
func (sm *SharedMemory) Put(key string, results []string, timestamp int64) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	// 准备要存储的数据
	entry := SharedMemoryEntry{
		Key:       key,
		Results:   results,
		Timestamp: timestamp,
	}

	// 序列化数据
	data, err := json.Marshal(entry)
	if err != nil {
		return formatError("序列化数据失败", err)
	}

	// 获取当前偏移量
	currentOffset := sm.getOffset()

	// 检查是否有足够空间
	if int(currentOffset)+len(data)+4 > SharedMemSize {
		// 空间不足，重置到头部之后
		currentOffset = HeaderSize
		sm.updateOffset(currentOffset)
	}

	// 写入数据
	sm.writeEntry(currentOffset, data)

	// 更新索引表
	sm.indexTable[key] = int64(currentOffset)

	// 更新偏移量
	newOffset := currentOffset + 4 + uint64(len(data))
	sm.updateOffset(newOffset)

	return nil
}

// Get 从共享内存获取数据
func (sm *SharedMemory) Get(key string) ([]string, int64, bool) {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()

	// 查找索引
	offset, exists := sm.indexTable[key]
	if !exists {
		return nil, 0, false
	}

	// 读取并反序列化条目
	entry, _, err := sm.readEntry(offset)
	if err != nil {
		return nil, 0, false
	}

	return entry.Results, entry.Timestamp, true
}

// Delete 从共享内存删除数据
func (sm *SharedMemory) Delete(key string) {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	// 从索引表中删除
	delete(sm.indexTable, key)

	// 注意：实际内存中的数据不会被删除，只是不再被索引
	// 当空间不足时，会被新数据覆盖
}

// GetAllKeys 获取所有键
func (sm *SharedMemory) GetAllKeys() []string {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()

	keys := make([]string, 0, len(sm.indexTable))
	for k := range sm.indexTable {
		keys = append(keys, k)
	}

	return keys
}

// loadIndexTable 加载索引表
func (sm *SharedMemory) loadIndexTable() {
	// 从头部开始扫描共享内存
	offset := uint64(HeaderSize)

	for offset < SharedMemSize {
		// 检查是否有足够空间读取数据长度
		if offset+4 > SharedMemSize {
			break
		}

		// 读取数据长度
		dataLen := binary.LittleEndian.Uint32(sm.memory[offset : offset+4])
		if dataLen == 0 {
			break
		}

		// 检查数据是否完整
		if offset+4+uint64(dataLen) > SharedMemSize {
			break
		}

		// 读取并反序列化条目
		entry, _, err := sm.readEntry(int64(offset))
		if err == nil && entry.Key != "" {
			// 更新索引表
			sm.indexTable[entry.Key] = int64(offset)
		}

		// 移动到下一条记录
		offset += 4 + uint64(dataLen)
	}
}

// Clear 清空共享内存
func (sm *SharedMemory) Clear() {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	// 重置头部
	sm.updateOffset(HeaderSize)

	// 清空索引表
	sm.indexTable = make(map[string]int64)
}

// Count 返回共享内存中的条目数量
func (sm *SharedMemory) Count() int {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()

	return len(sm.indexTable)
}
