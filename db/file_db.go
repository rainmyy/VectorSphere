package db

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
)

type FileDB struct {
	path string
	mu   sync.RWMutex
}

type KeyValueInfo struct {
	Key   []byte
	Value []byte
}

func (f *FileDB) NewInstance(path string) *FileDB {
	f.path = path
	return f
}

// Open 打开文件数据库
func (f *FileDB) Open() error {
	if err := os.MkdirAll(filepath.Dir(f.path), 0700); err != nil {
		return err
	}
	_, err := os.Stat(f.path)
	if os.IsNotExist(err) {
		_, err = os.Create(f.path)
		if err != nil {
			return err
		}
	}
	return nil
}

// Close 关闭文件数据库
func (f *FileDB) Close() error {
	// 对于文件操作，这里无需额外处理
	return nil
}

// Set 设置键值对
func (f *FileDB) Set(key, value []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	file, err := os.OpenFile(f.path, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = file.Write(append(key, value...))
	return err
}

// Get 获取键对应的值
func (f *FileDB) Get(key []byte) ([]byte, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	data, err := os.ReadFile(f.path)
	if err != nil {
		return nil, err
	}

	// 简单模拟查找，实际实现可能需要更复杂的逻辑
	start := 0
	for {
		index := findBytes(data[start:], key)
		if index == -1 {
			break
		}
		start += index + len(key)
		// 这里简单假设值紧跟在键后面
		return data[start:], nil
	}

	return nil, errors.New("key not found")
}

// BatchGet 批量获取键对应的值
func (f *FileDB) BatchGet(keys [][]byte) ([][]byte, error) {
	values := make([][]byte, len(keys))
	for i, key := range keys {
		val, err := f.Get(key)
		if err != nil {
			return nil, err
		}
		values[i] = val
	}
	return values, nil
}

// Del 删除键值对
func (f *FileDB) Del(key []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	data, err := os.ReadFile(f.path)
	if err != nil {
		return err
	}

	var newData []byte
	start := 0
	for {
		index := findBytes(data[start:], key)
		if index == -1 {
			newData = append(newData, data[start:]...)
			break
		}
		// 复制键之前的数据
		newData = append(newData, data[start:start+index]...)
		start += index + len(key)

		// 查找下一个键的位置
		nextIndex := findNextKey(data[start:], key)
		if nextIndex == -1 {
			break
		}
		start += nextIndex
	}

	return os.WriteFile(f.path, newData, 0644)
}

// findNextKey 查找下一个键的位置
func findNextKey(haystack, prevKey []byte) int {
	for i := 0; i <= len(haystack)-len(prevKey); i++ {
		if equalBytes(haystack[i:i+len(prevKey)], prevKey) {
			return i
		}
	}
	return -1
}

// BatchDel 批量删除键值对
func (f *FileDB) BatchDel(keys [][]byte) error {
	for _, key := range keys {
		if err := f.Del(key); err != nil {
			return err
		}
	}
	return nil
}

// Has 检查键是否存在
func (f *FileDB) Has(key []byte) bool {
	_, err := f.Get(key)
	return err == nil
}

// TotalDb 遍历数据库中的所有键值对
func (f *FileDB) TotalDb(fc func(k, v []byte) error) (int64, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	data, err := os.ReadFile(f.path)
	if err != nil {
		return 0, err
	}

	var count int64
	start := 0
	for {
		// 假设每个键值对的键是唯一的，且键值对之间没有其他数据
		// 查找下一个键的位置
		nextKeyIndex := findNextKey(data[start:], data[start:])
		if nextKeyIndex == -1 {
			// 处理最后一个键值对
			if start < len(data) {
				// 简单假设剩下的数据都是值
				value := data[start:]
				if err := fc(nil, value); err != nil {
					return count, err
				}
				count++
			}
			break
		}

		key := data[start : start+nextKeyIndex]
		start += nextKeyIndex
		// 查找值的结束位置（下一个键的开始位置）
		valueEndIndex := findNextKey(data[start:], data[start:])
		if valueEndIndex == -1 {
			value := data[start:]
			if err := fc(key, value); err != nil {
				return count, err
			}
			count++
			break
		}

		value := data[start : start+valueEndIndex]
		if err := fc(key, value); err != nil {
			return count, err
		}
		count++
		start += valueEndIndex
	}

	return count, nil
}

// TotalKey 遍历数据库中的所有键
func (f *FileDB) TotalKey(fc func(k []byte) error) (int64, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	data, err := os.ReadFile(f.path)
	if err != nil {
		return 0, err
	}

	var count int64
	start := 0
	for {
		nextKeyIndex := findNextKey(data[start:], data[start:])
		if nextKeyIndex == -1 {
			break
		}

		key := data[start : start+nextKeyIndex]
		if err := fc(key); err != nil {
			return count, err
		}
		count++
		start += nextKeyIndex
	}

	return count, nil
}

// findBytes 在 haystack 中查找 needle 第一次出现的位置
func findBytes(haystack, needle []byte) int {
	for i := 0; i <= len(haystack)-len(needle); i++ {
		if equalBytes(haystack[i:i+len(needle)], needle) {
			return i
		}
	}
	return -1
}

// equalBytes 比较两个字节切片是否相等
func equalBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
