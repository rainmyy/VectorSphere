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
	// 简单文件操作难以实现删除，这里返回错误
	return errors.New("delete operation not supported for file db")
}

// BatchDel 批量删除键值对
func (f *FileDB) BatchDel(keys [][]byte) error {
	// 简单文件操作难以实现批量删除，这里返回错误
	return errors.New("batch delete operation not supported for file db")
}

// Has 检查键是否存在
func (f *FileDB) Has(key []byte) bool {
	_, err := f.Get(key)
	return err == nil
}

// TotalDb 遍历数据库中的所有键值对
func (f *FileDB) TotalDb(fc func(k, v []byte) error) (int64, error) {
	// 简单文件操作难以实现遍历，这里返回错误
	return 0, errors.New("total db operation not supported for file db")
}

// TotalKey 遍历数据库中的所有键
func (f *FileDB) TotalKey(fc func(k []byte) error) (int64, error) {
	// 简单文件操作难以实现遍历，这里返回错误
	return 0, errors.New("total key operation not supported for file db")
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
