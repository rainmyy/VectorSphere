package db

import (
	"fmt"

	"VectorSphere/src/bootstrap"
)

// EncryptedDB 加密数据库包装器
type EncryptedDB struct {
	db              KvDb
	securityManager *bootstrap.EnhancedSecurityManager
}

// NewEncryptedDB 创建加密数据库
func NewEncryptedDB(db KvDb, sm *bootstrap.EnhancedSecurityManager) *EncryptedDB {
	return &EncryptedDB{
		db:              db,
		securityManager: sm,
	}
}

// Set 加密存储数据
func (edb *EncryptedDB) Set(key string, value []byte) error {
	// 加密数据
	encryptedValue, err := edb.securityManager.EncryptData(value)
	if err != nil {
		return fmt.Errorf("failed to encrypt data: %w", err)
	}

	// 存储加密数据
	return edb.db.Set([]byte(key), encryptedValue)
}

// Get 解密获取数据
func (edb *EncryptedDB) Get(key string) ([]byte, error) {
	// 获取加密数据
	encryptedValue, err := edb.db.Get([]byte(key))
	if err != nil {
		return nil, err
	}

	// 解密数据
	decryptedValue, err := edb.securityManager.DecryptData(encryptedValue)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt data: %w", err)
	}

	return decryptedValue, nil
}

// Delete 删除数据
func (edb *EncryptedDB) Delete(key string) error {
	return edb.db.Del([]byte(key))
}

// Close 关闭数据库
func (edb *EncryptedDB) Close() error {
	return edb.db.Close()
}
