package db

import (
	"os"
	"strings"
)

type KvDb interface {
	Open() error
	Close() error
	Set(key, values []byte) error
	Get(key []byte) ([]byte, error)
	BatchGet(keys [][]byte) ([][]byte, error)
	Del(key []byte) error
	BatchDel(keys [][]byte) error
	Has(key []byte) bool
	TotalDb(f func(k, v []byte) error) (int64, error)
	TotalKey(f func(k []byte) error) (int64, error)
}

const (
	BOLT = iota
	BADGER
	FILE
)

func GetDb(dbType int, path string, bucket string) (KvDb, error) {
	pathList := strings.Split(path, "/")
	parentPath := strings.Join(pathList[:len(pathList)-1], "/")
	stat, err := os.Stat(parentPath)
	if os.IsNotExist(err) {
		err = os.MkdirAll(parentPath, os.ModePerm)
		if err != nil {
			return nil, err
		}
	} else {
		if stat.Mode().IsRegular() {
			if err := os.MkdirAll(parentPath, os.ModePerm); err != nil {
				return nil, err
			}
		}
		if err := os.MkdirAll(parentPath, os.ModePerm); err != nil {
			return nil, err
		}
	}
	var db KvDb
	switch dbType {
	case BOLT:
		db = new(BBoltDB).NewInstance(path, bucket)
	case BADGER:
		db = new(BadgerDB).NewInstance(path, 1, ERROR, 0.5)
	case FILE:
		db = new(FileDB).NewInstance(path)
	default:
		db = new(BBoltDB).NewInstance(path, bucket)
	}

	return db, err
}
