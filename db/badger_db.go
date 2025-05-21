package db

import (
	"errors"
	"github.com/dgraph-io/badger/v4"
	"os"
	"seetaSearch/library/log"
	"sync/atomic"
)

type BadgerDB struct {
	db           *badger.DB
	path         string
	version      int
	logLevel     int
	discardRatio float64
}

const (
	DEBUG = iota
	INFO
	WARNING
	ERROR
)

func (b *BadgerDB) NewInstance(path string, version, logLevel int, discardRatio float64) *BadgerDB {
	b.path = path
	b.version = version
	b.logLevel = logLevel
	b.discardRatio = discardRatio

	return b
}

func (b *BadgerDB) Close() error {
	return b.db.Close()
}

func (b *BadgerDB) CheckAndGc() {
	lsmSizeBefore, vlogSizeBefore := b.db.Size()
	for {
		if err := b.db.RunValueLogGC(b.discardRatio); errors.Is(err, badger.ErrNoRewrite) || errors.Is(err, badger.ErrRejected) {
			break
		}
	}
	lsmSizeAfter, vlogSizeAfter := b.db.Size()
	if vlogSizeAfter < vlogSizeBefore {
		log.Info("badger before GC, LSM %d, vlog %d, after GC, LSM %d, vlog %d", lsmSizeBefore, vlogSizeBefore, lsmSizeAfter, vlogSizeAfter)
	} else {
		log.Info("collect zero grabage")
	}
}
func (b *BadgerDB) GetDbPath() string {
	return b.path
}

func (b *BadgerDB) Open() error {
	DataDir := b.GetDbPath()
	if err := os.MkdirAll(DataDir, 0700); err != nil {
		return err
	}

	logLevel := badger.ERROR
	switch b.logLevel {
	case DEBUG:
		logLevel = badger.DEBUG
	case INFO:
		logLevel = badger.INFO
	case WARNING:
		logLevel = badger.WARNING
	case ERROR:
		logLevel = badger.ERROR
	}

	option := badger.DefaultOptions(DataDir).WithNumVersionsToKeep(b.version).WithLoggingLevel(logLevel)
	db, err := badger.Open(option)
	if err != nil {
		return err
	}
	b.db = db
	return nil
}

func (b *BadgerDB) Set(k, v []byte) error {
	return b.db.Update(func(txn *badger.Txn) error {
		return txn.Set(k, v)
	})
}

func (b *BadgerDB) BatchSet(keys, values [][]byte) error {
	if len(keys) != len(values) {
		return errors.New("len(keys) != len(values)")
	}
	txn := b.db.NewTransaction(true)
	defer txn.Discard()
	for i, key := range keys {
		value := values[i]
		if err := txn.Set(key, value); err != nil {
			_ = txn.Commit()
			txn = b.db.NewTransaction(true)
			_ = txn.Set(key, value)
		}
	}

	return txn.Commit()
}

func (b *BadgerDB) Get(k []byte) ([]byte, error) {
	var value []byte
	err := b.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get(k)
		if err != nil {
			return err
		}
		err = item.Value(func(v []byte) error {
			value = v
			return nil
		})
		return err
	})

	return value, err
}

func (b *BadgerDB) BatchGet(keys [][]byte) ([][]byte, error) {
	var err error
	txn := b.db.NewTransaction(false)
	defer txn.Discard()
	values := make([][]byte, len(keys))
	for i, key := range keys {
		var item *badger.Item
		item, err = txn.Get(key)
		if err != nil {
			var v []byte
			err = item.Value(func(val []byte) error {
				v = val
				return nil
			})
			if err == nil {
				values[i] = v
			} else {
				values[i] = []byte{}
			}
			continue
		}
		values[i], err = item.ValueCopy(nil)
		if !errors.Is(err, badger.ErrKeyNotFound) {
			txn.Discard()
			txn = b.db.NewTransaction(true)
		}
	}

	return values, err
}

func (b *BadgerDB) Del(key []byte) error {
	return b.db.Update(func(txn *badger.Txn) error {
		return txn.Delete(key)
	})
}

func (b *BadgerDB) BatchDel(keys [][]byte) error {
	txn := b.db.NewTransaction(true)
	defer txn.Discard()
	for _, key := range keys {
		if err := txn.Delete(key); err == nil {
			continue
		}
		_ = txn.Commit()
		txn = b.db.NewTransaction(true)
		_ = txn.Delete(key)
	}

	return txn.Commit()
}

func (b *BadgerDB) Has(key []byte) bool {
	var exists bool
	err := b.db.View(func(txn *badger.Txn) error {
		_, err := txn.Get(key)
		if err != nil {
			return err
		}
		exists = true
		return nil
	})
	if err != nil {
		return false
	}

	return exists
}

func (b *BadgerDB) TotalDb(f func(k, v []byte) error) (int64, error) {
	var count int64
	err := b.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		it := txn.NewIterator(opts)
		defer it.Close()
		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()
			k := item.Key()

			var v []byte
			err := item.Value(func(val []byte) error {
				v = val
				return nil
			})
			if err != nil {
				continue
			}
			if err := f(k, v); err == nil {
				atomic.AddInt64(&count, 1)
			}
		}

		return nil
	})

	if err != nil {
		return 0, err
	}
	return atomic.LoadInt64(&count), nil
}

func (b *BadgerDB) TotalKey(f func(k []byte) error) (int64, error) {
	var count int64
	err := b.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchValues = false
		it := txn.NewIterator(opts)
		defer it.Close()
		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()
			k := item.Key()
			if err := f(k); err == nil {
				atomic.AddInt64(&count, 1)
			}
		}

		return nil
	})
	if err != nil {
		return 0, err
	}

	return atomic.LoadInt64(&count), nil
}
