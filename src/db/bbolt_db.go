package db

import (
	"errors"
	"fmt"
	"github.com/coreos/bbolt"
	"sync/atomic"
	"time"
)

type BBoltDB struct {
	db     *bbolt.DB
	path   string
	bucket []byte
}

func (b *BBoltDB) NewInstance(path, bucket string) *BBoltDB {
	b.path = path
	b.bucket = []byte(bucket)
	return b
}
func (b *BBoltDB) Close() error {
	return b.db.Close()
}

func (b *BBoltDB) GetDbPath() string {
	return b.path
}

func (b *BBoltDB) Open() error {
	dataDir := b.GetDbPath()
	db, err := bbolt.Open(dataDir, 0600, &bbolt.Options{Timeout: 10 * time.Second})
	if err != nil {
		return err
	}
	err = db.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(b.bucket)
		return err
	})
	if err != nil {
		db.Close()
		return err
	}

	b.db = db
	return nil
}

func (b *BBoltDB) CreateTable(tableName string) (*bbolt.Bucket, error) {
	ctx, err := b.db.Begin(true)
	if err != nil {
		return nil, err
	}
	defer func(ctx *bbolt.Tx) {
		err := ctx.Rollback()
		if err != nil {

		}
	}(ctx)
	table, err := ctx.CreateBucketIfNotExists([]byte(tableName))
	if err != nil {
		return nil, err
	}
	if err := ctx.Commit(); err != nil {
		return nil, err
	}

	return table, nil
}

func (b *BBoltDB) DeleteTable(tableName string) error {
	ctx, err := b.db.Begin(true)
	if err != nil {
		return err
	}
	defer func(ctx *bbolt.Tx) {
		err := ctx.Rollback()
		if err != nil {

		}
	}(ctx)
	err = ctx.DeleteBucket([]byte(tableName))
	if err != nil {

	}
	if err := ctx.Commit(); err != nil {
		return err
	}
	return nil
}
func (b *BBoltDB) GetBucket(tableName string) (*bbolt.Bucket, error) {
	ctx, err := b.db.Begin(true)
	if err != nil {
		return nil, err
	}
	defer func(ctx *bbolt.Tx) {
		err := ctx.Rollback()
		if err != nil {

		}
	}(ctx)
	bb := ctx.Bucket([]byte(tableName))
	if bb == nil {
		return nil, fmt.Errorf("TableName[%v] not found", tableName)
	}
	if err := ctx.Commit(); err != nil {
		return nil, err
	}

	return bb, nil
}
func (b *BBoltDB) Get(k []byte) ([]byte, error) {
	var data []byte
	err := b.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(b.bucket)
		data = b.Get(k)
		return nil
	})
	if len(data) == 0 {
		return nil, errors.New("no data")
	}

	return data, err
}

func (b *BBoltDB) Set(k, v []byte) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		bu := tx.Bucket(b.bucket)
		return bu.Put(k, v)
	})
}

func (b *BBoltDB) BatchGet(keys [][]byte) ([][]byte, error) {
	var results [][]byte
	err := b.db.View(func(tx *bbolt.Tx) error {
		bu := tx.Bucket(b.bucket)
		for _, k := range keys {
			v := bu.Get(k)
			results = append(results, v)
		}
		return nil
	})

	return results, err
}

func (b *BBoltDB) BatchSet(keys, values [][]byte) error {
	if len(keys) == 0 || len(keys) != len(values) {
		return errors.New("batch size mismatch")
	}
	return b.db.Batch(func(tx *bbolt.Tx) error {
		bu := tx.Bucket(b.bucket)
		for k := range keys {
			err := bu.Put(keys[k], values[k])
			if err != nil {
				return err
			}
		}
		return nil
	})
}

func (b *BBoltDB) BatchDel(keys [][]byte) error {
	return b.db.Batch(func(tx *bbolt.Tx) error {
		bu := tx.Bucket(b.bucket)
		for _, k := range keys {
			err := bu.Delete(k)
			if err != nil {
				return err
			}
		}
		return nil
	})
}

func (b *BBoltDB) Del(key []byte) error {
	return b.db.Update(func(tx *bbolt.Tx) error {
		bu := tx.Bucket(b.bucket)
		return bu.Delete(key)
	})
}

func (b *BBoltDB) Has(k []byte) bool {
	var data []byte
	err := b.db.View(func(tx *bbolt.Tx) error {
		bu := tx.Bucket(b.bucket)
		data = bu.Get(k)
		return nil
	})
	if err != nil || len(data) == 0 {
		return false
	} else {
		return true
	}
}

func (b *BBoltDB) TotalKey(f func(k []byte) error) (int64, error) {
	var total int64
	err := b.db.View(func(tx *bbolt.Tx) error {
		bu := tx.Bucket(b.bucket)
		c := bu.Cursor()
		for k, _ := c.First(); k != nil; k, _ = c.Next() {
			err := f(k)
			if err != nil {
				return err
			}
			atomic.AddInt64(&total, int64(1))
		}
		return nil
	})

	return total, err
}

func (b *BBoltDB) TotalDb(f func(k, v []byte) error) (int64, error) {
	var total int64
	err := b.db.View(func(tx *bbolt.Tx) error {
		bu := tx.Bucket(b.bucket)
		c := bu.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			err := f(k, v)
			if err != nil {
				return err
			}
			atomic.AddInt64(&total, int64(1))
		}
		return nil
	})

	return total, err
}
