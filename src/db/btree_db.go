package db

import (
	"github.com/coreos/bbolt"
	"os"
)

type BTreeDB struct {
	fileName  string
	mmapBytes []byte
	fd        *os.File
	bboltDb   *BBoltDB
	buckets   map[string]*bbolt.Tx
}

func NewBtreeDb(dbname string) *BTreeDB {
	return &BTreeDB{
		fileName: dbname,
		bboltDb:  new(BBoltDB).NewInstance(dbname, ""),
		buckets:  make(map[string]*bbolt.Tx),
	}
}

func (db *BTreeDB) AddBtree(name string) error {
	_, err := db.bboltDb.CreateTable(name)
	return err
}

func (db *BTreeDB) Sync() error {
	return nil
}

func (db *BTreeDB) Set(btName, key string, value uint64) error {
	return nil
}
