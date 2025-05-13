package index

import (
	"bytes"
	"encoding/gob"
	"errors"
	"github.com/rainmyy/seetaSearch/db"
	"strings"
	"sync/atomic"
)

type Index struct {
	db           db.KvDb
	reverseIndex IReverseIndex
	maxIntId     uint64
}
type IndexInterface interface {
	AddDoc(doc Document) (int, error)
	DelDoc(docId string) int
	Search(query *TermQuery, onFlag uint64, offFlag uint64, orFlags []uint64) []*Document
	Count() int
	Close() error
}

func (is *Index) NewIndexServer(docNumEstimate int, dbType int, bucket, dbDir string) error {
	kvDb, err := db.GetDb(dbType, dbDir, bucket)
	if err != nil {
		return err
	}
	reIndex := NewSkipListInvertedIndex(docNumEstimate)
	is.reverseIndex = reIndex
	is.db = kvDb
	return nil
}

func (is *Index) Close() error {
	return is.db.Close()
}

func (is *Index) AddDoc(doc Document) (int, error) {
	docId := strings.TrimSpace(doc.Id)
	if len(docId) == 0 {
		return 0, errors.New("empty doc id")
	}
	is.DelDoc(docId)
	doc.FloatId = float64(atomic.AddUint64(&is.maxIntId, 1))
	var val bytes.Buffer
	encode := gob.NewEncoder(&val)
	if err := encode.Encode(doc); err != nil {
		return 0, errors.New("error encoding doc: " + err.Error())
	}
	if err := is.db.Set([]byte(docId), val.Bytes()); err != nil {
		return 0, errors.New("error setting doc: " + err.Error())
	}
	is.reverseIndex.Add(doc)
	return val.Len(), nil
}

func (is *Index) DelDoc(docId string) int {
	if len(docId) == 0 {
		return -1
	}
	dbKey := []byte(docId)
	docBytes, err := is.db.Get(dbKey)
	if err != nil {
		return -1
	}
	if len(docBytes) == 0 {
		return -1
	}

	reader := bytes.NewBuffer(docBytes)
	var doc Document
	if err := gob.NewEncoder(reader); err != nil {
		return -1
	}

	for _, keyword := range doc.KeyWords {
		is.reverseIndex.Delete(doc.FloatId, keyword)
	}
	if err := is.db.Del(dbKey); err != nil {
		return -1
	}

	return 0
}

func (is *Index) LoadIndex() (int, error) {
	reader := bytes.NewReader([]byte{})
	n, err := is.db.TotalDb(func(k, v []byte) error {
		reader.Reset(v)
		var doc Document
		decoder := gob.NewDecoder(reader)
		if err := decoder.Decode(&doc); err != nil {
			return errors.New("")
		}
		_, err := is.AddDoc(doc)
		if err != nil {
			return err
		}
		return nil
	})

	if err != nil {
		return -1, err
	}

	return int(n), nil
}

func (is *Index) Search(query *TermQuery, onFlag, offFlag uint64, orFlags []uint64) ([]*Document, error) {
	docIds := is.reverseIndex.Search(query, onFlag, offFlag, orFlags)
	if len(docIds) == 0 {
		return nil, nil
	}
	keys := make([][]byte, 0, len(docIds))
	for _, docId := range docIds {
		keys = append(keys, []byte(docId))
	}
	docBytes, err := is.db.BatchGet(keys)
	if err != nil {
		return nil, err
	}
	result := make([]*Document, 0, len(docIds))
	reader := bytes.NewReader([]byte{})
	for _, docByte := range docBytes {
		reader.Reset(docByte)
		decoder := gob.NewDecoder(reader)
		var doc Document
		err = decoder.Decode(&doc)
		if err == nil {
			result = append(result, &doc)
		}
	}

	return result, nil
}

func (is *Index) Total() int {
	n, err := is.db.TotalKey(func(k []byte) error {
		return nil
	})
	if err != nil {
		return 0
	}

	return int(n)
}
