package llm

import (
	"VectorSphere/src/db"
	"VectorSphere/src/index"
	"VectorSphere/src/messages"
	"encoding/json"
	"fmt"
)

type SkipListSessionStore struct {
	Index *index.SkipListInvertedIndex
}

// DistributedKV 定义分布式KV存储接口
type DistributedKV interface {
	Get(key string) ([]byte, error)
	Set(key string, value []byte) error
}

// BadgerKV 使用BadgerDB实现DistributedKV接口
type BadgerKV struct {
	db *db.BadgerDB
}

// NewBadgerKV 创建BadgerKV实例
func NewBadgerKV(badgerDB *db.BadgerDB) *BadgerKV {
	return &BadgerKV{db: badgerDB}
}

// Get 实现DistributedKV接口的Get方法
func (b *BadgerKV) Get(key string) ([]byte, error) {
	return b.db.Get([]byte(key))
}

// Set 实现DistributedKV接口的Set方法
func (b *BadgerKV) Set(key string, value []byte) error {
	return b.db.Set([]byte(key), value)
}

type DistributedSessionStore struct {
	KVClient DistributedKV                // etcd/redis client
	Index    *index.SkipListInvertedIndex // 可选本地加速
}

// NewDistributedSessionStore 创建DistributedSessionStore实例
func NewDistributedSessionStore(kvClient DistributedKV, idx *index.SkipListInvertedIndex) *DistributedSessionStore {
	return &DistributedSessionStore{
		KVClient: kvClient,
		Index:    idx,
	}
}

// Get 获取会话历史记录
func (s *DistributedSessionStore) Get(sessionID string) []Message {
	val, err := s.KVClient.Get("session:" + sessionID)
	if err != nil || val == nil {
		return nil
	}

	var history []Message
	err = json.Unmarshal(val, &history)
	if err != nil {
		return nil
	}
	return history
}

// Set 设置会话历史记录
func (s *DistributedSessionStore) Set(meta SessionMeta, history []Message) error {
	data, err := json.Marshal(history)
	if err != nil {
		return err
	}

	// 存储到KV存储
	err = s.KVClient.Set("session:"+meta.SessionID, data)
	if err != nil {
		return err
	}

	// 可选：同步到本地索引
	if s.Index != nil {
		var keywords []*messages.KeyWord
		keywords = append(keywords, &messages.KeyWord{Word: meta.SessionID})
		if meta.UserID != "" {
			keywords = append(keywords, &messages.KeyWord{Word: meta.UserID})
		}
		if meta.DeviceID != "" {
			keywords = append(keywords, &messages.KeyWord{Word: meta.DeviceID})
		}
		for _, tag := range meta.Tags {
			keywords = append(keywords, &messages.KeyWord{Word: tag})
		}
		doc := messages.Document{
			Id:      meta.SessionID,
			KeWords: keywords,
			Bytes:   data,
		}
		s.Index.Add(doc)
	}

	return nil
}

// SetWithVersion 设置带版本的会话历史记录
func (s *DistributedSessionStore) SetWithVersion(meta SessionMeta, history []Message, version int64) error {
	data, err := json.Marshal(history)
	if err != nil {
		return err
	}

	// 存储到KV存储，带版本号
	err = s.KVClient.Set(fmt.Sprintf("session:%s:%d", meta.SessionID, version), data)
	if err != nil {
		return err
	}

	// 更新latest
	err = s.KVClient.Set(fmt.Sprintf("session:%s:latest", meta.SessionID), data)
	if err != nil {
		return err
	}

	// 可选：同步到本地索引
	if s.Index != nil {
		// 创建指针切片
		keywords := []*messages.KeyWord{{Word: meta.SessionID}}

		// 添加版本文档
		doc := messages.Document{
			Id:      fmt.Sprintf("%s:%d", meta.SessionID, version),
			KeWords: keywords,
			Bytes:   data,
		}
		s.Index.Add(doc)

		// 更新latest
		latestDoc := messages.Document{
			Id:      fmt.Sprintf("%s:latest", meta.SessionID),
			KeWords: keywords, // 复用同一个keywords切片
			Bytes:   data,
		}
		s.Index.Add(latestDoc)
	}

	return nil
}

// GetVersion 获取指定版本的会话历史记录
func (s *DistributedSessionStore) GetVersion(sessionID string, version int64) []Message {
	key := fmt.Sprintf("session:%s:%d", sessionID, version)
	val, err := s.KVClient.Get(key)
	if err != nil || val == nil {
		return nil
	}

	var history []Message
	err = json.Unmarshal(val, &history)
	if err != nil {
		return nil
	}
	return history
}

// SearchSessions 多字段检索，仅当本地索引可用时有效
func (s *DistributedSessionStore) SearchSessions(query *messages.TermQuery) [][]Message {
	if s.Index == nil {
		return nil
	}

	results := s.Index.Search(query, 0, 0, nil)
	var histories [][]Message
	for _, raw := range results {
		var history []Message
		_ = json.Unmarshal([]byte(raw), &history)
		histories = append(histories, history)
	}
	return histories
}

type SessionMeta struct {
	SessionID string
	UserID    string
	DeviceID  string
	Tags      []string
}

func (s *SkipListSessionStore) Set(meta SessionMeta, history []Message) error {
	data, _ := json.Marshal(history)
	var keywords []*messages.KeyWord
	keywords = append(keywords, &messages.KeyWord{Word: meta.SessionID})
	if meta.UserID != "" {
		keywords = append(keywords, &messages.KeyWord{Word: meta.UserID})
	}
	if meta.DeviceID != "" {
		keywords = append(keywords, &messages.KeyWord{Word: meta.DeviceID})
	}
	for _, tag := range meta.Tags {
		keywords = append(keywords, &messages.KeyWord{Word: tag})
	}
	doc := messages.Document{
		Id:      meta.SessionID,
		KeWords: keywords,
		Bytes:   data,
	}
	s.Index.Add(doc)
	return nil
}

// SearchSessions 多字段检索
func (s *SkipListSessionStore) SearchSessions(query *messages.TermQuery) [][]Message {
	results := s.Index.Search(query, 0, 0, nil)
	var histories [][]Message
	for _, raw := range results {
		var history []Message
		_ = json.Unmarshal([]byte(raw), &history)
		histories = append(histories, history)
	}
	return histories
}
