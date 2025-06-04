package service

import (
	"encoding/json"
	"fmt"
	"seetaSearch/index"
	"seetaSearch/messages"
)

type SkipListSessionStore struct {
	Index *index.SkipListInvertedIndex
}

func NewSkipListSessionStore(idx *index.SkipListInvertedIndex) *SkipListSessionStore {
	return &SkipListSessionStore{Index: idx}
}

func (s *SkipListSessionStore) Get(sessionID string) []Message {
	// 用sessionID作为keyword查找
	result := s.Index.Search(&messages.TermQuery{Keyword: &messages.KeyWord{Word: sessionID}}, 0, 0, nil)
	if len(result) == 0 {
		return nil
	}
	var history []Message
	_ = json.Unmarshal([]byte(result[0]), &history)
	return history
}

func (s *SkipListSessionStore) SetWithVersion(meta SessionMeta, history []Message, version int64) error {
	data, _ := json.Marshal(history)
	doc := messages.Document{
		Id:      fmt.Sprintf("%s:%d", meta.SessionID, version),
		KeWords: []messages.KeyWord{{Word: meta.SessionID}},
		Content: string(data),
	}
	s.Index.Add(doc)
	// 更新latest
	latestDoc := messages.Document{
		Id:      fmt.Sprintf("%s:latest", meta.SessionID),
		KeWords: []messages.KeyWord{{Word: meta.SessionID}},
		Content: string(data),
	}
	s.Index.Add(latestDoc)
	return nil
}

func (s *SkipListSessionStore) GetVersion(sessionID string, version int64) []Message {
	result := s.Index.Search(&messages.TermQuery{Keyword: &messages.KeyWord{Word: fmt.Sprintf("%s:%d", sessionID, version)}}, 0, 0, nil)
	if len(result) == 0 {
		return nil
	}
	var history []Message
	_ = json.Unmarshal([]byte(result[0]), &history)
	return history
}

type DistributedSessionStore struct {
	KVClient DistributedKV                // etcd/redis client
	Index    *index.SkipListInvertedIndex // 可选本地加速
}

func (s *DistributedSessionStore) Get(sessionID string) []Message {
	val, _ := s.KVClient.Get("session:" + sessionID)
	// ...反序列化...
}

func (s *DistributedSessionStore) Set(meta SessionMeta, history []Message) error {
	// ...序列化...
	s.KVClient.Set("session:"+meta.SessionID, data)
	// 可选：同步到本地索引
}

type SessionMeta struct {
	SessionID string
	UserID    string
	DeviceID  string
	Tags      []string
}

func (s *SkipListSessionStore) Set(meta SessionMeta, history []Message) error {
	data, _ := json.Marshal(history)
	var keywords []messages.KeyWord
	keywords = append(keywords, messages.KeyWord{Word: meta.SessionID})
	if meta.UserID != "" {
		keywords = append(keywords, messages.KeyWord{Word: meta.UserID})
	}
	if meta.DeviceID != "" {
		keywords = append(keywords, messages.KeyWord{Word: meta.DeviceID})
	}
	for _, tag := range meta.Tags {
		keywords = append(keywords, messages.KeyWord{Word: tag})
	}
	doc := messages.Document{
		Id:      meta.SessionID,
		KeWords: keywords,
		Bytes:   data,
	}
	s.Index.Add(doc)
	return nil
}

// 多字段检索
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
