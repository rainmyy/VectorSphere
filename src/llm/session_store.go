package llm

import (
	"VectorSphere/src/db"
	"VectorSphere/src/index"
	"VectorSphere/src/messages"
	"encoding/json"
	"fmt"
	"time"
)

type SkipListSessionStore struct {
	Index *index.SkipListInvertedIndex
}

// DistributedKV 定义分布式KV存储接口
type DistributedKV interface {
	Get(key string) ([]byte, error)
	Set(key string, value []byte) error
	Delete(key string) error
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

// Delete 实现DistributedKV接口的Delete方法
func (b *BadgerKV) Delete(key string) error {
	return b.db.Del([]byte(key))
}

type DistributedSessionStore struct {
	KVClient DistributedKV                // etcd/redis client
	Index    *index.SkipListInvertedIndex // 可选本地加速
	config   SessionConfig                // 会话配置
}

type SessionConfig struct {
	MaxSessionsPerUser int           // 每个用户的最大会话数
	DefaultTTL         time.Duration // 默认会话过期时间
	CleanupInterval    time.Duration // 清理间隔
}

// NewDistributedSessionStore 创建DistributedSessionStore实例
func NewDistributedSessionStore(kvClient DistributedKV, idx *index.SkipListInvertedIndex, config SessionConfig) *DistributedSessionStore {
	return &DistributedSessionStore{
		KVClient: kvClient,
		Index:    idx,
		config:   config,
	}
}

// EnforceSessionLimits 在创建新会话前检查限制
func (s *DistributedSessionStore) EnforceSessionLimits(userID string) error {
	userSessions := s.GetUserSessions(userID)
	if len(userSessions) >= s.config.MaxSessionsPerUser {
		// 可以选择删除最旧地会话或返回错误
		oldestSession := s.FindOldestSession(userSessions)
		return s.DeleteSession(oldestSession)
	}
	return nil
}

// GetUserSessions 获取用户的所有会话
func (s *DistributedSessionStore) GetUserSessions(userID string) []string {
	// 实现获取用户会话的逻辑
	// 可以通过在KV存储中使用特定前缀来存储用户会话映射
	// 例如: "user_sessions:" + userID
	key := "user_sessions:" + userID
	data, err := s.KVClient.Get(key)
	if err != nil || data == nil {
		return []string{}
	}

	var sessions []string
	err = json.Unmarshal(data, &sessions)
	if err != nil {
		return []string{}
	}
	return sessions
}

// FindOldestSession 查找最旧的会话
func (s *DistributedSessionStore) FindOldestSession(sessions []string) string {
	if len(sessions) == 0 {
		return ""
	}

	var oldestSession string
	var oldestTime time.Time

	// 初始化为未来的时间点
	oldestTime = time.Now().Add(100 * 365 * 24 * time.Hour) // 约100年后

	for _, sessionID := range sessions {
		// 获取会话元数据
		metaKey := "session_meta:" + sessionID
		metaData, err := s.KVClient.Get(metaKey)
		if err != nil || metaData == nil {
			continue
		}

		var meta SessionMeta
		err = json.Unmarshal(metaData, &meta)
		if err != nil {
			continue
		}

		// 比较创建时间
		if meta.ExpireTime.Before(oldestTime) {
			oldestTime = meta.ExpireTime
			oldestSession = sessionID
		}
	}

	return oldestSession
}

// DeleteSession 删除会话
func (s *DistributedSessionStore) DeleteSession(sessionID string) error {
	if sessionID == "" {
		return fmt.Errorf("无效的会话ID")
	}

	// 获取会话元数据以找到用户ID
	metaKey := "session_meta:" + sessionID
	metaData, err := s.KVClient.Get(metaKey)
	if err != nil || metaData == nil {
		return fmt.Errorf("会话元数据不存在")
	}

	var meta SessionMeta
	err = json.Unmarshal(metaData, &meta)
	if err != nil {
		return fmt.Errorf("解析会话元数据失败: %v", err)
	}

	// 1. 从用户会话列表中移除
	userSessionsKey := "user_sessions:" + meta.UserID
	userSessions := s.GetUserSessions(meta.UserID)

	// 过滤掉要删除的会话
	newSessions := make([]string, 0, len(userSessions))
	for _, id := range userSessions {
		if id != sessionID {
			newSessions = append(newSessions, id)
		}
	}

	// 更新用户会话列表
	updatedData, err := json.Marshal(newSessions)
	if err != nil {
		return fmt.Errorf("序列化会话列表失败: %v", err)
	}

	err = s.KVClient.Set(userSessionsKey, updatedData)
	if err != nil {
		return fmt.Errorf("更新用户会话列表失败: %v", err)
	}

	// 2. 删除会话数据
	sessionKey := "session:" + sessionID
	err = s.KVClient.Delete(sessionKey)
	if err != nil {
		return fmt.Errorf("删除会话数据失败: %v", err)
	}

	// 3. 删除会话元数据
	err = s.KVClient.Delete(metaKey)
	if err != nil {
		return fmt.Errorf("删除会话元数据失败: %v", err)
	}

	// 4. 如果使用本地索引，也需要从索引中删除
	if s.Index != nil {
		// 从索引中删除所有关键词
		doc := s.formatDocument(meta, metaData)
		// 从索引中删除所有关键词
		for _, keyword := range doc.KeWords {
			s.Index.Delete(doc.ScoreId, keyword)
		}
	}

	return nil
}

// SetWithExpiration 设置会话时添加过期时间
func (s *DistributedSessionStore) SetWithExpiration(meta SessionMeta, history []Message, ttl time.Duration) error {
	meta.ExpireTime = time.Now().Add(ttl)
	return s.Set(meta, history)
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
		doc := s.formatDocument(meta, data)
		s.Index.Add(doc)
	}

	return nil
}

// 修改formatDocument方法，添加ScoreId设置
func (s *DistributedSessionStore) formatDocument(meta SessionMeta, data []byte) messages.Document {
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

	// 将SessionID转换为int64作为ScoreId
	var scoreId int64
	// 简单的哈希方法，将字符串转换为int64
	for i, c := range meta.SessionID {
		scoreId = scoreId*31 + int64(c)
		if i > 10 { // 限制计算长度，避免溢出
			break
		}
	}

	doc := messages.Document{
		Id:      meta.SessionID,
		ScoreId: scoreId,
		KeWords: keywords,
		Bytes:   data,
	}
	return doc
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
	SessionID  string
	UserID     string
	DeviceID   string
	Tags       []string
	ExpireTime time.Time // 新增字段
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

// StartCleanupTask 在DistributedSessionStore中添加
func (s *DistributedSessionStore) StartCleanupTask(interval time.Duration) {
	ticker := time.NewTicker(interval)
	go func() {
		for range ticker.C {
			s.CleanupExpiredSessions()
		}
	}()
}

func (s *DistributedSessionStore) CleanupExpiredSessions() error {
	// 查询过期的会话
	expiredSessions := s.FindExpiredSessions()

	// 删除过期会话
	for _, sessionID := range expiredSessions {
		s.KVClient.Delete("session:" + sessionID)
		// 如果使用本地索引，也需要从索引中删除
		if s.Index != nil {
			// 从索引中删除
		}
	}

	return nil
}

func (s *DistributedSessionStore) FindExpiredSessions() []string {
	// 实现查找过期会话的逻辑
	// 可以在KV存储中使用特殊前缀存储过期时间信息
	// 或者定期扫描所有会话并检查元数据

	// 存储过期的会话ID
	var expiredSessionIDs []string

	// 当前时间
	now := time.Now()

	// 获取所有用户
	userIDs := s.getAllUserIDs()

	// 遍历所有用户的会话
	for _, userID := range userIDs {
		// 获取用户的所有会话
		userSessions := s.GetUserSessions(userID)

		// 检查每个会话是否过期
		for _, sessionID := range userSessions {
			// 获取会话元数据
			metaKey := "session_meta:" + sessionID
			metaData, err := s.KVClient.Get(metaKey)
			if err != nil || metaData == nil {
				continue
			}

			var meta SessionMeta
			err = json.Unmarshal(metaData, &meta)
			if err != nil {
				continue
			}

			// 检查是否过期
			if meta.ExpireTime.Before(now) {
				expiredSessionIDs = append(expiredSessionIDs, sessionID)
			}
		}
	}

	return expiredSessionIDs
}

// getAllUserIDs 获取所有用户ID
// 这是一个辅助方法，用于获取所有用户ID
func (s *DistributedSessionStore) getAllUserIDs() []string {
	// 实际实现中，可能需要根据存储引擎的特性来实现
	// 例如，如果使用的是支持前缀扫描的KV存储，可以扫描所有"user_sessions:"前缀的键

	// 这里我们假设有一个存储所有用户ID的键
	allUserIDsKey := "all_users"
	data, err := s.KVClient.Get(allUserIDsKey)
	if err != nil || data == nil {
		return []string{}
	}

	var userIDs []string
	err = json.Unmarshal(data, &userIDs)
	if err != nil {
		return []string{}
	}

	return userIDs
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
