package service

import (
	db2 "VectorSphere/src/db"
	"VectorSphere/src/index"
	"encoding/json"
	"fmt"
	"github.com/gorilla/websocket"
	"github.com/pkoukk/tiktoken-go"
	"net/http"
	"sync"
	"time"
)

/*
* 请求体
  - {
    "query": "请分析这份报告的主要结论",
    "history": [
    {"role": "user", "content": "你好", "time": 1710000000},
    {"role": "assistant", "content": "你好，有什么可以帮您？", "time": 1710000001}
    ],
    "system_instruction": "你是专业的数据分析助手",
    "top_k": 5
    }
*/

/**
* 响应体
{
  "result": "报告的主要结论是……",
  "history": [
    {"role": "user", "content": "你好", "time": 1710000000},
    {"role": "assistant", "content": "你好，有什么可以帮您？", "time": 1710000001},
    {"role": "user", "content": "请分析这份报告的主要结论", "time": 1710000020},
    {"role": "assistant", "content": "报告的主要结论是……", "time": 1710000021}
  ]
}
*/

type SessionStore struct {
	mu       sync.RWMutex
	sessions map[string][]Message // session_id -> history
}

func NewSessionStore() *SessionStore {
	return &SessionStore{sessions: make(map[string][]Message)}
}

func (s *SessionStore) Get(sessionID string) []Message {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sessions[sessionID]
}

func (s *SessionStore) Set(sessionID string, history []Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[sessionID] = history
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Time    int64  `json:"time"`
}

type TokenCounter interface {
	CountTokens(text string) int
}
type TokenizerManager struct {
	tokenizers map[string]TokenCounter // key: model or lang
	defaultTC  TokenCounter
}

func NewTokenizerManager() *TokenizerManager {
	return &TokenizerManager{
		tokenizers: make(map[string]TokenCounter),
		defaultTC:  &SimpleTokenizer{}, // fallback
	}
}

type SimpleTokenizer struct{}

func (s *SimpleTokenizer) CountTokens(text string) int {
	return len([]rune(text)) / 2 // 简单估算
}

// Register 注册tokenizer
func (m *TokenizerManager) Register(key string, tc TokenCounter) {
	m.tokenizers[key] = tc
}

// Get 获取tokenizer，优先模型名，其次语言，最后fallback
func (m *TokenizerManager) Get(model, lang string) TokenCounter {
	if tc, ok := m.tokenizers[model]; ok {
		return tc
	}
	if tc, ok := m.tokenizers[lang]; ok {
		return tc
	}
	return m.defaultTC
}

// 多模型Tokenizer
// 支持gpt-3.5-turbo、deepseek等

type TikTokenizer struct {
	enc *tiktoken.Tiktoken
}

func NewTikTokenizer(model string) *TikTokenizer {
	enc, _ := tiktoken.EncodingForModel(model)
	return &TikTokenizer{enc: enc}
}

func (t *TikTokenizer) CountTokens(text string) int {
	tokens := t.enc.Encode(text, nil, nil)
	return len(tokens)
}

// 示例tokenizer实现（可用gpt-tokenizer等库替换）
// type SimpleTokenizer struct{}
// func (s *SimpleTokenizer) CountTokens(text string) int {
// 	return len([]rune(text)) / 2 // 简单估算
// }

func trimHistory(history []Message, maxTurns int) []Message {
	if len(history) <= maxTurns {
		return history
	}
	return history[len(history)-maxTurns:]
}

func trimHistoryByToken(history []Message, tokenizer TokenCounter, maxTokens int) []Message {
	total := 0
	for i := len(history) - 1; i >= 0; i-- {
		total += tokenizer.CountTokens(history[i].Content)
		if total > maxTokens {
			return history[i+1:]
		}
	}
	return history
}

// 构建Prompt
func buildPrompt(contextText string, history []Message, userQuery string, systemInstruction string) string {
	prompt := ""
	if systemInstruction != "" {
		prompt += "系统指令：" + systemInstruction + "\n"
	}
	if len(history) > 0 {
		prompt += "历史对话：\n"
		for _, msg := range history {
			role := "用户"
			if msg.Role == "assistant" {
				role = "助手"
			}
			prompt += fmt.Sprintf("%s：%s\n", role, msg.Content)
		}
	}
	prompt += "知识库内容：\n" + contextText + "\n"
	prompt += "用户问题：" + userQuery
	return prompt
}

type LLMAnalysisService struct {
	VectorDB     *db2.VectorDB
	DeepSeek     *DeepSeekClient
	SessionStore *DistributedSessionStore
	Tokenizer    TokenCounter // 支持多模型
}

func NewLLMAnalysisService(vectorDBPath string, numClusters int, deepSeekConfig DeepSeekConfig, badgerDBPath string, skipListIndex *index.SkipListInvertedIndex) *LLMAnalysisService {
	// 初始化向量数据库
	vectorDB := db2.NewVectorDB(vectorDBPath, numClusters)

	// 初始化DeepSeek客户端
	deepSeek := NewDeepSeekClient(deepSeekConfig)

	// 初始化BadgerDB作为会话存储
	badgerDB := new(db2.BadgerDB).NewInstance(badgerDBPath, 1, db2.ERROR, 0.5)
	err := badgerDB.Open()
	if err != nil {
		panic(fmt.Sprintf("Failed to open BadgerDB: %v", err))
	}

	// 创建BadgerKV适配器
	badgerKV := NewBadgerKV(badgerDB)

	// 创建分布式会话存储
	sessionStore := NewDistributedSessionStore(badgerKV, skipListIndex)

	// 初始化分词器（使用默认的简单分词器）
	tokenizer := &SimpleTokenizer{}

	return &LLMAnalysisService{
		VectorDB:     vectorDB,
		DeepSeek:     deepSeek,
		SessionStore: sessionStore,
		Tokenizer:    tokenizer,
	}
}

func (s *LLMAnalysisService) HandleAnalyzeWithSession(w http.ResponseWriter, r *http.Request) {
	sessionID := r.Header.Get("X-Session-Id")
	if sessionID == "" {
		http.Error(w, "Missing session id", 400)
		return
	}
	var req struct {
		Query             string `json:"query"`
		SystemInstruction string `json:"system_instruction,omitempty"`
		TopK              int    `json:"top_k,omitempty"`
		MaxTokens         int    `json:"max_tokens,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", 400)
		return
	}

	// 获取历史对话
	history := s.SessionStore.Get(sessionID)

	// 设置默认值
	if req.TopK == 0 {
		req.TopK = 5
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = 2048
	}

	// 裁剪历史对话
	history = trimHistoryByToken(history, s.Tokenizer, req.MaxTokens)

	// 分析并获取结果
	result, newHistory, err := s.AnalyzeWithHistory(req.Query, req.TopK, history, req.SystemInstruction, req.MaxTokens)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	// 创建会话元数据
	meta := SessionMeta{
		SessionID: sessionID,
		// 可以从请求中获取其他元数据
		UserID:   r.Header.Get("X-User-Id"),
		DeviceID: r.Header.Get("X-Device-Id"),
		Tags:     []string{},
	}

	// 更新会话
	err = s.SessionStore.Set(meta, newHistory)
	if err != nil {
		// 记录错误但不中断响应
		fmt.Printf("Failed to update session: %v\n", err)
	}

	// 返回结果
	json.NewEncoder(w).Encode(map[string]interface{}{
		"result":  result,
		"history": newHistory,
	})
}

// AnalyzeWithHistory 支持多轮对话、token限制
func (s *LLMAnalysisService) AnalyzeWithHistory(userQuery string, topK int, history []Message, systemInstruction string, maxTokens int) (string, []Message, error) {
	history = trimHistoryByToken(history, s.Tokenizer, maxTokens)
	queryVec, err := s.DeepSeek.GetEmbedding(userQuery)
	if err != nil {
		return "", history, fmt.Errorf("embedding failed: %w", err)
	}
	ids, err := s.VectorDB.FindNearest(queryVec, topK, 10)
	if err != nil {
		return "", history, fmt.Errorf("vector search failed: %w", err)
	}
	var contextText string
	for _, id := range ids {
		meta, _ := s.VectorDB.GetMetadata(id.Id)
		content, _ := meta["content"].(string)
		contextText += content + "\n"
	}
	prompt := buildPrompt(contextText, history, userQuery, systemInstruction)
	answer, err := s.DeepSeek.Chat(prompt)
	if err != nil {
		return "", history, fmt.Errorf("LLM call failed: %w", err)
	}
	now := time.Now().Unix()
	history = append(history, Message{Role: "user", Content: userQuery, Time: now})
	history = append(history, Message{Role: "assistant", Content: answer, Time: now + 1})
	return answer, history, nil
}

var upgrader = websocket.Upgrader{}

func (s *LLMAnalysisService) HandleAnalyzeWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		http.Error(w, "upgrade failed", 400)
		return
	}
	defer conn.Close()
	for {
		var req struct {
			SessionID         string `json:"session_id"`
			Query             string `json:"query"`
			SystemInstruction string `json:"system_instruction,omitempty"`
			TopK              int    `json:"top_k,omitempty"`
			MaxTokens         int    `json:"max_tokens,omitempty"`
		}
		if err := conn.ReadJSON(&req); err != nil {
			break
		}
		history := s.SessionStore.Get(req.SessionID)
		if req.TopK == 0 {
			req.TopK = 5
		}
		if req.MaxTokens == 0 {
			req.MaxTokens = 2048
		}
		// 检索知识库内容
		queryVec, err := s.DeepSeek.GetEmbedding(req.Query)
		if err != nil {
			conn.WriteJSON(map[string]string{"error": err.Error()})
			continue
		}
		ids, err := s.VectorDB.FindNearest(queryVec, req.TopK, 10)
		if err != nil {
			conn.WriteJSON(map[string]string{"error": err.Error()})
			continue
		}
		var contextText string
		for _, id := range ids {
			meta, _ := s.VectorDB.GetMetadata(id.Id)
			content, _ := meta["content"].(string)
			contextText += content + "\n"
		}

		// 流式推送大模型输出
		prompt := buildPrompt(contextText, history, req.Query, req.SystemInstruction)
		stream := s.DeepSeek.ChatStream(prompt) // <-chan string
		var answer string
		for chunk := range stream {
			answer += chunk
			conn.WriteJSON(map[string]string{"delta": chunk})
		}
		now := time.Now().Unix()
		history = append(history, Message{Role: "user", Content: req.Query, Time: now})
		history = append(history, Message{Role: "assistant", Content: answer, Time: now + 1})
		// 创建会话元数据
		meta := SessionMeta{
			SessionID: req.SessionID,
			// 可以从请求中获取其他元数据
			UserID:   r.Header.Get("X-User-Id"),
			DeviceID: r.Header.Get("X-Device-Id"),
			Tags:     []string{},
		}
		err = s.SessionStore.Set(meta, history)
		if err != nil {
			return
		}
		conn.WriteJSON(map[string]interface{}{
			"result":  answer,
			"history": history,
		})
	}
}

func (s *LLMAnalysisService) HandleAnalyze(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query             string    `json:"query"`
		History           []Message `json:"history,omitempty"`
		SystemInstruction string    `json:"system_instruction,omitempty"`
		TopK              int       `json:"top_k,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", 400)
		return
	}
	if req.TopK == 0 {
		req.TopK = 5
	}
	// 裁剪历史
	history := trimHistory(req.History, 5)
	// 构建上下文
	result, _, err := s.AnalyzeWithHistory(req.Query, req.TopK, history, req.SystemInstruction, 10)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	// 追加本轮对话
	now := time.Now().Unix()
	history = append(history, Message{Role: "user", Content: req.Query, Time: now})
	history = append(history, Message{Role: "assistant", Content: result, Time: now + 1})
	json.NewEncoder(w).Encode(map[string]interface{}{
		"result":  result,
		"history": history,
	})
}
