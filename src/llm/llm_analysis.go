package llm

import (
	db2 "VectorSphere/src/db"
	"VectorSphere/src/index"
	"VectorSphere/src/library/entity"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/gorilla/websocket"
	"github.com/patrickmn/go-cache"
	"github.com/pkoukk/tiktoken-go"
	"golang.org/x/sync/errgroup"
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

// 缓存配置
const (
	defaultCacheTTL        = 15 * time.Minute
	defaultCleanupInterval = 30 * time.Minute
	defaultTimeout         = 30 * time.Second
	defaultRetryCount      = 3
	defaultRetryDelay      = 1 * time.Second
)

// 查询缓存键生成
func generateCacheKey(query string, topK int, systemInstruction string) string {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("%s:%d:%s", query, topK, systemInstruction)))
	return hex.EncodeToString(h.Sum(nil))
}

type AnalysisService struct {
	VectorDB         *db2.VectorDB
	DeepSeek         *DeepSeekClient
	SessionStore     *DistributedSessionStore
	Tokenizer        TokenCounter
	QueryCache       *cache.Cache  // 查询结果缓存
	EmbedCache       *cache.Cache  // 向量嵌入缓存
	ConcurrencyLimit int           // 并发限制
	Timeout          time.Duration // 超时设置
	RetryCount       int           // 重试次数
	RetryDelay       time.Duration // 重试延迟
}

func NewAnalysisService(vectorDBPath string, numClusters int, deepSeekConfig DeepSeekConfig, badgerDBPath string, skipListIndex *index.SkipListInvertedIndex) *AnalysisService {
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

	// 初始化缓存
	queryCache := cache.New(defaultCacheTTL, defaultCleanupInterval)
	embedCache := cache.New(defaultCacheTTL, defaultCleanupInterval)

	return &AnalysisService{
		VectorDB:         vectorDB,
		DeepSeek:         deepSeek,
		SessionStore:     sessionStore,
		Tokenizer:        tokenizer,
		QueryCache:       queryCache,
		EmbedCache:       embedCache,
		ConcurrencyLimit: 10, // 默认并发限制
		Timeout:          defaultTimeout,
		RetryCount:       defaultRetryCount,
		RetryDelay:       defaultRetryDelay,
	}
}

func (s *AnalysisService) HandleAnalyzeWithSession(w http.ResponseWriter, r *http.Request) {
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

	// 创建上下文，支持超时控制
	ctx, cancel := context.WithTimeout(r.Context(), s.Timeout)
	defer cancel()

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
	result, newHistory, err := s.AnalyzeWithHistoryContext(ctx, req.Query, req.TopK, history, req.SystemInstruction, req.MaxTokens)
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

// AnalyzeWithHistoryContext 支持上下文控制、并行处理和缓存的分析函数
func (s *AnalysisService) AnalyzeWithHistoryContext(ctx context.Context, userQuery string, topK int, history []Message, systemInstruction string, maxTokens int) (string, []Message, error) {
	// 检查缓存
	cacheKey := generateCacheKey(userQuery, topK, systemInstruction)
	if cachedResult, found := s.QueryCache.Get(cacheKey); found {
		result := cachedResult.(string)
		now := time.Now().Unix()
		newHistory := append(history, Message{Role: "user", Content: userQuery, Time: now})
		newHistory = append(newHistory, Message{Role: "assistant", Content: result, Time: now + 1})
		return result, newHistory, nil
	}

	// 裁剪历史
	history = trimHistoryByToken(history, s.Tokenizer, maxTokens)

	// 创建错误组，用于并行处理
	g, ctx := errgroup.WithContext(ctx)

	// 向量嵌入和检索可以并行执行
	var queryVec []float64
	var ids []entity.Result
	var embeddingErr, searchErr error

	// 并行获取向量嵌入
	g.Go(func() error {
		// 检查嵌入缓存
		if cachedVec, found := s.EmbedCache.Get(userQuery); found {
			queryVec = cachedVec.([]float64)
			return nil
		}

		// 使用重试机制获取嵌入
		for i := 0; i <= s.RetryCount; i++ {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				var err error
				queryVec, err = s.DeepSeek.GetEmbedding(userQuery)
				if err == nil {
					// 缓存嵌入结果
					s.EmbedCache.Set(userQuery, queryVec, cache.DefaultExpiration)
					return nil
				}
				embeddingErr = fmt.Errorf("embedding failed (attempt %d/%d): %w", i+1, s.RetryCount+1, err)
				if i < s.RetryCount {
					time.Sleep(s.RetryDelay * time.Duration(i+1))
				}
			}
		}
		return embeddingErr
	})

	// 等待嵌入完成
	if err := g.Wait(); err != nil {
		return "", history, err
	}

	// 重置错误组，用于下一阶段并行处理
	g, ctx = errgroup.WithContext(ctx)

	// 并行执行向量检索
	g.Go(func() error {
		// 使用重试机制进行向量检索
		for i := 0; i <= s.RetryCount; i++ {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				var err error
				ids, err = s.VectorDB.FindNearest(queryVec, topK, 10)
				if err == nil {
					return nil
				}
				searchErr = fmt.Errorf("vector search failed (attempt %d/%d): %w", i+1, s.RetryCount+1, err)
				if i < s.RetryCount {
					time.Sleep(s.RetryDelay * time.Duration(i+1))
				}
			}
		}
		return searchErr
	})

	// 等待检索完成
	if err := g.Wait(); err != nil {
		return "", history, err
	}

	// 并行获取元数据和内容
	type contentResult struct {
		id      string
		content string
	}
	contentChan := make(chan contentResult, topK)
	var contentWg sync.WaitGroup

	// 限制并发获取元数据的goroutine数量
	semaphore := make(chan struct{}, s.ConcurrencyLimit)
	for _, id := range ids {
		contentWg.Add(1)
		go func(id entity.Result) {
			defer contentWg.Done()
			semaphore <- struct{}{}        // 获取信号量
			defer func() { <-semaphore }() // 释放信号量

			meta, exists := s.VectorDB.GetMetadata(id.Id)
			if !exists {
				return
			}
			content, ok := meta["content"].(string)
			if !ok {
				return
			}
			select {
			case contentChan <- contentResult{id: id.Id, content: content}:
			case <-ctx.Done():
				return
			}
		}(id)
	}

	// 等待所有内容获取完成或上下文取消
	go func() {
		contentWg.Wait()
		close(contentChan)
	}()

	// 收集内容
	var contextText string
	for res := range contentChan {
		contextText += res.content + "\n"
	}

	// 构建提示词
	prompt := buildPrompt(contextText, history, userQuery, systemInstruction)

	// 调用LLM获取回答
	var answer string
	var llmErr error

	// 使用重试机制调用LLM
	for i := 0; i <= s.RetryCount; i++ {
		select {
		case <-ctx.Done():
			return "", history, ctx.Err()
		default:
			var err error
			answer, err = s.DeepSeek.Chat(prompt)
			if err == nil {
				break
			}
			llmErr = fmt.Errorf("LLM call failed (attempt %d/%d): %w", i+1, s.RetryCount+1, err)
			if i < s.RetryCount {
				time.Sleep(s.RetryDelay * time.Duration(i+1))
			}
		}
	}

	if llmErr != nil {
		return "", history, llmErr
	}

	// 缓存查询结果
	s.QueryCache.Set(cacheKey, answer, cache.DefaultExpiration)

	// 更新历史
	now := time.Now().Unix()
	history = append(history, Message{Role: "user", Content: userQuery, Time: now})
	history = append(history, Message{Role: "assistant", Content: answer, Time: now + 1})

	return answer, history, nil
}

// AnalyzeWithHistory 支持多轮对话、token限制
func (s *AnalysisService) AnalyzeWithHistory(userQuery string, topK int, history []Message, systemInstruction string, maxTokens int) (string, []Message, error) {
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

func (s *AnalysisService) HandleAnalyzeWS(w http.ResponseWriter, r *http.Request) {
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

		// 创建上下文，支持超时控制
		ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)

		history := s.SessionStore.Get(req.SessionID)
		if req.TopK == 0 {
			req.TopK = 5
		}
		if req.MaxTokens == 0 {
			req.MaxTokens = 2048
		}

		// 检查缓存
		cacheKey := generateCacheKey(req.Query, req.TopK, req.SystemInstruction)
		var answer string
		var useCache bool

		if cachedResult, found := s.QueryCache.Get(cacheKey); found {
			answer = cachedResult.(string)
			useCache = true
			// 即使使用缓存，也分块发送以模拟流式响应
			chunkSize := 10
			for i := 0; i < len(answer); i += chunkSize {
				end := i + chunkSize
				if end > len(answer) {
					end = len(answer)
				}
				chunk := answer[i:end]
				conn.WriteJSON(map[string]string{"delta": chunk})
				time.Sleep(50 * time.Millisecond) // 模拟流式传输的延迟
			}
		} else {
			// 并行获取向量嵌入
			var queryVec []float64
			var embeddingErr error

			// 检查嵌入缓存
			if cachedVec, found := s.EmbedCache.Get(req.Query); found {
				queryVec = cachedVec.([]float64)
			} else {
				// 使用重试机制获取嵌入
				for i := 0; i <= s.RetryCount; i++ {
					select {
					case <-ctx.Done():
						conn.WriteJSON(map[string]string{"error": "timeout getting embedding"})
						cancel()
						continue
					default:
						queryVec, embeddingErr = s.DeepSeek.GetEmbedding(req.Query)
						if embeddingErr == nil {
							// 缓存嵌入结果
							s.EmbedCache.Set(req.Query, queryVec, cache.DefaultExpiration)
							break
						}
						if i < s.RetryCount {
							time.Sleep(s.RetryDelay * time.Duration(i+1))
						}
					}
				}
				if embeddingErr != nil {
					conn.WriteJSON(map[string]string{"error": embeddingErr.Error()})
					cancel()
					continue
				}
			}

			// 向量检索
			ids, err := s.VectorDB.FindNearest(queryVec, req.TopK, 10)
			if err != nil {
				conn.WriteJSON(map[string]string{"error": err.Error()})
				cancel()
				continue
			}

			// 并行获取元数据和内容
			type contentResult struct {
				id      string
				content string
			}
			contentChan := make(chan contentResult, req.TopK)
			var contentWg sync.WaitGroup

			// 限制并发获取元数据的goroutine数量
			semaphore := make(chan struct{}, s.ConcurrencyLimit)
			for _, id := range ids {
				contentWg.Add(1)
				go func(id entity.Result) {
					defer contentWg.Done()
					semaphore <- struct{}{}        // 获取信号量
					defer func() { <-semaphore }() // 释放信号量

					meta, exists := s.VectorDB.GetMetadata(id.Id)
					if !exists {
						return
					}
					content, ok := meta["content"].(string)
					if !ok {
						return
					}
					select {
					case contentChan <- contentResult{id: id.Id, content: content}:
					case <-ctx.Done():
						return
					}
				}(id)
			}

			// 等待所有内容获取完成或上下文取消
			go func() {
				contentWg.Wait()
				close(contentChan)
			}()

			// 收集内容
			var contextText string
			for res := range contentChan {
				contextText += res.content + "\n"
			}

			// 构建提示词
			prompt := buildPrompt(contextText, history, req.Query, req.SystemInstruction)

			// 流式推送大模型输出
			stream := s.DeepSeek.ChatStream(prompt)
			answer = ""
			for chunk := range stream {
				answer += chunk
				conn.WriteJSON(map[string]string{"delta": chunk})
			}

			// 缓存查询结果
			s.QueryCache.Set(cacheKey, answer, cache.DefaultExpiration)
		}

		// 更新历史
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

		// 更新会话
		err = s.SessionStore.Set(meta, history)
		if err != nil {
			conn.WriteJSON(map[string]string{"error": "Failed to update session: " + err.Error()})
		}

		// 发送完整结果
		conn.WriteJSON(map[string]interface{}{
			"result":  answer,
			"history": history,
			"cached":  useCache,
		})

		// 取消上下文
		cancel()
	}
}

func (s *AnalysisService) HandleAnalyze(w http.ResponseWriter, r *http.Request) {
	// 创建上下文，支持超时控制
	ctx, cancel := context.WithTimeout(r.Context(), s.Timeout)
	defer cancel()

	var req struct {
		Query             string    `json:"query"`
		History           []Message `json:"history,omitempty"`
		SystemInstruction string    `json:"system_instruction,omitempty"`
		TopK              int       `json:"top_k,omitempty"`
		MaxTokens         int       `json:"max_tokens,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", 400)
		return
	}

	if req.TopK == 0 {
		req.TopK = 5
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = 2048
	}

	// 裁剪历史
	history := trimHistoryByToken(req.History, s.Tokenizer, req.MaxTokens)

	// 构建上下文
	result, newHistory, err := s.AnalyzeWithHistoryContext(ctx, req.Query, req.TopK, history, req.SystemInstruction, req.MaxTokens)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	// 返回结果
	json.NewEncoder(w).Encode(map[string]interface{}{
		"result":  result,
		"history": newHistory,
	})
}
