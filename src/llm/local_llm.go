package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// LocalConfig Python服务配置
type LocalConfig struct {
	// 基础配置继承自LLMConfig
	LLMConfig
	// 服务地址
	ServiceURL string `json:"service_url"`
	// 模型信息缓存
	modelInfo *ModelInfo
}

// LocalClient Python服务客户端
type LocalClient struct {
	Config     LocalConfig
	BaseClient *BaseHTTPClient
}

// NewLocalClient 创建Python服务客户端
func NewLocalClient(cfg LLMConfig) (*LocalClient, error) {
	// 从额外参数中获取Python服务配置
	serviceURL, ok := cfg.ExtraParams["service_url"].(string)
	if !ok || serviceURL == "" {
		serviceURL = "http://localhost:5000" // 默认服务地址
	}

	// 创建Python服务配置
	pythonConfig := LocalConfig{
		LLMConfig:  cfg,
		ServiceURL: serviceURL,
	}

	// 创建客户端
	client := &LocalClient{
		Config:     pythonConfig,
		BaseClient: NewBaseHTTPClient("", cfg.GetTimeout()), // Python服务不需要API密钥
	}

	// 获取模型信息
	modelInfo, err := client.fetchModelInfo()
	if err != nil {
		return nil, fmt.Errorf("获取Python服务模型信息失败: %w", err)
	}

	// 缓存模型信息
	client.Config.modelInfo = modelInfo

	return client, nil
}

// fetchModelInfo 获取模型信息
func (c *LocalClient) fetchModelInfo() (*ModelInfo, error) {
	// 构建请求URL
	url := fmt.Sprintf("%s/api/model_info", c.Config.ServiceURL)

	// 发送请求
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("请求模型信息失败: %w", err)
	}
	defer resp.Body.Close()

	// 读取响应
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取模型信息响应失败: %w", err)
	}

	// 解析响应
	var result struct {
		Embedding struct {
			Name         string `json:"name"`
			EmbeddingDim int    `json:"embedding_dim"`
		} `json:"embedding"`
		Generation struct {
			Name             string `json:"name"`
			MaxTokens        int    `json:"max_tokens"`
			SupportStreaming bool   `json:"support_streaming"`
		} `json:"generation"`
		Device string `json:"device"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("解析模型信息响应失败: %w", err)
	}

	// 创建模型信息
	modelInfo := &ModelInfo{
		Name:             result.Embedding.Name,
		Provider:         "PythonService",
		MaxTokens:        result.Generation.MaxTokens,
		ContextWindow:    result.Generation.MaxTokens,
		EmbeddingDim:     result.Embedding.EmbeddingDim,
		SupportStreaming: result.Generation.SupportStreaming,
	}

	return modelInfo, nil
}

// GetEmbedding 获取文本的向量嵌入
func (c *LocalClient) GetEmbedding(text string) ([]float64, error) {
	return c.GetEmbeddingWithContext(context.Background(), text)
}

// GetEmbeddingWithContext 带上下文控制的向量嵌入
func (c *LocalClient) GetEmbeddingWithContext(ctx context.Context, text string) ([]float64, error) {
	// 构建请求URL
	url := fmt.Sprintf("%s/api/embedding", c.Config.ServiceURL)

	// 构建请求体
	reqBody := map[string]interface{}{
		"text": text,
	}

	// 发送请求
	respBody, err := c.BaseClient.DoRequest(ctx, http.MethodPost, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("Python服务嵌入请求失败: %w", err)
	}

	// 解析响应
	var result struct {
		Embedding []float64 `json:"embedding"`
		Error     string    `json:"error,omitempty"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("解析Python服务嵌入响应失败: %w", err)
	}

	// 检查错误
	if result.Error != "" {
		return nil, fmt.Errorf("Python服务嵌入错误: %s", result.Error)
	}

	// 检查结果
	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("Python服务返回空嵌入")
	}

	return result.Embedding, nil
}

// Chat 调用大模型进行对话
func (c *LocalClient) Chat(prompt string) (string, error) {
	return c.ChatWithContext(context.Background(), prompt)
}

// ChatWithContext 带上下文控制的大模型对话
func (c *LocalClient) ChatWithContext(ctx context.Context, prompt string) (string, error) {
	// 构建请求URL
	url := fmt.Sprintf("%s/api/chat", c.Config.ServiceURL)

	// 构建请求体
	reqBody := map[string]interface{}{
		"prompt":      prompt,
		"temperature": c.Config.GetTemperature(),
	}

	// 添加可选参数
	if c.Config.MaxTokens > 0 {
		reqBody["max_tokens"] = c.Config.MaxTokens
	}
	if c.Config.TopP > 0 {
		reqBody["top_p"] = c.Config.TopP
	}

	// 发送请求
	respBody, err := c.BaseClient.DoRequest(ctx, http.MethodPost, url, reqBody)
	if err != nil {
		return "", fmt.Errorf("Python服务聊天请求失败: %w", err)
	}

	// 解析响应
	var result struct {
		Text  string `json:"text"`
		Error string `json:"error,omitempty"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("解析Python服务聊天响应失败: %w", err)
	}

	// 检查错误
	if result.Error != "" {
		return "", fmt.Errorf("Python服务聊天错误: %s", result.Error)
	}

	return result.Text, nil
}

// ChatStream 流式调用大模型
func (c *LocalClient) ChatStream(prompt string) <-chan string {
	return c.ChatStreamWithContext(context.Background(), prompt)
}

// ChatStreamWithContext 带上下文控制的流式调用
func (c *LocalClient) ChatStreamWithContext(ctx context.Context, prompt string) <-chan string {
	out := make(chan string)
	go func() {
		defer close(out)

		// 构建请求URL
		url := fmt.Sprintf("%s/api/chat_stream", c.Config.ServiceURL)

		// 构建请求体
		reqBody := map[string]interface{}{
			"prompt":      prompt,
			"temperature": c.Config.GetTemperature(),
		}

		// 添加可选参数
		if c.Config.MaxTokens > 0 {
			reqBody["max_tokens"] = c.Config.MaxTokens
		}
		if c.Config.TopP > 0 {
			reqBody["top_p"] = c.Config.TopP
		}

		// 将请求体转换为JSON
		reqJSON, err := json.Marshal(reqBody)
		if err != nil {
			out <- fmt.Sprintf("[ERROR] 序列化请求体失败: %v", err)
			return
		}

		// 创建请求
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(reqJSON)))
		if err != nil {
			out <- fmt.Sprintf("[ERROR] 创建请求失败: %v", err)
			return
		}

		// 设置请求头
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")

		// 发送请求
		client := &http.Client{Timeout: time.Minute * 10} // 流式请求需要更长的超时时间
		resp, err := client.Do(req)
		if err != nil {
			out <- fmt.Sprintf("[ERROR] 发送请求失败: %v", err)
			return
		}
		defer resp.Body.Close()

		// 检查响应状态
		if resp.StatusCode != http.StatusOK {
			out <- fmt.Sprintf("[ERROR] 服务返回错误状态码: %d", resp.StatusCode)
			return
		}

		// 读取SSE流
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()

			// 跳过空行和非数据行
			if line == "" || !strings.HasPrefix(line, "data: ") {
				continue
			}

			// 提取数据部分
			dataJSON := strings.TrimPrefix(line, "data: ")

			// 解析数据
			var data struct {
				Text  string `json:"text"`
				Error string `json:"error,omitempty"`
			}

			if err := json.Unmarshal([]byte(dataJSON), &data); err != nil {
				out <- fmt.Sprintf("[ERROR] 解析数据失败: %v", err)
				continue
			}

			// 检查错误
			if data.Error != "" {
				out <- fmt.Sprintf("[ERROR] %s", data.Error)
				continue
			}

			// 发送文本片段
			if data.Text != "" {
				out <- data.Text
			}
		}

		// 检查扫描错误
		if err := scanner.Err(); err != nil {
			out <- fmt.Sprintf("[ERROR] 读取流失败: %v", err)
		}
	}()
	return out
}

// GetModelInfo 获取模型信息
func (c *LocalClient) GetModelInfo() ModelInfo {
	// 如果已缓存模型信息，则直接返回
	if c.Config.modelInfo != nil {
		return *c.Config.modelInfo
	}

	// 否则返回默认信息
	return ModelInfo{
		Name:             "python-service-model",
		Provider:         "PythonService",
		MaxTokens:        1024,
		ContextWindow:    1024,
		EmbeddingDim:     384, // 默认维度
		SupportStreaming: true,
	}
}

/*
	// 创建Python服务LLM配置
	config := llm.LLMConfig{
		Type:      llm.PythonServiceLLM,
		MaxTokens: 100,
		ExtraParams: map[string]interface{}{
			"service_url": "http://localhost:5000", // Python服务地址
		},
	}

	// 创建Python服务LLM客户端
	client, err := llm.NewLLMClient(config)
	if err != nil {
		log.Fatalf("创建Python服务LLM客户端失败: %v", err)
	}

	// 获取模型信息
	modelInfo := client.GetModelInfo()
	fmt.Printf("模型信息: %+v\n", modelInfo)

	// 获取文本嵌入
	text := "这是一个测试文本，用于获取向量嵌入。"
	embedding, err := client.GetEmbedding(text)
	if err != nil {
		log.Fatalf("获取文本嵌入失败: %v", err)
	}
	fmt.Printf("嵌入维度: %d\n", len(embedding))

	// 调用大模型对话
	prompt := "请介绍一下自然语言处理技术。"
	response, err := client.Chat(prompt)
	if err != nil {
		log.Fatalf("调用大模型对话失败: %v", err)
	}
	fmt.Printf("大模型回复: %s\n", response)

	// 流式调用大模型
	fmt.Println("流式调用大模型:")
	stream := client.ChatStream("请列举三种常见的机器学习算法。")
	for text := range stream {
		if text[:7] == "[ERROR]" {
			fmt.Printf("流式调用错误: %s\n", text)
			break
		}
		fmt.Print(text)
	}
	fmt.Println()
*/
