package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type AnthropicClient struct {
	Config     LLMConfig
	BaseClient *BaseHTTPClient
}

func NewAnthropicClient(cfg LLMConfig) *AnthropicClient {
	// 设置默认URL
	if cfg.LLMURL == "" {
		cfg.LLMURL = "https://api.anthropic.com/v1/messages"
	}
	if cfg.EmbeddingURL == "" {
		// Anthropic 目前使用 Bedrock 提供嵌入服务，这里使用 OpenAI 兼容接口
		cfg.EmbeddingURL = "https://api.anthropic.com/v1/embeddings"
	}
	if cfg.Model == "" {
		cfg.Model = "claude-3-opus-20240229"
	}

	return &AnthropicClient{
		Config:     cfg,
		BaseClient: NewBaseHTTPClient(cfg.ApiKey, cfg.GetTimeout()),
	}
}

// GetEmbedding 获取文本embedding
func (c *AnthropicClient) GetEmbedding(text string) ([]float64, error) {
	return c.GetEmbeddingWithContext(context.Background(), text)
}

// GetEmbeddingWithContext 带上下文控制的向量嵌入
func (c *AnthropicClient) GetEmbeddingWithContext(ctx context.Context, text string) ([]float64, error) {
	// 构建请求体
	reqBody := map[string]interface{}{
		"input": text,
		"model": "claude-3-embedding", // Anthropic 嵌入模型
	}

	// 创建自定义请求以添加 Anthropic-Version 头
	var reqBodyReader *bytes.Reader
	if jsonData, err := json.Marshal(reqBody); err != nil {
		return nil, fmt.Errorf("marshal request body: %w", err)
	} else {
		reqBodyReader = bytes.NewReader(jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Config.EmbeddingURL, reqBodyReader)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// 添加 Anthropic 特定的请求头
	req.Header.Set("x-api-key", c.BaseClient.ApiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.BaseClient.HttpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Anthropic API returned status code %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Embedding []float64 `json:"embedding"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parse anthropic embedding response: %w", err)
	}

	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("Anthropic API returned empty embedding")
	}

	return result.Embedding, nil
}

// Chat 调用大模型
func (c *AnthropicClient) Chat(prompt string) (string, error) {
	return c.ChatWithContext(context.Background(), prompt)
}

// ChatWithContext 带上下文控制的大模型对话
func (c *AnthropicClient) ChatWithContext(ctx context.Context, prompt string) (string, error) {
	// 构建 Anthropic 消息请求格式
	reqBody := map[string]interface{}{
		"model": c.Config.Model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"temperature": c.Config.GetTemperature(),
	}

	// 添加可选参数
	if c.Config.MaxTokens > 0 {
		reqBody["max_tokens"] = c.Config.MaxTokens
	}
	if c.Config.TopP > 0 {
		reqBody["top_p"] = c.Config.TopP
	}
	if c.Config.TopK > 0 {
		reqBody["top_k"] = c.Config.TopK
	}

	// 添加额外参数
	for k, v := range c.Config.ExtraParams {
		reqBody[k] = v
	}

	// 创建自定义请求以添加 Anthropic-Version 头
	var reqBodyReader *bytes.Reader
	if jsonData, err := json.Marshal(reqBody); err != nil {
		return "", fmt.Errorf("marshal request body: %w", err)
	} else {
		reqBodyReader = bytes.NewReader(jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Config.LLMURL, reqBodyReader)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	// 添加 Anthropic 特定的请求头
	req.Header.Set("x-api-key", c.BaseClient.ApiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.BaseClient.HttpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Anthropic API returned status code %d: %s", resp.StatusCode, string(respBody))
	}

	// 解析 Anthropic 响应
	var result struct {
		Content []struct {
			Text string `json:"text"`
			Type string `json:"type"`
		} `json:"content"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("parse anthropic chat response: %w", err)
	}

	// 提取文本内容
	var textParts []string
	for _, content := range result.Content {
		if content.Type == "text" {
			textParts = append(textParts, content.Text)
		}
	}

	if len(textParts) == 0 {
		return "", fmt.Errorf("Anthropic API returned empty response")
	}

	return strings.Join(textParts, ""), nil
}

// ChatStream 流式调用大模型
func (c *AnthropicClient) ChatStream(prompt string) <-chan string {
	return c.ChatStreamWithContext(context.Background(), prompt)
}

// ChatStreamWithContext 带上下文控制的流式调用
func (c *AnthropicClient) ChatStreamWithContext(ctx context.Context, prompt string) <-chan string {
	out := make(chan string)
	go func() {
		defer close(out)

		// 构建 Anthropic 消息请求格式
		reqBody := map[string]interface{}{
			"model": c.Config.Model,
			"messages": []map[string]string{
				{"role": "user", "content": prompt},
			},
			"temperature": c.Config.GetTemperature(),
			"stream":      true,
		}

		// 添加可选参数
		if c.Config.MaxTokens > 0 {
			reqBody["max_tokens"] = c.Config.MaxTokens
		}
		if c.Config.TopP > 0 {
			reqBody["top_p"] = c.Config.TopP
		}
		if c.Config.TopK > 0 {
			reqBody["top_k"] = c.Config.TopK
		}

		// 添加额外参数
		for k, v := range c.Config.ExtraParams {
			reqBody[k] = v
		}

		// 创建自定义请求以添加 Anthropic-Version 头
		var reqBodyReader *bytes.Reader
		if jsonData, err := json.Marshal(reqBody); err != nil {
			out <- "[ERROR] " + err.Error()
			return
		} else {
			reqBodyReader = bytes.NewReader(jsonData)
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Config.LLMURL, reqBodyReader)
		if err != nil {
			out <- "[ERROR] " + err.Error()
			return
		}

		// 添加 Anthropic 特定的请求头
		req.Header.Set("x-api-key", c.BaseClient.ApiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")

		resp, err := c.BaseClient.HttpClient.Do(req)
		if err != nil {
			out <- "[ERROR] " + err.Error()
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			out <- fmt.Sprintf("[ERROR] Anthropic API returned status code %d", resp.StatusCode)
			return
		}

		// 处理 SSE 流
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			// 提取 JSON 数据
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "[DONE]" {
				break
			}

			var event struct {
				Type  string `json:"type"`
				Delta struct {
					Text string `json:"text"`
				} `json:"delta"`
				Content []struct {
					Text string `json:"text"`
					Type string `json:"type"`
				} `json:"content"`
			}

			if err := json.Unmarshal([]byte(jsonData), &event); err != nil {
				continue
			}

			// 根据事件类型处理
			switch event.Type {
			case "content_block_delta":
				if event.Delta.Text != "" {
					out <- event.Delta.Text
				}
			case "content_block_start":
				for _, content := range event.Content {
					if content.Type == "text" && content.Text != "" {
						out <- content.Text
					}
				}
			}

			select {
			case <-ctx.Done():
				out <- "[ERROR] Context canceled"
				return
			default:
			}
		}

		if err := scanner.Err(); err != nil {
			out <- "[ERROR] " + err.Error()
		}
	}()
	return out
}

// GetModelInfo 获取模型信息
func (c *AnthropicClient) GetModelInfo() ModelInfo {
	// 根据模型名称返回相应的信息
	switch c.Config.Model {
	case "claude-3-opus-20240229":
		return ModelInfo{
			Name:             "claude-3-opus",
			Provider:         "Anthropic",
			MaxTokens:        200000,
			ContextWindow:    200000,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	case "claude-3-sonnet-20240229":
		return ModelInfo{
			Name:             "claude-3-sonnet",
			Provider:         "Anthropic",
			MaxTokens:        200000,
			ContextWindow:    200000,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	case "claude-3-haiku-20240307":
		return ModelInfo{
			Name:             "claude-3-haiku",
			Provider:         "Anthropic",
			MaxTokens:        200000,
			ContextWindow:    200000,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	case "claude-2.1":
		return ModelInfo{
			Name:             "claude-2.1",
			Provider:         "Anthropic",
			MaxTokens:        100000,
			ContextWindow:    100000,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	case "claude-2.0":
		return ModelInfo{
			Name:             "claude-2.0",
			Provider:         "Anthropic",
			MaxTokens:        100000,
			ContextWindow:    100000,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	default:
		return ModelInfo{
			Name:             c.Config.Model,
			Provider:         "Anthropic",
			MaxTokens:        100000,
			ContextWindow:    100000,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	}
}

/*
// 创建 Anthropic 客户端
anthropicConfig := llm.LLMConfig{
	Type:        llm.AnthropicLLM,
	ApiKey:      "your-anthropic-api-key",
	Model:       "claude-3-opus-20240229",
	Temperature: 0.7,
	MaxTokens:   2000,
}

anthropicClient, err := llm.NewLLMClient(anthropicConfig)
if err != nil {
	panic(err)
}

// 使用客户端
response, err := anthropicClient.Chat("Tell me about Go programming")
if err != nil {
	fmt.Printf("Error: %v\n", err)
} else {
	fmt.Printf("Response: %s\n", response)
}

// 流式调用
stream := anthropicClient.ChatStream("Explain quantum computing")
for text := range stream {
	fmt.Print(text)
}
*/
