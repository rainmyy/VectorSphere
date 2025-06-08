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

type GeminiClient struct {
	Config     LLMConfig
	BaseClient *BaseHTTPClient
}

func NewGeminiClient(cfg LLMConfig) *GeminiClient {
	// 设置默认URL
	if cfg.EmbeddingURL == "" {
		cfg.EmbeddingURL = "https://generativelanguage.googleapis.com/v1/models/embedding-001:embedContent"
	}
	if cfg.LLMURL == "" {
		cfg.LLMURL = "https://generativelanguage.googleapis.com/v1/models"
	}
	if cfg.Model == "" {
		cfg.Model = "gemini-pro"
	}

	return &GeminiClient{
		Config:     cfg,
		BaseClient: NewBaseHTTPClient(cfg.ApiKey, cfg.GetTimeout()),
	}
}

// GetEmbedding 获取文本embedding
func (c *GeminiClient) GetEmbedding(text string) ([]float64, error) {
	return c.GetEmbeddingWithContext(context.Background(), text)
}

// GetEmbeddingWithContext 带上下文控制的向量嵌入
func (c *GeminiClient) GetEmbeddingWithContext(ctx context.Context, text string) ([]float64, error) {
	reqBody := map[string]interface{}{
		"content": map[string]interface{}{
			"parts": []map[string]string{
				{"text": text},
			},
		},
	}

	// 在URL中添加API密钥作为查询参数
	url := fmt.Sprintf("%s?key=%s", c.Config.EmbeddingURL, c.Config.ApiKey)

	// 使用不带Authorization头的请求
	respBody, err := c.doRequestWithoutAuth(ctx, http.MethodPost, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("gemini embedding request: %w", err)
	}

	var result struct {
		Embedding struct {
			Values []float64 `json:"values"`
		} `json:"embedding"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parse gemini embedding response: %w", err)
	}

	if len(result.Embedding.Values) == 0 {
		return nil, fmt.Errorf("Gemini API returned empty embedding")
	}

	return result.Embedding.Values, nil
}

// Chat 调用大模型
func (c *GeminiClient) Chat(prompt string) (string, error) {
	return c.ChatWithContext(context.Background(), prompt)
}

// ChatWithContext 带上下文控制的大模型对话
func (c *GeminiClient) ChatWithContext(ctx context.Context, prompt string) (string, error) {
	// 构建Gemini聊天请求格式
	reqBody := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"role": "user",
				"parts": []map[string]string{
					{"text": prompt},
				},
			},
		},
		"generationConfig": map[string]interface{}{
			"temperature": c.Config.GetTemperature(),
		},
	}

	// 添加可选参数
	generationConfig := reqBody["generationConfig"].(map[string]interface{})
	if c.Config.MaxTokens > 0 {
		generationConfig["maxOutputTokens"] = c.Config.MaxTokens
	}
	if c.Config.TopP > 0 {
		generationConfig["topP"] = c.Config.TopP
	}
	if c.Config.TopK > 0 {
		generationConfig["topK"] = c.Config.TopK
	}

	// 添加额外参数
	for k, v := range c.Config.ExtraParams {
		if strings.HasPrefix(k, "generationConfig.") {
			paramName := strings.TrimPrefix(k, "generationConfig.")
			generationConfig[paramName] = v
		} else {
			reqBody[k] = v
		}
	}

	// 在URL中添加API密钥作为查询参数
	url := fmt.Sprintf("%s/%s:generateContent?key=%s", c.Config.LLMURL, c.Config.Model, c.Config.ApiKey)

	// 使用不带Authorization头的请求
	respBody, err := c.doRequestWithoutAuth(ctx, http.MethodPost, url, reqBody)
	if err != nil {
		return "", fmt.Errorf("gemini chat request: %w", err)
	}

	var result struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("parse gemini chat response: %w", err)
	}

	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("Gemini API returned empty response")
	}

	return result.Candidates[0].Content.Parts[0].Text, nil
}

// ChatStream 流式调用大模型
func (c *GeminiClient) ChatStream(prompt string) <-chan string {
	return c.ChatStreamWithContext(context.Background(), prompt)
}

// ChatStreamWithContext 带上下文控制的流式调用
func (c *GeminiClient) ChatStreamWithContext(ctx context.Context, prompt string) <-chan string {
	out := make(chan string)
	go func() {
		defer close(out)

		// 构建Gemini聊天请求格式
		reqBody := map[string]interface{}{
			"contents": []map[string]interface{}{
				{
					"role": "user",
					"parts": []map[string]string{
						{"text": prompt},
					},
				},
			},
			"generationConfig": map[string]interface{}{
				"temperature": c.Config.GetTemperature(),
			},
		}

		// 添加可选参数
		generationConfig := reqBody["generationConfig"].(map[string]interface{})
		if c.Config.MaxTokens > 0 {
			generationConfig["maxOutputTokens"] = c.Config.MaxTokens
		}
		if c.Config.TopP > 0 {
			generationConfig["topP"] = c.Config.TopP
		}
		if c.Config.TopK > 0 {
			generationConfig["topK"] = c.Config.TopK
		}

		// 添加额外参数
		for k, v := range c.Config.ExtraParams {
			if strings.HasPrefix(k, "generationConfig.") {
				paramName := strings.TrimPrefix(k, "generationConfig.")
				generationConfig[paramName] = v
			} else {
				reqBody[k] = v
			}
		}

		// 在URL中添加API密钥和stream参数
		url := fmt.Sprintf("%s/%s:streamGenerateContent?key=%s&alt=sse", c.Config.LLMURL, c.Config.Model, c.Config.ApiKey)

		// 使用不带Authorization头的请求，但需要自定义处理SSE流
		resp, err := c.doStreamRequestSSE(ctx, http.MethodPost, url, reqBody)
		if err != nil {
			out <- "[ERROR] " + err.Error()
			return
		}
		defer resp.Body.Close()

		// 创建SSE解析器
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			// 提取JSON数据
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "[DONE]" {
				break
			}

			var chunk struct {
				Candidates []struct {
					Content struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					} `json:"content"`
				} `json:"candidates"`
			}

			if err := json.Unmarshal([]byte(jsonData), &chunk); err != nil {
				continue // 跳过无法解析的数据
			}

			if len(chunk.Candidates) > 0 && len(chunk.Candidates[0].Content.Parts) > 0 {
				out <- chunk.Candidates[0].Content.Parts[0].Text
			}
		}

		if err := scanner.Err(); err != nil {
			out <- "[ERROR] " + err.Error()
		}
	}()
	return out
}

// GetModelInfo 获取模型信息
func (c *GeminiClient) GetModelInfo() ModelInfo {
	// 根据模型名称返回相应的信息
	switch c.Config.Model {
	case "gemini-pro":
		return ModelInfo{
			Name:             "gemini-pro",
			Provider:         "Google",
			MaxTokens:        32768,
			ContextWindow:    32768,
			EmbeddingDim:     768,
			SupportStreaming: true,
		}
	case "gemini-pro-vision":
		return ModelInfo{
			Name:             "gemini-pro-vision",
			Provider:         "Google",
			MaxTokens:        16384,
			ContextWindow:    16384,
			EmbeddingDim:     768,
			SupportStreaming: true,
		}
	case "gemini-ultra":
		return ModelInfo{
			Name:             "gemini-ultra",
			Provider:         "Google",
			MaxTokens:        32768,
			ContextWindow:    32768,
			EmbeddingDim:     1024,
			SupportStreaming: true,
		}
	default:
		return ModelInfo{
			Name:             c.Config.Model,
			Provider:         "Google",
			MaxTokens:        32768,
			ContextWindow:    32768,
			EmbeddingDim:     768,
			SupportStreaming: true,
		}
	}
}

// doRequestWithoutAuth 执行不带Authorization头的HTTP请求
func (c *GeminiClient) doRequestWithoutAuth(ctx context.Context, method, url string, body interface{}) ([]byte, error) {
	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

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
		return nil, fmt.Errorf("API returned status code %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// doStreamRequestSSE 执行流式HTTP请求并返回原始响应
func (c *GeminiClient) doStreamRequestSSE(ctx context.Context, method, url string, body interface{}) (*http.Response, error) {
	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.BaseClient.HttpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("API returned status code %d", resp.StatusCode)
	}

	return resp, nil
}

/*
	// 创建 Gemini 客户端配置
	config := llm.LLMConfig{
		Type:        llm.GeminiLLM,
		ApiKey:      "YOUR_GEMINI_API_KEY",
		Model:       "gemini-pro",
		Temperature: 0.7,
		MaxTokens:   1024,
	}

	// 创建 Gemini 客户端
	client, err := llm.NewLLMClient(config)
	if err != nil {
		log.Fatalf("Failed to create Gemini client: %v", err)
	}

	// 获取模型信息
	modelInfo := client.GetModelInfo()
	fmt.Printf("Model: %s, Provider: %s, Context Window: %d\n",
		modelInfo.Name, modelInfo.Provider, modelInfo.ContextWindow)

	// 标准聊天调用
	response, err := client.Chat("请介绍一下你自己")
	if err != nil {
		log.Fatalf("Chat error: %v", err)
	}
	fmt.Printf("Response: %s\n", response)

	// 流式聊天调用
	fmt.Println("\nStreaming response:")
	stream := client.ChatStream("请列出5个世界上最高的山峰")
	for chunk := range stream {
		fmt.Print(chunk)
	}
	fmt.Println()
*/
