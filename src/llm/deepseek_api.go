package llm

import (
	"context"
	"encoding/json"
	"fmt"
	_ "fmt"
	"net/http"
)

type DeepSeekClient struct {
	Config     LLMConfig
	BaseClient *BaseHTTPClient
}

func NewDeepSeekClient(cfg LLMConfig) *DeepSeekClient {
	return &DeepSeekClient{
		Config:     cfg,
		BaseClient: NewBaseHTTPClient(cfg.ApiKey, cfg.GetTimeout()),
	}
}

// GetEmbedding 获取文本embedding
func (c *DeepSeekClient) GetEmbedding(text string) ([]float64, error) {
	return c.GetEmbeddingWithContext(context.Background(), text)
}

// GetEmbeddingWithContext 带上下文控制的向量嵌入
func (c *DeepSeekClient) GetEmbeddingWithContext(ctx context.Context, text string) ([]float64, error) {
	reqBody := map[string]string{"input": text}
	respBody, err := c.BaseClient.DoRequest(ctx, http.MethodPost, c.Config.EmbeddingURL, reqBody)
	if err != nil {
		return nil, fmt.Errorf("deepseek embedding request: %w", err)
	}

	var result struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parse deepseek embedding response: %w", err)
	}

	return result.Embedding, nil
}

// Chat 调用大模型
func (c *DeepSeekClient) Chat(prompt string) (string, error) {
	return c.ChatWithContext(context.Background(), prompt)
}

// ChatWithContext 带上下文控制的大模型对话
func (c *DeepSeekClient) ChatWithContext(ctx context.Context, prompt string) (string, error) {
	reqBody := map[string]interface{}{
		"prompt": prompt,
		"model":  c.Config.Model,
	}

	// 添加可选参数
	if c.Config.Temperature > 0 {
		reqBody["temperature"] = c.Config.GetTemperature()
	}
	if c.Config.MaxTokens > 0 {
		reqBody["max_tokens"] = c.Config.MaxTokens
	}
	if c.Config.TopP > 0 {
		reqBody["top_p"] = c.Config.TopP
	}

	// 添加额外参数
	for k, v := range c.Config.ExtraParams {
		reqBody[k] = v
	}

	respBody, err := c.BaseClient.DoRequest(ctx, http.MethodPost, c.Config.LLMURL, reqBody)
	if err != nil {
		return "", fmt.Errorf("deepseek chat request: %w", err)
	}

	var result struct {
		Output string `json:"output"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("parse deepseek chat response: %w", err)
	}

	return result.Output, nil
}

// ChatStream 流式调用大模型
func (c *DeepSeekClient) ChatStream(prompt string) <-chan string {
	return c.ChatStreamWithContext(context.Background(), prompt)
}

// ChatStreamWithContext 带上下文控制的流式调用
func (c *DeepSeekClient) ChatStreamWithContext(ctx context.Context, prompt string) <-chan string {
	out := make(chan string)
	go func() {
		defer close(out)

		reqBody := map[string]interface{}{
			"prompt": prompt,
			"model":  c.Config.Model,
			"stream": true,
		}

		// 添加可选参数
		if c.Config.Temperature > 0 {
			reqBody["temperature"] = c.Config.GetTemperature()
		}
		if c.Config.MaxTokens > 0 {
			reqBody["max_tokens"] = c.Config.MaxTokens
		}
		if c.Config.TopP > 0 {
			reqBody["top_p"] = c.Config.TopP
		}

		// 添加额外参数
		for k, v := range c.Config.ExtraParams {
			reqBody[k] = v
		}

		decoder, err := c.BaseClient.DoStreamRequest(ctx, http.MethodPost, c.Config.LLMURL, reqBody)
		if err != nil {
			out <- "[ERROR] " + err.Error()
			return
		}

		// 处理流式响应
		for decoder.More() {
			select {
			case <-ctx.Done():
				out <- "[ERROR] Context canceled"
				return
			default:
				var chunk struct {
					Delta string `json:"delta"`
				}
				if err := decoder.Decode(&chunk); err != nil {
					break
				}
				out <- chunk.Delta
			}
		}
	}()
	return out
}

// GetModelInfo 获取模型信息
// GetModelInfo 获取模型信息
func (c *DeepSeekClient) GetModelInfo() ModelInfo {
	// 根据模型名称返回相应的信息
	switch c.Config.Model {
	case "deepseek-chat":
		return ModelInfo{
			Name:             "deepseek-chat",
			Provider:         "DeepSeek",
			MaxTokens:        8192,
			ContextWindow:    8192,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	case "deepseek-coder":
		return ModelInfo{
			Name:             "deepseek-coder",
			Provider:         "DeepSeek",
			MaxTokens:        16384,
			ContextWindow:    16384,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	default:
		return ModelInfo{
			Name:             c.Config.Model,
			Provider:         "DeepSeek",
			MaxTokens:        8192,
			ContextWindow:    8192,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	}
}
