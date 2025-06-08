package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

type OpenAIConfig struct {
	ApiKey       string // OpenAI API Key
	EmbeddingURL string // 嵌入API URL，默认为 "https://api.openai.com/v1/embeddings"
	LLMURL       string // LLM API URL，默认为 "https://api.openai.com/v1/chat/completions"
	Model        string // 模型名称，如 "gpt-3.5-turbo"
}

type OpenAIClient struct {
	Config     LLMConfig
	BaseClient *BaseHTTPClient
}

func NewOpenAIClient(cfg LLMConfig) *OpenAIClient {
	// 设置默认URL
	if cfg.EmbeddingURL == "" {
		cfg.EmbeddingURL = "https://api.openai.com/v1/embeddings"
	}
	if cfg.LLMURL == "" {
		cfg.LLMURL = "https://api.openai.com/v1/chat/completions"
	}
	if cfg.Model == "" {
		cfg.Model = "gpt-3.5-turbo"
	}

	return &OpenAIClient{
		Config:     cfg,
		BaseClient: NewBaseHTTPClient(cfg.ApiKey, cfg.GetTimeout()),
	}
}

// GetEmbedding 获取文本embedding
func (c *OpenAIClient) GetEmbedding(text string) ([]float64, error) {
	return c.GetEmbeddingWithContext(context.Background(), text)
}

// GetEmbeddingWithContext 带上下文控制的向量嵌入
func (c *OpenAIClient) GetEmbeddingWithContext(ctx context.Context, text string) ([]float64, error) {
	// 确定使用的嵌入模型
	embeddingModel := "text-embedding-ada-002"
	if c.Config.Model == "gpt-4" || c.Config.Model == "gpt-4-turbo" {
		embeddingModel = "text-embedding-3-large"
	} else if c.Config.Model == "gpt-3.5-turbo" {
		embeddingModel = "text-embedding-3-small"
	}

	reqBody := map[string]interface{}{
		"input": text,
		"model": embeddingModel,
	}

	respBody, err := c.BaseClient.DoRequest(ctx, http.MethodPost, c.Config.EmbeddingURL, reqBody)
	if err != nil {
		return nil, fmt.Errorf("openai embedding request: %w", err)
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parse openai embedding response: %w", err)
	}

	if len(result.Data) == 0 || len(result.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("OpenAI API returned empty embedding")
	}

	return result.Data[0].Embedding, nil
}

// Chat 调用大模型
func (c *OpenAIClient) Chat(prompt string) (string, error) {
	return c.ChatWithContext(context.Background(), prompt)
}

// ChatWithContext 带上下文控制的大模型对话
func (c *OpenAIClient) ChatWithContext(ctx context.Context, prompt string) (string, error) {
	// 构建OpenAI聊天请求格式
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

	// 添加额外参数
	for k, v := range c.Config.ExtraParams {
		reqBody[k] = v
	}

	respBody, err := c.BaseClient.DoRequest(ctx, http.MethodPost, c.Config.LLMURL, reqBody)
	if err != nil {
		return "", fmt.Errorf("openai chat request: %w", err)
	}

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("parse openai chat response: %w", err)
	}

	if len(result.Choices) == 0 {
		return "", fmt.Errorf("OpenAI API returned empty response")
	}

	return result.Choices[0].Message.Content, nil
}

// ChatStream 流式调用大模型
func (c *OpenAIClient) ChatStream(prompt string) <-chan string {
	return c.ChatStreamWithContext(context.Background(), prompt)
}

// ChatStreamWithContext 带上下文控制的流式调用
func (c *OpenAIClient) ChatStreamWithContext(ctx context.Context, prompt string) <-chan string {
	out := make(chan string)
	go func() {
		defer close(out)

		// 构建OpenAI聊天请求格式
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

		// 添加额外参数
		for k, v := range c.Config.ExtraParams {
			reqBody[k] = v
		}

		decoder, err := c.BaseClient.DoStreamRequest(ctx, http.MethodPost, c.Config.LLMURL, reqBody)
		if err != nil {
			out <- "[ERROR] " + err.Error()
			return
		}

		for {
			select {
			case <-ctx.Done():
				out <- "[ERROR] Context canceled"
				return
			default:
				// 读取流式响应
				var line struct {
					Choices []struct {
						Delta struct {
							Content string `json:"content"`
						} `json:"delta"`
					} `json:"choices"`
				}

				if err := decoder.Decode(&line); err != nil {
					return // 流结束或出错
				}

				if len(line.Choices) > 0 && line.Choices[0].Delta.Content != "" {
					out <- line.Choices[0].Delta.Content
				}
			}
		}
	}()
	return out
}

// GetModelInfo 获取模型信息
func (c *OpenAIClient) GetModelInfo() ModelInfo {
	// 根据模型名称返回相应的信息
	switch c.Config.Model {
	case "gpt-3.5-turbo":
		return ModelInfo{
			Name:             "gpt-3.5-turbo",
			Provider:         "OpenAI",
			MaxTokens:        4096,
			ContextWindow:    4096,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	case "gpt-4":
		return ModelInfo{
			Name:             "gpt-4",
			Provider:         "OpenAI",
			MaxTokens:        8192,
			ContextWindow:    8192,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	case "gpt-4-turbo":
		return ModelInfo{
			Name:             "gpt-4-turbo",
			Provider:         "OpenAI",
			MaxTokens:        128000,
			ContextWindow:    128000,
			EmbeddingDim:     3072,
			SupportStreaming: true,
		}
	default:
		return ModelInfo{
			Name:             c.Config.Model,
			Provider:         "OpenAI",
			MaxTokens:        4096,
			ContextWindow:    4096,
			EmbeddingDim:     1536,
			SupportStreaming: true,
		}
	}
}
