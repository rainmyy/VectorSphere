package llm

import (
	"context"
	"fmt"
	"time"
)

// LLMClient 定义了所有LLM客户端必须实现的接口
type LLMClient interface {
	// GetEmbedding 获取文本的向量嵌入
	GetEmbedding(text string) ([]float64, error)
	// GetEmbeddingWithContext 带上下文控制的向量嵌入
	GetEmbeddingWithContext(ctx context.Context, text string) ([]float64, error)
	// Chat 调用大模型进行对话
	Chat(prompt string) (string, error)
	// ChatWithContext 带上下文控制的大模型对话
	ChatWithContext(ctx context.Context, prompt string) (string, error)
	// ChatStream 流式调用大模型
	ChatStream(prompt string) <-chan string
	// ChatStreamWithContext 带上下文控制的流式调用
	ChatStreamWithContext(ctx context.Context, prompt string) <-chan string
	// GetModelInfo 获取模型信息
	GetModelInfo() ModelInfo
}

// ModelInfo 模型信息
type ModelInfo struct {
	Name             string // 模型名称
	Provider         string // 提供商
	MaxTokens        int    // 最大token数
	ContextWindow    int    // 上下文窗口大小
	EmbeddingDim     int    // 嵌入维度
	SupportStreaming bool   // 是否支持流式输出
}

// LLMType 定义LLM类型
type LLMType string

const (
	DeepSeekLLM LLMType = "deepseek"
	OpenAILLM   LLMType = "openai"
	// 可以在此添加更多模型类型
	AnthropicLLM LLMType = "anthropic"
	GeminiLLM    LLMType = "gemini"
	LocalLLM     LLMType = "local"
)

// LLMConfig 通用LLM配置
type LLMConfig struct {
	Type         LLMType                `json:"type"`          // LLM类型
	ApiKey       string                 `json:"api_key"`       // API密钥
	EmbeddingURL string                 `json:"embedding_url"` // 嵌入API URL
	LLMURL       string                 `json:"llm_url"`       // LLM API URL
	Model        string                 `json:"model"`         // 模型名称
	Timeout      int                    `json:"timeout"`       // 超时时间(秒)
	MaxRetries   int                    `json:"max_retries"`   // 最大重试次数
	Temperature  float64                `json:"temperature"`   // 温度参数
	TopP         float64                `json:"top_p"`         // Top-P 参数
	TopK         int                    `json:"top_k"`         // Top-K 参数
	MaxTokens    int                    `json:"max_tokens"`    // 最大生成token数
	ProxyURL     string                 `json:"proxy_url"`     // 代理URL
	ExtraParams  map[string]interface{} `json:"extra_params"`  // 额外参数
}

// GetTimeout 获取超时时间，如果未设置则返回默认值
func (c *LLMConfig) GetTimeout() time.Duration {
	if c.Timeout <= 0 {
		return 30 * time.Second
	}
	return time.Duration(c.Timeout) * time.Second
}

// GetTemperature 获取温度参数，如果未设置则返回默认值
func (c *LLMConfig) GetTemperature() float64 {
	if c.Temperature <= 0 {
		return 0.7
	}
	return c.Temperature
}

// NewLLMClient 创建LLM客户端的工厂方法
func NewLLMClient(config LLMConfig) (LLMClient, error) {
	switch config.Type {
	case DeepSeekLLM:
		return NewDeepSeekClient(config), nil
	case OpenAILLM:
		return NewOpenAIClient(config), nil
	case AnthropicLLM:
		return NewAnthropicClient(config), nil
	case GeminiLLM:
		return NewGeminiClient(config), nil
	case LocalLLM:
		return NewLocalClient(config)
	default:
		return nil, fmt.Errorf("unsupported LLM type: %s", config.Type)
	}
}
