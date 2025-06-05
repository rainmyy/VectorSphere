package service

import (
	"bytes"
	"encoding/json"
	_ "fmt"
	"net/http"
)

type DeepSeekConfig struct {
	EmbeddingURL string // DeepSeek embedding API
	LLMURL       string // DeepSeek LLM API
	ApiKey       string
}

type DeepSeekClient struct {
	Config DeepSeekConfig
}

func NewDeepSeekClient(cfg DeepSeekConfig) *DeepSeekClient {
	return &DeepSeekClient{Config: cfg}
}

// GetEmbedding 获取文本embedding
func (c *DeepSeekClient) GetEmbedding(text string) ([]float64, error) {
	reqBody, _ := json.Marshal(map[string]string{"input": text})
	req, _ := http.NewRequest("POST", c.Config.EmbeddingURL, bytes.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer "+c.Config.ApiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Embedding, nil
}

// Chat 调用大模型
func (c *DeepSeekClient) Chat(prompt string) (string, error) {
	reqBody, _ := json.Marshal(map[string]string{"prompt": prompt})
	req, _ := http.NewRequest("POST", c.Config.LLMURL, bytes.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer "+c.Config.ApiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var result struct {
		Output string `json:"output"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	return result.Output, nil
}

func (c *DeepSeekClient) ChatStream(prompt string) <-chan string {
	out := make(chan string)
	go func() {
		defer close(out)
		reqBody, _ := json.Marshal(map[string]interface{}{
			"prompt": prompt,
			"stream": true,
		})
		req, _ := http.NewRequest("POST", c.Config.LLMURL, bytes.NewReader(reqBody))
		req.Header.Set("Authorization", "Bearer "+c.Config.ApiKey)
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			out <- "[ERROR] " + err.Error()
			return
		}
		defer resp.Body.Close()
		decoder := json.NewDecoder(resp.Body)
		// 如果是每行一个JSON
		for decoder.More() {
			var chunk struct {
				Delta string `json:"delta"`
			}
			if err := decoder.Decode(&chunk); err != nil {
				break
			}
			out <- chunk.Delta
		}
	}()
	return out
}
