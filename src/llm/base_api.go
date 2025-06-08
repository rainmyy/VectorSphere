package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// BaseHTTPClient 提供基础的 HTTP 请求功能
type BaseHTTPClient struct {
	HttpClient *http.Client
	ApiKey     string
}

// NewBaseHTTPClient 创建一个新的基础 HTTP 客户端
func NewBaseHTTPClient(apiKey string, timeout time.Duration) *BaseHTTPClient {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &BaseHTTPClient{
		HttpClient: &http.Client{Timeout: timeout},
		ApiKey:     apiKey,
	}
}

// DoRequest 执行 HTTP 请求并返回响应体
func (c *BaseHTTPClient) DoRequest(ctx context.Context, method, url string, body interface{}) ([]byte, error) {
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

	req.Header.Set("Authorization", "Bearer "+c.ApiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HttpClient.Do(req)
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

// DoStreamRequest 执行流式 HTTP 请求并返回响应解码器
func (c *BaseHTTPClient) DoStreamRequest(ctx context.Context, method, url string, body interface{}) (*json.Decoder, error) {
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

	req.Header.Set("Authorization", "Bearer "+c.ApiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HttpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("API returned status code %d", resp.StatusCode)
	}

	return json.NewDecoder(resp.Body), nil
}
