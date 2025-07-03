package distributed

import (
	"strconv"
	"strings"
)

// ParseEndpoints 解析etcd端点字符串
// 输入格式: "host1:port1,host2:port2,host3:port3"
// 输出: []string{"host1:port1", "host2:port2", "host3:port3"}
func ParseEndpoints(endpoints string) []string {
	if endpoints == "" {
		return []string{}
	}
	
	// 按逗号分隔
	parts := strings.Split(endpoints, ",")
	
	// 去除空白
	result := make([]string, 0, len(parts))
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	
	return result
}

// ParseHostPort 解析host:port格式的字符串
// 输入格式: "host:port"
// 输出: host (string), port (int)
func ParseHostPort(endpoint string) (string, int) {
	parts := strings.Split(endpoint, ":")
	if len(parts) != 2 {
		// 如果格式不正确，返回默认值
		return "localhost", 2379
	}
	
	host := strings.TrimSpace(parts[0])
	portStr := strings.TrimSpace(parts[1])
	
	port, err := strconv.Atoi(portStr)
	if err != nil {
		// 如果端口解析失败，返回默认端口
		return host, 2379
	}
	
	return host, port
}