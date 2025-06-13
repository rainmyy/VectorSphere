package distributed

import (
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