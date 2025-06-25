package entity

// SearchResult 搜索结果结构体
type SearchResult struct {
	ID         string
	Similarity float64
	Metadata   map[string]interface{}
}
