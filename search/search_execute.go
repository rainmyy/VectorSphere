package search

import (
	"fmt"
	"seetaSearch/messages"
	"sort"
	// "strings" // 可能会用到
)

// SearchResultItem 定义了包含ID和可能的其他字段（如分数）的搜索结果项
type SearchResultItem struct {
	ID     string
	Fields map[string]interface{} // 用于存储 SELECT 出来的字段值
	Score  float64                // 用于排序
}

// SearchExecutor 结构体负责执行解析后的查询
type SearchExecutor struct {
	service *MultiTableSearchService
}

// NewSearchExecutor 创建一个新的 SearchExecutor 实例
func NewSearchExecutor(service *MultiTableSearchService) *SearchExecutor {
	return &SearchExecutor{service: service}
}

// ExecuteSearchPlan 根据 ParsedQuery 执行搜索
func (se *SearchExecutor) ExecuteSearchPlan(parsedQuery *ParsedQuery) ([]SearchResultItem, error) {
	if se.service == nil {
		return nil, fmt.Errorf("SearchExecutor service is not initialized")
	}

	if parsedQuery == nil {
		return nil, fmt.Errorf("parsedQuery is nil")
	}

	tableName := parsedQuery.TableName
	keywordQuery := parsedQuery.KeywordQuery
	vectorQueryText := parsedQuery.VectorQueryText
	limit := parsedQuery.Limit
	offset := parsedQuery.Offset
	orderByField := parsedQuery.OrderByField
	orderByDirection := parsedQuery.OrderByDirection
	useANN := parsedQuery.UseANN
	k := parsedQuery.K

	// 1. 根据 ParsedQuery.Filters 构建更复杂的过滤条件 (onFlag, offFlag, orFlags)
	var onFlag, offFlag uint64
	var orFlags []uint64
	// TODO: 实现从 parsedQuery.Filters 到 onFlag, offFlag, orFlags 的转换逻辑
	// 这部分需要详细定义 FilterCondition 的结构以及如何映射到这些位掩码标志
	// 例如：
	// for _, filter := range parsedQuery.Filters {
	//     if filter.Operator == "=" && filter.Field == "status" && filter.Value == "active" {
	//         onFlag |= (1 << 0) // 假设 status active 是第0位
	//     } else if filter.Operator == "=" && filter.Field == "category" && filter.Value == "A" {
	//         orFlags = append(orFlags, (1 << 1)) // 假设 category A 是第1位
	//     } else if filter.Operator == "!=" && filter.Field == "type" && filter.Value == "internal" {
	//         offFlag |= (1 << 2) // 假设 type internal 是第2位
	//     }
	// }

	// 2. 决策调用策略
	var invertedIndexResults []string
	var vectorResults []string
	var err error
	var finalDocIDs []string

	// 决策逻辑：
	// - 如果有 keywordQuery，则调用倒排索引
	// - 如果有 vectorQueryText，则调用向量搜索
	// - 如果两者都有，则都调用并合并
	callInvertedIndex := keywordQuery != ""
	callVectorSearch := vectorQueryText != "" // 或者根据 useANN 和 k > 0

	if callInvertedIndex {
		termQuery := &messages.TermQuery{
			Keyword: &messages.KeyWord{Word: keywordQuery},
		}
		invertedIndexResults, err = se.service.Search(
			tableName,
			termQuery,
			0,  // vectorizedType (示例，可能需要从 parsedQuery 获取)
			k,  // k (可能主要用于向量搜索，但 Search 接口需要)
			10, // probe (示例)
			onFlag,
			offFlag,
			orFlags,
			false, // useANN for inverted index part is usually false, vector part handles it
		)
		if err != nil {
			return nil, fmt.Errorf("inverted index search failed: %w", err)
		}
	}

	if callVectorSearch {
		// 假设 Search 方法内部能处理 vectorQueryText，或者需要一个专门的向量搜索接口
		// MultiTableSearchService.Search 似乎混合了两者，需要确保其行为符合预期
		// 这里我们假设 Search 方法能利用 vectorQueryText (如果 useANN 为 true)
		// 或者，我们可能需要一个独立的 vector search 方法调用
		// For simplicity, let's assume Search handles it if useANN is true and keyword is vector text
		// This part might need significant refinement based on actual service capabilities.

		// 模拟向量搜索调用 (实际应调用服务)
		// 如果 MultiTableSearchService.Search 能够根据 useANN 和 keyword(作为vector query) 进行向量搜索，则可以复用
		// 否则，需要一个类似 table.VectorDB.Search(...) 的调用
		termQueryForVector := &messages.TermQuery{
			Keyword: &messages.KeyWord{Word: vectorQueryText}, // 使用 vectorQueryText
		}
		vectorResults, err = se.service.Search(
			tableName,
			termQueryForVector,
			0, // vectorizedType
			k,
			10,     // probe
			0,      // onFlag for vector search might be different or not applicable
			0,      // offFlag
			nil,    // orFlags
			useANN, // Crucial for vector search
		)
		if err != nil {
			return nil, fmt.Errorf("vector search failed: %w", err)
		}
	}

	// 合并结果
	if callInvertedIndex && callVectorSearch {
		// 取交集
		set := make(map[string]struct{})
		for _, id := range invertedIndexResults {
			set[id] = struct{}{}
		}
		for _, id := range vectorResults {
			if _, found := set[id]; found {
				finalDocIDs = append(finalDocIDs, id)
			}
		}
	} else if callInvertedIndex {
		finalDocIDs = invertedIndexResults
	} else if callVectorSearch {
		finalDocIDs = vectorResults
	} else {
		// 没有指定查询条件，可以返回空或者错误，或者所有文档（不推荐）
		return []SearchResultItem{}, nil
	}

	// 3. 处理 SELECT 字段 (如果不仅仅是返回文档 ID)
	// 这需要从数据库或其他存储中获取每个文档的实际字段值
	// 当前 MultiTableSearchService.Search 只返回 []string (文档ID)
	// 我们需要扩展它或添加新方法来获取文档的详细信息
	processedResults := make([]SearchResultItem, 0, len(finalDocIDs))
	for _, id := range finalDocIDs {
		item := SearchResultItem{ID: id, Fields: make(map[string]interface{}), Score: 0.0 /* TODO: Get actual score */}
		if parsedQuery.SelectFields != nil && len(parsedQuery.SelectFields) > 0 {
			// 检查是否是 SELECT *
			isSelectAll := false
			for _, sf := range parsedQuery.SelectFields {
				if sf.FieldName == "*" {
					isSelectAll = true
					break
				}
			}

			// TODO: 获取文档的实际数据
			// docData, err := se.service.GetDocumentDetails(tableName, id, parsedQuery.SelectFields, isSelectAll)
			// if err != nil { /* handle error, maybe skip this doc */ continue }
			// item.Fields = docData.Fields
			// item.Score = docData.Score // 如果服务能返回分数
			// 示例：填充假数据
			if isSelectAll {
				item.Fields["title"] = "Sample Title for " + id
				item.Fields["content"] = "Sample content..."
			} else {
				for _, sf := range parsedQuery.SelectFields {
					item.Fields[sf.FieldName] = "Value for " + sf.FieldName
				}
			}
		}
		processedResults = append(processedResults, item)
	}

	// 4. 处理 ORDER BY
	if orderByField != "" {
		sort.SliceStable(processedResults, func(i, j int) bool {
			valI, okI := processedResults[i].Fields[orderByField]
			valJ, okJ := processedResults[j].Fields[orderByField]

			if orderByField == "score" { // 特殊处理按分数排序
				valI = processedResults[i].Score
				valJ = processedResults[j].Score
				okI = true
				okJ = true
			}

			if !okI && !okJ {
				return false
			} // 两者都没有该字段
			if !okI {
				return orderByDirection != "ASC"
			} // i 没有，j 有。升序则 j 在前，降序则 i 在前
			if !okJ {
				return orderByDirection == "ASC"
			} // j 没有，i 有。升序则 i 在前，降序则 j 在前

			// 比较逻辑 (简化版，需要根据字段类型进行转换和比较)
			switch vI := valI.(type) {
			case string:
				vJ, ok := valJ.(string)
				if !ok {
					return false
				} // 类型不匹配
				if orderByDirection == "ASC" {
					return vI < vJ
				}
				return vI > vJ
			case int, int32, int64:
				// TODO: Convert to int64 and compare
			case float32, float64:
				// TODO: Convert to float64 and compare
				// Example for score (float64)
				fI, okfI := valI.(float64)
				fJ, okfJ := valJ.(float64)
				if okfI && okfJ {
					if orderByDirection == "ASC" {
						return fI < fJ
					}
					return fI > fJ
				}
			}
			return false // 默认不排序或类型不支持
		})
	}

	// 5. 应用 LIMIT 和 OFFSET
	start := int(offset)
	end := int(offset + limit)

	if limit == 0 { // limit 0 可能表示不限制，或者需要根据具体语义调整
		end = len(processedResults)
	} else if limit < 0 { // limit < 0 通常表示不限制
		end = len(processedResults)
	}

	if end > len(processedResults) {
		end = len(processedResults)
	}
	if start < 0 {
		start = 0
	}

	if start > len(processedResults) {
		return []SearchResultItem{}, nil // Offset is beyond results
	}
	if start > end {
		return []SearchResultItem{}, nil
	}

	return processedResults[start:end], nil
}
