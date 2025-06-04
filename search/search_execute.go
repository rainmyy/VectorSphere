package search

import (
	"fmt"
	tree "seetaSearch/library/tree"
	"seetaSearch/messages"
	"sort"
	"strconv"
)

// SearchResultItem 定义了包含ID和可能的其他字段（如分数）的搜索结果项
type SearchResultItem struct {
	ID     string
	Fields map[string]interface{} // 用于存储 SELECT 出来的字段值
	Score  float64                // 用于排序
}

// SelectField 结构体用于表示查询中的选择字段
type SelectField struct {
	FieldName string
	Alias     string // 可选的字段别名
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

	// 将 Fields 转换为 SelectFields
	var selectFields []SelectField
	for _, field := range parsedQuery.Fields {
		selectFields = append(selectFields, SelectField{
			FieldName: field,
			Alias:     field, // 默认别名与字段名相同
		})
	}

	// 1. 根据 ParsedQuery.Filters 构建更复杂的过滤条件 (onFlag, offFlag, orFlags)
	var onFlag, offFlag uint64
	var orFlags []uint64

	// 实现从 parsedQuery.Filters 到 onFlag, offFlag, orFlags 的转换逻辑
	// 这里假设有一个映射关系，将字段和值映射到位掩码
	for _, filter := range parsedQuery.Filters {
		// 这里需要根据实际的字段和值定义映射规则
		// 例如：status=active 映射到第0位，category=A 映射到第1位，等等

		// 这里只是示例，实际实现需要根据业务逻辑定义
		bitPosition := getBitPositionForField(filter.Field, filter.Value)
		if bitPosition >= 0 {
			switch filter.Operator {
			case "=":
				onFlag |= 1 << uint(bitPosition)
			case "!=":
				offFlag |= 1 << uint(bitPosition)
			case "IN":
				// 对于 IN 操作符，可以添加到 orFlags
				orFlags = append(orFlags, 1<<uint(bitPosition))
			}
		}
	}

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
	callVectorSearch := vectorQueryText != "" && k > 0

	// 获取表实例
	table, err := se.service.GetTable(tableName)
	if err != nil {
		return nil, fmt.Errorf("failed to get table '%s': %w", tableName, err)
	}

	// 开启事务
	tx := se.service.TxMgr.Begin(tree.Serializable)
	defer se.service.TxMgr.Commit(tx)

	if callInvertedIndex {
		// 创建 TermQuery
		termQuery := &messages.TermQuery{
			Keyword: &messages.KeyWord{Word: keywordQuery},
		}

		// 直接使用 InvertedIndex 的 Search 方法
		invertedIndexResults, err = table.InvertedIndex.Search(
			tx,
			termQuery,
			onFlag,
			offFlag,
			orFlags,
			"",    // 不使用向量查询文本
			0,     // 不使用 kNearest
			false, // 不使用 ANN
		)
		if err != nil {
			return nil, fmt.Errorf("inverted index search failed: %w", err)
		}
	}

	if callVectorSearch {
		// 直接使用 InvertedIndex 的 Search 方法进行向量搜索
		// 创建一个空的 TermQuery，因为我们只关心向量搜索部分
		emptyTermQuery := &messages.TermQuery{}

		vectorResults, err = table.InvertedIndex.Search(
			tx,
			emptyTermQuery,
			0,               // 向量搜索不使用 onFlag
			0,               // 向量搜索不使用 offFlag
			nil,             // 向量搜索不使用 orFlags
			vectorQueryText, // 使用向量查询文本
			k,               // 使用 kNearest
			useANN,          // 使用 ANN
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

		// 如果交集为空但两个结果集都不为空，可能是因为过滤条件太严格
		// 在这种情况下，可以考虑放宽条件或使用其中一个结果集
		if len(finalDocIDs) == 0 {
			if len(invertedIndexResults) > 0 {
				// 策略1：使用倒排索引结果（更精确的关键词匹配）
				finalDocIDs = invertedIndexResults
			} else if len(vectorResults) > 0 {
				// 策略2：使用向量搜索结果（更好的语义相似性）
				finalDocIDs = vectorResults
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
	processedResults := make([]SearchResultItem, 0, len(finalDocIDs))
	for _, id := range finalDocIDs {
		item := SearchResultItem{ID: id, Fields: make(map[string]interface{}), Score: 0.0}

		// 检查是否是 SELECT *
		isSelectAll := false
		for _, sf := range selectFields {
			if sf.FieldName == "*" {
				isSelectAll = true
				break
			}
		}

		// TODO: 获取文档的实际数据
		// 这里需要实现从存储中获取文档详细信息的逻辑
		// 可以添加一个 GetDocumentDetails 方法到 MultiTableSearchService
		docData, err := se.service.GetDocumentDetails(tableName, id, selectFields, isSelectAll)
		if err != nil {
			continue
		}
		item.Fields = docData.Fields
		item.Score = docData.Score

		// 示例：填充假数据
		if isSelectAll {
			item.Fields["title"] = "Sample Title for " + id
			item.Fields["content"] = "Sample content..."
			item.Score = 0.95 // 示例分数
		} else {
			for _, sf := range selectFields {
				item.Fields[sf.FieldName] = "Value for " + sf.FieldName
			}
			item.Score = 0.85 // 示例分数
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

			// 比较逻辑 (完善版，根据字段类型进行转换和比较)
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
			case int:
				// 转换为 int64 进行比较
				iI := int64(vI)
				var iJ int64
				switch vJ := valJ.(type) {
				case int:
					iJ = int64(vJ)
				case int32:
					iJ = int64(vJ)
				case int64:
					iJ = vJ
				case string:
					// 尝试将字符串转换为整数
					if v, err := strconv.ParseInt(vJ, 10, 64); err == nil {
						iJ = v
					} else {
						return false // 转换失败
					}
				default:
					return false // 类型不匹配
				}
				if orderByDirection == "ASC" {
					return iI < iJ
				}
				return iI > iJ
			case int32:
				// 转换为 int64 进行比较
				iI := int64(vI)
				var iJ int64
				switch vJ := valJ.(type) {
				case int:
					iJ = int64(vJ)
				case int32:
					iJ = int64(vJ)
				case int64:
					iJ = vJ
				case string:
					// 尝试将字符串转换为整数
					if v, err := strconv.ParseInt(vJ, 10, 64); err == nil {
						iJ = v
					} else {
						return false // 转换失败
					}
				default:
					return false // 类型不匹配
				}
				if orderByDirection == "ASC" {
					return iI < iJ
				}
				return iI > iJ
			case int64:
				// 直接使用 int64 进行比较
				var iJ int64
				switch vJ := valJ.(type) {
				case int:
					iJ = int64(vJ)
				case int32:
					iJ = int64(vJ)
				case int64:
					iJ = vJ
				case string:
					// 尝试将字符串转换为整数
					if v, err := strconv.ParseInt(vJ, 10, 64); err == nil {
						iJ = v
					} else {
						return false // 转换失败
					}
				default:
					return false // 类型不匹配
				}
				if orderByDirection == "ASC" {
					return vI < iJ
				}
				return vI > iJ
			case float32:
				// 转换为 float64 进行比较
				fI := float64(vI)
				var fJ float64
				switch vJ := valJ.(type) {
				case float32:
					fJ = float64(vJ)
				case float64:
					fJ = vJ
				case string:
					// 尝试将字符串转换为浮点数
					if v, err := strconv.ParseFloat(vJ, 64); err == nil {
						fJ = v
					} else {
						return false // 转换失败
					}
				default:
					return false // 类型不匹配
				}
				if orderByDirection == "ASC" {
					return fI < fJ
				}
				return fI > fJ
			case float64:
				// 直接使用 float64 进行比较
				var fJ float64
				switch vJ := valJ.(type) {
				case float32:
					fJ = float64(vJ)
				case float64:
					fJ = vJ
				case string:
					// 尝试将字符串转换为浮点数
					if v, err := strconv.ParseFloat(vJ, 64); err == nil {
						fJ = v
					} else {
						return false // 转换失败
					}
				default:
					return false // 类型不匹配
				}
				if orderByDirection == "ASC" {
					return vI < fJ
				}
				return vI > fJ
			}
			return false // 默认不排序或类型不支持
		})
	}

	// 5. 应用 LIMIT 和 OFFSET
	start := int(offset)
	end := int(offset + limit)

	if limit == 0 { // limit 0 表示不限制
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

// getBitPositionForField 根据字段名和值获取对应的位位置
// 这个函数需要根据实际业务逻辑实现
func getBitPositionForField(field string, value interface{}) int {
	// 这里只是示例，实际实现需要根据业务逻辑定义
	// 例如：可以使用一个映射表将字段名和值映射到位位置

	// 示例映射逻辑
	switch field {
	case "status":
		if strValue, ok := value.(string); ok {
			switch strValue {
			case "active":
				return 0
			case "inactive":
				return 1
			}
		}
	case "category":
		if strValue, ok := value.(string); ok {
			switch strValue {
			case "A":
				return 2
			case "B":
				return 3
			case "C":
				return 4
			}
		}
	case "type":
		if strValue, ok := value.(string); ok {
			switch strValue {
			case "internal":
				return 5
			case "external":
				return 6
			}
		}
	}

	return -1 // 表示没有找到对应的位位置
}
