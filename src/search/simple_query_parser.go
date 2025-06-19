package search

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// 简化的SQL解析器，专门用于基本分页查询
// 支持的查询格式：SELECT * FROM table_name [WHERE conditions] [ORDER BY field [ASC|DESC]] [LIMIT number] [OFFSET number]

// SimpleQuery 简化的查询结构体
type SimpleQuery struct {
	TableName        string
	Fields           []string
	WhereConditions  []SimpleFilterCondition
	OrderByField     string
	OrderByDirection string // "ASC" or "DESC"
	Limit            int64
	Offset           int64
}

// SimpleFilterCondition 表示一个简单的过滤条件
type SimpleFilterCondition struct {
	Field    string
	Operator string // "=", ">", "<", ">=", "<=", "!=", "LIKE"
	Value    string
}

// SimpleParser 简化的SQL解析器
type SimpleParser struct {
	query string
}

// NewSimpleParser 创建一个新的简化解析器
func NewSimpleParser(query string) *SimpleParser {
	return &SimpleParser{query: strings.TrimSpace(query)}
}

// Parse 解析SQL查询并返回SimpleQuery结构体
func (p *SimpleParser) Parse() (*SimpleQuery, error) {
	query := strings.ToLower(p.query)
	result := &SimpleQuery{
		Fields:           []string{"*"},
		OrderByDirection: "ASC",
		Limit:            -1,
		Offset:           0,
	}

	// 解析SELECT子句
	if err := p.parseSelect(query, result); err != nil {
		return nil, err
	}

	// 解析FROM子句
	if err := p.parseFrom(query, result); err != nil {
		return nil, err
	}

	// 解析WHERE子句
	if err := p.parseWhere(query, result); err != nil {
		return nil, err
	}

	// 解析ORDER BY子句
	if err := p.parseOrderBy(query, result); err != nil {
		return nil, err
	}

	// 解析LIMIT子句
	if err := p.parseLimit(query, result); err != nil {
		return nil, err
	}

	// 解析OFFSET子句
	if err := p.parseOffset(query, result); err != nil {
		return nil, err
	}

	return result, nil
}

// parseSelect 解析SELECT子句
func (p *SimpleParser) parseSelect(query string, result *SimpleQuery) error {
	selectRegex := regexp.MustCompile(`select\s+([^\s]+(?:\s*,\s*[^\s]+)*)\s+from`)
	matches := selectRegex.FindStringSubmatch(query)
	if len(matches) < 2 {
		return fmt.Errorf("invalid SELECT clause")
	}

	fieldsStr := strings.TrimSpace(matches[1])
	if fieldsStr == "*" {
		result.Fields = []string{"*"}
	} else {
		fields := strings.Split(fieldsStr, ",")
		result.Fields = make([]string, len(fields))
		for i, field := range fields {
			result.Fields[i] = strings.TrimSpace(field)
		}
	}
	return nil
}

// parseFrom 解析FROM子句
func (p *SimpleParser) parseFrom(query string, result *SimpleQuery) error {
	fromRegex := regexp.MustCompile(`from\s+([a-zA-Z_][a-zA-Z0-9_]*)`)
	matches := fromRegex.FindStringSubmatch(query)
	if len(matches) < 2 {
		return fmt.Errorf("invalid FROM clause")
	}
	result.TableName = matches[1]
	return nil
}

// parseWhere 解析WHERE子句
func (p *SimpleParser) parseWhere(query string, result *SimpleQuery) error {
	whereRegex := regexp.MustCompile(`where\s+(.+?)(?:\s+order\s+by|\s+limit|\s+offset|$)`)
	matches := whereRegex.FindStringSubmatch(query)
	if len(matches) < 2 {
		return nil // WHERE子句是可选的
	}

	whereClause := strings.TrimSpace(matches[1])
	return p.parseSimpleConditions(whereClause, result)
}

// parseSimpleConditions 解析简单的WHERE条件
func (p *SimpleParser) parseSimpleConditions(whereClause string, result *SimpleQuery) error {
	// 支持简单的条件：field = 'value', field > 123, etc.
	conditionRegex := regexp.MustCompile(`([a-zA-Z_][a-zA-Z0-9_]*)\s*(=|!=|>=|<=|>|<|like)\s*('([^']*)'|([0-9.]+))`)
	matches := conditionRegex.FindAllStringSubmatch(whereClause, -1)

	for _, match := range matches {
		if len(match) >= 6 {
			condition := SimpleFilterCondition{
				Field:    match[1],
				Operator: strings.ToUpper(match[2]),
			}
			// 判断是字符串值还是数字值
			if match[4] != "" {
				condition.Value = match[4] // 字符串值
			} else {
				condition.Value = match[5] // 数字值
			}
			result.WhereConditions = append(result.WhereConditions, condition)
		}
	}
	return nil
}

// parseOrderBy 解析ORDER BY子句
func (p *SimpleParser) parseOrderBy(query string, result *SimpleQuery) error {
	orderByRegex := regexp.MustCompile(`order\s+by\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(asc|desc))?`)
	matches := orderByRegex.FindStringSubmatch(query)
	if len(matches) >= 2 {
		result.OrderByField = matches[1]
		if len(matches) >= 3 && matches[2] != "" {
			result.OrderByDirection = strings.ToUpper(matches[2])
		} else {
			result.OrderByDirection = "ASC"
		}
	}
	return nil
}

// parseLimit 解析LIMIT子句
func (p *SimpleParser) parseLimit(query string, result *SimpleQuery) error {
	limitRegex := regexp.MustCompile(`limit\s+(\d+)`)
	matches := limitRegex.FindStringSubmatch(query)
	if len(matches) >= 2 {
		limit, err := strconv.ParseInt(matches[1], 10, 64)
		if err != nil {
			return fmt.Errorf("invalid LIMIT value: %s", matches[1])
		}
		result.Limit = limit
	}
	return nil
}

// parseOffset 解析OFFSET子句
func (p *SimpleParser) parseOffset(query string, result *SimpleQuery) error {
	offsetRegex := regexp.MustCompile(`offset\s+(\d+)`)
	matches := offsetRegex.FindStringSubmatch(query)
	if len(matches) >= 2 {
		offset, err := strconv.ParseInt(matches[1], 10, 64)
		if err != nil {
			return fmt.Errorf("invalid OFFSET value: %s", matches[1])
		}
		result.Offset = offset
	}
	return nil
}

// ParseQuery 解析SQL查询的主要入口函数
func ParseQuery(query string) (*SimpleQuery, error) {
	parser := NewSimpleParser(query)
	return parser.Parse()
}

// ParseQueryWithDefaults 解析SQL查询并设置默认LIMIT值
func ParseQueryWithDefaults(query string, defaultLimit int64) (*SimpleQuery, error) {
	result, err := ParseQuery(query)
	if err != nil {
		return nil, err
	}

	if result.Limit == -1 {
		result.Limit = defaultLimit
	}

	return result, nil
}

// 辅助函数：提取字符串中的数字
func extractNumber(s string) (int64, error) {
	return strconv.ParseInt(strings.TrimSpace(s), 10, 64)
}

// 辅助函数：清理字符串（去除引号）
func cleanString(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && s[0] == '\'' && s[len(s)-1] == '\'' {
		return s[1 : len(s)-1]
	}
	return s
}

// SimpleParsedQuery 保持与原有系统的兼容性
type SimpleParsedQuery struct {
	TableName        string
	Fields           []string
	KeywordQuery     string
	VectorQueryText  string
	Filters          []SimpleFilterCondition
	OrderByField     string
	OrderByDirection string
	Limit            int64
	Offset           int64
	UseANN           bool
	K                int
}

// ConvertSimpleQueryToParsedQuery 将SimpleQuery转换为ParsedQuery以保持兼容性
func ConvertSimpleQueryToParsedQuery(sq *SimpleQuery) *SimpleParsedQuery {
	return &SimpleParsedQuery{
		TableName:        sq.TableName,
		Fields:           sq.Fields,
		Filters:          sq.WhereConditions,
		OrderByField:     sq.OrderByField,
		OrderByDirection: sq.OrderByDirection,
		Limit:            sq.Limit,
		Offset:           sq.Offset,
		UseANN:           false,
		K:                0,
	}
}
