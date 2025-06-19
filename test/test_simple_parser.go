package test

import (
	"VectorSphere/src/search"
	"fmt"
	"log"
	"strings"
)

func main() {
	fmt.Println("测试简化的SQL解析器")
	fmt.Println("========================")

	// 测试用例
	testQueries := []string{
		"SELECT * FROM users LIMIT 10",
		"SELECT id, name FROM products WHERE category = 'electronics' ORDER BY price DESC LIMIT 20 OFFSET 5",
		"SELECT * FROM orders WHERE status = 'active' AND amount > 100 ORDER BY created_at ASC",
		"SELECT title FROM articles WHERE author = 'john' LIMIT 5",
		"SELECT * FROM logs ORDER BY timestamp DESC LIMIT 100",
	}

	for i, query := range testQueries {
		fmt.Printf("\n测试 %d: %s\n", i+1, query)
		fmt.Println(strings.Repeat("-", 50))

		// 使用新的简化解析器
		simpleQuery, err := search.ParseQuery(query)
		if err != nil {
			log.Printf("解析错误: %v", err)
			continue
		}

		// 打印解析结果
		fmt.Printf("表名: %s\n", simpleQuery.TableName)
		fmt.Printf("字段: %v\n", simpleQuery.Fields)
		fmt.Printf("WHERE条件: %v\n", simpleQuery.WhereConditions)
		fmt.Printf("排序字段: %s\n", simpleQuery.OrderByField)
		fmt.Printf("排序方向: %s\n", simpleQuery.OrderByDirection)
		fmt.Printf("限制: %d\n", simpleQuery.Limit)
		fmt.Printf("偏移: %d\n", simpleQuery.Offset)

		// 测试兼容性转换
		parsedQuery := search.ConvertSimpleQueryToParsedQuery(simpleQuery)
		fmt.Printf("\n兼容性转换结果:\n")
		fmt.Printf("  表名: %s\n", parsedQuery.TableName)
		fmt.Printf("  字段: %v\n", parsedQuery.Fields)
		fmt.Printf("  过滤条件: %v\n", parsedQuery.Filters)
		fmt.Printf("  排序: %s %s\n", parsedQuery.OrderByField, parsedQuery.OrderByDirection)
		fmt.Printf("  分页: LIMIT %d OFFSET %d\n", parsedQuery.Limit, parsedQuery.Offset)

		// 测试ParseQueryWithAST兼容性函数
		_, compatParsedQuery, errs, err := search.ParseQueryWithAST(query)
		if err != nil {
			log.Printf("兼容性解析错误: %v", err)
		} else if len(errs) > 0 {
			log.Printf("兼容性解析警告: %v", errs)
		} else {
			fmt.Printf("\n兼容性函数测试成功!\n")
			fmt.Printf("  表名: %s\n", compatParsedQuery.TableName)
			fmt.Printf("  字段数量: %d\n", len(compatParsedQuery.Fields))
		}
	}

	fmt.Println("\n========================")
	fmt.Println("测试完成!")
}
