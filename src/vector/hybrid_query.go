package vector

import (
	"fmt"
	"strconv"
	"sync"
	"time"

	"VectorSphere/src/library/entity"
)

// HybridQueryEngine 混合查询引擎
type HybridQueryEngine struct {
	vectorDB       *VectorDB
	scalarIndex    *ScalarIndex
	queryOptimizer *QueryOptimizer
	resultMerger   *ResultMerger
}

// ScalarIndex 标量索引
type ScalarIndex struct {
	fields map[string]*FieldIndex
	mu     sync.RWMutex
}

// FieldIndex 字段索引
type FieldIndex struct {
	FieldName string
	FieldType FieldType
	Index     interface{} // 可以是BTree、Hash等不同类型的索引
	mu        sync.RWMutex
}

// FieldType 字段类型
type FieldType string

const (
	FieldTypeString   FieldType = "string"
	FieldTypeInt      FieldType = "int"
	FieldTypeFloat    FieldType = "float"
	FieldTypeBool     FieldType = "bool"
	FieldTypeDateTime FieldType = "datetime"
	FieldTypeArray    FieldType = "array"
)

// QueryCondition 查询条件
type QueryCondition struct {
	Field    string
	Operator Operator
	Value    interface{}
	Logic    LogicOperator
}

// Operator 操作符
type Operator string

const (
	OpEqual              Operator = "eq"
	OpNotEqual           Operator = "ne"
	OpGreaterThan        Operator = "gt"
	OpGreaterThanOrEqual Operator = "gte"
	OpLessThan           Operator = "lt"
	OpLessThanOrEqual    Operator = "lte"
	OpIn                 Operator = "in"
	OpNotIn              Operator = "not_in"
	OpLike               Operator = "like"
	OpNotLike            Operator = "not_like"
	OpIsNull             Operator = "is_null"
	OpIsNotNull          Operator = "is_not_null"
	OpContains           Operator = "contains"
	OpNotContains        Operator = "not_contains"
)

// LogicOperator 逻辑操作符
type LogicOperator string

const (
	LogicAnd LogicOperator = "and"
	LogicOr  LogicOperator = "or"
	LogicNot LogicOperator = "not"
)

// HybridQuery 混合查询
type HybridQuery struct {
	// 向量查询部分
	QueryVector   []float64
	K             int
	VectorOptions entity.SearchOptions

	// 标量查询部分
	Conditions []QueryCondition
	OrderBy    []OrderByClause
	Limit      int
	Offset     int

	// 混合查询选项
	MergeStrategy MergeStrategy
	Weights       QueryWeights
}

// OrderByClause 排序子句
type OrderByClause struct {
	Field     string
	Direction SortDirection
}

// SortDirection 排序方向
type SortDirection string

const (
	SortAsc  SortDirection = "asc"
	SortDesc SortDirection = "desc"
)

// MergeStrategy 合并策略
type MergeStrategy string

const (
	MergeIntersection MergeStrategy = "intersection" // 交集
	MergeUnion        MergeStrategy = "union"        // 并集
	MergeVectorFirst  MergeStrategy = "vector_first" // 向量优先
	MergeScalarFirst  MergeStrategy = "scalar_first" // 标量优先
	MergeWeighted     MergeStrategy = "weighted"     // 加权合并
)

// QueryWeights 查询权重
type QueryWeights struct {
	VectorWeight float64 // 向量查询权重
	ScalarWeight float64 // 标量查询权重
}

// QueryOptimizer 查询优化器
type QueryOptimizer struct {
	statistics *QueryStatistics
	costModel  *CostModel
}

// QueryStatistics 查询统计
type QueryStatistics struct {
	fieldSelectivity map[string]float64 // 字段选择性
	indexUsage       map[string]int     // 索引使用次数
	queryPatterns    map[string]int     // 查询模式
	mu               sync.RWMutex
}

// CostModel 成本模型
type CostModel struct {
	vectorSearchCost float64
	scalarSearchCost float64
	mergeCost        float64
	indexSeekCost    float64
	indexScanCost    float64
}

// ResultMerger 结果合并器
type ResultMerger struct {
	strategies map[MergeStrategy]MergeFunction
}

// MergeFunction 合并函数
type MergeFunction func(vectorResults []entity.Result, scalarResults []int, weights QueryWeights) []entity.Result

// NewHybridQueryEngine 创建混合查询引擎
func NewHybridQueryEngine(vectorDB *VectorDB) *HybridQueryEngine {
	return &HybridQueryEngine{
		vectorDB:       vectorDB,
		scalarIndex:    NewScalarIndex(),
		queryOptimizer: NewQueryOptimizer(),
		resultMerger:   NewResultMerger(),
	}
}

// NewScalarIndex 创建标量索引
func NewScalarIndex() *ScalarIndex {
	return &ScalarIndex{
		fields: make(map[string]*FieldIndex),
	}
}

// NewQueryOptimizer 创建查询优化器
func NewQueryOptimizer() *QueryOptimizer {
	return &QueryOptimizer{
		statistics: &QueryStatistics{
			fieldSelectivity: make(map[string]float64),
			indexUsage:       make(map[string]int),
			queryPatterns:    make(map[string]int),
		},
		costModel: &CostModel{
			vectorSearchCost: 1.0,
			scalarSearchCost: 0.1,
			mergeCost:        0.05,
			indexSeekCost:    0.01,
			indexScanCost:    0.001,
		},
	}
}

// NewResultMerger 创建结果合并器
func NewResultMerger() *ResultMerger {
	merger := &ResultMerger{
		strategies: make(map[MergeStrategy]MergeFunction),
	}

	// 注册合并策略
	merger.strategies[MergeIntersection] = merger.mergeIntersection
	merger.strategies[MergeUnion] = merger.mergeUnion
	merger.strategies[MergeVectorFirst] = merger.mergeVectorFirst
	merger.strategies[MergeScalarFirst] = merger.mergeScalarFirst
	merger.strategies[MergeWeighted] = merger.mergeWeighted

	return merger
}

// ExecuteHybridQuery 执行混合查询
func (hqe *HybridQueryEngine) ExecuteHybridQuery(query *HybridQuery) ([]entity.Result, error) {
	// 查询优化
	optimizedQuery := hqe.queryOptimizer.OptimizeQuery(query)

	// 并行执行向量查询和标量查询
	vectorResultsChan := make(chan []entity.Result, 1)
	scalarResultsChan := make(chan []int, 1)
	errorChan := make(chan error, 2)

	// 执行向量查询
	go func() {
		if len(optimizedQuery.QueryVector) > 0 {
			results, err := hqe.vectorDB.OptimizedSearch(optimizedQuery.QueryVector, optimizedQuery.K, optimizedQuery.VectorOptions)
			if err != nil {
				errorChan <- fmt.Errorf("向量查询失败: %v", err)
				return
			}
			vectorResultsChan <- results
		} else {
			vectorResultsChan <- []entity.Result{}
		}
	}()

	// 执行标量查询
	go func() {
		if len(optimizedQuery.Conditions) > 0 {
			results, err := hqe.executeScalarQuery(optimizedQuery.Conditions)
			if err != nil {
				errorChan <- fmt.Errorf("标量查询失败: %v", err)
				return
			}
			scalarResultsChan <- results
		} else {
			scalarResultsChan <- []int{}
		}
	}()

	// 等待查询结果
	var vectorResults []entity.Result
	var scalarResults []int
	var errors []error

	for i := 0; i < 2; i++ {
		select {
		case vr := <-vectorResultsChan:
			vectorResults = vr
		case sr := <-scalarResultsChan:
			scalarResults = sr
		case err := <-errorChan:
			errors = append(errors, err)
		case <-time.After(30 * time.Second):
			return nil, fmt.Errorf("查询超时")
		}
	}

	if len(errors) > 0 {
		return nil, fmt.Errorf("查询执行失败: %v", errors)
	}

	// 合并结果
	mergedResults := hqe.resultMerger.MergeResults(vectorResults, scalarResults, optimizedQuery.MergeStrategy, optimizedQuery.Weights)

	// 应用排序和分页
	finalResults := hqe.applyOrderAndPaging(mergedResults, optimizedQuery.OrderBy, optimizedQuery.Limit, optimizedQuery.Offset)

	return finalResults, nil
}

// executeScalarQuery 执行标量查询
func (hqe *HybridQueryEngine) executeScalarQuery(conditions []QueryCondition) ([]int, error) {
	if len(conditions) == 0 {
		return []int{}, nil
	}

	// 简化实现：遍历所有向量的元数据进行过滤
	results := make([]int, 0)

	// 这里应该根据实际的元数据存储结构来实现
	// 目前返回空结果作为占位符
	return results, nil
}

// OptimizeQuery 优化查询
func (qo *QueryOptimizer) OptimizeQuery(query *HybridQuery) *HybridQuery {
	// 创建查询副本
	optimizedQuery := *query

	// 条件重排序：将选择性高的条件放在前面
	optimizedQuery.Conditions = qo.reorderConditions(query.Conditions)

	// 索引选择：为每个条件选择最优索引
	qo.selectOptimalIndexes(optimizedQuery.Conditions)

	// 查询策略选择
	if len(query.QueryVector) > 0 && len(query.Conditions) > 0 {
		// 根据成本模型选择最优执行策略
		vectorCost := qo.estimateVectorQueryCost(query.QueryVector, query.K)
		scalarCost := qo.estimateScalarQueryCost(query.Conditions)

		if scalarCost < vectorCost {
			// 标量查询成本更低，优先执行标量查询
			optimizedQuery.MergeStrategy = MergeScalarFirst
		} else {
			// 向量查询成本更低，优先执行向量查询
			optimizedQuery.MergeStrategy = MergeVectorFirst
		}
	}

	return &optimizedQuery
}

// reorderConditions 重排序条件
func (qo *QueryOptimizer) reorderConditions(conditions []QueryCondition) []QueryCondition {
	qo.statistics.mu.RLock()
	defer qo.statistics.mu.RUnlock()

	// 按选择性排序（选择性高的在前）
	reorderedConditions := make([]QueryCondition, len(conditions))
	copy(reorderedConditions, conditions)

	// 简单的冒泡排序，按选择性排序
	for i := 0; i < len(reorderedConditions); i++ {
		for j := i + 1; j < len(reorderedConditions); j++ {
			selectivity1 := qo.statistics.fieldSelectivity[reorderedConditions[i].Field]
			selectivity2 := qo.statistics.fieldSelectivity[reorderedConditions[j].Field]

			if selectivity1 < selectivity2 { // 选择性高的在前
				reorderedConditions[i], reorderedConditions[j] = reorderedConditions[j], reorderedConditions[i]
			}
		}
	}

	return reorderedConditions
}

// selectOptimalIndexes 选择最优索引
func (qo *QueryOptimizer) selectOptimalIndexes(conditions []QueryCondition) {
	// 为每个条件选择最优的索引
	for _, condition := range conditions {
		// 更新索引使用统计
		qo.statistics.mu.Lock()
		qo.statistics.indexUsage[condition.Field]++
		qo.statistics.mu.Unlock()
	}
}

// estimateVectorQueryCost 估算向量查询成本
func (qo *QueryOptimizer) estimateVectorQueryCost(queryVector []float64, k int) float64 {
	// 简化的成本估算
	vectorDim := len(queryVector)
	cost := qo.costModel.vectorSearchCost * float64(vectorDim) * float64(k)
	return cost
}

// estimateScalarQueryCost 估算标量查询成本
func (qo *QueryOptimizer) estimateScalarQueryCost(conditions []QueryCondition) float64 {
	// 简化的成本估算
	cost := 0.0
	for _, condition := range conditions {
		// 根据操作符类型估算成本
		switch condition.Operator {
		case OpEqual, OpNotEqual:
			cost += qo.costModel.indexSeekCost
		case OpGreaterThan, OpLessThan, OpGreaterThanOrEqual, OpLessThanOrEqual:
			cost += qo.costModel.indexScanCost * 10 // 范围查询成本更高
		case OpIn, OpNotIn:
			cost += qo.costModel.indexSeekCost * 5 // IN查询成本中等
		case OpLike, OpNotLike:
			cost += qo.costModel.scalarSearchCost * 100 // 模糊查询成本很高
		default:
			cost += qo.costModel.scalarSearchCost
		}
	}
	return cost
}

// MergeResults 合并结果
func (rm *ResultMerger) MergeResults(vectorResults []entity.Result, scalarResults []int, strategy MergeStrategy, weights QueryWeights) []entity.Result {
	if mergeFunc, exists := rm.strategies[strategy]; exists {
		return mergeFunc(vectorResults, scalarResults, weights)
	}
	return rm.mergeIntersection(vectorResults, scalarResults, weights)
}

// mergeIntersection 交集合并
func (rm *ResultMerger) mergeIntersection(vectorResults []entity.Result, scalarResults []int, weights QueryWeights) []entity.Result {
	if len(vectorResults) == 0 || len(scalarResults) == 0 {
		return []entity.Result{}
	}

	scalarSet := make(map[int]bool)
	for _, id := range scalarResults {
		scalarSet[id] = true
	}

	intersection := make([]entity.Result, 0)
	for _, result := range vectorResults {
		if id, err := strconv.Atoi(result.Id); err == nil && scalarSet[id] {
			intersection = append(intersection, result)
		}
	}

	return intersection
}

// mergeUnion 并集合并
func (rm *ResultMerger) mergeUnion(vectorResults []entity.Result, scalarResults []int, weights QueryWeights) []entity.Result {
	resultMap := make(map[int]entity.Result)

	// 添加向量结果
	for _, result := range vectorResults {
		if id, err := strconv.Atoi(result.Id); err == nil {
			resultMap[id] = result
		}
	}

	// 添加标量结果（如果不存在）
	for _, id := range scalarResults {
		if _, exists := resultMap[id]; !exists {
			resultMap[id] = entity.Result{Id: strconv.Itoa(id), Distance: 1.0} // 默认距离
		}
	}

	// 转换为切片
	union := make([]entity.Result, 0, len(resultMap))
	for _, result := range resultMap {
		union = append(union, result)
	}

	return union
}

// mergeVectorFirst 向量优先合并
func (rm *ResultMerger) mergeVectorFirst(vectorResults []entity.Result, scalarResults []int, weights QueryWeights) []entity.Result {
	if len(vectorResults) == 0 {
		// 如果没有向量结果，返回标量结果
		results := make([]entity.Result, len(scalarResults))
		for i, id := range scalarResults {
			results[i] = entity.Result{Id: strconv.Itoa(id), Distance: 1.0}
		}
		return results
	}
	return vectorResults
}

// mergeScalarFirst 标量优先合并
func (rm *ResultMerger) mergeScalarFirst(vectorResults []entity.Result, scalarResults []int, weights QueryWeights) []entity.Result {
	if len(scalarResults) == 0 {
		return vectorResults
	}

	// 将标量结果转换为Result格式
	results := make([]entity.Result, len(scalarResults))
	for i, id := range scalarResults {
		results[i] = entity.Result{Id: strconv.Itoa(id), Distance: 0.0} // 标量匹配距离为0
	}

	return results
}

// mergeWeighted 加权合并
func (rm *ResultMerger) mergeWeighted(vectorResults []entity.Result, scalarResults []int, weights QueryWeights) []entity.Result {
	resultMap := make(map[int]*entity.Result)

	// 处理向量结果
	for _, result := range vectorResults {
		if id, err := strconv.Atoi(result.Id); err == nil {
			weightedResult := result
			weightedResult.Distance *= weights.VectorWeight
			resultMap[id] = &weightedResult
		}
	}

	// 处理标量结果
	for _, id := range scalarResults {
		if existing, exists := resultMap[id]; exists {
			// 如果已存在，合并分数
			existing.Distance += weights.ScalarWeight * 0.1 // 标量匹配给予小的加分
		} else {
			// 新结果
			resultMap[id] = &entity.Result{
				Id:       strconv.Itoa(id),
				Distance: weights.ScalarWeight,
			}
		}
	}

	// 转换为切片并排序
	weightedResults := make([]entity.Result, 0, len(resultMap))
	for _, result := range resultMap {
		weightedResults = append(weightedResults, *result)
	}

	// 按加权分数排序
	for i := 0; i < len(weightedResults); i++ {
		for j := i + 1; j < len(weightedResults); j++ {
			if weightedResults[i].Distance > weightedResults[j].Distance {
				weightedResults[i], weightedResults[j] = weightedResults[j], weightedResults[i]
			}
		}
	}

	return weightedResults
}

// applyOrderAndPaging 应用排序和分页
func (hqe *HybridQueryEngine) applyOrderAndPaging(results []entity.Result, orderBy []OrderByClause, limit, offset int) []entity.Result {
	// 应用排序
	if len(orderBy) > 0 {
		// 这里应该根据orderBy子句对结果进行排序
		// 目前保持原有排序
	}

	// 应用分页
	if offset >= len(results) {
		return []entity.Result{}
	}

	start := offset
	end := len(results)
	if limit > 0 && start+limit < end {
		end = start + limit
	}

	return results[start:end]
}

// AddFieldIndex 添加字段索引
func (si *ScalarIndex) AddFieldIndex(fieldName string, fieldType FieldType) error {
	si.mu.Lock()
	defer si.mu.Unlock()

	if _, exists := si.fields[fieldName]; exists {
		return fmt.Errorf("字段 %s 的索引已存在", fieldName)
	}

	fieldIndex := &FieldIndex{
		FieldName: fieldName,
		FieldType: fieldType,
		Index:     createIndexForType(fieldType),
	}

	si.fields[fieldName] = fieldIndex
	return nil
}

// createIndexForType 为字段类型创建索引
func createIndexForType(fieldType FieldType) interface{} {
	// 这里应该根据字段类型创建相应的索引结构
	// 目前返回简单的map作为占位符
	switch fieldType {
	case FieldTypeString:
		return make(map[string][]int) // 字符串到ID列表的映射
	case FieldTypeInt:
		return make(map[int][]int) // 整数到ID列表的映射
	case FieldTypeFloat:
		return make(map[float64][]int) // 浮点数到ID列表的映射
	case FieldTypeBool:
		return make(map[bool][]int) // 布尔值到ID列表的映射
	case FieldTypeDateTime:
		return make(map[time.Time][]int) // 时间到ID列表的映射
	default:
		return make(map[interface{}][]int) // 通用映射
	}
}

// QueryBuilder 查询构建器
type QueryBuilder struct {
	query *HybridQuery
}

// NewQueryBuilder 创建查询构建器
func NewQueryBuilder() *QueryBuilder {
	return &QueryBuilder{
		query: &HybridQuery{
			Conditions:    make([]QueryCondition, 0),
			OrderBy:       make([]OrderByClause, 0),
			MergeStrategy: MergeIntersection,
			Weights:       QueryWeights{VectorWeight: 0.7, ScalarWeight: 0.3},
		},
	}
}

// Vector 设置向量查询
func (qb *QueryBuilder) Vector(queryVector []float64, k int, options entity.SearchOptions) *QueryBuilder {
	qb.query.QueryVector = queryVector
	qb.query.K = k
	qb.query.VectorOptions = options
	return qb
}

// Where 添加查询条件
func (qb *QueryBuilder) Where(field string, operator Operator, value interface{}) *QueryBuilder {
	condition := QueryCondition{
		Field:    field,
		Operator: operator,
		Value:    value,
		Logic:    LogicAnd,
	}
	qb.query.Conditions = append(qb.query.Conditions, condition)
	return qb
}

// Or 添加OR条件
func (qb *QueryBuilder) Or(field string, operator Operator, value interface{}) *QueryBuilder {
	condition := QueryCondition{
		Field:    field,
		Operator: operator,
		Value:    value,
		Logic:    LogicOr,
	}
	qb.query.Conditions = append(qb.query.Conditions, condition)
	return qb
}

// OrderBy 添加排序
func (qb *QueryBuilder) OrderBy(field string, direction SortDirection) *QueryBuilder {
	orderClause := OrderByClause{
		Field:     field,
		Direction: direction,
	}
	qb.query.OrderBy = append(qb.query.OrderBy, orderClause)
	return qb
}

// Limit 设置限制
func (qb *QueryBuilder) Limit(limit int) *QueryBuilder {
	qb.query.Limit = limit
	return qb
}

// Offset 设置偏移
func (qb *QueryBuilder) Offset(offset int) *QueryBuilder {
	qb.query.Offset = offset
	return qb
}

// MergeStrategy 设置合并策略
func (qb *QueryBuilder) MergeStrategy(strategy MergeStrategy) *QueryBuilder {
	qb.query.MergeStrategy = strategy
	return qb
}

// Weights 设置权重
func (qb *QueryBuilder) Weights(vectorWeight, scalarWeight float64) *QueryBuilder {
	qb.query.Weights = QueryWeights{
		VectorWeight: vectorWeight,
		ScalarWeight: scalarWeight,
	}
	return qb
}

// Build 构建查询
func (qb *QueryBuilder) Build() *HybridQuery {
	return qb.query
}
