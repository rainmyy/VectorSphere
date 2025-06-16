package service

import (
	"VectorSphere/src/library/logger"
	"VectorSphere/src/search"
	"context"
	"database/sql"
	"fmt"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"time"

	"VectorSphere/src/messages"
	_ "github.com/go-sql-driver/mysql"
	_ "github.com/lib/pq"
	_ "github.com/mattn/go-sqlite3"
)

// DataTransformHook 数据转换钩子函数类型
type DataTransformHook func(columns []string, values []interface{}) ([]interface{}, error)

// DocumentProcessHook 文档处理钩子函数类型
type DocumentProcessHook func(doc *messages.Document) error

// RetryStrategy 重试策略
type RetryStrategy struct {
	MaxRetries      int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	BackoffFactor   float64
	RetryableErrors []string
}

// DBImportConfig 数据库导入配置
type DBImportConfig struct {
	DriverName      string            // 数据库驱动名称：mysql, postgres, sqlite3
	DataSource      string            // 数据源连接字符串
	BatchSize       int               // 批量处理大小
	WorkerCount     int               // 工作协程数
	VectorizedType  int               // 向量化类型
	FieldMappings   map[string]string // 字段映射
	KeywordFields   []string          // 关键词字段列表
	ContentField    string            // 内容字段
	IdField         string            // ID字段
	FeatureField    string            // 特征字段
	MaxRetries      int               // 最大重试次数
	RetryDelay      time.Duration     // 重试延迟
	TransactionSize int               // 事务大小（每个事务处理的批次数）
	MaxMemoryUsage  int64             // 最大内存使用量（字节）
	DBMaxIdleConns  int               // 数据库最大空闲连接数
	DBMaxOpenConns  int               // 数据库最大打开连接数
	DBConnTimeout   time.Duration     // 数据库连接超时时间
	VectorBatchSize int               // 向量处理批次大小

	// 增量导入相关配置
	IncrementalImport bool                 // 是否启用增量导入
	LastImportTime    map[string]time.Time // 每个表的上次导入时间
	UpdateTimeField   string               // 数据库中的更新时间字段
	PrimaryKeyField   string               // 主键字段，用于增量更新
	ChecksumField     string               // 校验和字段，用于检测记录是否变更

	// 数据转换钩子
	PreTransformHook  DataTransformHook   // 数据转换前的钩子
	PostTransformHook DataTransformHook   // 数据转换后的钩子
	PreDocumentHook   DocumentProcessHook // 文档创建前的钩子
	PostDocumentHook  DocumentProcessHook // 文档创建后的钩子

	// 重试策略
	RetryStrategy *RetryStrategy
}

// DBImportService 数据库导入服务
type DBImportService struct {
	config           *DBImportConfig
	searchService    *search.MultiTableSearchService
	importStats      *ImportStats
	ctx              context.Context
	cancelFunc       context.CancelFunc
	importInProgress bool
	importMutex      sync.Mutex
	memoryMonitor    *MemoryMonitor
}

// MemoryMonitor 内存监控器
type MemoryMonitor struct {
	maxMemoryUsage int64
	currentUsage   int64
	mu             sync.Mutex
	ctx            context.Context
	cancel         context.CancelFunc
}

// NewRetryStrategy 创建默认重试策略
func NewRetryStrategy() *RetryStrategy {
	return &RetryStrategy{
		MaxRetries:    5,
		InitialDelay:  500 * time.Millisecond,
		MaxDelay:      30 * time.Second,
		BackoffFactor: 2.0,
		RetryableErrors: []string{
			"connection reset",
			"connection refused",
			"deadline exceeded",
			"server closed",
			"temporary failure",
			"timeout",
		},
	}
}

// IsRetryable 判断错误是否可重试
func (s *RetryStrategy) IsRetryable(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())
	for _, retryableErr := range s.RetryableErrors {
		if strings.Contains(errStr, strings.ToLower(retryableErr)) {
			return true
		}
	}

	return false
}

// NewMemoryMonitor 创建内存监控器
func NewMemoryMonitor(maxUsage int64, parentCtx context.Context) *MemoryMonitor {
	ctx, cancel := context.WithCancel(parentCtx)
	return &MemoryMonitor{
		maxMemoryUsage: maxUsage,
		ctx:            ctx,
		cancel:         cancel,
	}
}

// CheckMemoryUsage 检查内存使用情况
func (m *MemoryMonitor) CheckMemoryUsage() {
	if m == nil || m.maxMemoryUsage <= 0 {
		return
	}

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	m.mu.Lock()
	defer m.mu.Unlock()

	m.currentUsage = int64(memStats.Alloc)
	if m.currentUsage > m.maxMemoryUsage {
		logger.Warning("内存使用超过阈值: 当前 %d MB, 最大 %d MB", m.currentUsage/(1024*1024), m.maxMemoryUsage/(1024*1024))
		m.cancel() // 取消相关操作
	}
}

// ImportTableIncremental 增量导入表数据
func (s *DBImportService) ImportTableIncremental(tableName string, sqlQuery string) error {
	// 检查是否启用增量导入
	if !s.config.IncrementalImport {
		return s.ImportTable(tableName, sqlQuery)
	}

	// 获取上次导入时间
	lastImportTime, exists := s.config.LastImportTime[tableName]
	if !exists {
		// 首次导入，执行全量导入
		err := s.ImportTable(tableName, sqlQuery)
		if err == nil {
			// 更新上次导入时间
			if s.config.LastImportTime == nil {
				s.config.LastImportTime = make(map[string]time.Time)
			}
			s.config.LastImportTime[tableName] = time.Now()
		}
		return err
	}

	// 构建增量查询
	incrementalQuery := sqlQuery
	if s.config.UpdateTimeField != "" {
		// 检查SQL是否已包含WHERE子句
		if strings.Contains(strings.ToUpper(sqlQuery), " WHERE ") {
			incrementalQuery = fmt.Sprintf("%s AND %s > '%s'", sqlQuery,
				s.config.UpdateTimeField, lastImportTime.Format("2006-01-02 15:04:05"))
		} else {
			incrementalQuery = fmt.Sprintf("%s WHERE %s > '%s'", sqlQuery,
				s.config.UpdateTimeField, lastImportTime.Format("2006-01-02 15:04:05"))
		}
	}

	// 执行增量导入
	logger.Info("执行增量导入，表: %s，上次导入时间: %s", tableName, lastImportTime.Format("2006-01-02 15:04:05"))
	err := s.ImportTable(tableName, incrementalQuery)
	if err == nil {
		// 更新上次导入时间
		s.config.LastImportTime[tableName] = time.Now()
	}

	return err
}

// 增强的重试操作方法
func (s *DBImportService) retryOperationWithBackoff(operation func() error) error {
	if s.config.RetryStrategy == nil {
		s.config.RetryStrategy = NewRetryStrategy()
	}

	var err error
	delay := s.config.RetryStrategy.InitialDelay

	for attempt := 0; attempt < s.config.RetryStrategy.MaxRetries; attempt++ {
		err = operation()
		if err == nil {
			return nil
		}

		// 检查错误是否可重试
		if !s.config.RetryStrategy.IsRetryable(err) {
			return fmt.Errorf("不可重试的错误: %w", err)
		}

		if attempt < s.config.RetryStrategy.MaxRetries-1 {
			logger.Warning("操作失败，将在 %v 后重试 (尝试 %d/%d): %v",
				delay, attempt+1, s.config.RetryStrategy.MaxRetries, err)

			// 添加抖动以避免惊群效应
			jitter := time.Duration(rand.Int63n(int64(delay) / 4))
			time.Sleep(delay + jitter)

			// 指数退避
			delay = time.Duration(float64(delay) * s.config.RetryStrategy.BackoffFactor)
			if delay > s.config.RetryStrategy.MaxDelay {
				delay = s.config.RetryStrategy.MaxDelay
			}
		}
	}

	return fmt.Errorf("%d 次尝试后操作仍然失败: %w", s.config.RetryStrategy.MaxRetries, err)
}

// GetChangedRecords 获取自上次导入以来变更的记录
func (s *DBImportService) GetChangedRecords(db *sql.DB, tableName string, lastImportTime time.Time) ([]string, error) {
	var idList []string

	// 构建查询
	query := fmt.Sprintf("SELECT %s FROM %s WHERE %s > ?",
		s.config.PrimaryKeyField, tableName, s.config.UpdateTimeField)

	// 执行查询
	rows, err := db.Query(query, lastImportTime.Format("2006-01-02 15:04:05"))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	// 收集ID
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		idList = append(idList, id)
	}

	return idList, rows.Err()
}

// StartMonitoring 开始监控内存
func (m *MemoryMonitor) StartMonitoring(interval time.Duration) {
	if m == nil || m.maxMemoryUsage <= 0 {
		return
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				m.CheckMemoryUsage()
			case <-m.ctx.Done():
				return
			}
		}
	}()
}

// ImportStats 导入统计信息
type ImportStats struct {
	TotalDocuments    int
	SuccessDocuments  int
	FailedDocuments   int
	StartTime         time.Time
	EndTime           time.Time
	CurrentBatch      int
	TotalBatches      int
	CurrentTable      string
	Errors            []string
	MemoryUsage       int64         // 当前内存使用量
	AvgProcessingTime time.Duration // 平均处理时间
	mu                sync.Mutex

	// 进度跟踪
	Progress          float64       // 导入进度百分比 (0-100)
	EstimatedTimeLeft time.Duration // 预计剩余时间
	ProcessedBytes    int64         // 已处理的数据量（字节）
	TotalBytes        int64         // 总数据量（字节）
	CurrentSpeed      float64       // 当前处理速度（文档/秒）
	AverageSpeed      float64       // 平均处理速度（文档/秒）
	StartedAt         time.Time     // 开始时间
	UpdatedAt         time.Time     // 最后更新时间
	ImportType        string        // 导入类型（全量/增量）
	StageInfo         string        // 当前阶段信息
}

// NewDBImportService 创建新的数据库导入服务
func NewDBImportService(searchService *search.MultiTableSearchService, config *DBImportConfig) *DBImportService {
	// 设置默认值
	if config.BatchSize <= 0 {
		// 根据可用内存动态调整批处理大小
		var memStats runtime.MemStats
		runtime.ReadMemStats(&memStats)
		availableMem := int64(memStats.Alloc)

		// 根据可用内存计算合理地批处理大小，假设每个文档平均占用10KB
		config.BatchSize = int(availableMem / (10 * 1024) / 10) // 使用可用内存的1/10
		if config.BatchSize < 50 {
			config.BatchSize = 50 // 最小批处理大小
		} else if config.BatchSize > 1000 {
			config.BatchSize = 1000 // 最大批处理大小
		}
		logger.Info("根据系统内存自动设置批处理大小: %d", config.BatchSize)
	}

	if config.WorkerCount <= 0 {
		// 根据CPU核心数设置工作协程数
		config.WorkerCount = runtime.NumCPU()
		if config.WorkerCount > 8 {
			config.WorkerCount = 8 // 限制最大工作协程数
		}
	}

	if config.VectorizedType <= 0 {
		config.VectorizedType = 1 // 默认使用简单词袋模型
	}

	if config.MaxRetries <= 0 {
		config.MaxRetries = 3
	}

	if config.RetryDelay <= 0 {
		config.RetryDelay = 500 * time.Millisecond
	}

	if config.TransactionSize <= 0 {
		// 根据批处理大小动态调整事务大小
		config.TransactionSize = 5
		if config.BatchSize > 500 {
			config.TransactionSize = 2 // 大批次使用小事务
		} else if config.BatchSize < 100 {
			config.TransactionSize = 10 // 小批次使用大事务
		}
	}

	if config.MaxMemoryUsage <= 0 {
		// 默认使用系统可用内存的70%作为最大内存使用量
		var memStats runtime.MemStats
		runtime.ReadMemStats(&memStats)
		config.MaxMemoryUsage = int64(float64(memStats.Alloc) * 0.7)
	}

	if config.DBMaxIdleConns <= 0 {
		config.DBMaxIdleConns = 5
	}

	if config.DBMaxOpenConns <= 0 {
		config.DBMaxOpenConns = 10
	}

	if config.DBConnTimeout <= 0 {
		config.DBConnTimeout = 30 * time.Second
	}

	if config.VectorBatchSize <= 0 {
		// 向量处理批次大小默认为普通批次大小的1/4
		config.VectorBatchSize = config.BatchSize / 4
		if config.VectorBatchSize < 10 {
			config.VectorBatchSize = 10
		}
	}

	// 初始化重试策略
	if config.RetryStrategy == nil {
		config.RetryStrategy = NewRetryStrategy()
	}

	ctx, cancel := context.WithCancel(context.Background())
	memoryMonitor := NewMemoryMonitor(config.MaxMemoryUsage, ctx)

	return &DBImportService{
		config:        config,
		searchService: searchService,
		importStats:   &ImportStats{StartTime: time.Now()},
		ctx:           ctx,
		cancelFunc:    cancel,
		memoryMonitor: memoryMonitor,
	}
}

// UpdateProgress 更新导入进度
func (s *DBImportService) UpdateProgress(processed, total int, stage string) {
	s.importStats.mu.Lock()
	defer s.importStats.mu.Unlock()

	now := time.Now()
	duration := now.Sub(s.importStats.StartTime)

	// 更新进度信息
	s.importStats.Progress = float64(processed) / float64(total) * 100
	s.importStats.CurrentSpeed = float64(processed) / duration.Seconds()
	s.importStats.AverageSpeed = float64(s.importStats.SuccessDocuments) / duration.Seconds()

	// 计算预计剩余时间
	if s.importStats.CurrentSpeed > 0 {
		remaining := float64(total-processed) / s.importStats.CurrentSpeed
		s.importStats.EstimatedTimeLeft = time.Duration(remaining * float64(time.Second))
	}

	s.importStats.UpdatedAt = now
	s.importStats.StageInfo = stage

	// 记录日志
	logger.Info("导入进度: %.2f%%, 阶段: %s, 预计剩余时间: %v",
		s.importStats.Progress, stage, s.importStats.EstimatedTimeLeft)
}

// GetProgressReport 获取格式化的进度报告
func (s *DBImportService) GetProgressReport() string {
	s.importStats.mu.Lock()
	defer s.importStats.mu.Unlock()

	report := fmt.Sprintf(
		"导入进度报告:\n"+
			"表: %s\n"+
			"类型: %s\n"+
			"进度: %.2f%%\n"+
			"已处理: %d/%d 文档\n"+
			"成功: %d, 失败: %d\n"+
			"当前速度: %.2f 文档/秒\n"+
			"平均速度: %.2f 文档/秒\n"+
			"已用时间: %v\n"+
			"预计剩余: %v\n"+
			"当前阶段: %s\n"+
			"内存使用: %d MB",
		s.importStats.CurrentTable,
		s.importStats.ImportType,
		s.importStats.Progress,
		s.importStats.SuccessDocuments+s.importStats.FailedDocuments,
		s.importStats.TotalDocuments,
		s.importStats.SuccessDocuments,
		s.importStats.FailedDocuments,
		s.importStats.CurrentSpeed,
		s.importStats.AverageSpeed,
		time.Since(s.importStats.StartTime),
		s.importStats.EstimatedTimeLeft,
		s.importStats.StageInfo,
		s.importStats.MemoryUsage/(1024*1024),
	)

	return report
}

// ImportTable 从数据库表导入数据
func (s *DBImportService) ImportTable(tableName string, sqlQuery string) error {
	s.importMutex.Lock()
	if s.importInProgress {
		s.importMutex.Unlock()
		return fmt.Errorf("导入已在进行中，请等待当前导入完成")
	}
	s.importInProgress = true
	s.importMutex.Unlock()

	defer func() {
		s.importMutex.Lock()
		s.importInProgress = false
		s.importMutex.Unlock()
	}()

	// 重置统计信息
	s.importStats.mu.Lock()
	s.importStats = &ImportStats{
		StartTime:    time.Now(),
		CurrentTable: tableName,
	}
	s.importStats.mu.Unlock()

	// 开始内存监控
	s.memoryMonitor.StartMonitoring(5 * time.Second)

	// 连接数据库并配置连接池
	db, err := sql.Open(s.config.DriverName, s.config.DataSource)
	if err != nil {
		return fmt.Errorf("连接数据库失败: %w", err)
	}
	defer db.Close()

	// 配置数据库连接池
	db.SetMaxIdleConns(s.config.DBMaxIdleConns)
	db.SetMaxOpenConns(s.config.DBMaxOpenConns)
	db.SetConnMaxLifetime(s.config.DBConnTimeout)

	// 测试连接
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		return fmt.Errorf("数据库连接测试失败: %w", err)
	}

	// 查询数据
	rows, err := db.QueryContext(s.ctx, sqlQuery)
	if err != nil {
		return fmt.Errorf("执行查询失败: %w", err)
	}
	defer rows.Close()

	// 获取列名
	columns, err := rows.Columns()
	if err != nil {
		return fmt.Errorf("获取列名失败: %w", err)
	}

	// 验证必要的字段映射
	if err := s.validateFieldMappings(columns); err != nil {
		return err
	}

	// 创建工作池
	workChan := make(chan []messages.Document, s.config.WorkerCount*2)
	errChan := make(chan error, s.config.WorkerCount)
	doneChan := make(chan bool, 1)
	ctx, cancel = context.WithCancel(s.ctx)
	defer cancel()

	// 启动工作协程
	var wg sync.WaitGroup
	for i := 0; i < s.config.WorkerCount; i++ {
		wg.Add(1)
		go s.processDocumentBatch(ctx, tableName, workChan, errChan, &wg)
	}

	// 启动错误处理协程
	go func() {
		for err := range errChan {
			s.importStats.mu.Lock()
			s.importStats.Errors = append(s.importStats.Errors, err.Error())
			s.importStats.mu.Unlock()
			logger.Error("导入错误: %v", err)

			// 如果是严重错误，取消所有操作
			cancel()
			break
		}
	}()

	// 等待所有工作完成
	go func() {
		wg.Wait()
		close(errChan)
		doneChan <- true
	}()

	// 读取数据并创建文档批次
	batch := make([]messages.Document, 0, s.config.BatchSize)
	batchCount := 0
	totalRows := 0

	// 准备扫描行数据
	values := make([]interface{}, len(columns))
	valuePtrs := make([]interface{}, len(columns))
	for i := range columns {
		valuePtrs[i] = &values[i]
	}

	// 记录处理开始时间
	batchStartTime := time.Now()
	processedBatches := 0
	totalProcessingTime := time.Duration(0)

	for rows.Next() {
		// 检查是否被取消
		select {
		case <-ctx.Done():
			return fmt.Errorf("导入操作被取消: %v", ctx.Err())
		default:
			// 继续处理
		}

		// 扫描行数据
		if err := rows.Scan(valuePtrs...); err != nil {
			return fmt.Errorf("扫描行数据失败: %w", err)
		}

		// 创建文档
		doc, err := s.createDocument(columns, values)
		if err != nil {
			s.importStats.mu.Lock()
			s.importStats.FailedDocuments++
			s.importStats.Errors = append(s.importStats.Errors, fmt.Sprintf("创建文档失败: %v", err))
			s.importStats.mu.Unlock()
			logger.Warning("创建文档失败: %v", err)
			continue
		}

		// 添加到批次
		batch = append(batch, doc)
		totalRows++

		// 达到批处理大小，发送到工作通道
		if len(batch) >= s.config.BatchSize {
			// 创建批次副本，避免数据竞争
			batchCopy := make([]messages.Document, len(batch))
			copy(batchCopy, batch)
			workChan <- batchCopy
			batch = batch[:0] // 清空批次，但保留底层数组
			batchCount++

			// 更新统计信息
			s.importStats.mu.Lock()
			s.importStats.CurrentBatch = batchCount
			s.importStats.mu.Unlock()

			// 计算批处理时间
			processedBatches++
			batchProcessTime := time.Since(batchStartTime)
			totalProcessingTime += batchProcessTime
			batchStartTime = time.Now()

			// 动态调整批处理大小
			if processedBatches >= 5 { // 每5个批次调整一次
				avgTime := totalProcessingTime / time.Duration(processedBatches)
				s.adjustBatchSize(avgTime)
				processedBatches = 0
				totalProcessingTime = 0
			}

			// 检查内存使用情况
			s.memoryMonitor.CheckMemoryUsage()
		}
	}

	// 处理最后一个不完整的批次
	if len(batch) > 0 {
		workChan <- batch
		batchCount++
		s.importStats.mu.Lock()
		s.importStats.CurrentBatch = batchCount
		s.importStats.TotalBatches = batchCount
		s.importStats.mu.Unlock()
	}

	// 检查行扫描错误
	if err := rows.Err(); err != nil {
		return fmt.Errorf("行扫描过程中发生错误: %w", err)
	}

	// 关闭工作通道
	close(workChan)

	// 等待所有工作完成或出错
	select {
	case <-doneChan:
		// 所有工作完成
	case <-ctx.Done():
		return fmt.Errorf("导入操作被取消: %v", ctx.Err())
	}

	// 更新统计信息
	s.importStats.mu.Lock()
	s.importStats.EndTime = time.Now()
	s.importStats.TotalDocuments = totalRows

	// 获取当前内存使用情况
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	s.importStats.MemoryUsage = int64(memStats.Alloc)

	s.importStats.mu.Unlock()

	logger.Info("成功导入表 %s 的数据: 总计 %d 条，成功 %d 条，失败 %d 条，内存使用 %d MB",
		tableName, totalRows, s.importStats.SuccessDocuments, s.importStats.FailedDocuments, s.importStats.MemoryUsage/(1024*1024))

	return nil
}

// validateFieldMappings 验证字段映射配置
func (s *DBImportService) validateFieldMappings(columns []string) error {
	// 验证ID字段
	idFieldFound := false
	for _, col := range columns {
		if col == s.config.IdField {
			idFieldFound = true
			break
		}
	}

	if !idFieldFound && s.config.IdField != "" {
		return fmt.Errorf("配置的ID字段 '%s' 在查询结果中不存在", s.config.IdField)
	}

	// 验证内容字段
	if s.config.ContentField != "*" {
		contentFieldFound := false
		for _, col := range columns {
			if col == s.config.ContentField {
				contentFieldFound = true
				break
			}
		}

		if !contentFieldFound && s.config.ContentField != "" {
			return fmt.Errorf("配置的内容字段 '%s' 在查询结果中不存在", s.config.ContentField)
		}
	}

	// 验证关键词字段
	for _, kwField := range s.config.KeywordFields {
		kwFieldFound := false
		for _, col := range columns {
			if col == kwField {
				kwFieldFound = true
				break
			}
		}

		if !kwFieldFound {
			return fmt.Errorf("配置的关键词字段 '%s' 在查询结果中不存在", kwField)
		}
	}

	return nil
}

// adjustBatchSize 根据处理时间动态调整批处理大小
func (s *DBImportService) adjustBatchSize(avgProcessTime time.Duration) {
	// 目标处理时间：500ms-2s之间
	const targetMinTime = 500 * time.Millisecond
	const targetMaxTime = 2 * time.Second

	if avgProcessTime < targetMinTime && s.config.BatchSize < 1000 {
		// 处理太快，增加批处理大小
		newSize := int(float64(s.config.BatchSize) * 1.2) // 增加20%
		if newSize > 1000 {
			newSize = 1000 // 最大批处理大小
		}
		if newSize != s.config.BatchSize {
			logger.Info("动态调整批处理大小: %d -> %d (处理时间: %v)", s.config.BatchSize, newSize, avgProcessTime)
			s.config.BatchSize = newSize
		}
	} else if avgProcessTime > targetMaxTime && s.config.BatchSize > 50 {
		// 处理太慢，减少批处理大小
		newSize := int(float64(s.config.BatchSize) * 0.8) // 减少20%
		if newSize < 50 {
			newSize = 50 // 最小批处理大小
		}
		if newSize != s.config.BatchSize {
			logger.Info("动态调整批处理大小: %d -> %d (处理时间: %v)", s.config.BatchSize, newSize, avgProcessTime)
			s.config.BatchSize = newSize
		}
	}

	// 同时调整事务大小
	if s.config.BatchSize > 500 && s.config.TransactionSize > 2 {
		s.config.TransactionSize = 2 // 大批次使用小事务
	} else if s.config.BatchSize < 100 && s.config.TransactionSize < 10 {
		s.config.TransactionSize = 10 // 小批次使用大事务
	}
}

// processDocumentBatch 处理文档批次
func (s *DBImportService) processDocumentBatch(ctx context.Context, tableName string, workChan <-chan []messages.Document, errChan chan<- error, wg *sync.WaitGroup) {
	defer wg.Done()

	var batchesInTx int
	var docBuffer []messages.Document

	for batch := range workChan {
		// 检查是否被取消
		select {
		case <-ctx.Done():
			errChan <- fmt.Errorf("批处理被取消: %v", ctx.Err())
			return
		default:
			// 继续处理
		}

		// 将当前批次添加到缓冲区
		docBuffer = append(docBuffer, batch...)
		batchesInTx++

		// 当达到事务大小或没有更多批次时，提交事务
		if batchesInTx >= s.config.TransactionSize || len(workChan) == 0 {
			// 分离向量处理和索引处理，减少单个事务的大小
			if len(docBuffer) > s.config.VectorBatchSize {
				// 将文档分成多个向量批次处理
				for i := 0; i < len(docBuffer); i += s.config.VectorBatchSize {
					end := i + s.config.VectorBatchSize
					if end > len(docBuffer) {
						end = len(docBuffer)
					}

					// 使用重试机制添加文档
					err := s.retryOperation(func() error {
						return s.searchService.BatchAddDocuments(tableName, docBuffer[i:end], s.config.VectorizedType)
					}, s.config.MaxRetries, s.config.RetryDelay)

					if err != nil {
						errChan <- fmt.Errorf("批量添加文档失败: %w", err)
						return
					}

					// 更新统计信息
					s.importStats.mu.Lock()
					s.importStats.SuccessDocuments += end - i
					s.importStats.mu.Unlock()

					// 检查内存使用情况
					s.memoryMonitor.CheckMemoryUsage()
				}
			} else {
				// 使用重试机制添加文档
				err := s.retryOperation(func() error {
					return s.searchService.BatchAddDocuments(tableName, docBuffer, s.config.VectorizedType)
				}, s.config.MaxRetries, s.config.RetryDelay)

				if err != nil {
					errChan <- fmt.Errorf("批量添加文档失败: %w", err)
					return
				}

				// 更新统计信息
				s.importStats.mu.Lock()
				s.importStats.SuccessDocuments += len(docBuffer)
				s.importStats.mu.Unlock()
			}

			// 重置缓冲区和事务计数
			docBuffer = nil
			batchesInTx = 0

			// 主动触发GC，减少内存压力
			if s.config.BatchSize > 500 {
				runtime.GC()
			}
		}
	}
}

// retryOperation 重试操作
func (s *DBImportService) retryOperation(operation func() error, maxRetries int, delay time.Duration) error {
	var err error
	for attempt := 0; attempt < maxRetries; attempt++ {
		err = operation()
		if err == nil {
			return nil
		}

		if attempt < maxRetries-1 {
			logger.Warning("操作失败，将在 %v 后重试 (尝试 %d/%d): %v", delay, attempt+1, maxRetries, err)
			time.Sleep(delay)
			// 指数退避
			delay *= 2
		}
	}
	return fmt.Errorf("%d 次尝试后操作仍然失败: %w", maxRetries, err)
}

// createDocument 从数据库行创建文档
func (s *DBImportService) createDocument(columns []string, rowValues []interface{}) (messages.Document, error) {
	// 应用前置数据转换钩子
	if s.config.PreTransformHook != nil {
		var err error
		rowValues, err = s.config.PreTransformHook(columns, rowValues)
		if err != nil {
			return messages.Document{}, fmt.Errorf("前置数据转换钩子执行失败: %w", err)
		}
	}

	doc := messages.Document{}
	var contentBuilder strings.Builder

	// 遍历所有列
	for i, colName := range columns {
		val := rowValues[i]

		// 跳过空值
		if val == nil {
			continue
		}

		// 转换为字符串
		strVal := fmt.Sprintf("%v", val)

		// 设置ID
		if colName == s.config.IdField {
			doc.Id = strVal
		}

		// 设置特征位
		if colName == s.config.FeatureField {
			// 尝试将字符串转换为uint64
			var bitsFeature uint64
			_, err := fmt.Sscanf(strVal, "%d", &bitsFeature)
			if err == nil {
				doc.BitsFeature = bitsFeature
			}
		}

		// 添加关键词
		for _, kwField := range s.config.KeywordFields {
			if colName == kwField && strVal != "" {
				// 使用字段映射获取索引字段名
				fieldName := colName
				if mapped, ok := s.config.FieldMappings[colName]; ok {
					fieldName = mapped
				}

				// 对字段值进行分词
				words := strings.Fields(strVal)
				for _, word := range words {
					doc.KeWords = append(doc.KeWords, &messages.KeyWord{
						Field: fieldName,
						Word:  word,
					})
				}

				// 同时将整个字段值作为一个关键词
				doc.KeWords = append(doc.KeWords, &messages.KeyWord{
					Field: fieldName,
					Word:  strVal,
				})
			}
		}

		// 构建内容字段
		if colName == s.config.ContentField || s.config.ContentField == "*" {
			if contentBuilder.Len() > 0 {
				contentBuilder.WriteString(" ")
			}
			contentBuilder.WriteString(strVal)
		}
	}

	// 设置文档内容
	doc.Bytes = []byte(contentBuilder.String())

	// 验证文档ID
	if doc.Id == "" {
		return doc, fmt.Errorf("无法从数据中提取文档ID，请确保IdField配置正确")
	}

	// 应用后置数据转换钩子
	if s.config.PostTransformHook != nil {
		transformedValues, err := s.config.PostTransformHook(columns, rowValues)
		if err != nil {
			return doc, fmt.Errorf("后置数据转换钩子执行失败: %w", err)
		}

		// 使用转换后的值更新文档
		for i, colName := range columns {
			if i < len(transformedValues) && transformedValues[i] != nil {
				val := transformedValues[i]
				strVal := fmt.Sprintf("%v", val)

				// 更新文档字段
				if colName == s.config.IdField {
					doc.Id = strVal
				}
				// ... 其他字段更新 ...
			}
		}
	}

	// 应用前置文档处理钩子
	if s.config.PreDocumentHook != nil {
		if err := s.config.PreDocumentHook(&doc); err != nil {
			return doc, fmt.Errorf("前置文档处理钩子执行失败: %w", err)
		}
	}

	// 验证文档ID
	if doc.Id == "" {
		return doc, fmt.Errorf("无法从数据中提取文档ID，请确保IdField配置正确")
	}

	// 应用后置文档处理钩子
	if s.config.PostDocumentHook != nil {
		if err := s.config.PostDocumentHook(&doc); err != nil {
			return doc, fmt.Errorf("后置文档处理钩子执行失败: %w", err)
		}
	}
	return doc, nil
}

// ImportFromMultipleTables 从多个表导入数据
func (s *DBImportService) ImportFromMultipleTables(tableMapping map[string]string) error {
	for tableName, sqlQuery := range tableMapping {
		logger.Info("开始导入表 %s 的数据", tableName)
		if err := s.ImportTable(tableName, sqlQuery); err != nil {
			return fmt.Errorf("导入表 %s 失败: %w", tableName, err)
		}
	}
	return nil
}

// GetImportStats 获取导入统计信息
func (s *DBImportService) GetImportStats() ImportStats {
	s.importStats.mu.Lock()
	defer s.importStats.mu.Unlock()
	stats := *s.importStats
	return stats
}

// CancelImport 取消导入操作
func (s *DBImportService) CancelImport() {
	s.cancelFunc()
}

// ImportScheduler 导入调度器
type ImportScheduler struct {
	service   *DBImportService
	schedules map[string]*ImportSchedule
	stopChan  chan struct{}
	wg        sync.WaitGroup
	mu        sync.Mutex
}

// ImportSchedule 导入计划
type ImportSchedule struct {
	TableName   string
	SQLQuery    string
	Interval    time.Duration
	Incremental bool
	LastRun     time.Time
	NextRun     time.Time
	Running     bool
	Enabled     bool
}

// NewImportScheduler 创建新的导入调度器
func NewImportScheduler(service *DBImportService) *ImportScheduler {
	return &ImportScheduler{
		service:   service,
		schedules: make(map[string]*ImportSchedule),
		stopChan:  make(chan struct{}),
	}
}

// AddSchedule 添加导入计划
func (s *ImportScheduler) AddSchedule(tableName, sqlQuery string, interval time.Duration, incremental bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.schedules[tableName] = &ImportSchedule{
		TableName:   tableName,
		SQLQuery:    sqlQuery,
		Interval:    interval,
		Incremental: incremental,
		NextRun:     time.Now(),
		Enabled:     true,
	}

	logger.Info("已添加导入计划: 表=%s, 间隔=%v, 增量=%v", tableName, interval, incremental)
}

// RemoveSchedule 移除导入计划
func (s *ImportScheduler) RemoveSchedule(tableName string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.schedules, tableName)
	logger.Info("已移除导入计划: 表=%s", tableName)
}

// Start 启动调度器
func (s *ImportScheduler) Start() {
	s.wg.Add(1)
	go s.run()
	logger.Info("导入调度器已启动")
}

// Stop 停止调度器
func (s *ImportScheduler) Stop() {
	close(s.stopChan)
	s.wg.Wait()
	logger.Info("导入调度器已停止")
}

// run 运行调度器
func (s *ImportScheduler) run() {
	defer s.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.checkSchedules()
		case <-s.stopChan:
			return
		}
	}
}

// checkSchedules 检查并执行计划
func (s *ImportScheduler) checkSchedules() {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()

	for tableName, schedule := range s.schedules {
		if !schedule.Enabled || schedule.Running {
			continue
		}

		if now.After(schedule.NextRun) {
			schedule.Running = true
			s.wg.Add(1)

			go func(tableName string, schedule *ImportSchedule) {
				defer s.wg.Done()
				defer func() {
					s.mu.Lock()
					schedule.Running = false
					schedule.LastRun = time.Now()
					schedule.NextRun = schedule.LastRun.Add(schedule.Interval)
					s.mu.Unlock()
				}()

				logger.Info("执行计划导入: 表=%s, 增量=%v", tableName, schedule.Incremental)

				var err error
				if schedule.Incremental {
					err = s.service.ImportTableIncremental(tableName, schedule.SQLQuery)
				} else {
					err = s.service.ImportTable(tableName, schedule.SQLQuery)
				}

				if err != nil {
					logger.Error("计划导入失败: 表=%s, 错误=%v", tableName, err)
				} else {
					logger.Info("计划导入成功: 表=%s", tableName)
				}
			}(tableName, schedule)
		}
	}
}

// GetScheduleStatus 获取计划状态
func (s *ImportScheduler) GetScheduleStatus() map[string]map[string]interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()

	status := make(map[string]map[string]interface{})

	for tableName, schedule := range s.schedules {
		status[tableName] = map[string]interface{}{
			"enabled":     schedule.Enabled,
			"running":     schedule.Running,
			"interval":    schedule.Interval.String(),
			"incremental": schedule.Incremental,
			"lastRun":     schedule.LastRun,
			"nextRun":     schedule.NextRun,
		}
	}

	return status
}
