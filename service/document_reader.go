package service

import (
	"VectorSphere/db"
	"bytes"
	"fmt"
	"github.com/PuerkitoBio/goquery"
	"github.com/go-ego/gse"
	"github.com/ledongthuc/pdf"
	"github.com/unidoc/unioffice/document"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
	"unicode"
)

// ProgressCallback 进度报告回调函数类型
type ProgressCallback func(current, total int, phase string, elapsedTime time.Duration)

// 向量化类型常量，与VectorSphere中定义一致
const (
	TfidfVectorized         = 1
	SimpleVectorized        = 2
	WordEmbeddingVectorized = 3
	DefaultVectorized       = 0
)

// 文档类型常量
const (
	TXT_DOC  = "txt"
	HTML_DOC = "html"
	PDF_DOC  = "pdf"
	DOCX_DOC = "docx"
	MD_DOC   = "md"
)

// DocumentProcessor 文档处理器接口
type DocumentProcessor interface {
	ExtractText(filePath string) (string, error)
	CleanText(text string) string
	SplitText(text string, maxChunkSize int) []string
}

// DefaultDocumentProcessor 默认文档处理器实现
type DefaultDocumentProcessor struct {
	segmenter      gse.Segmenter
	stopWords      map[string]struct{}
	cleaningRegexp *regexp.Regexp
}

// NewDefaultDocumentProcessor 创建新的默认文档处理器
func NewDefaultDocumentProcessor() (*DefaultDocumentProcessor, error) {
	var seg gse.Segmenter
	err := seg.LoadDict()
	if err != nil {
		return nil, fmt.Errorf("加载分词词典失败: %w", err)
	}

	// 初始化停用词表
	stopWords := make(map[string]struct{})
	// 添加常见停用词
	for _, word := range []string{"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"} {
		stopWords[word] = struct{}{}
	}

	// 初始化清洗正则表达式
	cleaningRegexp := regexp.MustCompile(`[\s\p{P}]+`)

	return &DefaultDocumentProcessor{
		segmenter:      seg,
		stopWords:      stopWords,
		cleaningRegexp: cleaningRegexp,
	}, nil
}

// ExtractText 从文件中提取文本
func (p *DefaultDocumentProcessor) ExtractText(filePath string) (string, error) {
	ext := strings.ToLower(filepath.Ext(filePath))
	switch ext {
	case ".txt", ".md":
		return p.extractFromTextFile(filePath)
	case ".html", ".htm":
		return p.extractFromHTML(filePath)
	case ".pdf":
		return p.extractFromPDF(filePath)
	case ".docx":
		return p.extractFromDOCX(filePath)
	default:
		return "", fmt.Errorf("不支持的文件类型: %s", ext)
	}
}

// extractFromTextFile 从文本文件提取内容
func (p *DefaultDocumentProcessor) extractFromTextFile(filePath string) (string, error) {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("读取文件失败: %w", err)
	}
	return string(data), nil
}

// extractFromHTML 从HTML文件提取内容
func (p *DefaultDocumentProcessor) extractFromHTML(filePath string) (string, error) {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("读取HTML文件失败: %w", err)
	}

	doc, err := goquery.NewDocumentFromReader(bytes.NewReader(data))
	if err != nil {
		return "", fmt.Errorf("解析HTML失败: %w", err)
	}

	// 提取文本内容，去除脚本和样式
	doc.Find("script, style").Each(func(i int, s *goquery.Selection) {
		s.Remove()
	})

	return doc.Text(), nil
}

// extractFromPDF 从PDF文件提取内容
func (p *DefaultDocumentProcessor) extractFromPDF(filePath string) (string, error) {
	f, r, err := pdf.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("打开PDF文件失败: %w", err)
	}
	defer f.Close()

	var buf bytes.Buffer
	b, err := r.GetPlainText()
	if err != nil {
		return "", fmt.Errorf("提取PDF文本失败: %w", err)
	}

	_, err = buf.ReadFrom(b)
	if err != nil {
		return "", err
	}
	return buf.String(), nil
}

// extractFromDOCX 从DOCX文件提取内容
func (p *DefaultDocumentProcessor) extractFromDOCX(filePath string) (string, error) {
	doc, err := document.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("打开DOCX文件失败: %w", err)
	}

	var text strings.Builder
	for _, para := range doc.Paragraphs() {
		for _, run := range para.Runs() {
			text.WriteString(run.Text())
		}
		text.WriteString("\n")
	}

	return text.String(), nil
}

// CleanText 清洗文本
func (p *DefaultDocumentProcessor) CleanText(text string) string {
	// 替换多余的空白字符为单个空格
	text = p.cleaningRegexp.ReplaceAllString(text, " ")

	// 分词
	words := p.segmenter.Cut(text, true)

	// 过滤停用词和非文本字符
	var filtered []string
	for _, word := range words {
		// 跳过停用词
		if _, isStopWord := p.stopWords[word]; isStopWord {
			continue
		}

		// 跳过只包含标点或空白的词
		hasContent := false
		for _, r := range word {
			if !unicode.IsPunct(r) && !unicode.IsSpace(r) {
				hasContent = true
				break
			}
		}

		if hasContent {
			filtered = append(filtered, word)
		}
	}

	return strings.Join(filtered, " ")
}

// SplitText 将文本分割成固定大小的块
func (p *DefaultDocumentProcessor) SplitText(text string, maxChunkSize int) []string {
	if maxChunkSize <= 0 {
		maxChunkSize = 512 // 默认块大小
	}

	// 分词
	words := p.segmenter.Cut(text, true)

	// 按最大块大小分割
	var chunks []string
	var currentChunk []string
	currentSize := 0

	for _, word := range words {
		wordSize := len(word)

		// 如果当前块加上新词超过最大大小，创建新块
		if currentSize+wordSize > maxChunkSize && len(currentChunk) > 0 {
			chunks = append(chunks, strings.Join(currentChunk, " "))
			currentChunk = []string{word}
			currentSize = wordSize
		} else {
			currentChunk = append(currentChunk, word)
			currentSize += wordSize
		}
	}

	// 添加最后一个块
	if len(currentChunk) > 0 {
		chunks = append(chunks, strings.Join(currentChunk, " "))
	}

	return chunks
}

// EnhancedDocumentInfo 增强的文档信息结构体
type EnhancedDocumentInfo struct {
	ID           string
	Path         string
	Size         int64
	Content      string
	Chunks       []string
	Loaded       bool
	Processed    bool
	Metadata     map[string]interface{}
	LastModified time.Time
}

// LoadAndProcess 加载并处理文档
func (d *EnhancedDocumentInfo) LoadAndProcess(processor DocumentProcessor, chunkSize int) error {
	if d.Processed {
		return nil
	}

	// 提取文本
	content, err := processor.ExtractText(d.Path)
	if err != nil {
		return err
	}

	// 清洗文本
	cleanedText := processor.CleanText(content)

	// 分块
	d.Chunks = processor.SplitText(cleanedText, chunkSize)

	// 保存原始内容
	d.Content = content
	d.Loaded = true
	d.Processed = true

	return nil
}

// ProcessDocumentCollection 处理文档集合并构建向量数据库
func ProcessDocumentCollection(config DocumentProcessingConfig) error {
	// 初始化向量数据库
	vectorDB := db.NewVectorDB(config.DBPath, config.NumClusters)

	// 尝试从文件加载现有数据库
	err := vectorDB.LoadFromFile(config.DBPath)
	if err != nil {
		// 如果文件不存在，创建新的数据库
		if os.IsNotExist(err) {
			fmt.Printf("创建新的向量数据库: %s\n", config.DBPath)
		} else {
			return fmt.Errorf("加载向量数据库失败: %v", err)
		}
	}

	// 启用HNSW索引以提高搜索性能
	if config.UseHNSW {
		vectorDB.EnableHNSWIndex(config.HNSWMaxConnections, config.HNSWEFConstruction, config.HNSWEFSearch)
	}

	// 启用PQ压缩以减少内存使用
	if config.UsePQCompression {
		err := vectorDB.EnablePQCompression(config.DBPath, config.PQSubvectors, config.PQCentroids)
		if err != nil {
			return err
		}
	}

	// 创建文档处理器
	processor, err := NewDefaultDocumentProcessor()
	if err != nil {
		return fmt.Errorf("创建文档处理器失败: %v", err)
	}

	// 扫描文档目录
	docInfos, err := ScanDocumentsFromDirectory(config.DocsDir, config.AllowedExtensions)
	if err != nil {
		return fmt.Errorf("扫描文档目录失败: %v", err)
	}

	// 转换为增强的文档信息
	enhancedDocs := make([]*EnhancedDocumentInfo, len(docInfos))
	for i, doc := range docInfos {
		enhancedDocs[i] = &EnhancedDocumentInfo{
			ID:        doc.ID,
			Path:      doc.Path,
			Size:      doc.Size,
			Loaded:    false,
			Processed: false,
			Metadata:  make(map[string]interface{}),
		}

		// 获取文件最后修改时间
		fileInfo, err := os.Stat(doc.Path)
		if err == nil {
			enhancedDocs[i].LastModified = fileInfo.ModTime()
		}
	}

	fmt.Printf("扫描到 %d 个文档\n", len(enhancedDocs))

	// 处理文档集合
	err = ProcessEnhancedDocuments(vectorDB, enhancedDocs, processor, config)
	if err != nil {
		return fmt.Errorf("处理文档失败: %v", err)
	}

	// 保存数据库
	fmt.Println("正在保存向量数据库...")
	err = vectorDB.SaveToFile(config.DBPath)
	if err != nil {
		return fmt.Errorf("保存向量数据库失败: %v", err)
	}
	fmt.Println("向量数据库保存成功")

	return nil
}

// DocumentProcessingConfig 文档处理配置
type DocumentProcessingConfig struct {
	// 文档目录路径
	DocsDir string
	// 数据库文件路径
	DBPath string
	// 向量化类型
	VectorizedType int
	// 允许的文件扩展名
	AllowedExtensions []string
	// 批处理配置
	BatchConfig BatchConfig
	// 文档块大小
	ChunkSize int
	// 是否使用增量更新
	IncrementalUpdate bool
	// 聚类数量
	NumClusters int
	// 是否使用HNSW索引
	UseHNSW bool
	// HNSW最大连接数
	HNSWMaxConnections int
	// HNSW构建扩展因子
	HNSWEFConstruction float64
	// HNSW搜索扩展因子
	HNSWEFSearch float64
	// 是否使用PQ压缩
	UsePQCompression bool
	// PQ子向量数量
	PQSubvectors int
	// PQ每个子向量的质心数量
	PQCentroids  int
	CodebookPath string
}

// DefaultDocumentProcessingConfig 默认文档处理配置
func DefaultDocumentProcessingConfig() DocumentProcessingConfig {
	return DocumentProcessingConfig{
		DocsDir:            "./documents",
		DBPath:             "./vector_db.dat",
		CodebookPath:       "./code_book.dat",
		VectorizedType:     TfidfVectorized,
		AllowedExtensions:  []string{".txt", ".md", ".html", ".pdf", ".docx"},
		BatchConfig:        DefaultBatchConfig(),
		ChunkSize:          512,
		IncrementalUpdate:  true,
		NumClusters:        10,
		UseHNSW:            true,
		HNSWMaxConnections: 16,
		HNSWEFConstruction: 200,
		HNSWEFSearch:       50,
		UsePQCompression:   false,
		PQSubvectors:       8,
		PQCentroids:        256,
	}
}

// ProcessEnhancedDocuments 处理增强的文档集合
func ProcessEnhancedDocuments(vectorDB *db.VectorDB, docInfos []*EnhancedDocumentInfo, processor DocumentProcessor, config DocumentProcessingConfig) error {
	totalDocs := len(docInfos)
	if totalDocs == 0 {
		return nil
	}

	// 设置批处理配置
	batchConfig := config.BatchConfig

	// 初始化进度统计
	startTime := time.Now()
	processedCount := 0
	failedCount := 0

	// 如果启用增量更新，过滤掉已处理的文档
	var docsToProcess []*EnhancedDocumentInfo
	if config.IncrementalUpdate {
		// 获取数据库中已有的文档ID
		existingIDs := vectorDB.GetAllIDs()
		existingIDMap := make(map[string]bool)
		for _, id := range existingIDs {
			existingIDMap[id] = true
		}

		// 过滤需要处理的文档
		for _, doc := range docInfos {
			// 如果文档ID不在数据库中，或者文件修改时间比上次索引时间新，则需要处理
			if !existingIDMap[doc.ID] || (doc.LastModified.After(vectorDB.GetStats().LastReindexTime)) {
				docsToProcess = append(docsToProcess, doc)
			}
		}

		fmt.Printf("增量更新: 需要处理 %d/%d 个文档\n", len(docsToProcess), totalDocs)
	} else {
		docsToProcess = docInfos
	}

	// 如果没有需要处理的文档，直接返回
	if len(docsToProcess) == 0 {
		fmt.Println("没有新文档需要处理")
		return nil
	}

	// 分批处理
	for i := 0; i < len(docsToProcess); i += batchConfig.BatchSize {
		// 确定当前批次的结束索引
		end := i + batchConfig.BatchSize
		if end > len(docsToProcess) {
			end = len(docsToProcess)
		}

		// 当前批次的文档
		currentBatch := docsToProcess[i:end]
		batchSize := len(currentBatch)

		// 报告批次开始
		if batchConfig.ProgressCallback != nil {
			batchConfig.ProgressCallback(processedCount, len(docsToProcess), "批次开始", time.Since(startTime))
		}

		// 并行加载和处理当前批次的文档
		var wg sync.WaitGroup
		processErrors := make(chan error, batchSize)
		processSem := make(chan struct{}, batchConfig.NumWorkers)

		for j := range currentBatch {
			wg.Add(1)
			go func(doc *EnhancedDocumentInfo) {
				defer wg.Done()
				processSem <- struct{}{}
				defer func() { <-processSem }()

				// 加载并处理文档
				err := doc.LoadAndProcess(processor, config.ChunkSize)
				if err != nil {
					processErrors <- fmt.Errorf("处理文档 %s 失败: %w", doc.ID, err)
				}
			}(currentBatch[j])
		}

		// 等待所有处理完成
		wg.Wait()
		close(processErrors)

		// 检查处理错误
		processErrorsCount := 0
		for err := range processErrors {
			processErrorsCount++
			log.Printf("错误: %v", err)
			if batchConfig.ErrorHandling == "abort" {
				return fmt.Errorf("批处理中止: %w", err)
			}
		}

		// 更新失败计数
		failedCount += processErrorsCount

		// 批量添加文档块到向量数据库
		successCount := 0
		for _, doc := range currentBatch {
			if doc.Processed {
				// 为每个块创建唯一ID
				for i, chunk := range doc.Chunks {
					chunkID := fmt.Sprintf("%s_chunk_%d", doc.ID, i)

					// 添加块到向量数据库
					err := vectorDB.AddDocument(chunkID, chunk, config.VectorizedType)
					if err != nil {
						log.Printf("警告: 添加文档块 %s 失败: %v", chunkID, err)
						continue
					}

					// 添加元数据
					vectorDB.AddMetadata(chunkID, map[string]interface{}{
						"source_doc":    doc.ID,
						"chunk_index":   i,
						"total_chunks":  len(doc.Chunks),
						"path":          doc.Path,
						"last_modified": doc.LastModified,
					})

					successCount++
				}
			}
		}

		// 更新处理计数
		processedCount += successCount

		// 报告批次完成
		if batchConfig.ProgressCallback != nil {
			batchConfig.ProgressCallback(processedCount, len(docsToProcess), "批次完成", time.Since(startTime))
		}

		// 如果配置了内存限制，检查内存使用并可能触发GC
		if batchConfig.MemoryLimit > 0 {
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			memoryUsageMB := int(m.Alloc / (1024 * 1024))

			if memoryUsageMB > batchConfig.MemoryLimit {
				log.Printf("内存使用达到 %d MB，触发垃圾回收", memoryUsageMB)
				runtime.GC()
			}
		}

		// 如果配置了每批保存，则保存数据库
		if batchConfig.SaveAfterEachBatch {
			err := vectorDB.SaveToFile(vectorDB.GetFilePath())
			if err != nil {
				log.Printf("警告: 保存数据库失败: %v", err)
			}
		}

		// 清理当前批次的内存
		for j := range currentBatch {
			if currentBatch[j].Processed {
				currentBatch[j].Content = "" // 释放内容内存
				currentBatch[j].Chunks = nil // 释放块内存
			}
		}
	}

	// 构建或更新索引
	if processedCount > 0 {
		if batchConfig.ProgressCallback != nil {
			batchConfig.ProgressCallback(processedCount, len(docsToProcess), "开始构建索引", time.Since(startTime))
		}

		var err error
		indexStartTime := time.Now()

		// 使用并行索引构建
		if vectorDB.IsHNSWEnabled() {
			err = vectorDB.BuildHNSWIndexParallel(batchConfig.NumWorkers)
		} else {
			err = vectorDB.RebuildIndex()
		}

		if err != nil {
			return fmt.Errorf("构建索引失败: %w", err)
		}

		if batchConfig.ProgressCallback != nil {
			batchConfig.ProgressCallback(processedCount, len(docsToProcess), "索引构建完成", time.Since(startTime))
		}

		log.Printf("索引构建完成，耗时: %v", time.Since(indexStartTime))
	}

	// 最终统计
	elapsedTime := time.Since(startTime)
	log.Printf("处理完成: 成功添加 %d 个文档块，失败 %d 个文档，总耗时: %v",
		processedCount, failedCount, elapsedTime)

	return nil
}

// BatchConfig 批处理配置
type BatchConfig struct {
	// 每批处理的文档数量
	BatchSize int
	// 并行工作协程数量，默认为CPU核心数
	NumWorkers int
	// 是否启用增量索引构建
	IncrementalIndexing bool
	// 进度报告回调函数
	ProgressCallback ProgressCallback
	// 内存使用限制（MB），达到限制时触发GC
	MemoryLimit int
	// 是否在每批处理后保存数据库
	SaveAfterEachBatch bool
	// 错误处理策略："continue"(继续处理), "abort"(中止处理)
	ErrorHandling string
}

// DefaultBatchConfig 默认批处理配置
func DefaultBatchConfig() BatchConfig {
	return BatchConfig{
		BatchSize:           1000,
		NumWorkers:          runtime.NumCPU(),
		IncrementalIndexing: true,
		ProgressCallback:    nil,
		MemoryLimit:         4096, // 4GB
		SaveAfterEachBatch:  false,
		ErrorHandling:       "continue",
	}
}

// DocumentReader 定义文档读取接口
type DocumentReader interface {
	ReadDocument(path string) (string, error)
}

// FileDocumentReader 实现从文件系统读取文档
type FileDocumentReader struct{}

// ReadDocument 从文件读取文档内容
func (r *FileDocumentReader) ReadDocument(path string) (string, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("读取文件失败: %w", err)
	}
	return string(data), nil
}

// DocumentInfo 文档信息结构体，用于延迟加载文档内容
type DocumentInfo struct {
	ID       string
	Path     string
	Size     int64
	Content  string
	Loaded   bool
	Metadata map[string]interface{}
}

// LoadContent 加载文档内容
func (d *DocumentInfo) LoadContent(reader DocumentReader) error {
	if d.Loaded {
		return nil
	}

	content, err := reader.ReadDocument(d.Path)
	if err != nil {
		return err
	}

	d.Content = content
	d.Loaded = true
	return nil
}

// ScanDocumentsFromDirectory 扫描目录中的文档，但不立即加载内容
func ScanDocumentsFromDirectory(dirPath string, extensions []string) ([]*DocumentInfo, error) {
	var documents []*DocumentInfo

	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// 跳过目录
		if info.IsDir() {
			return nil
		}

		// 检查文件扩展名
		ext := strings.ToLower(filepath.Ext(path))
		validExt := false
		for _, allowedExt := range extensions {
			if ext == allowedExt {
				validExt = true
				break
			}
		}

		if !validExt {
			return nil
		}

		// 创建文档信息，但不加载内容
		docID := filepath.Base(path)
		documents = append(documents, &DocumentInfo{
			ID:      docID,
			Path:    path,
			Size:    info.Size(),
			Loaded:  false,
			Content: "",
		})

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("遍历目录失败: %w", err)
	}

	return documents, nil
}

// ReadDocumentsFromDirectory 从目录读取所有文档
func ReadDocumentsFromDirectory(dirPath string, extensions []string) (map[string]string, error) {
	reader := &FileDocumentReader{}
	documents := make(map[string]string)

	docInfos, err := ScanDocumentsFromDirectory(dirPath, extensions)
	if err != nil {
		return nil, err
	}

	for _, docInfo := range docInfos {
		err := docInfo.LoadContent(reader)
		if err != nil {
			return nil, fmt.Errorf("读取文档 %s 失败: %w", docInfo.Path, err)
		}
		documents[docInfo.ID] = docInfo.Content
	}

	return documents, nil
}

// BatchProcessDocuments 分批处理文档并添加到向量数据库
func BatchProcessDocuments(vectorDB *db.VectorDB, docInfos []*DocumentInfo, vectorizedType int, config BatchConfig) error {
	totalDocs := len(docInfos)
	if totalDocs == 0 {
		return nil
	}

	// 设置默认值
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 1000
	}

	// 初始化进度统计
	startTime := time.Now()
	processedCount := 0
	failedCount := 0

	// 创建文档读取器
	reader := &FileDocumentReader{}

	// 分批处理
	for i := 0; i < totalDocs; i += config.BatchSize {
		// 确定当前批次的结束索引
		end := i + config.BatchSize
		if end > totalDocs {
			end = totalDocs
		}

		// 当前批次的文档
		currentBatch := docInfos[i:end]
		batchSize := len(currentBatch)

		// 报告批次开始
		if config.ProgressCallback != nil {
			config.ProgressCallback(processedCount, totalDocs, "批次开始", time.Since(startTime))
		}

		// 加载当前批次的文档内容
		var wg sync.WaitGroup
		loadErrors := make(chan error, batchSize)
		loadSem := make(chan struct{}, config.NumWorkers) // 限制并发加载数量

		for j := range currentBatch {
			wg.Add(1)
			go func(doc *DocumentInfo) {
				defer wg.Done()
				loadSem <- struct{}{}        // 获取信号量
				defer func() { <-loadSem }() // 释放信号量

				if !doc.Loaded {
					err := doc.LoadContent(reader)
					if err != nil {
						loadErrors <- fmt.Errorf("加载文档 %s 失败: %w", doc.ID, err)
					}
				}
			}(currentBatch[j])
		}

		// 等待所有加载完成
		wg.Wait()
		close(loadErrors)

		// 检查加载错误
		loadErrorsCount := 0
		for err := range loadErrors {
			loadErrorsCount++
			log.Printf("错误: %v", err)
			if config.ErrorHandling == "abort" {
				return fmt.Errorf("批处理中止: %w", err)
			}
		}

		// 更新失败计数
		failedCount += loadErrorsCount

		// 准备批量添加的文档
		docs := make(map[string]string, batchSize-loadErrorsCount)
		for _, doc := range currentBatch {
			if doc.Loaded {
				docs[doc.ID] = doc.Content
			}
		}

		// 批量添加文档到向量数据库
		err := BatchAddDocuments(vectorDB, docs, vectorizedType)
		if err != nil {
			if config.ErrorHandling == "abort" {
				return fmt.Errorf("批量添加文档失败: %w", err)
			}
			log.Printf("警告: 批量添加文档部分失败: %v", err)
		}

		// 更新处理计数
		processedCount += len(docs)

		// 报告批次完成
		if config.ProgressCallback != nil {
			config.ProgressCallback(processedCount, totalDocs, "批次完成", time.Since(startTime))
		}

		// 如果配置了内存限制，检查内存使用并可能触发GC
		if config.MemoryLimit > 0 {
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			memoryUsageMB := int(m.Alloc / (1024 * 1024))

			if memoryUsageMB > config.MemoryLimit {
				log.Printf("内存使用达到 %d MB，触发垃圾回收", memoryUsageMB)
				runtime.GC()
			}
		}

		// 如果配置了每批保存，则保存数据库
		if config.SaveAfterEachBatch {
			err := vectorDB.SaveToFile(vectorDB.GetFilePath())
			if err != nil {
				log.Printf("警告: 保存数据库失败: %v", err)
			}
		}

		// 清理当前批次的内存
		for j := range currentBatch {
			if currentBatch[j].Loaded {
				currentBatch[j].Content = "" // 释放内容内存
				currentBatch[j].Loaded = false
			}
		}
	}

	// 构建或更新索引
	if processedCount > 0 {
		if config.ProgressCallback != nil {
			config.ProgressCallback(processedCount, totalDocs, "开始构建索引", time.Since(startTime))
		}

		var err error
		indexStartTime := time.Now()

		// 使用并行索引构建
		if vectorDB.IsHNSWEnabled() {
			err = vectorDB.BuildHNSWIndexParallel(config.NumWorkers)
		} else {
			err = vectorDB.RebuildIndex()
		}

		if err != nil {
			return fmt.Errorf("构建索引失败: %w", err)
		}

		if config.ProgressCallback != nil {
			config.ProgressCallback(processedCount, totalDocs, "索引构建完成", time.Since(startTime))
		}

		log.Printf("索引构建完成，耗时: %v", time.Since(indexStartTime))
	}

	// 最终统计
	elapsedTime := time.Since(startTime)
	log.Printf("处理完成: 成功添加 %d/%d 个文档，失败 %d 个，总耗时: %v",
		processedCount, totalDocs, failedCount, elapsedTime)

	return nil
}

// BatchAddDocuments 批量添加文档到向量数据库（并行版本）
func BatchAddDocuments(vectorDB *db.VectorDB, docs map[string]string, vectorizedType int) error {
	// 使用工作池并行处理
	numWorkers := runtime.NumCPU()
	workChan := make(chan struct{ id, content string }, len(docs))
	errChan := make(chan error, len(docs))
	doneChan := make(chan bool, 1)

	// 启动工作协程
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for doc := range workChan {
				err := vectorDB.AddDocument(doc.id, doc.content, vectorizedType)
				if err != nil {
					errChan <- fmt.Errorf("添加文档 %s 到向量数据库失败: %w", doc.id, err)
					return
				}
			}
		}()
	}

	// 发送工作
	go func() {
		for id, content := range docs {
			workChan <- struct{ id, content string }{id, content}
		}
		close(workChan)
	}()

	// 等待所有工作完成
	go func() {
		wg.Wait()
		close(errChan)
		doneChan <- true
	}()

	// 收集错误
	select {
	case err := <-errChan:
		if err != nil {
			return err
		}
	case <-doneChan:
		// 所有工作完成，无错误
	}

	return nil
}

// ProcessLargeDocumentCollection 处理大型文档集合的主函数
func ProcessLargeDocumentCollection(docsDir string, dbPath string, vectorizedType int, config BatchConfig) error {
	// 初始化向量数据库
	vectorDB := db.NewVectorDB(dbPath, 10)

	// 尝试从文件加载现有数据库
	err := vectorDB.LoadFromFile(dbPath)
	if err != nil {
		// 如果文件不存在，创建新的数据库
		if os.IsNotExist(err) {
			fmt.Printf("创建新的向量数据库: %s\n", dbPath)
		} else {
			return fmt.Errorf("加载向量数据库失败: %v", err)
		}
	}

	// 启用HNSW索引以提高搜索性能
	vectorDB.EnableHNSWIndex(16, 200, 50)

	// 扫描文档目录
	allowedExtensions := []string{".txt", ".md", ".html"}
	docInfos, err := ScanDocumentsFromDirectory(docsDir, allowedExtensions)
	if err != nil {
		return fmt.Errorf("扫描文档目录失败: %v", err)
	}

	fmt.Printf("扫描到 %d 个文档\n", len(docInfos))

	// 分批处理文档
	err = BatchProcessDocuments(vectorDB, docInfos, vectorizedType, config)
	if err != nil {
		return fmt.Errorf("处理文档失败: %v", err)
	}

	// 保存数据库
	fmt.Println("正在保存向量数据库...")
	err = vectorDB.SaveToFile(dbPath)
	if err != nil {
		return fmt.Errorf("保存向量数据库失败: %v", err)
	}
	fmt.Println("向量数据库保存成功")

	return nil
}
func insert() {
	// 配置参数
	docsDir := "./documents"
	dbPath := "./vector_db.dat"
	vectorizedType := TfidfVectorized

	// 使用新的处理大型文档集合的函数
	config := DefaultBatchConfig()
	config.BatchSize = 100 // 较小的批次，兼容旧版本行为
	config.ProgressCallback = func(current, total int, phase string, elapsed time.Duration) {
		fmt.Printf("[%s] 进度: %d/%d (%.2f%%), 耗时: %v\n",
			phase, current, total, float64(current)/float64(total)*100, elapsed)
	}

	err := ProcessLargeDocumentCollection(docsDir, dbPath, vectorizedType, config)
	if err != nil {
		log.Fatalf("处理文档集合失败: %v", err)
	}

	// 示例：搜索最相似的文档
	vectorDB := db.NewVectorDB(dbPath, 10)
	err = vectorDB.LoadFromFile(dbPath)
	if err != nil {
		log.Fatalf("加载向量数据库失败: %v", err)
	}

	queryText := "示例查询文本"
	fmt.Printf("搜索与 '%s' 最相似的文档:\n", queryText)

	// 将查询文本向量化
	queryVector, err := vectorDB.GetVectorForText(queryText, vectorizedType)
	if err != nil {
		log.Fatalf("查询文本向量化失败: %v", err)
	}

	// 查找最近的5个文档
	results, err := vectorDB.FindNearest(queryVector, 5, 10)
	if err != nil {
		log.Fatalf("搜索失败: %v", err)
	}

	// 打印结果
	for i, id := range results {
		fmt.Printf("%d. %s\n", i+1, id)
	}
}
func BatchInsert() {
	// 创建文档处理配置
	config := DefaultDocumentProcessingConfig()
	config.DocsDir = "./documents"
	config.DBPath = "./vector_db.dat"
	config.VectorizedType = TfidfVectorized
	config.ChunkSize = 512
	config.IncrementalUpdate = true

	// 设置批处理配置
	config.BatchConfig.BatchSize = 100
	config.BatchConfig.NumWorkers = 4
	config.BatchConfig.ProgressCallback = func(current, total int, phase string, elapsed time.Duration) {
		fmt.Printf("[%s] 进度: %d/%d (%.2f%%), 耗时: %v\n",
			phase, current, total, float64(current)/float64(total)*100, elapsed)
	}

	// 处理文档集合
	err := ProcessDocumentCollection(config)
	if err != nil {
		log.Fatalf("处理文档集合失败: %v", err)
	}

	// 示例：搜索最相似的文档
	vectorDB := db.NewVectorDB(config.DBPath, config.NumClusters)
	err = vectorDB.LoadFromFile(config.DBPath)
	if err != nil {
		log.Fatalf("加载向量数据库失败: %v", err)
	}

	queryText := "示例查询文本"
	fmt.Printf("搜索与 '%s' 最相似的文档:\n", queryText)

	// 使用过滤器搜索
	results, err := vectorDB.SearchWithFilter(queryText, 5, func(metadata map[string]interface{}) bool {
		// 示例：只返回特定路径的文档
		if path, ok := metadata["path"].(string); ok {
			return strings.Contains(path, "important")
		}
		return true
	})

	if err != nil {
		log.Fatalf("搜索失败: %v", err)
	}

	// 打印结果
	for i, result := range results {
		fmt.Printf("%d. %s (相似度: %.4f)\n", i+1, result.ID, result.Similarity)

		// 获取原始文档
		if sourceDoc, ok := result.Metadata["source_doc"].(string); ok {
			fmt.Printf("   来源文档: %s\n", sourceDoc)
		}
	}

}
