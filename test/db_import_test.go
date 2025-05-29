package test

import (
	"fmt"
	"seetaSearch/db"
	"seetaSearch/library/log"
	"seetaSearch/library/tree"
	"seetaSearch/messages"
	"seetaSearch/search"
	"time"
)

func test_2() {
	// 创建导入配置
	config := &search.DBImportConfig{
		DriverName:     "mysql",
		DataSource:     "user:password@tcp(localhost:3306)/dbname",
		BatchSize:      100,
		WorkerCount:    4,
		VectorizedType: 1,
		FieldMappings:  map[string]string{"title": "title_field", "content": "content_field"},
		KeywordFields:  []string{"title", "tags"},
		ContentField:   "content",
		IdField:        "id",

		// 增量导入配置
		IncrementalImport: true,
		UpdateTimeField:   "updated_at",
		PrimaryKeyField:   "id",

		// 数据转换钩子
		PreDocumentHook: func(doc *messages.Document) error {
			// 自定义文档处理逻辑
			return nil
		},
	}
	// 初始化事务管理器、锁管理器和WAL管理器
	txMgr := tree.NewTransactionManager()
	lockMgr := tree.NewLockManager()
	walMgr, _ := tree.NewWALManager("./wal")

	// 创建多表搜索服务
	searchService := search.NewMultiTableSearchService(txMgr, lockMgr, walMgr)
	// 创建导入服务
	service := search.NewDBImportService(searchService, config)

	// 创建导入调度器
	scheduler := search.NewImportScheduler(service)

	// 添加导入计划
	scheduler.AddSchedule("products", "SELECT * FROM products", 1*time.Hour, true)
	scheduler.AddSchedule("users", "SELECT * FROM users", 24*time.Hour, false)

	// 启动调度器
	scheduler.Start()

	// 手动执行增量导入
	err := service.ImportTableIncremental("products", "SELECT * FROM products")
	if err != nil {
		log.Error("增量导入失败: %v", err)
	}

	// 获取进度报告
	progressReport := service.GetProgressReport()
	fmt.Println(progressReport)

	// 停止调度器
	scheduler.Stop()
}

func test_1() {
	// 初始化事务管理器、锁管理器和WAL管理器
	txMgr := tree.NewTransactionManager()
	lockMgr := tree.NewLockManager()
	walMgr, _ := tree.NewWALManager("./wal")

	// 创建多表搜索服务
	searchService := search.NewMultiTableSearchService(txMgr, lockMgr, walMgr)

	// 创建或获取表
	tableName := "products"
	_, err := searchService.GetTable(tableName)
	if err != nil {
		// 表不存在，创建新表
		log.Info("创建新表: %s", tableName)
		if err := searchService.CreateTable(tableName, "./vector_db/"+tableName, 10, 10); err != nil {
			log.Fatal("创建表失败: %v", err)
		}
	}

	// 配置数据库导入
	importConfig := search.DBImportConfig{
		DriverName:     "mysql",
		DataSource:     "user:password@tcp(localhost:3306)/mydatabase",
		BatchSize:      100,
		WorkerCount:    4,
		VectorizedType: db.DefaultVectorized,
		FieldMappings: map[string]string{
			"product_name":        "name",
			"product_description": "description",
			"product_category":    "category",
		},
		KeywordFields: []string{"product_name", "product_category", "product_tags"},
		ContentField:  "product_description", // 用于向量化的字段
		IdField:       "product_id",          // 作为文档ID的字段
		FeatureField:  "product_features",    // 特征位字段
	}

	// 创建导入服务
	importService := search.NewDBImportService(searchService, &importConfig)

	// 定义SQL查询
	sqlQuery := `SELECT product_id, product_name, product_description, product_category, product_tags, product_features
		FROM products WHERE is_active = 1`

	// 执行导入
	log.Info("开始导入数据到表: %s", tableName)
	if err := importService.ImportTable(tableName, sqlQuery); err != nil {
		log.Fatal("导入数据失败: %v", err)
	}

	// 导入多个表
	tableMapping := map[string]string{
		"products": sqlQuery,
		"articles": "SELECT article_id, title, content, category FROM articles WHERE status = 'published'",
	}

	if err := importService.ImportFromMultipleTables(tableMapping); err != nil {
		log.Fatal("导入多表数据失败: %v", err)
	}

	log.Info("数据导入完成")
}
