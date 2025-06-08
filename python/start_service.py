import os
import argparse
from model_service import app, init_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the transformer model service")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--generation-model", default="gpt2", help="Text generation model name")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--port", type=int, default=5000, help="Service port")
    parser.add_argument("--memory-db", default="memory.db", help="Path to memory database")
    parser.add_argument("--fine-tune-dir", default="fine_tuned_models", help="Directory for fine-tuned models")
    parser.add_argument("--disable-learning", action="store_true", help="Disable learning capabilities")
    parser.add_argument("--memory-retention", type=int, default=1000, help="Number of memories to retain")
    parser.add_argument("--similarity-threshold", type=float, default=0.75, help="Similarity threshold for memory retrieval")
    # 新增数据库相关参数
    parser.add_argument("--db-connection", default="", help="Database connection string")
    parser.add_argument("--db-schema", default="public", help="Database schema")
    parser.add_argument("--db-tables-limit", type=int, default=10, help="Maximum number of tables to process")
    parser.add_argument("--db-rows-limit", type=int, default=1000, help="Maximum number of rows to fetch per table")
    parser.add_argument("--enable-db-training", action="store_true", help="Enable database training capabilities")
    parser.add_argument("--db-cache-dir", default="db_cache", help="Directory for database cache")

    args = parser.parse_args()

    # 设置环境变量
    os.environ["EMBEDDING_MODEL"] = args.embedding_model
    os.environ["GENERATION_MODEL"] = args.generation_model
    os.environ["USE_GPU"] = "true" if args.use_gpu else "false"
    os.environ["PORT"] = str(args.port)
    os.environ["MEMORY_DB_PATH"] = args.memory_db
    os.environ["FINE_TUNE_DIR"] = args.fine_tune_dir
    os.environ["ENABLE_LEARNING"] = "false" if args.disable_learning else "true"
    os.environ["MEMORY_RETENTION"] = str(args.memory_retention)
    os.environ["SIMILARITY_THRESHOLD"] = str(args.similarity_threshold)
    # 设置数据库相关环境变量
    os.environ["DB_CONNECTION_STRING"] = args.db_connection
    os.environ["DB_SCHEMA"] = args.db_schema
    os.environ["DB_TABLES_LIMIT"] = str(args.db_tables_limit)
    os.environ["DB_ROWS_LIMIT"] = str(args.db_rows_limit)
    os.environ["DB_TRAINING_ENABLED"] = "true" if args.enable_db_training else "false"
    os.environ["DB_CACHE_DIR"] = args.db_cache_dir

    # 初始化模型
    init_models()

    # 启动服务
    app.run(host="0.0.0.0", port=args.port)