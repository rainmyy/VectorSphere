import os
import json
import torch
import numpy as np
import datetime
import sqlite3
import uuid
import logging
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from pathlib import Path
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_service.log")
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 全局变量存储加载的模型
embedding_model = None
generation_model = None
generation_tokenizer = None

# 模型配置
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GENERATION_MODEL_NAME = os.environ.get("GENERATION_MODEL", "gpt2")
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# 自学习配置
MEMORY_DB_PATH = os.environ.get("MEMORY_DB_PATH", "memory.db")
FINE_TUNE_DIR = os.environ.get("FINE_TUNE_DIR", "fine_tuned_models")
ENABLE_LEARNING = os.environ.get("ENABLE_LEARNING", "true").lower() == "true"
MEMORY_RETENTION = int(os.environ.get("MEMORY_RETENTION", "1000"))  # 记忆保留的条目数
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.75"))  # 相似度阈值

# 数据库训练配置
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING", "")
DB_SCHEMA = os.environ.get("DB_SCHEMA", "public")
DB_TABLES_LIMIT = int(os.environ.get("DB_TABLES_LIMIT", "10"))
DB_ROWS_LIMIT = int(os.environ.get("DB_ROWS_LIMIT", "1000"))
DB_TRAINING_ENABLED = os.environ.get("DB_TRAINING_ENABLED", "false").lower() == "true"
DB_CACHE_DIR = os.environ.get("DB_CACHE_DIR", "db_cache")

# 确保目录存在
Path(FINE_TUNE_DIR).mkdir(parents=True, exist_ok=True)

# 数据库连接池
db_engines = {}


# 获取数据库连接
def get_db_engine(connection_string=None):
    if not connection_string and not DB_CONNECTION_STRING:
        raise ValueError("No database connection string provided")

    conn_str = connection_string or DB_CONNECTION_STRING

    if conn_str not in db_engines:
        try:
            engine = create_engine(conn_str)
            db_engines[conn_str] = engine
            logger.info(
                f"Created new database connection for {conn_str.split('@')[-1] if '@' in conn_str else 'database'}")
        except Exception as e:
            logger.error(f"Error creating database connection: {str(e)}")
            raise

    return db_engines[conn_str]


# 获取数据库表信息
def get_db_tables(connection_string=None, schema=None):
    engine = get_db_engine(connection_string)
    schema_name = schema or DB_SCHEMA

    try:
        inspector = sqlalchemy.inspect(engine)
        tables = inspector.get_table_names(schema=schema_name)

        table_info = []
        for table in tables[:DB_TABLES_LIMIT]:
            columns = inspector.get_columns(table, schema=schema_name)
            primary_keys = inspector.get_pk_constraint(table, schema=schema_name)
            foreign_keys = inspector.get_foreign_keys(table, schema=schema_name)

            # 获取表行数
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {schema_name}.{table}"))
                row_count = result.scalar()

            table_info.append({
                "name": table,
                "schema": schema_name,
                "columns": [{
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True)
                } for col in columns],
                "primary_keys": primary_keys.get("constrained_columns", []),
                "foreign_keys": [{
                    "columns": fk.get("constrained_columns", []),
                    "referred_table": fk.get("referred_table", ""),
                    "referred_columns": fk.get("referred_columns", [])
                } for fk in foreign_keys],
                "row_count": row_count
            })

        return table_info
    except Exception as e:
        logger.error(f"Error getting database tables: {str(e)}")
        raise


# 从数据库表获取数据
def get_table_data(table_name, schema=None, limit=None, connection_string=None):
    engine = get_db_engine(connection_string)
    schema_name = schema or DB_SCHEMA
    row_limit = limit or DB_ROWS_LIMIT

    try:
        query = text(f"SELECT * FROM {schema_name}.{table_name} LIMIT {row_limit}")
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Error getting data from table {table_name}: {str(e)}")
        raise


# 将数据库表结构转换为文本描述
def table_to_text(table_info):
    text = f"Table: {table_info['schema']}.{table_info['name']}\n"
    text += "Columns:\n"

    for col in table_info["columns"]:
        pk_marker = "(PK)" if col["name"] in table_info["primary_keys"] else ""
        nullable = "" if col.get("nullable") else "NOT NULL"
        text += f"  - {col['name']} {col['type']} {pk_marker} {nullable}\n"

    if table_info["foreign_keys"]:
        text += "Foreign Keys:\n"
        for fk in table_info["foreign_keys"]:
            text += f"  - {', '.join(fk['columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}\n"

    text += f"Row Count: {table_info['row_count']}\n"

    return text


# 将数据库表数据转换为训练文本
def data_to_training_text(table_name, df, table_info=None):
    # 表头描述
    if table_info:
        header = table_to_text(table_info)
    else:
        header = f"Table: {table_name}\nColumns: {', '.join(df.columns)}\n"

    # 数据示例
    examples = []
    for _, row in df.iterrows():
        example = "Row: " + ", ".join([f"{col}={row[col]}" for col in df.columns])
        examples.append(example)

    # 生成SQL查询示例
    sql_examples = [
        f"SELECT * FROM {table_name} LIMIT 5;",
        f"SELECT COUNT(*) FROM {table_name};"
    ]

    if table_info and table_info["primary_keys"]:
        pk = table_info["primary_keys"][0]
        sql_examples.append(f"SELECT * FROM {table_name} WHERE {pk} = <value>;")

    # 组合成训练文本
    training_text = header + "\n\nData Examples:\n" + "\n".join(examples[:5])
    training_text += "\n\nSQL Query Examples:\n" + "\n".join(sql_examples)

    return training_text


# 从数据库生成知识
def generate_db_knowledge(connection_string=None, schema=None):
    try:
        # 获取表信息
        tables = get_db_tables(connection_string, schema)

        knowledge_items = []

        # 为每个表生成知识
        for table_info in tables:
            # 表结构知识
            table_structure = table_to_text(table_info)
            knowledge_items.append({
                "content": f"Database Schema Information:\n{table_structure}",
                "source": "database_schema",
                "confidence": 1.0
            })

            # 表数据知识
            df = get_table_data(table_info["name"], table_info["schema"], connection_string=connection_string)
            data_text = data_to_training_text(table_info["name"], df, table_info)
            knowledge_items.append({
                "content": f"Database Content Information:\n{data_text}",
                "source": "database_content",
                "confidence": 0.9
            })

            # 表关系知识
            if table_info["foreign_keys"]:
                relations = []
                for fk in table_info["foreign_keys"]:
                    relations.append(
                        f"Table {table_info['name']} has a relationship with table {fk['referred_table']} through columns {', '.join(fk['columns'])} referencing {', '.join(fk['referred_columns'])}.")

                if relations:
                    knowledge_items.append({
                        "content": f"Database Relationship Information:\n{chr(10).join(relations)}",
                        "source": "database_relations",
                        "confidence": 0.95
                    })

        return knowledge_items
    except Exception as e:
        logger.error(f"Error generating database knowledge: {str(e)}")
        raise


# 从数据库生成训练数据
def generate_db_training_data(connection_string=None, schema=None):
    try:
        # 获取表信息
        tables = get_db_tables(connection_string, schema)

        training_texts = []

        # 为每个表生成训练数据
        for table_info in tables:
            # 获取表数据
            df = get_table_data(table_info["name"], table_info["schema"], connection_string=connection_string)

            # 生成表描述训练文本
            table_text = data_to_training_text(table_info["name"], df, table_info)
            training_texts.append(table_text)

            # 生成问答对
            qa_pairs = generate_qa_pairs(table_info, df)
            training_texts.extend(qa_pairs)

        return training_texts
    except Exception as e:
        logger.error(f"Error generating database training data: {str(e)}")
        raise


# 生成问答对
def generate_qa_pairs(table_info, df):
    qa_pairs = []
    table_name = table_info["name"]

    # 基本问题模板
    templates = [
        (f"What columns are in the {table_name} table?",
         f"The {table_name} table has the following columns: {', '.join([col['name'] for col in table_info['columns']])}"),

        (f"How many rows are in the {table_name} table?",
         f"The {table_name} table has {table_info['row_count']} rows."),

        (f"What is the primary key of the {table_name} table?",
         f"The primary key of the {table_name} table is {', '.join(table_info['primary_keys'])}" if table_info[
             'primary_keys'] else f"The {table_name} table does not have a primary key defined.")
    ]

    # 添加基本问答对
    for question, answer in templates:
        qa_pairs.append(f"<s>[INST] {question} [/INST] {answer}</s>")

    # 如果有数据，添加数据示例问答对
    if not df.empty:
        # 选择一个示例行
        sample_row = df.iloc[0]

        # 添加行数据问答对
        row_question = f"Show me an example row from the {table_name} table."
        row_answer = "Here is an example row: " + ", ".join([f"{col}={sample_row[col]}" for col in df.columns])
        qa_pairs.append(f"<s>[INST] {row_question} [/INST] {row_answer}</s>")

        # 如果有主键，添加查询示例
        if table_info["primary_keys"]:
            pk = table_info["primary_keys"][0]
            pk_value = sample_row[pk]
            query_question = f"How can I query the {table_name} table for a record with {pk}={pk_value}?"
            query_answer = f"You can use the following SQL query:\nSELECT * FROM {table_name} WHERE {pk} = {pk_value};"
            qa_pairs.append(f"<s>[INST] {query_question} [/INST] {query_answer}</s>")

    return qa_pairs


# 数据库训练API
@app.route("/api/db_train", methods=["POST"])
def train_from_database():
    if not DB_TRAINING_ENABLED:
        return {"error": "Database training is disabled"}, 400

    if not generation_model or not generation_tokenizer:
        return {"error": "Generation model not loaded"}, 500

    data = request.json
    if not data:
        return {"error": "Missing request data"}, 400

    connection_string = data.get("connection_string", DB_CONNECTION_STRING)
    schema = data.get("schema", DB_SCHEMA)
    add_to_knowledge = data.get("add_to_knowledge", True)
    fine_tune = data.get("fine_tune", True)

    if not connection_string:
        return {"error": "No database connection string provided"}, 400

    try:
        # 生成数据库知识
        if add_to_knowledge:
            knowledge_items = generate_db_knowledge(connection_string, schema)

            # 添加到知识库
            for item in knowledge_items:
                # 使用现有的添加知识API
                content = item["content"]
                source = item["source"]
                confidence = item["confidence"]

                # 生成嵌入
                embedding = embedding_model.encode(content)

                # 连接数据库
                conn = sqlite3.connect(MEMORY_DB_PATH)
                cursor = conn.cursor()

                # 存储知识
                knowledge_id = str(uuid.uuid4())
                timestamp = datetime.datetime.now().isoformat()
                cursor.execute(
                    "INSERT INTO knowledge (id, timestamp, content, embedding, source, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                    (knowledge_id, timestamp, content, embedding.tobytes(), source, confidence)
                )

                conn.commit()
                conn.close()

        # 微调模型
        if fine_tune:
            # 生成训练数据
            training_texts = generate_db_training_data(connection_string, schema)

            # 创建数据集
            train_dataset = Dataset.from_dict({"text": training_texts})

            # 对数据集进行标记化处理
            def tokenize_function(examples):
                return generation_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

            tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

            # 数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=generation_tokenizer,
                mlm=False
            )

            # 训练参数
            training_args = TrainingArguments(
                output_dir=os.path.join(FINE_TUNE_DIR, "db_checkpoints"),
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                save_steps=100,
                save_total_limit=2,
                logging_dir="./logs",
                logging_steps=10,
                learning_rate=5e-5,
            )

            # 创建训练器
            trainer = Trainer(
                model=generation_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_dataset,
            )

            # 开始训练
            trainer.train()

            # 保存微调后的模型
            fine_tuned_path = os.path.join(FINE_TUNE_DIR, f"{GENERATION_MODEL_NAME}_db_tuned")
            trainer.save_model(fine_tuned_path)
            generation_tokenizer.save_pretrained(fine_tuned_path)

            # 更新当前加载的模型
            global generation_model
            generation_model = trainer.model

        return {
            "status": "success",
            "knowledge_added": add_to_knowledge,
            "model_fine_tuned": fine_tune,
            "tables_processed": len(get_db_tables(connection_string, schema))
        }
    except Exception as e:
        logger.error(f"Error during database training: {str(e)}")
        return {"error": str(e)}, 500


# 数据库查询API
@app.route("/api/db_query", methods=["POST"])
def query_database():
    data = request.json
    if not data or "query" not in data:
        return {"error": "Missing 'query' field in request"}, 400

    query = data["query"]
    connection_string = data.get("connection_string", DB_CONNECTION_STRING)

    if not connection_string:
        return {"error": "No database connection string provided"}, 400

    try:
        engine = get_db_engine(connection_string)

        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()

            # 转换为字典列表
            results = [dict(zip(columns, row)) for row in rows]

            return {"results": results}
    except Exception as e:
        logger.error(f"Error executing database query: {str(e)}")
        return {"error": str(e)}, 500


# 数据库表信息API
@app.route("/api/db_info", methods=["POST"])
def get_database_info():
    data = request.json or {}
    connection_string = data.get("connection_string", DB_CONNECTION_STRING)
    schema = data.get("schema", DB_SCHEMA)

    if not connection_string:
        return {"error": "No database connection string provided"}, 400

    try:
        tables = get_db_tables(connection_string, schema)
        return {"tables": tables}
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        return {"error": str(e)}, 500


# 初始化记忆数据库
def init_memory_db():
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        prompt TEXT,
        response TEXT,
        prompt_embedding BLOB,
        feedback INTEGER DEFAULT 0,
        used_count INTEGER DEFAULT 0
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS knowledge (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        content TEXT,
        embedding BLOB,
        source TEXT,
        confidence REAL DEFAULT 1.0
    )
    ''')
    conn.commit()
    conn.close()
    logger.info(f"Memory database initialized at {MEMORY_DB_PATH}")

# 初始化模型
def init_models():
    global embedding_model, generation_model, generation_tokenizer

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on {DEVICE}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

    if GENERATION_MODEL_NAME:
        logger.info(f"Loading generation model: {GENERATION_MODEL_NAME} on {DEVICE}")
        generation_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)

        # 检查是否有微调模型
        fine_tuned_path = os.path.join(FINE_TUNE_DIR, GENERATION_MODEL_NAME)
        if os.path.exists(fine_tuned_path):
            logger.info(f"Loading fine-tuned model from {fine_tuned_path}")
            generation_model = AutoModelForCausalLM.from_pretrained(fine_tuned_path).to(DEVICE)
        else:
            logger.info(f"Loading base model: {GENERATION_MODEL_NAME}")
            generation_model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_NAME).to(DEVICE)

    # 初始化记忆数据库
    init_memory_db()

# 存储交互到记忆
def store_interaction(prompt, response):
    if not ENABLE_LEARNING:
        return

    try:
        # 生成嵌入
        prompt_embedding = embedding_model.encode(prompt)

        # 连接数据库
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()

        # 存储交互
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO interactions (id, timestamp, prompt, response, prompt_embedding) VALUES (?, ?, ?, ?, ?)",
            (interaction_id, timestamp, prompt, response, prompt_embedding.tobytes())
        )

        # 清理旧记忆，保留最新的MEMORY_RETENTION条
        cursor.execute(
            "DELETE FROM interactions WHERE id NOT IN (SELECT id FROM interactions ORDER BY timestamp DESC LIMIT ?)",
            (MEMORY_RETENTION,)
        )

        conn.commit()
        conn.close()
        logger.info(f"Stored interaction with ID: {interaction_id}")
    except Exception as e:
        logger.error(f"Error storing interaction: {str(e)}")

# 查找相似的记忆
def find_similar_memories(prompt, limit=5):
    if not ENABLE_LEARNING:
        return []

    try:
        # 生成嵌入
        prompt_embedding = embedding_model.encode(prompt)

        # 连接数据库
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()

        # 获取所有记忆的嵌入
        cursor.execute("SELECT id, prompt, response, prompt_embedding FROM interactions")
        memories = cursor.fetchall()

        if not memories:
            conn.close()
            return []

        # 计算相似度
        similar_memories = []
        for memory_id, memory_prompt, memory_response, memory_embedding_bytes in memories:
            memory_embedding = np.frombuffer(memory_embedding_bytes, dtype=np.float32)
            similarity = cosine_similarity([prompt_embedding], [memory_embedding])[0][0]

            if similarity >= SIMILARITY_THRESHOLD:
                similar_memories.append({
                    "id": memory_id,
                    "prompt": memory_prompt,
                    "response": memory_response,
                    "similarity": float(similarity)
                })

                # 更新使用计数
                cursor.execute("UPDATE interactions SET used_count = used_count + 1 WHERE id = ?", (memory_id,))

        conn.commit()
        conn.close()

        # 按相似度排序并限制数量
        similar_memories.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_memories[:limit]
    except Exception as e:
        logger.error(f"Error finding similar memories: {str(e)}")
        return []

# 添加知识到知识库
@app.route("/api/add_knowledge", methods=["POST"])
def add_knowledge():
    if not ENABLE_LEARNING:
        return {"error": "Learning is disabled"}, 400

    data = request.json
    if not data or "content" not in data:
        return {"error": "Missing 'content' field in request"}, 400

    content = data["content"]
    source = data.get("source", "user")
    confidence = data.get("confidence", 1.0)

    try:
        # 生成嵌入
        embedding = embedding_model.encode(content)

        # 连接数据库
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()

        # 存储知识
        knowledge_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO knowledge (id, timestamp, content, embedding, source, confidence) VALUES (?, ?, ?, ?, ?, ?)",
            (knowledge_id, timestamp, content, embedding.tobytes(), source, confidence)
        )

        conn.commit()
        conn.close()

        return {"id": knowledge_id, "status": "success"}
    except Exception as e:
        return {"error": str(e)}, 500

# 查询知识库
def query_knowledge(query, limit=3):
    if not ENABLE_LEARNING:
        return []

    try:
        # 生成嵌入
        query_embedding = embedding_model.encode(query)

        # 连接数据库
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()

        # 获取所有知识的嵌入
        cursor.execute("SELECT id, content, embedding, confidence FROM knowledge")
        knowledge_items = cursor.fetchall()

        if not knowledge_items:
            conn.close()
            return []

        # 计算相似度
        relevant_knowledge = []
        for knowledge_id, content, embedding_bytes, confidence in knowledge_items:
            knowledge_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            similarity = cosine_similarity([query_embedding], [knowledge_embedding])[0][0]

            if similarity >= SIMILARITY_THRESHOLD:
                relevant_knowledge.append({
                    "id": knowledge_id,
                    "content": content,
                    "similarity": float(similarity),
                    "confidence": confidence
                })

        conn.close()

        # 按相似度排序并限制数量
        relevant_knowledge.sort(key=lambda x: x["similarity"] * x["confidence"], reverse=True)
        return relevant_knowledge[:limit]
    except Exception as e:
        logger.error(f"Error querying knowledge: {str(e)}")
        return []

# 提供反馈
@app.route("/api/feedback", methods=["POST"])
def provide_feedback():
    if not ENABLE_LEARNING:
        return {"error": "Learning is disabled"}, 400

    data = request.json
    if not data or "interaction_id" not in data or "feedback" not in data:
        return {"error": "Missing required fields in request"}, 400

    interaction_id = data["interaction_id"]
    feedback = int(data["feedback"])  # 1 for positive, -1 for negative

    try:
        # 连接数据库
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()

        # 更新反馈
        cursor.execute("UPDATE interactions SET feedback = ? WHERE id = ?", (feedback, interaction_id))

        if cursor.rowcount == 0:
            conn.close()
            return {"error": "Interaction not found"}, 404

        conn.commit()
        conn.close()

        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}, 500

# 微调模型
@app.route("/api/fine_tune", methods=["POST"])
def fine_tune_model():
    if not ENABLE_LEARNING:
        return {"error": "Learning is disabled"}, 400

    if not generation_model or not generation_tokenizer:
        return {"error": "Generation model not loaded"}, 500

    try:
        # 连接数据库
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()

        # 获取正面反馈的交互
        cursor.execute("SELECT prompt, response FROM interactions WHERE feedback > 0")
        positive_interactions = cursor.fetchall()

        if len(positive_interactions) < 10:
            return {"error": "Not enough positive interactions for fine-tuning (minimum 10 required)"}, 400

        # 准备训练数据
        train_texts = []
        for prompt, response in positive_interactions:
            # 根据模型类型格式化训练文本
            train_text = f"<s>[INST] {prompt} [/INST] {response}</s>"
            train_texts.append(train_text)

        # 创建数据集
        train_dataset = Dataset.from_dict({"text": train_texts})

        # 对数据集进行标记化处理
        def tokenize_function(examples):
            return generation_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=generation_tokenizer,
            mlm=False
        )

        # 训练参数
        training_args = TrainingArguments(
            output_dir=os.path.join(FINE_TUNE_DIR, "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=100,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=10,
            learning_rate=5e-5,
        )

        # 创建训练器
        trainer = Trainer(
            model=generation_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
        )

        # 开始训练
        trainer.train()

        # 保存微调后的模型
        fine_tuned_path = os.path.join(FINE_TUNE_DIR, GENERATION_MODEL_NAME)
        trainer.save_model(fine_tuned_path)
        generation_tokenizer.save_pretrained(fine_tuned_path)

        conn.close()

        return {"status": "success", "model_path": fine_tuned_path}
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        return {"error": str(e)}, 500

# 获取模型信息
@app.route("/api/model_info", methods=["GET"])
def get_model_info():
    embedding_dim = embedding_model.get_sentence_embedding_dimension() if embedding_model else 0
    generation_info = {}

    if generation_model:
        # 检查是否是微调模型
        fine_tuned_path = os.path.join(FINE_TUNE_DIR, GENERATION_MODEL_NAME)
        is_fine_tuned = os.path.exists(fine_tuned_path)

        generation_info = {
            "name": GENERATION_MODEL_NAME,
            "max_tokens": generation_tokenizer.model_max_length,
            "support_streaming": True,
            "is_fine_tuned": is_fine_tuned
        }

    # 获取记忆统计信息
    memory_stats = {}
    try:
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM interactions")
        memory_stats["interactions_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM knowledge")
        memory_stats["knowledge_count"] = cursor.fetchone()[0]

        conn.close()
    except Exception as e:
        memory_stats["error"] = str(e)

    return {
        "embedding": {
            "name": EMBEDDING_MODEL_NAME,
            "embedding_dim": embedding_dim,
        },
        "generation": generation_info,
        "device": DEVICE,
        "learning_enabled": ENABLE_LEARNING,
        "memory": memory_stats
    }

# 文本嵌入接口
@app.route("/api/embedding", methods=["POST"])
def get_embedding():
    if not embedding_model:
        return {"error": "Embedding model not loaded"}, 500

    data = request.json
    if not data or "text" not in data:
        return {"error": "Missing 'text' field in request"}, 400

    text = data["text"]
    try:
        # 获取文本嵌入
        embedding = embedding_model.encode(text)
        # 转换为Python列表并返回
        return {"embedding": embedding.tolist()}
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return {"error": str(e)}, 500

# 增强提示词
def enhance_prompt(prompt):
    # 查找相关知识
    relevant_knowledge = query_knowledge(prompt)

    # 查找相似记忆
    similar_memories = find_similar_memories(prompt)

    # 构建增强提示词
    enhanced_prompt = prompt

    # 添加相关知识
    if relevant_knowledge:
        knowledge_text = "\n\nRelevant knowledge:\n"
        for i, k in enumerate(relevant_knowledge):
            knowledge_text += f"{i+1}. {k['content']}\n"
        enhanced_prompt = knowledge_text + "\n" + enhanced_prompt

    # 添加相似记忆作为上下文
    if similar_memories:
        memory_text = "\n\nSimilar past interactions:\n"
        for i, m in enumerate(similar_memories):
            memory_text += f"{i+1}. Q: {m['prompt']}\n   A: {m['response']}\n"
        enhanced_prompt = memory_text + "\n" + enhanced_prompt

    logger.info(f"Enhanced prompt with {len(relevant_knowledge)} knowledge items and {len(similar_memories)} memories")
    return enhanced_prompt

# 文本生成接口
@app.route("/api/chat", methods=["POST"])
def generate_text():
    if not generation_model:
        return {"error": "Generation model not loaded"}, 500

    data = request.json
    if not data or "prompt" not in data:
        return {"error": "Missing 'prompt' field in request"}, 400

    prompt = data["prompt"]
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    use_memory = data.get("use_memory", True)

    # 生成交互ID
    interaction_id = str(uuid.uuid4())

    try:
        # 增强提示词
        if use_memory and ENABLE_LEARNING:
            enhanced_prompt = enhance_prompt(prompt)
        else:
            enhanced_prompt = prompt

        # 编码输入文本
        inputs = generation_tokenizer(enhanced_prompt, return_tensors="pt").to(DEVICE)

        # 生成文本
        outputs = generation_model.generate(
            inputs["input_ids"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

        # 解码生成的文本
        generated_text = generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除原始提示，只返回新生成的内容
        response_text = generated_text[len(enhanced_prompt):]

        # 存储交互到记忆
        if ENABLE_LEARNING:
            store_interaction(prompt, response_text)

        return {
            "text": response_text,
            "interaction_id": interaction_id,
            "enhanced": use_memory and ENABLE_LEARNING
        }
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return {"error": str(e)}, 500

# 流式文本生成接口
@app.route("/api/chat_stream", methods=["POST"])
def generate_text_stream():
    if not generation_model:
        return {"error": "Generation model not loaded"}, 500

    data = request.json
    if not data or "prompt" not in data:
        return {"error": "Missing 'prompt' field in request"}, 400

    prompt = data["prompt"]
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    use_memory = data.get("use_memory", True)

    # 生成交互ID
    interaction_id = str(uuid.uuid4())

    # 增强提示词
    if use_memory and ENABLE_LEARNING:
        enhanced_prompt = enhance_prompt(prompt)
    else:
        enhanced_prompt = prompt

    def generate():
        try:
            # 编码输入文本
            inputs = generation_tokenizer(enhanced_prompt, return_tensors="pt").to(DEVICE)
            generated = ""
            past = None

            for _ in range(max_tokens):
                with torch.no_grad():
                    if past:
                        outputs = generation_model(
                            inputs["input_ids"][:, -1:],
                            past_key_values=past,
                            temperature=temperature,
                            top_p=top_p
                        )
                    else:
                        outputs = generation_model(
                            inputs["input_ids"],
                            temperature=temperature,
                            top_p=top_p
                        )

                past = outputs.past_key_values
                token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

                if token.item() == generation_tokenizer.eos_token_id:
                    break

                new_text = generation_tokenizer.decode(token.item())
                generated += new_text

                # 流式返回生成的文本片段
                yield f"data: {json.dumps({'text': new_text, 'interaction_id': interaction_id})}\n\n"

            # 存储完整交互到记忆
            if ENABLE_LEARNING:
                store_interaction(prompt, generated)

        except Exception as e:
            logger.error(f"Error in stream generation: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

# 健康检查接口
@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # 初始化模型
    init_models()

    # 启动服务
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)