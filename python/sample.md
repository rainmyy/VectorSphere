# 安装依赖
pip install -r requirements.txt

# 启动服务（使用CPU）
python start_service.py --embedding-model all-MiniLM-L6-v2 --generation-model gpt2 --port 5000

# 或者使用GPU（如果可用）
python start_service.py --embedding-model all-MiniLM-L6-v2 --generation-model gpt2 --use-gpu --port 5000

## 四、Docker部署
### 1. 构建Python服务镜像
```
构建镜像
docker build -t transformer-service:latest .

运行容器
docker run -d --name transformer-service -p 5000:5000 transformer-service:latest
```

### 2. 使用GPU版本（需要NVIDIA Docker支持
```
构建支持GPU的镜像
docker build -t transformer-service:gpu -f Dockerfile.gpu .

# 运行GPU容器
docker run -d --gpus all --name transformer-service-gpu -p 5000:5000 \
-e USE_GPU=true \
transformer-service:gpu
```
## 使用方法
### 1.启动服务
```
python start_service.py --embedding-model all-MiniLM-L6-v2 --generation-model gpt2 --max-conversation-history 30 --max-context-length 2048
```
### 2.创建新会话并发送消息
````
curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" -d '{"prompt": "你好，请介绍一下自己", "use_memory": true}'
````
### 3.在现有会话中继续对话
```
curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" -d '{"prompt": "继续我们的对话", "conversation_id": "上一步返回的conversation_id", "use_memory": true}'
```
### 4.获取会话历史
```
curl -X GET http://localhost:5000/api/conversation/会话ID
```
### 5.删除会话
```
curl -X DELETE http://localhost:5000/api/conversation/会话ID
```

## 训练关系型数据库
### 启动服务
```
python start_service.py --generation-model gpt2 --embedding-model all-MiniLM-L6-v2 --enable-db-training --db-connection "postgresql://username:password@localhost:5432/dbname"
```
### API 使用示例
```
curl -X POST http://localhost:5000/api/db_info -H "Content-Type: application/json" -d '{"connection_string": "postgresql://username:password@localhost:5432/dbname", "schema": "public"}'
```
### 从数据库训练模型
```
curl -X POST http://localhost:5000/api/db_train -H "Content-Type: application/json" -d '{"connection_string": "postgresql://username:password@localhost:5432/dbname", "schema": "public", "add_to_knowledge": true, "fine_tune": true}'
```
### 执行数据库查询
```
curl -X POST http://localhost:5000/api/db_query -H "Content-Type: application/json" -d '{"connection_string": "postgresql://username:password@localhost:5432/dbname", "query": "SELECT * FROM users LIMIT 5"}'
```