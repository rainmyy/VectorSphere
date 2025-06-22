
## 概述

VectorSphere 分布式系统是一个基于 etcd 的分布式向量数据库系统，支持自动的 Master-Slave 架构，提供高可用性和水平扩展能力。

## 架构设计

### 核心组件

1. **DistributedManager** - 分布式管理器
   - 管理 etcd 连接和 leader 选举
   - 协调 Master/Slave 服务的启动和切换
   - 处理节点故障转移

2. **ServiceDiscovery** - 服务发现
   - 基于 etcd 的服务注册和发现
   - 实时监控节点状态变化
   - 提供负载均衡支持

3. **CommunicationService** - 通信服务
   - 管理 gRPC 连接池
   - 处理 Master 到 Slave 的请求转发
   - 支持广播和点对点通信

4. **APIGateway** - API 网关
   - 统一的 HTTP API 接口
   - 请求路由和负载均衡
   - 认证和限流中间件

5. **ConfigManager** - 配置管理
   - 统一的配置加载和验证
   - 支持环境变量覆盖
   - 配置热更新

6. **AppLauncher** - 应用启动器
   - 整合所有组件的启动逻辑
   - 优雅的启动和停止流程
   - 健康检查和状态监控

### 系统架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HTTP Client   │    │   HTTP Client   │    │   HTTP Client   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      API Gateway         │
                    │    (Master Node Only)    │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  Communication Service   │
                    │   (gRPC Load Balancer)   │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼────────┐    ┌───────────▼────────┐    ┌───────────▼────────┐
│  Slave Node 1  │    │   Slave Node 2     │    │   Slave Node N     │
│                │    │                    │    │                    │
│ ┌────────────┐ │    │ ┌────────────────┐ │    │ ┌────────────────┐ │
│ │Index Service│ │    │ │ Index Service  │ │    │ │ Index Service  │ │
│ └────────────┘ │    │ └────────────────┘ │    │ └────────────────┘ │
└────────────────┘    └────────────────────┘    └────────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │         etcd             │
                    │   (Service Discovery)    │
                    └─────────────────────────────┘
```

## 功能特性

### 分布式特性
- **自动 Leader 选举**: 基于 etcd 的分布式锁实现
- **服务发现**: 自动注册和发现集群中的节点
- **故障转移**: Master 节点故障时自动选举新的 Master
- **负载均衡**: 请求在多个 Slave 节点间均匀分布
- **水平扩展**: 支持动态添加和移除节点

### API 功能
- **表管理**: 创建、删除、列出表
- **文档操作**: 添加、删除、更新文档
- **向量搜索**: 高性能的相似性搜索
- **统计信息**: 集群和表的统计数据
- **健康检查**: 节点和服务的健康状态

### 高可用性
- **多副本**: 数据在多个节点间复制
- **故障检测**: 实时监控节点健康状态
- **自动恢复**: 节点故障后自动重新加入集群
- **数据一致性**: 确保分布式环境下的数据一致性

## 快速开始

### 1. 环境准备

确保已安装以下依赖：
- Go 1.19+
- etcd 3.5+
- FAISS (可选，用于 GPU 加速)

### 2. 配置文件

创建配置文件 `c/c.yaml`：

```yaml
service_name: "vectorsphere"
node_type: "auto"  # auto, master, slave
http_port: 8080
default_port: 9090
data_dir: "./data"
task_timeout: 30
heartbeat: 10
health_check_interval: 30
scheduler_worker_count: 4

etcd:
  endpoints:
    - "localhost:2379"
  username: ""
  password: ""
  dial_timeout: 5
  request_timeout: 10

master:
  election_timeout: 15
  session_ttl: 30
  max_retry_count: 3
  retry_interval: 5

slave:
  register_retry_count: 3
  register_retry_interval: 5
  heartbeat_interval: 10
```

### 3. 启动 etcd

```bash
# 单节点模式
etcd --data-dir=./etcd-data

# 集群模式（推荐生产环境）
# 节点1
etcd --name node1 --data-dir ./etcd-data/node1 \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://192.168.1.10:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://192.168.1.10:2380 \
  --initial-cluster node1=http://192.168.1.10:2380,node2=http://192.168.1.11:2380,node3=http://192.168.1.12:2380 \
  --initial-cluster-state new
```

### 4. 编译和运行

```bash
# 编译
cd src/distributed
go build -o vectorsphere main.go

# 运行（自动模式）
./vectorsphere

# 指定配置文件
./vectorsphere -c /path/to/c.yaml

# 强制指定节点类型
./vectorsphere -type master
./vectorsphere -type slave

# 启用调试日志
./vectorsphere -log debug
```

### 5. 多节点部署

```bash
# 节点1（会自动成为 Master）
./vectorsphere -c config1.yaml

# 节点2（Slave）
./vectorsphere -c config2.yaml

# 节点3（Slave）
./vectorsphere -c config3.yaml
```

## API 使用示例

### 创建表

```bash
curl -X POST http://localhost:8080/api/v1/tables \
  -H "Content-Type: application/json" \
  -d '{
    "table_name": "my_vectors",
    "dimension": 128,
    "index_type": "flat"
  }'
```

### 添加文档

```bash
curl -X POST http://localhost:8080/api/v1/tables/my_vectors/documents \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc1",
    "vector": [0.1, 0.2, 0.3, ...],
    "metadata": {"title": "Example Document"}
  }'
```

### 搜索向量

```bash
curl -X POST http://localhost:8080/api/v1/tables/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "top_k": 10,
    "threshold": 0.8
  }'
```

### 获取集群状态

```bash
curl http://localhost:8080/api/v1/cluster/status
```

## 监控和运维

### 健康检查

```bash
# 检查节点健康状态
curl http://localhost:8080/health

# 检查集群状态
curl http://localhost:8080/api/v1/cluster/status

# 获取节点信息
curl http://localhost:8080/api/v1/cluster/nodes
```

### 日志监控

系统提供详细的日志输出，包括：
- 节点启动和停止事件
- Leader 选举过程
- 服务注册和发现
- 请求处理和错误信息
- 性能指标

### 故障排查

1. **etcd 连接问题**
   ```bash
   # 检查 etcd 状态
   etcdctl endpoint health
   etcdctl member list
   ```

2. **节点无法加入集群**
   - 检查网络连接
   - 验证配置文件
   - 查看日志输出

3. **Master 选举失败**
   - 确保 etcd 集群正常
   - 检查网络分区
   - 验证节点时间同步

## 性能优化

### 配置调优

1. **etcd 配置**
   ```yaml
   etcd:
     dial_timeout: 5
     request_timeout: 10
     # 增加连接池大小
     max_call_send_msg_size: 10485760  # 10MB
     max_call_recv_msg_size: 10485760  # 10MB
   ```

2. **任务调度**
   ```yaml
   scheduler_worker_count: 8  # 根据 CPU 核心数调整
   task_timeout: 60          # 增加超时时间
   ```

3. **网络优化**
   ```yaml
   # 启用 gRPC 连接复用
   grpc_max_connections: 10
   grpc_keepalive_time: 30
   grpc_keepalive_timeout: 5
   ```

### 硬件建议

- **CPU**: 4+ 核心，推荐 8+ 核心
- **内存**: 8GB+，推荐 16GB+
- **存储**: SSD，推荐 NVMe
- **网络**: 千兆以太网，推荐万兆

## 安全配置

### etcd 安全

```yaml
etcd:
  # 启用 TLS
  ca_file: "/path/to/ca.crt"
  cert_file: "/path/to/client.crt"
  key_file: "/path/to/client.key"
  
  # 启用认证
  username: "vectorsphere"
  password: "your_password"
```

### API 安全

```yaml
# 启用认证中间件
auth:
  enabled: true
  jwt_secret: "your_jwt_secret"
  token_expiry: 3600

# 启用限流
rate_limit:
  enabled: true
  requests_per_second: 100
  burst_size: 200
```

## 开发指南

### 项目结构

```
src/distributed/
├── main.go                    # 主程序入口
├── app_launcher.go           # 应用启动器
├── distributed_manager.go    # 分布式管理器
├── service_discovery.go      # 服务发现
├── communication_service.go  # 通信服务
├── api_gateway.go           # API 网关
├── config_manager.go        # 配置管理
└── README.md               # 文档
```

### 扩展开发

1. **添加新的 API 接口**
   - 在 `api_gateway.go` 中添加路由
   - 实现对应的处理函数
   - 更新 protobuf 定义

2. **自定义中间件**
   - 实现 `http.Handler` 接口
   - 在 API 网关中注册

3. **扩展服务发现**
   - 实现自定义的服务注册逻辑
   - 添加新的元数据字段

## 常见问题

### Q: 如何处理网络分区？
A: 系统使用 etcd 的 Raft 算法处理网络分区，确保只有拥有多数节点的分区能够继续提供服务。

### Q: 如何扩展集群？
A: 直接启动新的节点即可，系统会自动发现并加入集群。

### Q: 如何备份数据？
A: 定期备份 etcd 数据和各节点的数据目录。

### Q: 如何升级系统？
A: 采用滚动升级方式，逐个替换节点，确保服务不中断。

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！