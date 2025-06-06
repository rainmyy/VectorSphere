# VectorSphere
## 项目简介
VectorSphere是一个高性能分布式搜索引擎系统，支持传统关键词搜索和向量相似度搜索，采用主从架构设计，具备高可用性和可扩展性。系统提供了丰富的索引和搜索功能，适用于大规模文本和向量数据的存储、索引和检索。

向量星环（VectorSphere）——高性能分布式向量化索引与智能检索平台
如需更技术化的副标题，可用：
> “面向企业的高性能分布式向量化索引与智能检索系统”

## 系统架构
### 整体架构
VectorSphere采用主从（Master-Slave）分布式架构：

- 主节点（Master） ：负责任务调度、负载均衡和集群管理
- 从节点（Slave） ：负责数据存储、索引构建和查询执行
- 独立模式（Standalone） ：支持单机部署，适合开发和小规模应用场景
  系统使用etcd进行服务发现和分布式协调，通过gRPC进行节点间通信。

### 核心组件
1. 索引系统

    - 倒排索引：支持关键词搜索和布尔查询
    - 向量索引：支持高维向量相似度搜索
    - MVCC（多版本并发控制）：保证数据一致性
2. 数据存储

    - 支持多种数据库后端（BTree、Badger、BBolt等）
    - 向量数据库：高效存储和检索高维向量
3. 搜索引擎

    - 查询解析器：解析复杂查询表达式
    - 搜索执行器：执行混合搜索策略
    - 多表搜索：支持跨表查询
4. 任务调度

    - 任务池管理：高效调度和执行分布式任务
    - 定时任务：支持Cron表达式的定时任务
5. 负载均衡

    - 支持多种负载均衡策略（随机、轮询、加权轮询、一致性哈希）
#### 集成FAISS-GPU 步骤:
- 安装 FAISS-GPU 库和 CUDA 工具包
- 配置 CGO 绑定和 C/C++ 头文件
- 链接 CUDA 和 FAISS 库
- 处理 C/C++ 与 Go 的数据类型转换
## 主要功能
### 1. 混合搜索
- 关键词搜索 ：基于倒排索引的传统文本搜索
- 向量搜索 ：基于向量相似度的语义搜索
- 混合搜索 ：结合关键词和向量搜索的混合策略
### 2. 向量处理
- 多种向量化方法 ：

    - 简单词袋模型（Bag of Words）
    - TF-IDF向量化
    - 词嵌入向量化（Word Embedding）
- 向量索引优化 ：

    - K-means聚类
    - 近似最近邻（ANN）搜索
    - 向量压缩和量化
### 3. 分布式任务
- 任务分发和调度
- 任务状态监控和结果汇总
- 失败任务重试和超时处理
### 4. 高可用性
- 主节点选举和故障转移
- 健康检查和节点状态监控
- 负载均衡和动态扩缩容
## 技术栈
- 语言 ：Go
- 通信 ：gRPC
- 服务发现 ：etcd
- 数据结构 ：B+树、跳表、KD树、Trie树
- 算法 ：K-means、向量量化、优先队列

## 使用方法
### 安装 CUDA 工具包:
# 下载并安装 CUDA Toolkit (推荐 11.8 或 12.x 版本)
# 从 NVIDIA 官网下载：https://developer.nvidia.com/cuda-downloads
# 安装后验证
nvcc --version
### 安装 FAISS-GPU：
# 方法1：使用 conda (推荐)
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 cudatoolkit=11.8

# 方法2：从源码编译
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --prefix C:/faiss
### 安装和编译
```
# 克隆仓库
git clone https://github.com/yourusername/VectorSphere.git
cd VectorSphere

# 安装依赖
go mod download

# 编译
go build -o VectorSphere
```
makefile 编译
```
# 查看系统信息
make info
# 测试编译环境
make test-env
# 编译项目
make build
# 清理构建文件
make clean
# 查看帮助
make help
```

### 运行模式 主节点模式
```
./VectorSphere -mode master -port 8080 -etcd localhost:2379 -service VectorSphere
``` 
从节点模式
```
./VectorSphere -mode slave -port 8081 -etcd localhost:2379 -service VectorSphere -data ./data -doc-num 10000 -db-type 0
``` 
独立模式
```
./VectorSphere -mode standalone -port 8080 -data ./data -doc-num 10000 -db-type 0
```
### 配置说明
- mode ：运行模式（master/slave/standalone）
- port ：服务端口
- etcd ：etcd服务地址，多个地址用逗号分隔
- service ：服务名称
- data ：数据目录
- doc-num ：文档数量估计
- db-type ：数据库类型
## 开发指南
### 项目结构
- bootstrap/ ：系统启动和初始化
- conf/ ：配置文件
- db/ ：数据库实现
- index/ ：索引实现
- library/ ：通用库和工具
- messages/ ：消息定义和协议
- scheduler/ ：任务调度
- search/ ：搜索实现
- server/ ：服务实现
### 扩展指南 添加新的向量化方法
在 db/vectorized.go 中实现新的 DocumentVectorized 函数。
 添加新的索引类型
实现 index/index.go 中定义的 IndexInterface 接口。
 添加新的调度任务
实现 scheduler/task_pool.go 中定义的 ScheduledTask 接口。

## 性能优化
- 使用向量量化和压缩减少内存占用
- 实现缓存机制提高查询性能
- 使用多级索引加速向量搜索
- 采用MVCC机制保证并发安全
## 贡献指南
欢迎贡献代码、报告问题或提出新功能建议。请遵循以下步骤：

1. Fork 仓库
2. 创建特性分支 ( git checkout -b feature/amazing-feature )
3. 提交更改 ( git commit -m 'Add some amazing feature' )
4. 推送到分支 ( git push origin feature/amazing-feature )
5. 创建 Pull Request
## 许可证
本项目采用 LICENSE 许可证。

## 联系方式
如有问题或建议，请通过 Issues 或 Pull Requests 与我们联系。

开始使用 VectorSphere，构建高性能的分布式搜索系统！