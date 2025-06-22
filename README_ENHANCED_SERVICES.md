# VectorSphere 增强型分布式服务集成指南

## 概述

本文档描述了如何使用 VectorSphere 中集成的增强型分布式服务组件。这些组件基于 etcd 实现，提供了完整的分布式系统解决方案。

## 增强型服务组件

### 1. 增强的服务注册与发现 (EnhancedServiceRegistry)
- **功能**: 自适应租约管理、服务元数据增强、本地缓存优化
- **特性**: 租约自动续期、服务状态监控、标签过滤、变更通知

### 2. 增强的配置管理 (EnhancedConfigManager)
- **功能**: 配置版本控制、热更新机制、多环境支持
- **特性**: 配置继承、变更审计、备份恢复、敏感信息加密

### 3. 增强的分布式锁 (EnhancedDistributedLock)
- **功能**: 可重入锁、读写锁分离、死锁检测
- **特性**: 锁统计监控、事件通知、自动释放、超时保护

### 4. 增强的领导者选举 (EnhancedLeaderElection)
- **功能**: 高可用设计、任期管理、多种选举策略
- **特性**: 故障转移、选举监控、候选者管理、事件回调

### 5. 增强的错误处理 (EnhancedErrorHandler)
- **功能**: 智能重试策略、熔断器机制、限流控制
- **特性**: 错误分类、模式匹配、统计分析、配置管理

### 6. 增强的健康检查 (EnhancedHealthChecker)
- **功能**: 多层次检查、自适应间隔、健康评分
- **特性**: 故障预测、告警机制、指标收集、自动恢复

### 7. 增强的负载均衡 (EnhancedLoadBalancer)
- **功能**: 多种算法支持、健康检查集成、会话亲和性
- **特性**: 慢启动机制、熔断器集成、动态权重、后端监控

### 8. 增强的安全管理 (EnhancedSecurityManager)
- **功能**: TLS/SSL 加密、RBAC 权限控制、数据加密
- **特性**: 用户管理、会话管理、审计日志、安全策略

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HTTP Client   │    │   HTTP Client   │    │   HTTP Client   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      API Gateway         │
                    │  (Load Balancer +        │
                    │   Security Manager)      │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │     Master Node          │
                    │  (Leader Election +      │
                    │   Service Registry)      │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │        etcd Cluster       │
                    │   (Service Discovery +    │
                    │    Config Management)     │
                    └───────────────────────────┘
        ┌─────────────────────────┼─────────────────────────┐
        │                        │                         │
┌───────┴───────┐    ┌───────────┴───────┐    ┌───────────┴───────┐
│  Worker Node  │    │   Worker Node     │    │   Worker Node     │
│ (Vector Index │    │  (Vector Index    │    │  (Vector Index    │
│  + Search)    │    │   + Search)       │    │   + Search)       │
└───────────────┘    └───────────────────┘    └───────────────────┘
```

## 快速开始

### 1. 环境准备

```bash
# 启动 etcd
etcd --listen-client-urls http://localhost:2379 --advertise-client-urls http://localhost:2379
```

### 2. 配置文件

确保 `conf/app.yaml` 配置文件存在并正确配置。主要配置项：

```yaml
# etcd 配置
etcdEndpoints:
  - "localhost:2379"

# 服务注册配置
serviceRegistryPath: "/services/vectorsphere/"
serviceTTL: 30

# 配置管理
configPathPrefix: "/c/vectorsphere/"

# 领导者选举
electionPathPrefix: "/election/vectorsphere/"
```

### 3. 启动服务

#### Windows
```cmd
# 启动主节点
scripts\start.bat master 8080 localhost:2379 VectorSphere

# 启动从节点
scripts\start.bat slave 8081 localhost:2379 VectorSphere
scripts\start.bat slave 8082 localhost:2379 VectorSphere

# 启动独立节点
scripts\start.bat standalone 8080 localhost:2379 VectorSphere
```

#### Linux/macOS
```bash
# 启动主节点
./scripts/start.sh master 8080 localhost:2379 VectorSphere

# 启动从节点
./scripts/start.sh slave 8081 localhost:2379 VectorSphere
./scripts/start.sh slave 8082 localhost:2379 VectorSphere

# 启动独立节点
./scripts/start.sh standalone 8080 localhost:2379 VectorSphere
```

#### 直接使用 Go
```bash
# 启动主节点
go run src/main.go --mode=master --port=8080 --etcd=localhost:2379 --service=VectorSphere --c=conf/app.yaml

# 启动从节点
go run src/main.go --mode=slave --port=8081 --etcd=localhost:2379 --service=VectorSphere --c=conf/app.yaml

# 启动独立节点
go run src/main.go --mode=standalone --port=8080 --etcd=localhost:2379 --service=VectorSphere --c=conf/app.yaml
```

## 服务启动流程

### 1. 初始化阶段
1. 加载应用配置 (`conf/app.yaml`)
2. 创建应用上下文 (`bootstrap.AppContext`)
3. 初始化所有增强服务组件
4. 设置优雅关闭机制

### 2. 启动阶段
1. 启动增强服务组件（按依赖顺序）
   - 服务注册与发现
   - 配置管理
   - 分布式锁
   - 领导者选举
   - 错误处理
   - 健康检查
   - 负载均衡
   - 安全管理
2. 启动业务服务（Master/Slave/Standalone）
3. 启动 gRPC 服务器

### 3. 运行阶段
- 所有增强组件在后台运行
- 提供分布式系统的各种能力
- 监控服务健康状态
- 处理故障和恢复

### 4. 关闭阶段
- 接收关闭信号
- 按相反顺序关闭增强组件
- 关闭业务服务
- 清理资源

## 监控和运维

### 健康检查
```bash
# 检查服务健康状态
curl http://localhost:9090/health
```

### 指标监控
```bash
# 获取 Prometheus 指标
curl http://localhost:9090/metrics
```

### 日志查看
```bash
# 查看应用日志
tail -f logs/vectorsphere.log
```

## 故障排除

### 常见问题

1. **etcd 连接失败**
   - 检查 etcd 是否正在运行
   - 验证 etcd 端点配置
   - 检查网络连接

2. **服务注册失败**
   - 检查服务注册路径配置
   - 验证 etcd 权限
   - 查看租约状态

3. **领导者选举异常**
   - 检查选举路径配置
   - 验证候选者 ID 唯一性
   - 查看选举日志

4. **配置加载失败**
   - 检查配置文件路径
   - 验证 YAML 格式
   - 查看配置权限

### 调试模式

在配置文件中启用调试模式：

```yaml
development:
  enabled: true
  debugMode: true
  profileEnabled: true
  profilePort: 6060
```

## 性能优化

### 1. 缓存优化
- 启用本地缓存
- 调整缓存大小和 TTL
- 使用 Redis 作为分布式缓存

### 2. 连接池优化
- 调整 etcd 客户端连接池大小
- 优化 HTTP 客户端连接池
- 配置数据库连接池

### 3. 异步处理
- 启用异步日志写入
- 使用异步健康检查
- 配置异步配置变更通知

### 4. 批量操作
- 启用批量写入
- 使用批量查询
- 配置批量处理大小

## 安全配置

### 1. TLS 加密
```yaml
security:
  tls:
    enabled: true
    certFile: "certs/server.crt"
    keyFile: "certs/server.key"
```

### 2. 认证授权
```yaml
security:
  auth:
    enabled: true
    jwtSecret: "your-jwt-secret-key"
    tokenExpiry: 24h
```

### 3. RBAC 权限控制
- 配置用户角色
- 设置资源权限
- 启用审计日志

## 扩展和定制

### 1. 自定义组件
- 实现自定义负载均衡算法
- 添加自定义健康检查
- 扩展配置管理功能

### 2. 插件机制
- 开发自定义插件
- 注册插件钩子
- 配置插件参数

### 3. 集成第三方服务
- 集成监控系统
- 连接消息队列
- 对接外部存储

## 最佳实践

1. **配置管理**
   - 使用环境变量覆盖配置
   - 分离敏感配置
   - 版本化配置文件

2. **服务部署**
   - 使用容器化部署
   - 配置健康检查
   - 实现滚动更新

3. **监控告警**
   - 设置关键指标监控
   - 配置告警规则
   - 建立故障响应流程

4. **安全防护**
   - 启用所有安全特性
   - 定期更新证书
   - 审计访问日志

## 参考资料

- [etcd 官方文档](https://etcd.io/docs/)
- [gRPC 官方文档](https://grpc.io/docs/)
- [Prometheus 监控](https://prometheus.io/docs/)
- [Go 最佳实践](https://golang.org/doc/effective_go.html)