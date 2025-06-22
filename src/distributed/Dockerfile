# VectorSphere 分布式系统 Dockerfile
# 多阶段构建，优化镜像大小

# 构建阶段
FROM golang:1.21-alpine AS builder

# 设置工作目录
WORKDIR /app

# 安装构建依赖
RUN apk add --no-cache git make gcc musl-dev

# 复制 go mod 文件
COPY go.mod go.sum ./

# 下载依赖
RUN go mod download

# 复制源代码
COPY . .

# 构建应用
ARG VERSION=1.0.0
ARG BUILD_TIME
ARG GIT_COMMIT
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags "-X main.Version=${VERSION} -X main.BuildTime=${BUILD_TIME} -X main.GitCommit=${GIT_COMMIT} -w -s" \
    -o vectorsphere main.go

# 运行阶段
FROM alpine:3.18

# 安装运行时依赖
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    curl \
    jq

# 创建非 root 用户
RUN addgroup -g 1001 -S vectorsphere && \
    adduser -u 1001 -S vectorsphere -G vectorsphere

# 设置工作目录
WORKDIR /app

# 创建必要的目录
RUN mkdir -p /app/c /app/data /app/logs && \
    chown -R vectorsphere:vectorsphere /app

# 从构建阶段复制二进制文件
COPY --from=builder /app/vectorsphere /app/vectorsphere
COPY --from=builder /app/c/c.yaml /app/c/c.yaml
COPY --from=builder /app/README.md /app/README.md

# 设置权限
RUN chmod +x /app/vectorsphere && \
    chown vectorsphere:vectorsphere /app/vectorsphere

# 切换到非 root 用户
USER vectorsphere

# 暴露端口
EXPOSE 8080 9090 9091

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 设置环境变量
ENV VECTORSPHERE_CONFIG_PATH=/app/c/c.yaml
ENV VECTORSPHERE_DATA_DIR=/app/data
ENV VECTORSPHERE_LOG_LEVEL=info

# 启动命令
CMD ["/app/vectorsphere", "-c", "/app/c/c.yaml"]