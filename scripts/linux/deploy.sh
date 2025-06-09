#!/bin/bash

set -e

# 配置变量
NAMESPACE="vectorsphere"
REGISTRY="ghcr.io/your-org/vectorsphere"
VERSION="${1:-latest}"

echo "Deploying VectorSphere version: $VERSION"

# 创建命名空间
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 应用ConfigMap和Secret
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# 部署基础设施
echo "Deploying infrastructure..."
kubectl apply -f k8s/etcd-statefulset.yaml
kubectl apply -f k8s/redis-deployment.yaml

# 等待基础设施就绪
echo "Waiting for infrastructure to be ready..."
kubectl wait --for=condition=ready pod -l app=etcd -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s

# 部署应用服务
echo "Deploying application services..."
kubectl apply -f k8s/master-deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml
kubectl apply -f k8s/vectordb-statefulset.yaml

# 等待应用服务就绪
echo "Waiting for application services to be ready..."
kubectl wait --for=condition=ready pod -l app=master -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=worker -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=vectordb -n $NAMESPACE --timeout=300s

# 部署监控
echo "Deploying monitoring..."
kubectl apply -f k8s/monitoring/

# 部署Ingress
echo "Deploying ingress..."
kubectl apply -f k8s/ingress.yaml

echo "Deployment completed successfully!"
echo "Services:"
kubectl get services -n $NAMESPACE
echo "Pods:"
kubectl get pods -n $NAMESPACE