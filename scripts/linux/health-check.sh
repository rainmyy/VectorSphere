#!/bin/bash

NAMESPACE="vectorsphere"

echo "Checking VectorSphere health..."

# 检查所有Pod状态
echo "Pod Status:"
kubectl get pods -n $NAMESPACE

# 检查服务端点
echo "\nService Endpoints:"
kubectl get endpoints -n $NAMESPACE

# 检查HPA状态
echo "\nHPA Status:"
kubectl get hpa -n $NAMESPACE

# 检查PVC状态
echo "\nPVC Status:"
kubectl get pvc -n $NAMESPACE

# 执行健康检查
echo "\nHealth Check Results:"
for service in master worker vectordb; do
    echo "Checking $service..."
    kubectl exec -n $NAMESPACE deployment/$service -- wget -q --spider http://localhost:808${service: -1}/health && echo "$service: OK" || echo "$service: FAILED"
done

echo "\nHealth check completed."