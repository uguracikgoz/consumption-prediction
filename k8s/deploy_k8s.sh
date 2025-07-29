#!/bin/bash
# Script to deploy the pod prediction service to Kubernetes

NAMESPACE="default"  # Change this to your namespace if needed

echo "=================================================="
echo "  Pod Prediction Service - Kubernetes Deployment"
echo "=================================================="

# Ensure script runs from the correct directory
cd "$(dirname "$0")"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH."
    exit 1
fi

# Create namespace if it doesn't exist
echo "Ensuring namespace exists: $NAMESPACE"
kubectl get namespace $NAMESPACE &> /dev/null || kubectl create namespace $NAMESPACE

# Deploy Redis
echo "Deploying Redis..."
kubectl apply -f k8s/redis.yaml -n $NAMESPACE

# Deploy the API service
echo "Deploying Pod Prediction API..."
kubectl apply -f k8s/deployment.yaml -n $NAMESPACE
kubectl apply -f k8s/service.yaml -n $NAMESPACE

# Wait for deployments to be ready
echo "Waiting for Redis deployment to be ready..."
kubectl rollout status deployment/pod-prediction-redis -n $NAMESPACE

echo "Waiting for Pod Prediction API deployment to be ready..."
kubectl rollout status deployment/pod-prediction-api -n $NAMESPACE

# Get service details
echo "Getting service details..."
kubectl get service pod-prediction-service -n $NAMESPACE

echo "Deployment completed successfully!"
echo "You can access the API at http://<cluster-ip>/predict"
echo "For external access, you may need to set up an Ingress or use port-forwarding:"
echo "kubectl port-forward service/pod-prediction-service 8000:80 -n $NAMESPACE"
