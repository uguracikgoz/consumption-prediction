#!/bin/bash
# Script to create a Kubernetes secret from the Google credentials file

# Ensure we have the credentials file
if [ ! -f "../google-creds.json" ]; then
  echo "Error: google-creds.json file not found in the parent directory"
  exit 1
fi

# Create the secret
kubectl create secret generic pod-prediction-google-creds \
  --from-file=google-creds.json=../google-creds.json \
  --dry-run=client -o yaml > google_creds_secret.yaml

echo "Secret YAML file created as google_creds_secret.yaml"
echo "Review the file and then apply it with: kubectl apply -f google_creds_secret.yaml"
