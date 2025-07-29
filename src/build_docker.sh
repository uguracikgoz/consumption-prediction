#!/bin/bash
# Script to build and push the Docker image for the pod prediction service

# Set build arguments
IMAGE_NAME="pod-prediction-service"
TAG="latest"
REGISTRY=""  # Set this to your container registry if needed

echo "=========================================="
echo "  Pod Prediction Service - Docker Build"
echo "=========================================="

# Enable automatic error handling
set -e

# Ensure script runs from the correct directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Debug information
echo "Current directory: $SCRIPT_DIR"
echo "Dockerfile location: $SCRIPT_DIR/Dockerfile"

# Check for .env file
if [ -f "./.env" ]; then
    echo "Found .env file, will use environment variables from it"
    # Export .env variables to be available in the script
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: No .env file found, using default configuration"
fi

# Check for Google credentials file
CREDS_FILE=${GOOGLE_CREDENTIALS_FILE:-"./google-test-account-service.json"}
if [ -f "$CREDS_FILE" ]; then
    echo "Found Google credentials file at $CREDS_FILE"
    mkdir -p ./credentials
    cp "$CREDS_FILE" ./credentials/google-creds.json
else
    echo "Error: Google credentials file not found at $CREDS_FILE"
    echo "Google Sheets integration requires valid credentials."
    exit 1
fi

# Make sure we're using the correct Dockerfile
echo "Using Dockerfile at: $SCRIPT_DIR/Dockerfile"

# Build the Docker image with explicit context and Dockerfile path
echo "Building Docker image: $IMAGE_NAME:$TAG..."
docker build -t $IMAGE_NAME:$TAG -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"

# Clean up
if [ -f "../credentials/google-creds.json" ]; then
    echo "Cleaning up temporary Google credentials copy"
    rm -f ../credentials/google-creds.json 2>/dev/null || true
    rmdir ../credentials 2>/dev/null || true
fi

echo "Cleaning up..."

# If registry is specified, tag and push the image
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$TAG"
    echo "Tagging image as: $FULL_IMAGE_NAME..."
    docker tag $IMAGE_NAME:$TAG $FULL_IMAGE_NAME
    
    echo "Pushing image to registry..."
    docker push $FULL_IMAGE_NAME
    
    echo "Image pushed successfully to: $FULL_IMAGE_NAME"
else
    echo "No registry specified. Image built locally: $IMAGE_NAME:$TAG"
fi

# Ask user if they want to run the container now
echo
echo "Do you want to run the container now and see logs? (y/n)"
read -r run_container

if [[ "$run_container" =~ ^[Yy] ]]; then
    # Stop and remove existing container if it exists
    echo "Stopping and removing any existing container..."
    docker stop $IMAGE_NAME 2>/dev/null || true
    docker rm $IMAGE_NAME 2>/dev/null || true
    
    # Run the container
    echo "Starting container $IMAGE_NAME..."
    
    # Run the docker container using run_docker.sh
    ./run_docker.sh
    
    # Show logs in follow mode
    echo "Showing container logs (Ctrl+C to exit logs, container will keep running):"
    echo "---------------------------------------------------------------------"
    docker logs -f $IMAGE_NAME
else
    echo "Container not started. You can run it later with ./run_docker.sh"
fi

echo "Done!"
