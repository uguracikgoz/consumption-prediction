#!/bin/bash
# Script to run the Docker container for pod prediction service using .env configuration

echo "=========================================="
echo "  Pod Prediction Service - Docker Run"
echo "=========================================="

cd "$(dirname "$0")"

if [ -f "../../.env" ]; then
    echo "Found .env file, will use for Docker environment"
    export $(grep -v '^#' ../../.env | xargs)
else
    echo "Warning: No .env file found. Using default configuration."
fi

IMAGE_NAME="${IMAGE_NAME:-pod-prediction-service}"
TAG="${TAG:-latest}"
PORT="${PORT:-8000}"
HOST_PORT="${HOST_PORT:-8000}"
GOOGLE_CREDENTIALS_FILE="${GOOGLE_CREDENTIALS_FILE:-./google-test-account-service.json}"
CONTAINER_NAME="${CONTAINER_NAME:-pod-prediction-service}"

if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container with name $CONTAINER_NAME already exists. Stopping and removing..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

if [ -f "$GOOGLE_CREDENTIALS_FILE" ]; then
    echo "Found Google credentials file at $GOOGLE_CREDENTIALS_FILE"
    CREDS_MOUNT="-v $(realpath $GOOGLE_CREDENTIALS_FILE):/app/credentials/google-creds.json"
    GOOGLE_ENV="-e GOOGLE_CREDENTIALS_FILE=/app/credentials/google-creds.json"
else
    echo "Warning: Google credentials file not found at $GOOGLE_CREDENTIALS_FILE"
    echo "Will use local data.csv file for training instead"
    CREDS_MOUNT=""
    GOOGLE_ENV=""
fi

echo "Running Docker container: $IMAGE_NAME:$TAG..."
echo "Mapping port $HOST_PORT -> $PORT"

if [ -f ".env" ]; then
    ENV_ARGS=$(grep -v '^#' .env | xargs -I{} echo "-e {}")
    docker run -d \
        --name $CONTAINER_NAME \
        -p $HOST_PORT:$PORT \
        $CREDS_MOUNT \
        $ENV_ARGS \
        $IMAGE_NAME:$TAG
else
    docker run -d \
        --name $CONTAINER_NAME \
        -p $HOST_PORT:$PORT \
        $CREDS_MOUNT \
        $GOOGLE_ENV \
        $IMAGE_NAME:$TAG
fi

echo "Container started! API is available at http://localhost:$HOST_PORT"
echo "Run 'docker logs $CONTAINER_NAME' to view container logs"
