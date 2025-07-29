FROM python:3.11-slim

RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install python-dotenv --no-cache-dir

# Copy the rest of the application
COPY . .

# Create model directory if it doesn't exist
RUN mkdir -p /app/src/model/saved_models

# Create directory for Google credentials
RUN mkdir -p /app/credentials

# Set correct permissions for the app directories
RUN chown -R appuser:appuser /app

# Set environment variables (will be overridden by .env if present)
ENV PORT=8000
ENV PYTHONPATH=/app
ENV GOOGLE_CREDENTIALS_FILE=/app/credentials/google-creds.json

# Expose the port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Initialize the model and start the API
CMD ["sh", "-c", "python init_model.py && python api/main.py"]
