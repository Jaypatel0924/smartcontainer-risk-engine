# SmartContainer Risk Engine - Docker Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building, then remove after pip install
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# Copy application code
COPY . .

# Create directories
RUN mkdir -p output models

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:5000/api/stats || exit 1

# Run with gunicorn for production (Render sets PORT env var)
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 300 app:app
