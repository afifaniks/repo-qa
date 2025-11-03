# Use Python base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Install Ollama ---
RUN curl -fsSL https://ollama.com/install.sh | bash

# Expose Ollama's API port
EXPOSE 11434

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# --- Start Ollama + your app together ---
# Start Ollama as a background process before running the app
CMD ollama serve & sleep 5 && python -m repoqa.app
