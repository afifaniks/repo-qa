#!/bin/bash

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
# Wait for Ollama to be ready by checking its health endpoint
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "Ollama not ready yet, waiting..."
    sleep 1
done
echo "Ollama is ready!"

echo "Pulling qwen3:1.7b model..."
ollama pull qwen3:1.7b

echo "Starting RepoQA API server..."
python -m uvicorn repoqa.api:app --host 0.0.0.0 --port 8000
