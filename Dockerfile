FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    wget \
    git \
    gcc \
    curl \
    gfortran \
    python3-pkgconfig \
    libopenblas-dev \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* --verbose


RUN curl -fsSL https://ollama.com/install.sh | bash


WORKDIR /app


COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install the package in editable mode
RUN pip install -e .

# Copy and set up the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create directory for Ollama data
RUN mkdir -p /root/.ollama

# Define volume for Ollama models
VOLUME ["/root/.ollama"]

CMD ["/app/start.sh"]
