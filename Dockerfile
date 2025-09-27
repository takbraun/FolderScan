# Fixed Dockerfile with proper permissions for PyTorch cache
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Create a user to run the application
RUN useradd -m -u 1000 appuser

# Update package lists first
RUN apt-get update

# Install essential packages in smaller groups
RUN apt-get install -y \
    python3-opencv \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install build tools separately
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install image processing libraries
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory with proper permissions
RUN mkdir -p /home/appuser/.cache/torch && \
    chown -R appuser:appuser /home/appuser/.cache

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and set ownership
COPY . .
RUN chown -R appuser:appuser /app

# Create directories with proper permissions
RUN mkdir -p /app/models /app/temp /app/output /app/logs && \
    chown -R appuser:appuser /app

# Set environment variables for PyTorch cache
ENV TORCH_HOME=/home/appuser/.cache/torch
ENV HOME=/home/appuser

# Switch to non-root user
USER appuser

EXPOSE 8000
CMD ["python3", "app.py"]
