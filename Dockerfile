# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies (if needed, e.g., for lxml, Pillow, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Copy only requirements first (Docker layer caching trick)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Default command (adjust to your app entrypoint)
CMD ["python", "main.py"]
