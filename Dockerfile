FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential cmake libopenblas-dev

# Copy your project
COPY . /app
WORKDIR /app

# Upgrade pip and install Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
