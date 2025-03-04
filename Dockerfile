FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV MODEL_ID="deepseek-ai/deepseek-coder-7b-instruct"
ENV PORT=8000
ENV DEVICE="cuda"
ENV PRECISION="bfloat16"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"] 