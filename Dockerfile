FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
# Explicitly update bitsandbytes first, then install requirements
RUN pip install --no-cache-dir -U bitsandbytes>=0.42.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
ENV PORT=8000
ENV DEVICE="cuda"
ENV PRECISION="bfloat16"
# Limit GPU memory usage to prevent OOM errors - increased for 32B model
ENV MAX_GPU_MEMORY=24
# Enable 8-bit quantization for memory efficiency
ENV LOAD_IN_8BIT=true
# Set temperature for DeepSeek-R1 (recommended: 0.6)
ENV TEMPERATURE=0.6
# Ensure trust_remote_code is enabled
ENV TRUST_REMOTE_CODE="true"
# Enable tensor parallelism for large models
ENV TENSOR_PARALLEL_SIZE="2"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"] 