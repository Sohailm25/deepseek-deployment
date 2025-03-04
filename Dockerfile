FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install libstdc++ to fix GLIBCXX dependency
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies - use a smaller model on Railway to avoid memory issues
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables - using a smaller model for better compatibility
ENV MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
ENV PORT=8000
ENV DEVICE="cuda"
ENV PRECISION="bfloat16"
# Limit GPU memory usage 
ENV MAX_GPU_MEMORY=6
# Enable 8-bit quantization for memory efficiency
ENV LOAD_IN_8BIT=true
# Set temperature for DeepSeek-R1 (recommended: 0.6)
ENV TEMPERATURE=0.6
# Ensure trust_remote_code is enabled
ENV TRUST_REMOTE_CODE="true"
# Setup tensor parallelism but let the app decide if it's available
ENV TENSOR_PARALLEL_SIZE="1"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"] 