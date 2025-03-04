#!/bin/bash

# Helper script for setting up DeepSeek-R1-Distill-Qwen-32B deployment

# Detect available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

echo "Detected $NUM_GPUS GPUs with approximately ${GPU_MEMORY}MB of memory each"

# Set tensor parallel size based on available GPUs
TENSOR_PARALLEL_SIZE=1
if [ $NUM_GPUS -ge 2 ]; then
    TENSOR_PARALLEL_SIZE=2
    echo "Setting tensor parallelism to use $TENSOR_PARALLEL_SIZE GPUs"
fi
if [ $NUM_GPUS -ge 4 ]; then
    TENSOR_PARALLEL_SIZE=4
    echo "Setting tensor parallelism to use $TENSOR_PARALLEL_SIZE GPUs"
fi

# Set memory limit based on available GPU memory
# For 32B models, we need significant memory
if [ $GPU_MEMORY -ge 80000 ]; then
    # A100 80GB or similar
    MAX_GPU_MEMORY=70
elif [ $GPU_MEMORY -ge 40000 ]; then
    # A100 40GB or similar
    MAX_GPU_MEMORY=35
elif [ $GPU_MEMORY -ge 24000 ]; then
    # A10 24GB or similar
    MAX_GPU_MEMORY=20
else
    # Smaller GPUs - will need aggressive memory optimization
    MAX_GPU_MEMORY=6
    echo "Warning: Your GPU has limited memory. Consider using a smaller model."
fi

echo "Setting MAX_GPU_MEMORY=$MAX_GPU_MEMORY GB per GPU"

# Create or update .env file with the correct configuration
cat > .env << EOL
MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
DEVICE=cuda
PRECISION=bfloat16
MAX_GPU_MEMORY=$MAX_GPU_MEMORY
LOAD_IN_8BIT=true
TEMPERATURE=0.6
TRUST_REMOTE_CODE=true
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
PORT=8000
EOL

echo "Configuration saved to .env file"
echo "You can now deploy your application with: docker-compose up -d" 