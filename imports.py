"""
This file is used to pre-import all necessary packages and verify they are installed correctly.
"""

# Basic imports
import os
import sys
import logging
import time

# Web framework
import fastapi
import uvicorn
import pydantic

# ML framework
import torch

# Verify CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Transformers
import transformers
print(f"Transformers version: {transformers.__version__}")

# Other ML libraries
try:
    import accelerate
    print(f"Accelerate version: {accelerate.__version__}")
except ImportError:
    print("Accelerate not available")

try:
    import bitsandbytes
    print(f"BitsAndBytes version: {bitsandbytes.__version__}")
except ImportError:
    print("BitsAndBytes not available")

try:
    import xformers
    print(f"xformers version: {xformers.__version__}")
except ImportError:
    print("xformers not available")

# Test tokenization
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("Tokenizer test: Successful")
except Exception as e:
    print(f"Tokenizer test failed: {str(e)}")

# Verify system info
import platform
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")

print("All imports completed successfully") 