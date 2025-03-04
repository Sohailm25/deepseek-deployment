# Deepseek Model API

A RESTful API service for the Deepseek LLM, ready to deploy on Railway.

## Overview

This repository provides a production-ready API for the Deepseek Large Language Model. It's designed to be easily deployed on Railway and provides a simple interface for making inference requests to the model.

By default, it uses the `deepseek-ai/deepseek-coder-7b-instruct` model, which is optimized for code-related tasks, but you can easily configure it to use other Deepseek models.

## Features

- 🚀 RESTful API with FastAPI
- 🔄 Simple text generation endpoint
- ⚙️ Configurable model parameters
- 🐳 Docker containerization
- 📊 Token usage statistics
- 🔍 Health check endpoint
- 📝 Comprehensive logging

## Deployment on Railway

### Quick Deploy

1. Fork this repository
2. Create a new project on Railway
3. Connect to your forked repository
4. Deploy
5. (Optional) Set environment variables to customize configuration

### Environment Variables

You can customize the deployment by setting the following environment variables:

- `MODEL_ID`: The Hugging Face model ID (default: "deepseek-ai/deepseek-coder-7b-instruct")
- `DEVICE`: Device to run the model on (default: "cuda" if available, else "cpu")
- `PRECISION`: Model precision (default: "bfloat16" if on GPU, else "float32")
- `MAX_GPU_MEMORY`: Optional limit for GPU memory usage in GB
- `PORT`: Port to run the server on (default: 8000)

## Hardware Requirements

The default model (`deepseek-coder-7b-instruct`) requires:
- At least 14GB of GPU memory for inference
- Railway's GPU plan should be sufficient

## API Usage

### Text Generation

```bash
curl -X POST "https://your-railway-url/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to calculate Fibonacci numbers",
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50
  }'
```

### Health Check

```bash
curl "https://your-railway-url/health"
```

## API Reference

### POST /generate

Generate text based on a prompt.

**Request Body:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| prompt | string | Input text | (required) |
| max_tokens | integer | Maximum tokens to generate | 256 |
| temperature | float | Sampling temperature (0-1) | 0.7 |
| top_p | float | Nucleus sampling parameter | 0.95 |
| top_k | integer | Top-k sampling parameter | 50 |
| stop_sequences | array | List of sequences to stop generation | null |

**Response:**

```json
{
  "text": "Generated text...",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 100,
    "total_tokens": 110
  }
}
```

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure as needed
4. Run the server:
   ```bash
   python main.py
   ```

## Using Different Deepseek Models

You can use any of the Deepseek models available on Hugging Face by changing the `MODEL_ID` environment variable. Some options include:

- `deepseek-ai/deepseek-coder-7b-instruct`
- `deepseek-ai/deepseek-coder-33b-instruct`
- `deepseek-ai/deepseek-llm-7b-chat`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Deepseek AI](https://github.com/deepseek-ai) for the amazing models
- [Hugging Face](https://huggingface.co/) for hosting the models
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Railway](https://railway.app/) for the deployment platform 