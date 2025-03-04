.PHONY: build-cpu build-gpu run-cpu run-gpu test verify-imports deploy-railway help

help:
	@echo "Available commands:"
	@echo "  make build-cpu        Build the CPU Docker image"
	@echo "  make build-gpu        Build the GPU Docker image"
	@echo "  make run-cpu          Run the CPU Docker image"
	@echo "  make run-gpu          Run the GPU Docker image"
	@echo "  make test             Test the API with a simple request"
	@echo "  make verify-imports   Run the imports verification"
	@echo "  make deploy-railway   Deploy to Railway"

build-cpu:
	docker build -t deepseek-cpu -f Dockerfile.cpu .

build-gpu:
	docker build -t deepseek-gpu -f Dockerfile .

run-cpu: build-cpu
	docker run -p 8000:8000 deepseek-cpu

run-gpu: build-gpu
	docker run --gpus all -p 8000:8000 deepseek-gpu

test:
	@echo "Testing API..."
	curl -X POST http://localhost:8000/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Hello, tell me about yourself.", "max_tokens": 25, "temperature": 0.7}'

verify-imports:
	python imports.py

deploy-railway:
	@echo "Deploying to Railway..."
	railway up 