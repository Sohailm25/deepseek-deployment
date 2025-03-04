import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Check if CUDA is actually available regardless of what the environment says
CUDA_AVAILABLE = torch.cuda.is_available()
if not CUDA_AVAILABLE:
    logger.warning("CUDA was requested but is not available! Falling back to CPU.")

# Default model settings - use a smaller model if CUDA isn't available
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" if not CUDA_AVAILABLE else "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
DEFAULT_PRECISION = "bfloat16" if DEFAULT_DEVICE == "cuda" else "float32"
DEFAULT_TEMPERATURE = 0.6  # Recommended temperature for DeepSeek-R1
DEFAULT_TRUST_REMOTE_CODE = "true"  # Required for Qwen models
DEFAULT_TENSOR_PARALLEL_SIZE = "1"  # Default to no tensor parallelism
DEFAULT_LOAD_IN_8BIT = "true" if CUDA_AVAILABLE else "false"  # Only use 8-bit with CUDA

# Environment variable overrides - but validate them against reality
MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
DEVICE = "cuda" if os.environ.get("DEVICE", DEFAULT_DEVICE) == "cuda" and CUDA_AVAILABLE else "cpu"
PRECISION = os.environ.get("PRECISION", DEFAULT_PRECISION)
MAX_GPU_MEMORY = os.environ.get("MAX_GPU_MEMORY", None) if CUDA_AVAILABLE else None
LOAD_IN_8BIT = os.environ.get("LOAD_IN_8BIT", DEFAULT_LOAD_IN_8BIT).lower() == "true" and CUDA_AVAILABLE
TEMPERATURE = float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE))
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", DEFAULT_TRUST_REMOTE_CODE).lower() == "true"
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", DEFAULT_TENSOR_PARALLEL_SIZE)) if CUDA_AVAILABLE else 1

# If we're not on CUDA but the model requested is too large, switch to a smaller one
if DEVICE == "cpu" and "32B" in MODEL_ID:
    logger.warning(f"Model {MODEL_ID} is too large for CPU. Switching to DeepSeek-R1-Distill-Qwen-1.5B")
    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        # Log configuration
        logger.info(f"Model configuration: model_id={MODEL_ID}, device={DEVICE}, cuda_available={CUDA_AVAILABLE}, "
                   f"precision={PRECISION}, 8bit={LOAD_IN_8BIT}, temperature={TEMPERATURE}, "
                   f"trust_remote_code={TRUST_REMOTE_CODE}, tensor_parallel_size={TENSOR_PARALLEL_SIZE}")
        
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def load_model(self):
        """Load the DeepSeek-R1 model and tokenizer"""
        logger.info(f"Loading model {MODEL_ID} on {DEVICE}...")
        
        try:
            # Load tokenizer first
            logger.info(f"Loading tokenizer with trust_remote_code={TRUST_REMOTE_CODE}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                padding_side="left",
                trust_remote_code=TRUST_REMOTE_CODE
            )
            
            # Configure device mapping and quantization
            model_kwargs = {}
            
            if DEVICE == "cuda":
                device_map = "auto"
                torch_dtype = torch.bfloat16 if PRECISION == "bfloat16" else torch.float16
                
                # Configure GPU memory limit if specified
                if MAX_GPU_MEMORY:
                    memory_gb = int(MAX_GPU_MEMORY)
                    max_memory = {i: f"{memory_gb}GiB" for i in range(torch.cuda.device_count())}
                    model_kwargs["max_memory"] = max_memory
                    logger.info(f"Setting GPU memory limit to {memory_gb}GB per GPU")
                
                # Configure quantization with BitsAndBytesConfig
                if LOAD_IN_8BIT:
                    try:
                        logger.info("Loading model with 8-bit quantization via BitsAndBytesConfig")
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False
                        )
                        model_kwargs["quantization_config"] = quantization_config
                    except Exception as e:
                        logger.warning(f"8-bit quantization failed, continuing without it: {str(e)}")
                
                # Configure tensor parallelism
                if TENSOR_PARALLEL_SIZE > 1:
                    logger.info(f"Enabling tensor parallelism with {TENSOR_PARALLEL_SIZE} GPUs")
                    if "model_kwargs" not in model_kwargs:
                        model_kwargs["model_kwargs"] = {}
                    
                    # Add tensor parallel configuration to model kwargs
                    model_kwargs["model_kwargs"]["tensor_parallel_size"] = TENSOR_PARALLEL_SIZE
            else:
                logger.info("Running on CPU - disabling all GPU-specific optimizations")
                device_map = "cpu"
                torch_dtype = torch.float32
                
                # If model is large and we're on CPU, load with low_cpu_mem_usage=True
                if "7B" in MODEL_ID or "14B" in MODEL_ID or "32B" in MODEL_ID:
                    logger.info("Large model on CPU - enabling low_cpu_mem_usage")
                    model_kwargs["low_cpu_mem_usage"] = True
            
            # Load model
            logger.info(f"Loading model with trust_remote_code={TRUST_REMOTE_CODE} and device={DEVICE}")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=TRUST_REMOTE_CODE,
                **model_kwargs
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully on {DEVICE}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _format_prompt(self, prompt: str) -> str:
        """
        Format the prompt according to DeepSeek-R1 format requirements.
        DeepSeek-R1 uses a simple prompt format: "<human>: {prompt}\n<bot>: "
        """
        return f"<human>: {prompt}\n<bot>: "
    
    def unload_model(self):
        """Unload the model to free up resources"""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Force CUDA memory cleanup if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._is_loaded = False
        logger.info("Model unloaded")
    
    def generate(self, 
                prompt: str, 
                max_tokens: int = 256,
                temperature: float = None,
                top_p: float = 0.95,
                top_k: int = 50,
                stop_sequences: Optional[List[str]] = None) -> Tuple[str, Dict[str, int]]:
        """
        Generate text based on the prompt
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (uses default if None)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of sequences that stop generation
            
        Returns:
            Tuple of (generated_text, usage_info)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Use default temperature if not specified
        if temperature is None:
            temperature = TEMPERATURE
        
        # Log generation parameters
        logger.info(f"Generation parameters: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, top_k={top_k}")
        
        # Format the prompt according to DeepSeek-R1 format
        formatted_prompt = self._format_prompt(prompt)
        
        # Encode the input
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.model.device)
        input_length = input_ids.shape[1]
        
        # Configure generation parameters
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add stopping criteria if provided
        if stop_sequences:
            from transformers import StoppingCriteriaList, StoppingCriteria
            
            class StopSequenceCriteria(StoppingCriteria):
                def __init__(self, tokenizer, stop_sequences, input_length):
                    self.tokenizer = tokenizer
                    self.stop_sequences = stop_sequences
                    self.input_length = input_length
                
                def __call__(self, input_ids, scores, **kwargs):
                    generated_text = self.tokenizer.decode(input_ids[0][self.input_length:])
                    for stop_seq in self.stop_sequences:
                        if stop_seq in generated_text:
                            return True
                    return False
            
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopSequenceCriteria(self.tokenizer, stop_sequences, input_length)
            ])
        
        # Generate text
        with torch.no_grad():
            output = self.model.generate(**gen_kwargs)
        
        # Decode and process output
        generated_text = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        
        # Calculate token usage
        input_tokens = input_ids.shape[1]
        output_tokens = output.shape[1] - input_tokens
        total_tokens = input_tokens + output_tokens
        
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens
        }
        
        return generated_text, usage 