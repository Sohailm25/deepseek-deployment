import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Default model settings
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PRECISION = "bfloat16" if DEFAULT_DEVICE == "cuda" else "float32"
DEFAULT_TEMPERATURE = 0.6  # Recommended temperature for DeepSeek-R1
DEFAULT_TRUST_REMOTE_CODE = "true"  # Required for Qwen models
DEFAULT_TENSOR_PARALLEL_SIZE = "1"  # Default to no tensor parallelism

# Environment variable overrides
MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
DEVICE = os.environ.get("DEVICE", DEFAULT_DEVICE)
PRECISION = os.environ.get("PRECISION", DEFAULT_PRECISION)
MAX_GPU_MEMORY = os.environ.get("MAX_GPU_MEMORY", None)  # In GB, None means use all available
LOAD_IN_8BIT = os.environ.get("LOAD_IN_8BIT", "false").lower() == "true"  # Parse 8-bit loading option
TEMPERATURE = float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE))  # Default temperature for generation
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", DEFAULT_TRUST_REMOTE_CODE).lower() == "true"
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", DEFAULT_TENSOR_PARALLEL_SIZE))

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        # Log configuration
        logger.info(f"Model configuration: model_id={MODEL_ID}, device={DEVICE}, precision={PRECISION}, "
                   f"8bit={LOAD_IN_8BIT}, temperature={TEMPERATURE}, trust_remote_code={TRUST_REMOTE_CODE}, "
                   f"tensor_parallel_size={TENSOR_PARALLEL_SIZE}")
        
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def load_model(self):
        """Load the DeepSeek-R1 model and tokenizer"""
        logger.info(f"Loading model {MODEL_ID} on {DEVICE}...")
        
        try:
            # Configure device mapping and quantization
            if DEVICE == "cuda":
                device_map = "auto"
                torch_dtype = torch.bfloat16 if PRECISION == "bfloat16" else torch.float16
                
                # Configure GPU memory limit if specified
                gpu_kwargs = {}
                if MAX_GPU_MEMORY:
                    memory_gb = int(MAX_GPU_MEMORY)
                    max_memory = {i: f"{memory_gb}GiB" for i in range(torch.cuda.device_count())}
                    gpu_kwargs["max_memory"] = max_memory
                    logger.info(f"Setting GPU memory limit to {memory_gb}GB per GPU")
                
                # Configure quantization with BitsAndBytesConfig
                if LOAD_IN_8BIT:
                    logger.info("Loading model with 8-bit quantization via BitsAndBytesConfig")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False
                    )
                    gpu_kwargs["quantization_config"] = quantization_config
                
                # Configure tensor parallelism
                if TENSOR_PARALLEL_SIZE > 1:
                    logger.info(f"Enabling tensor parallelism with {TENSOR_PARALLEL_SIZE} GPUs")
                    if "model_kwargs" not in gpu_kwargs:
                        gpu_kwargs["model_kwargs"] = {}
                    
                    # Add tensor parallel configuration to model kwargs
                    gpu_kwargs["model_kwargs"]["tensor_parallel_size"] = TENSOR_PARALLEL_SIZE
                    # We need to use device_map="auto" when using tensor parallelism
                    device_map = "auto"
            else:
                device_map = "cpu"
                torch_dtype = torch.float32
                gpu_kwargs = {}
            
            # Load tokenizer
            logger.info(f"Loading tokenizer with trust_remote_code={TRUST_REMOTE_CODE}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                padding_side="left",
                trust_remote_code=TRUST_REMOTE_CODE
            )
            
            # Load model
            logger.info(f"Loading model with trust_remote_code={TRUST_REMOTE_CODE}")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=TRUST_REMOTE_CODE,
                **gpu_kwargs
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully")
        
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