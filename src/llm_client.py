"""
LLM Client Wrapper
Supports: OpenAI, Anthropic Claude, and HuggingFace models
"""

import os
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
import time


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 2000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response with conversation history"""
        pass



class OpenAIClient(BaseLLMClient):
    """OpenAI API Client (compatible with max_tokens and max_completion_tokens)"""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7,
                 max_tokens: int = 2000, api_key: Optional[str] = None):
        super().__init__(model_name, temperature, max_tokens)
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def _chat_create(self, messages: List[Dict[str, str]]) -> str:
        base_kwargs = dict(
            model=self.model_name,
            messages=messages,
        )

        # We'll try combinations in decreasing "richness"
        trials = [
            dict(base_kwargs, temperature=self.temperature, max_completion_tokens=self.max_tokens),
            dict(base_kwargs, temperature=self.temperature, max_tokens=self.max_tokens),
            dict(base_kwargs, max_completion_tokens=self.max_tokens),
            dict(base_kwargs, max_tokens=self.max_tokens),
            dict(base_kwargs),  # ultimate fallback: only model + messages
        ]

        last_error = None
        for kwargs in trials:
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                last_error = e
                msg = str(e)
                # Only swallow errors related to unsupported params/values
                if any(k in msg for k in ["Unsupported parameter", "Unsupported value"]):
                    continue
                # Other errors (auth, rate limit, etc.) should surface
                raise

        # If all trials failed, raise the last error
        raise last_error


    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._chat_create(messages)

    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        return self._chat_create(messages)



class ClaudeClient(BaseLLMClient):
    """Anthropic Claude API Client"""
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", temperature: float = 0.7,
                 max_tokens: int = 2000, api_key: Optional[str] = None):
        super().__init__(model_name, temperature, max_tokens)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from Claude"""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response with conversation history
        
        Note: Claude requires alternating user/assistant messages.
        System prompt should be passed separately.
        """
        system_prompt = ""
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered_messages.append(msg)
        
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=filtered_messages
        )
        return message.content[0].text


class HuggingFaceClient(BaseLLMClient):
    """HuggingFace Inference API Client"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-1B-Instruct", 
                 temperature: float = 0.7, max_tokens: int = 2000, 
                 api_key: Optional[str] = None):
        super().__init__(model_name, temperature, max_tokens)
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(
                model=model_name,
                token=api_key or os.getenv("HUGGINGFACE_API_KEY")
            )
        except ImportError:
            raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from HuggingFace"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response with conversation history"""
        response = self.client.chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content


class HuggingFaceLocalClient(BaseLLMClient):
    """
    HuggingFace Local Model Client
    Download and run models locally on your server
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 temperature: float = 0.7, max_tokens: int = 2000,
                 device: str = "auto", load_in_8bit: bool = True,
                 load_in_4bit: bool = False, use_flash_attention: bool = False):
        """
        Args:
            model_name: HuggingFace model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            device: Device to use ('auto', 'cuda', 'cpu')
            load_in_8bit: Use 8-bit quantization (saves ~50% memory)
            load_in_4bit: Use 4-bit quantization (saves ~75% memory)
            use_flash_attention: Use flash attention 2 (faster, requires Ampere+ GPU)
        """
        super().__init__(model_name, temperature, max_tokens)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "Please install transformers and torch:\n"
                "pip install transformers torch accelerate"
            )
        
        print(f"Loading local model: {model_name}")
        print(f"  Device: {device}")
        if load_in_8bit:
            print(f"  Quantization: 8-bit")
        elif load_in_4bit:
            print(f"  Quantization: 4-bit")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model loading
        load_kwargs = {
            "trust_remote_code": True,
        }
        
        # Quantization configuration
        if load_in_8bit:
            try:
                import bitsandbytes
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"
            except ImportError:
                raise ImportError("8-bit requires bitsandbytes: pip install bitsandbytes")
        elif load_in_4bit:
            try:
                import bitsandbytes
                load_kwargs["load_in_4bit"] = True
                load_kwargs["device_map"] = "auto"
            except ImportError:
                raise ImportError("4-bit requires bitsandbytes: pip install bitsandbytes")
        else:
            # Standard loading
            import torch
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.bfloat16
            else:
                load_kwargs["torch_dtype"] = torch.float32
            
            if device == "auto":
                load_kwargs["device_map"] = "auto"
        
        # Flash attention
        if use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Set device if not using device_map
        if "device_map" not in load_kwargs:
            import torch
            self.device = torch.device(device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
        else:
            self.device = None  # Managed by device_map
        
        print(f"✓ Model loaded successfully")
    
    def _prepare_chat_input(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to chat format"""
        # Try using tokenizer's chat template
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                chat_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return chat_input
            except Exception:
                pass  # Fall back to manual formatting
        
        # Manual formatting (fallback)
        chat_input = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                chat_input += f"<|system|>\n{content}\n"
            elif role == "user":
                chat_input += f"<|user|>\n{content}\n"
            elif role == "assistant":
                chat_input += f"<|assistant|>\n{content}\n"
        chat_input += "<|assistant|>\n"
        
        return chat_input
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from local model"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.generate_with_history(messages)
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        """Generate response with conversation history"""
        import torch
        
        # Prepare input
        chat_input = self._prepare_chat_input(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            chat_input,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        if self.device is not None:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # device_map handles this
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()



class VLLMClient(BaseLLMClient):
    """vLLM Client """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 temperature: float = 0.7, max_tokens: int = 2000,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 visible_devices: Optional[str] = None):
        """
        Args:
            model_name: model name
            temperature: sampling temperature
            max_tokens: max tokens to generate
            tensor_parallel_size: parallel size for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            visible_devices: specify visible GPU devices
        """
        super().__init__(model_name, temperature, max_tokens)
        
        
        if visible_devices is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            print(f"Setting CUDA_VISIBLE_DEVICES={visible_devices}")
        
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Please install vllm: pip install vllm")
        
        print(f"Loading vLLM model: {model_name}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")
        if visible_devices:
            print(f"  Visible devices: {visible_devices}")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16"
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95
        )
        
        print(f"✓ vLLM model loaded successfully")
    
    def _prepare_prompt(self, messages: List[Dict[str, str]]) -> str:
        """take messages and convert to vLLM prompt format"""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        prompt += "<|assistant|>\n"
        return prompt
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """generate single response"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.generate_with_history(messages)
    
    def generate_with_history(self, messages: List[Dict[str, str]]) -> str:
        prompt_text = self._prepare_prompt(messages)
        
        outputs = self.llm.generate([prompt_text], self.sampling_params)
        
        return outputs[0].outputs[0].text.strip()
    
    def batch_generate(self, prompts: List[str], system_prompt: Optional[str] = None) -> List[str]:
        """
        batch generate multiple prompts
        """
        messages_batch = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages_batch.append(self._prepare_prompt(messages))
        
        outputs = self.llm.generate(messages_batch, self.sampling_params)
        
        return [output.outputs[0].text.strip() for output in outputs]



class LLMClientFactory:
    @staticmethod
    def create_client(
        provider: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        
        # Transformers local parameters
        use_local: bool = True,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
        
        # vLLM parameters
        use_vllm: bool = True,              
        tensor_parallel_size: int = 1,       
        gpu_memory_utilization: float = 0.9,
        visible_devices: Optional[str] = "0"
    ) -> BaseLLMClient:
        """
        Create LLM client
        """
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAIClient(
                model_name=model_name or "gpt-4o",
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )
        
        elif provider == "claude":
            return ClaudeClient(
                model_name=model_name or "claude-sonnet-4-20250514",
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )
        
        elif provider == "huggingface":
            # 优先级: vLLM > Local > API
            if use_vllm:
                return VLLMClient(
                    model_name=model_name or "meta-llama/Llama-3.1-8B-Instruct",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    visible_devices=visible_devices
                )
            elif use_local:
                return HuggingFaceLocalClient(
                    model_name=model_name or "meta-llama/Llama-3.1-8B-Instruct",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    device=device,
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    use_flash_attention=use_flash_attention
                )
            else:
                return HuggingFaceClient(
                    model_name=model_name or "meta-llama/Llama-3.1-8B-Instruct",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")


# Convenience function for quick usage
def get_llm_client(provider: str = "openai", **kwargs) -> BaseLLMClient:
    """Quick function to get an LLM client"""
    return LLMClientFactory.create_client(provider, **kwargs)


