# DeSTA2.5-Audio vLLM Integration
"""
vLLM support for DeSTA2.5-Audio model.

This module provides vLLM-compatible implementation for high-throughput
multimodal inference with DeSTA2.5-Audio.

Usage:
    # Offline inference
    from vllm import LLM
    import desta.vllm  # This registers the model
    llm = LLM(model="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")

    # Online serving
    vllm serve DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B

Reference: https://arxiv.org/abs/2507.02768
"""

# Use the original config from desta/models/modeling_desta25.py
from desta.models.modeling_desta25 import DeSTA25Config

# Register config with transformers AutoConfig and CONFIG_MAPPING
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

try:
    AutoConfig.register("desta25", DeSTA25Config)
except ValueError:
    pass  # Already registered

# Also register with CONFIG_MAPPING directly for TOKENIZER_MAPPING compatibility
try:
    CONFIG_MAPPING.register("desta25", DeSTA25Config)
except ValueError:
    pass  # Already registered

# Register with TOKENIZER_MAPPING directly (more reliable than AutoTokenizer.register)
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.llama import LlamaTokenizerFast

try:
    # Register (slow_tokenizer_class, fast_tokenizer_class) tuple
    TOKENIZER_MAPPING.register(DeSTA25Config, (None, LlamaTokenizerFast))
except ValueError:
    pass  # Already registered

from desta.vllm.modeling_desta25 import (
    DeSTA25ForConditionalGeneration,
    DeSTA25ProcessingInfo,
    DeSTA25MultiModalProcessor,
    DeSTA25DummyInputsBuilder,
)

# Register the model with vLLM's model registry
from vllm.model_executor.models.registry import ModelRegistry

# Register both the original HuggingFace architecture name and our vLLM class
ModelRegistry.register_model(
    "DeSTA25AudioModel",
    "desta.vllm.modeling_desta25:DeSTA25ForConditionalGeneration"
)
ModelRegistry.register_model(
    "DeSTA25ForConditionalGeneration",
    "desta.vllm.modeling_desta25:DeSTA25ForConditionalGeneration"
)

__all__ = [
    "DeSTA25Config",
    "DeSTA25ForConditionalGeneration",
    "DeSTA25ProcessingInfo",
    "DeSTA25MultiModalProcessor",
    "DeSTA25DummyInputsBuilder",
]
