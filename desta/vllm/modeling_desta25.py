# SPDX-License-Identifier: Apache-2.0
# DeSTA2.5-Audio vLLM Implementation
"""
Inference-only DeSTA2.5-Audio model compatible with vLLM.

This module wraps the existing DeSTA25AudioModel components for vLLM compatibility,
supporting high-throughput multimodal inference.

Reference: https://arxiv.org/abs/2507.02768
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
from transformers import AutoProcessor, BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

# Import original DeSTA25 components
from desta.models.modeling_desta25 import (
    DeSTA25Config as OriginalDeSTA25Config,
    WhisperPerception,
    QformerConnector,
)
from desta.vllm.asr_engine import get_asr_engine, ASREngine

_AUDIO_PLACEHOLDER = "<|AUDIO|>"
_DEFAULT_PROMPT_SIZE = 64
_DEFAULT_MAX_TRANSCRIPTION_TOKENS = 128


# === Audio Input Types === #

class DeSTA25AudioFeatureInputs:
    """Audio feature inputs for DeSTA25."""
    type: Literal["audio_features"]
    data: torch.Tensor | list[torch.Tensor]


class DeSTA25AudioEmbeddingInputs:
    """Pre-computed audio embedding inputs."""
    type: Literal["audio_embeds"]
    data: list[torch.Tensor]


DeSTA25AudioInputs: TypeAlias = dict


# === Processing Classes === #

# Module-level cache for processors to avoid repeated loading
_PROCESSOR_CACHE: dict[str, object] = {}
_FEATURE_EXTRACTOR_CACHE: dict[str, WhisperFeatureExtractor] = {}


class DeSTA25ProcessingInfo(BaseProcessingInfo):
    """Processing info for DeSTA25 audio model."""

    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(self, **kwargs):
        config = self.get_hf_config()
        encoder_model_id = getattr(config, "encoder_model_id", "openai/whisper-large-v3")

        # Use cached processor if available (ignore kwargs for cache key)
        if encoder_model_id not in _PROCESSOR_CACHE:
            _PROCESSOR_CACHE[encoder_model_id] = AutoProcessor.from_pretrained(
                encoder_model_id,
                cache_dir=os.getenv("HF_HOME"),
                **kwargs,
            )
        return _PROCESSOR_CACHE[encoder_model_id]

    def get_feature_extractor(self, **kwargs) -> WhisperFeatureExtractor:
        config = self.get_hf_config()
        encoder_model_id = getattr(config, "encoder_model_id", "openai/whisper-large-v3")

        # Use cached feature extractor if available
        if encoder_model_id not in _FEATURE_EXTRACTOR_CACHE:
            hf_processor = self.get_hf_processor(**kwargs)
            if hasattr(hf_processor, "feature_extractor"):
                _FEATURE_EXTRACTOR_CACHE[encoder_model_id] = hf_processor.feature_extractor
            else:
                _FEATURE_EXTRACTOR_CACHE[encoder_model_id] = hf_processor
        return _FEATURE_EXTRACTOR_CACHE[encoder_model_id]

    def get_target_channels(self) -> int:
        """DeSTA uses mono audio."""
        return 1

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_asr_engine(self) -> ASREngine:
        """Get ASR engine (faster-whisper small for CPU, good speed/accuracy balance)."""
        return get_asr_engine()

    def get_llm_tokenizer(self):
        """Get the LLM tokenizer (same as the main tokenizer)."""
        return self.get_tokenizer()


class DeSTA25DummyInputsBuilder(BaseDummyInputsBuilder[DeSTA25ProcessingInfo]):
    """Builds dummy inputs for profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return _AUDIO_PLACEHOLDER * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


class DeSTA25MultiModalProcessor(BaseMultiModalProcessor[DeSTA25ProcessingInfo]):
    """Multimodal processor for DeSTA25."""

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs,
        tokenization_kwargs,
    ) -> bool:
        # Our _call_hf_processor does NOT apply placeholder expansion.
        # We keep <|AUDIO|> as-is and let vLLM's _apply_prompt_updates handle it.
        return False

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.info.get_target_channels(),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        tokenizer = self.info.get_tokenizer()

        if not audios:
            text_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                **tok_kwargs,
            )
            prompt_ids = text_inputs["input_ids"][0].tolist()
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # Process audio features
        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        audio_features = feature_extractor(
            audios,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
        ).input_features

        # Run ASR and insert transcriptions into prompt
        # Format: <|AUDIO|>{transcription}
        asr_engine = self.info.get_asr_engine()
        if asr_engine.is_ready:
            # Pass raw audio arrays to faster-whisper
            transcriptions = asr_engine.transcribe(audios)

            mm_kwargs["transcriptions"] = transcriptions


        # Tokenize the (possibly modified) prompt
        text_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            **tok_kwargs,
        )

        return BatchFeature(
            dict(
                input_ids=text_inputs["input_ids"],
                audio_features=audio_features,
            ),
            tensor_type="pt",
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_features=MultiModalFieldConfig.batched("audio"),
            audio_embeds=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.get_hf_config()
        prompt_size = getattr(config, "prompt_size", _DEFAULT_PROMPT_SIZE)

        transcriptions = hf_processor_mm_kwargs.get("transcriptions")


        def get_replacement_desta(item_idx: int):
            
            if not transcriptions:
                transcription = " "
            else:
                transcription = transcriptions[item_idx]
                if transcription == "":
                    transcription = " "


            return PromptUpdateDetails.select_text(
                seq=f"<start_audio>{_AUDIO_PLACEHOLDER * prompt_size}{transcription}<end_audio>",
                embed_text=_AUDIO_PLACEHOLDER,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=_AUDIO_PLACEHOLDER,
                replacement=get_replacement_desta,
            )
        ]


# === Main Model Class === #

@MULTIMODAL_REGISTRY.register_processor(
    DeSTA25MultiModalProcessor,
    info=DeSTA25ProcessingInfo,
    dummy_inputs=DeSTA25DummyInputsBuilder,
)
class DeSTA25ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    DeSTA2.5-Audio model for vLLM inference.

    This class wraps the original WhisperPerception from desta/models/modeling_desta25.py
    and integrates it with vLLM's model loading and inference system.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "perception.": "audio_tower.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return _AUDIO_PLACEHOLDER
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Get config values
        llm_model_id = getattr(config, "llm_model_id", "DeSTA-ntu/Llama-3.1-8B-Instruct")
        prompt_size = getattr(config, "prompt_size", _DEFAULT_PROMPT_SIZE)

        # Use the original WhisperPerception from desta/models/modeling_desta25.py
        self.audio_tower = WhisperPerception(config)

        # Cast audio_tower to bfloat16 to match vLLM's default dtype
        self.audio_tower = self.audio_tower.to(torch.bfloat16)

        # Set audio tower to eval mode (inference only, no dropout)
        self.audio_tower.eval()

        # Initialize vLLM language model
        llm_config = getattr(config, "llm_config", None)
        if llm_config is None:
            from transformers import AutoConfig
            llm_config = AutoConfig.from_pretrained(
                llm_model_id, cache_dir=os.getenv("HF_HOME")
            )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=llm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Track secondary weights for loading
        encoder_model_id = getattr(config, "encoder_model_id", "openai/whisper-large-v3")
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=encoder_model_id,
                revision=None,
                prefix="audio_tower.whisper.",
            ),
            DefaultModelLoader.Source(
                model_or_path=llm_model_id,
                revision=None,
                prefix="language_model.",
            ),
        ]

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Cache encoder dtype/device to avoid repeated lookups during inference
        self._encoder_dtype: torch.dtype | None = None
        self._encoder_device: torch.device | None = None

    def _get_encoder_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        """Get cached encoder dtype and device."""
        if self._encoder_dtype is None:
            self._encoder_dtype = self.audio_tower.whisper.model.encoder.conv1.weight.dtype
            self._encoder_device = self.audio_tower.whisper.model.encoder.conv1.weight.device
        return self._encoder_dtype, self._encoder_device

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get module prefix mapping for multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="audio_tower.connector.",
            tower_model="audio_tower.whisper.",
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> DeSTA25AudioInputs | None:
        # Use .get() instead of .pop() to avoid dict modification overhead
        audio_embeds = kwargs.get("audio_embeds")
        if audio_embeds is not None:
            return {"type": "audio_embeds", "data": audio_embeds}

        audio_features = kwargs.get("audio_features")
        if audio_features is not None:
            return {"type": "audio_features", "data": audio_features}

        return None

    @torch.inference_mode()
    def _process_audio_input(
        self, audio_input: DeSTA25AudioInputs
    ) -> NestedTensors | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return tuple(audio_input["data"])

        # Process audio features through WhisperPerception
        audio_features = audio_input["data"]

        if isinstance(audio_features, list):
            audio_features = torch.stack(audio_features)

        # Use cached dtype/device to avoid repeated attribute lookups
        encoder_dtype, encoder_device = self._get_encoder_dtype_device()
        audio_features = audio_features.to(dtype=encoder_dtype, device=encoder_device)

        # Use original WhisperPerception forward
        # It returns (audio_embeddings, feature_lengths)
        audio_embeddings, _ = self.audio_tower(input_features=audio_features)

        # Return as tuple of individual embeddings (unbind is faster than indexing)
        return tuple(audio_embeddings.unbind(0))

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        return self._process_audio_input(audio_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass through the model."""
        if intermediate_tensors is not None:
            inputs_embeds = None

        # Get the underlying language model
        language_model = self.language_model
        if hasattr(language_model, "model"):
            hidden_states = language_model.model(
                input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
            )
        else:
            hidden_states = language_model(
                input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
            )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["audio_tower.whisper."]
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Move audio_tower to same device as language_model
        # (Whisper was loaded via from_pretrained on CPU, vLLM ignores it)
        device = next(self.language_model.parameters()).device
        self.audio_tower.to(device)

        return loaded
