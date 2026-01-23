# SPDX-License-Identifier: Apache-2.0
"""
ASR Engine using faster-whisper for DeSTA2.5-Audio vLLM integration.
"""

import logging
from typing import Optional
import numpy as np
import os

# Suppress faster-whisper and ctranslate2 logs
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("ctranslate2").setLevel(logging.WARNING)

_ASR_ENGINES: dict[str, "ASREngine"] = {}


class ASREngine:
    """ASR engine using faster-whisper."""

    def __init__(self, model_id: str = "large-v3"):
        self.model_id = model_id
        self._model = None

    def _ensure_loaded(self):
        """Lazy load the model."""
        if self._model is None:
            import torch
            from faster_whisper import WhisperModel

            # Use GPU if available, otherwise CPU
            if torch.cuda.is_available():
                print(f"[ASR] Loading faster-whisper {self.model_id} on CUDA (float16)")
                self._model = WhisperModel(
                    self.model_id,
                    device="cuda",
                    compute_type="float16",
                )
            else:
                print(f"[ASR] Loading faster-whisper {self.model_id} on CPU (int8)")
                self._model = WhisperModel(
                    self.model_id,
                    device="cpu",
                    compute_type="int8",
                )

    @property
    def is_ready(self) -> bool:
        return True  # Always ready, will lazy load

    def transcribe(self, audio_arrays: list[np.ndarray], sampling_rate: int = 16000) -> list[str]:
        """
        Run ASR on audio arrays.

        Args:
            audio_arrays: List of audio arrays (numpy, float32, mono)
            sampling_rate: Audio sampling rate (should be 16000)

        Returns:
            List of transcription strings
        """
        self._ensure_loaded()

        transcriptions = []
        for audio in audio_arrays:
            # Ensure float32 numpy array (faster-whisper requirement)
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            audio = audio.astype(np.float32)

            # faster-whisper transcribe
            segments, _ = self._model.transcribe(audio, beam_size=1, vad_filter=True)
            text = " ".join(segment.text for segment in segments)
            if text == "":
                text = " "
            transcriptions.append(text.strip())

        if os.getenv("ENABLE_ASR_DEBUG", "0") == "1":
            print(f"[ASR] Transcriptions: {transcriptions}")

        return transcriptions


def get_asr_engine(model_id: str = "large-v3") -> ASREngine:
    """Get or create ASR engine singleton."""
    if model_id not in _ASR_ENGINES:
        _ASR_ENGINES[model_id] = ASREngine(model_id)
    return _ASR_ENGINES[model_id]


def clear_asr_engines():
    """Clear all cached ASR engines."""
    _ASR_ENGINES.clear()
