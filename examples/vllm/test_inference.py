#!/usr/bin/env python3
"""
Test DeSTA2.5-Audio vLLM Inference with example audio files.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, "/mnt/data/khlu/DeSTA2.5-Audio-vllm")
os.chdir("/mnt/data/khlu/DeSTA2.5-Audio-vllm")

import librosa
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Import to register DeSTA25 model with vLLM
import desta.vllm

# Audio files
AUDIO_DIR = "/mnt/data/khlu/DeSTA2.5-Audio-vllm/assets/audios"
DOG_AUDIO = f"{AUDIO_DIR}/dog10.wav"
CAT_AUDIO = f"{AUDIO_DIR}/cat14.wav"

TARGET_SR = 16000  # Whisper expects 16kHz
AUDIO_LOCATOR = "<|AUDIO|>"

def load_audio(path):
    """Load audio file with resampling to 16kHz mono (consistent with original)."""
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio, sr

def format_prompt(tokenizer, user_content):
    """Apply chat template to format the prompt correctly."""
    messages = [
        {"role": "user", "content": user_content}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Add <start_audio> and <end_audio> markers around <|AUDIO|> (as in training)
    prompt = prompt.replace(AUDIO_LOCATOR, f"<start_audio>{AUDIO_LOCATOR}<end_audio>")
    return prompt

def main():
    print("=" * 60)
    print("DeSTA2.5-Audio vLLM Test")
    print("=" * 60)

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained("DeSTA-ntu/Llama-3.1-8B-Instruct")

    # Load audio files
    print(f"\nLoading audio files...")
    dog_audio, dog_sr = load_audio(DOG_AUDIO)
    cat_audio, cat_sr = load_audio(CAT_AUDIO)
    print(f"  Dog audio: {dog_audio.shape}, SR: {dog_sr}")
    print(f"  Cat audio: {cat_audio.shape}, SR: {cat_sr}")

    # Initialize vLLM
    print(f"\nInitializing vLLM with DeSTA2.5-Audio...")
    llm = LLM(
        model="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B",
        tokenizer="DeSTA-ntu/Llama-3.1-8B-Instruct",
        trust_remote_code=True,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.0,  # Deterministic for testing
    )

    # Test 1: Single audio - Dog
    print("\n" + "=" * 60)
    print("Test 1: Describe dog audio")
    print("=" * 60)
    prompt = format_prompt(tokenizer, f"Describe the audio. {AUDIO_LOCATOR}")
    print(f"Formatted prompt:\n{prompt}\n")
    outputs = llm.generate(
        [{
            "prompt": prompt,
            "multi_modal_data": {"audio": (dog_audio, dog_sr)},
        }],
        sampling_params=sampling_params,
    )
    print(f"Response: {outputs[0].outputs[0].text}")

    # Test 2: Single audio - Cat
    print("\n" + "=" * 60)
    print("Test 2: Describe cat audio")
    print("=" * 60)
    prompt = format_prompt(tokenizer, f"Describe the audio. {AUDIO_LOCATOR}")
    outputs = llm.generate(
        [{
            "prompt": prompt,
            "multi_modal_data": {"audio": (cat_audio, cat_sr)},
        }],
        sampling_params=sampling_params,
    )
    print(f"Response: {outputs[0].outputs[0].text}")

    # Test 3: Multi-audio comparison
    print("\n" + "=" * 60)
    print("Test 3: Compare two audio clips")
    print("=" * 60)
    prompt = format_prompt(
        tokenizer,
        f"Compare these two sounds. First sound: {AUDIO_LOCATOR} Second sound: {AUDIO_LOCATOR} What are the differences?"
    )
    outputs = llm.generate(
        [{
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [
                    (dog_audio, dog_sr),
                    (cat_audio, cat_sr),
                ]
            },
        }],
        sampling_params=sampling_params,
    )
    print(f"Response: {outputs[0].outputs[0].text}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
