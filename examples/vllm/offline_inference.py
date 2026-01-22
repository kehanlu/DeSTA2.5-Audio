#!/usr/bin/env python3
"""
MMAU Evaluation with DeSTA2.5-Audio using vLLM backend.
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, "/mnt/data/khlu/DeSTA2.5-Audio-vllm")

import librosa
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Register DeSTA25 model with vLLM
import desta.vllm
# from desta.vllm import print_timing_summary

TARGET_SR = 16000
# AUDIO_LOCATOR = "<|AUDIO|>"
AUDIO_LOCATOR = "<|reserved_special_token_87|>"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
    return parser.parse_args()


def load_audio(path):
    """Load audio file with resampling to 16kHz mono."""
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio, sr


def main(args):
    # Load tokenizer for chat template

    # Initialize vLLM
    print(f"Initializing vLLM with {args.model_id}...")
    llm = LLM(
        model=args.model_id,
        tokenizer="DeSTA-ntu/Llama-3.1-8B-Instruct",
        trust_remote_code=True,
        max_model_len=4096,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.0,  # Deterministic for evaluation
    )

    
    def prepare_prompt(messages, audio_filepaths=None):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False,add_generation_prompt=True,)

        if audio_filepaths is None:
            return {
                "prompt": prompt,
            }
        else:
            audios = []
            for audio_filepath in audio_filepaths:
                audio, sr = load_audio(audio_filepath)
                audios.append((audio, sr))

            return {
                "prompt": prompt,
                "multi_modal_data": {"audio": audios},
            }
    
    vllm_inputs = []

    ### Example 1
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Hello, how are you?"},
    ]
    vllm_inputs.append(prepare_prompt(messages))

    ### Example 2
    messages = [
        {"role": "system", "content": "Focus on the audio clips and instructions."},
        {"role": "user", "content": f"{AUDIO_LOCATOR}\n\nDescribe the audio."},
    ]
    vllm_inputs.append(prepare_prompt(messages, ["/root/lab/DeSTA2.5-Audio-vllm/assets/audios/dog10.wav"]))

    ### Example 2
    messages = [
        {"role": "system", "content": "Focus on the audio clips and instructions."},
        {"role": "user", "content": f"{AUDIO_LOCATOR}\n\nDescribe the audio."},
    ]
    vllm_inputs.append(prepare_prompt(messages, ["/root/lab/DeSTA2.5-Audio-vllm/assets/audios/cat14.wav"]))

    ### Example 3
    messages = [
        {"role": "system", "content": "Focus on the audio clips and instructions."},
        {"role": "user", "content": f"{AUDIO_LOCATOR}\n\nDescribe the audio."},
    ]
    vllm_inputs.append(prepare_prompt(messages, ["/root/lab/DeSTA2.5-Audio-vllm/assets/audios/72fb5481-73ae-409d-8e16-c94ac48d2ee4.wav"]))

    ### Example 3
    messages = [
        {"role": "system", "content": "Focus on these two audio clips and instructions."},
        {"role": "user", "content": f"Compare the two audio clips.\n\nFirst: {AUDIO_LOCATOR}\n\nSecond: {AUDIO_LOCATOR}.\n\nDescribe them one by one."},
    ]
    vllm_inputs.append(prepare_prompt(messages, ["/root/lab/DeSTA2.5-Audio-vllm/assets/audios/dog10.wav", "/root/lab/DeSTA2.5-Audio-vllm/assets/audios/cat14.wav"]))

    # Example 4
    messages = [
        {"role": "system", "content": "Focus on the audio clips and instructions."},
        {"role": "user", "content": f"{AUDIO_LOCATOR}\n\nDescribe the audio in detail."},
    ]
    vllm_inputs.append(prepare_prompt(messages, ["/root/lab/DeSTA2.5-Audio-vllm/assets/audios/bf50d3fb-4454-4eea-9336-6acc0e8d34fa.wav"]))


    # Example 5
    messages = [
        {"role": "system", "content": "Focus on the audio clips and instructions."},
        {"role": "user", "content": f"{AUDIO_LOCATOR}\n\nDescribe the audio in detail.\n\n{AUDIO_LOCATOR}"},
    ]
    vllm_inputs.append(prepare_prompt(messages, ["/root/lab/DeSTA2.5-Audio-vllm/assets/audios/bf50d3fb-4454-4eea-9336-6acc0e8d34fa.wav", "/root/lab/DeSTA2.5-Audio-vllm/assets/audios/72fb5481-73ae-409d-8e16-c94ac48d2ee4.wav"]))

    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

    for vllm_input, output in zip(vllm_inputs, outputs):
        print(tokenizer.decode(output.prompt_token_ids).replace("<|reserved_special_token_87|>"*64, "<|AUDIO|>"))
        print(f"Output: {output.outputs[0].text}")
        print("-" * 100)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    