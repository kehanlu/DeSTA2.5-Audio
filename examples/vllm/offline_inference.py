#!/usr/bin/env python3
"""
DeSTA2.5-Audio Offline Inference with vLLM

This script demonstrates how to run offline inference with DeSTA2.5-Audio
using vLLM for high-throughput audio-language processing.

Usage:
    python offline_inference.py --audio_path /path/to/audio.wav
"""

import argparse
import torchaudio
from vllm import LLM, SamplingParams

# Register DeSTA25 model with vLLM
import desta.vllm  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="DeSTA2.5-Audio vLLM Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B",
        help="Model name or path",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What do you hear in this audio? <|AUDIO|>",
        help="Prompt with <|AUDIO|> placeholder",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    # Load audio
    print(f"Loading audio from: {args.audio_path}")
    audio, sr = torchaudio.load(args.audio_path)
    print(f"Audio shape: {audio.shape}, Sample rate: {sr}")

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        print(f"Converted to mono: {audio.shape}")

    # Initialize vLLM
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Prepare input
    inputs = {
        "prompt": args.prompt,
        "multi_modal_data": {"audio": (audio.numpy(), sr)},
    }

    # Generate
    print("Generating response...")
    outputs = llm.generate([inputs], sampling_params=sampling_params)

    # Print results
    for output in outputs:
        print("\n" + "=" * 50)
        print("Prompt:", output.prompt)
        print("=" * 50)
        for completion in output.outputs:
            print("Response:", completion.text)
            print("=" * 50)


if __name__ == "__main__":
    main()
