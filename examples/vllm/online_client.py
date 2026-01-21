#!/usr/bin/env python3
"""
DeSTA2.5-Audio Online Inference Client

This script demonstrates how to call DeSTA2.5-Audio via vLLM's OpenAI-compatible API.

First, start the vLLM server:
    vllm serve DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B --trust-remote-code

Then run this client:
    python online_client.py --audio_path /path/to/audio.wav

Reference: https://docs.vllm.ai/en/stable/features/multimodal_inputs/
"""

import argparse
import base64
from pathlib import Path

from openai import OpenAI


def encode_audio_to_base64(audio_path: str) -> tuple[str, str]:
    """Encode audio file to base64 string."""
    path = Path(audio_path)
    with open(path, "rb") as f:
        audio_bytes = f.read()

    # Determine format from extension
    suffix = path.suffix.lower()
    format_map = {
        ".wav": "wav",
        ".mp3": "mp3",
        ".flac": "flac",
        ".ogg": "ogg",
        ".m4a": "m4a",
    }
    audio_format = format_map.get(suffix, "wav")

    return base64.b64encode(audio_bytes).decode("utf-8"), audio_format


def main():
    parser = argparse.ArgumentParser(description="DeSTA2.5-Audio vLLM Online Client")
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
        "--api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B",
        help="Model name",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="Focus on the audio clips and instructions.",
        help="System prompt",
    )
    args = parser.parse_args()

    # Initialize OpenAI client pointing to vLLM server
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require API key
        base_url=args.api_base,
    )

    # Encode audio to base64
    print(f"Loading audio from: {args.audio_path}")
    audio_base64, audio_format = encode_audio_to_base64(args.audio_path)
    print(f"Audio format: {audio_format}")

    # Create chat completion request
    print("Sending request to vLLM server...")
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": args.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": args.prompt,
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": audio_format,
                        },
                    },
                ],
            },
        ],
        max_completion_tokens=args.max_tokens,
    )

    # Print response
    print("\n" + "=" * 50)
    print("Response:")
    print("=" * 50)
    print(response.choices[0].message.content)
    print("=" * 50)


if __name__ == "__main__":
    main()
