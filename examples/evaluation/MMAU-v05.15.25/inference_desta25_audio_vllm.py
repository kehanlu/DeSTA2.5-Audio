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

TARGET_SR = 16000
AUDIO_LOCATOR = "<|AUDIO|>"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_path", "-o", type=str, default="results_vllm")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_audio(path):
    """Load audio file with resampling to 16kHz mono."""
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio, sr


def format_prompt(tokenizer, system_content, user_content):
    """Apply chat template to format the prompt correctly."""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Add <start_audio> and <end_audio> markers around <|AUDIO|> (as in training)
    prompt = prompt.replace(AUDIO_LOCATOR, f"<start_audio>{AUDIO_LOCATOR}<end_audio>")
    return prompt


def main(args):
    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained("DeSTA-ntu/Llama-3.1-8B-Instruct")

    # Initialize vLLM
    print(f"Initializing vLLM with {args.model_id}...")
    llm = LLM(
        model=args.model_id,
        tokenizer="DeSTA-ntu/Llama-3.1-8B-Instruct",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.0,  # Deterministic for evaluation
    )

    # Load MMAU data
    with open(args.input_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {args.input_path}")

    # System prompt (same as original)
    system_prompt = 'Focus on the audio clips and instructions. Put your answer in the format "The correct answer is: "___" ".'

    # Inference
    results = []
    vllm_inputs = []
    for item in tqdm(data[:args.limit], desc="Processing"):
        audio_path = os.path.join(
            args.data_root,
            item["audio_id"].replace("./", "", 1)
        )

        # Load audio
        try:
            audio, sr = load_audio(audio_path)
        except Exception as e:
            print(f"Warning: Failed to load audio {audio_path}: {e}")
            item["model_output"] = f"ERROR: {e}"
            item["model_prediction"] = ""
            results.append(item)
            continue


        # Build question (same format as original)
        question = f"{item['question']} "
        question += "Choose from the following options: "
        for i, option in enumerate(item["choices"]):
            question += f'"{option}"'
            if i == len(item["choices"]) - 2:
                question += " or "
            elif i < len(item["choices"]) - 1:
                question += ", "

        # Format user content with audio placeholder
        user_content = f"{AUDIO_LOCATOR}\n\n{question}"

        # Apply chat template
        prompt = format_prompt(tokenizer, system_prompt, user_content)
        vllm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"audio": (audio, sr)},
        })

    outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

    for output, item in zip(outputs, data):
        response = output.outputs[0].text
        item["model_output"] = response
        item["model_prediction"] = response.replace("The correct answer is: ", "").strip()
        results.append(item)

        # Store results
        item["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content, "audio": audio_path},
        ]
        item["model_output"] = response
        item["model_prediction"] = response.replace("The correct answer is: ", "").strip()

        results.append(item)

        # Print progress
        if len(results) % 10 == 0:
            print(f"\nProcessed {len(results)}/{len(data)} items")
            print(f"Last response: {response[:100]}...")

    # Save results
    os.makedirs("results", exist_ok=True)
    output_file = f"results/{args.output_path}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Total items processed: {len(results)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
