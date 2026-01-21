# DeSTA2.5-Audio vLLM Examples

This directory contains examples for running DeSTA2.5-Audio with vLLM for high-throughput inference.

## Prerequisites

```bash
pip install vllm>=0.14.0
pip install -e .  # Install desta package
```

## Offline Inference

Run inference directly using vLLM's LLM class:

```bash
python offline_inference.py --audio_path /path/to/audio.wav
```

Options:
- `--model`: Model name or path (default: `DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B`)
- `--audio_path`: Path to audio file (required)
- `--prompt`: Prompt with `<|AUDIO|>` placeholder
- `--max_tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)

## Online Serving

### Start the Server

```bash
vllm serve DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B --trust-remote-code
```

### Call the API

Using the provided client:

```bash
python online_client.py --audio_path /path/to/audio.wav
```

Or using curl:

```bash
# Encode audio to base64
AUDIO_BASE64=$(base64 -w 0 /path/to/audio.wav)

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B\",
    \"messages\": [
      {
        \"role\": \"system\",
        \"content\": \"Focus on the audio clips and instructions.\"
      },
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"text\", \"text\": \"What do you hear? <|AUDIO|>\"},
          {\"type\": \"input_audio\", \"input_audio\": {\"data\": \"$AUDIO_BASE64\", \"format\": \"wav\"}}
        ]
      }
    ],
    \"max_tokens\": 512
  }"
```

## Multi-Audio Example

DeSTA2.5-Audio supports multiple audio inputs in a single request:

```python
from vllm import LLM, SamplingParams
import desta.vllm  # Register model

llm = LLM(model="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B", trust_remote_code=True)

# Compare two audio files
inputs = {
    "prompt": "Compare these two audio clips. First: <|AUDIO|> Second: <|AUDIO|>",
    "multi_modal_data": {
        "audio": [
            (audio1.numpy(), sr1),
            (audio2.numpy(), sr2),
        ]
    },
}

outputs = llm.generate([inputs], SamplingParams(max_tokens=512))
```

## Docker

Use the provided docker-compose for containerized deployment:

```bash
docker-compose up -d
```

See `docker-compose.yaml` in the repository root.
