# DeSTA2.5-Audio vLLM Integration


```bash
git clone https://github.com/kehanlu/DeSTA2.5-Audio.git
cd DeSTA2.5-Audio
```

## Environment

### using pip

```
pip install -e .
pip install vllm
```

### using docker

Use official vllm docker image as a base image.

```bash
docker run -it --rm \
  --name vllm-dev \
  --gpus all \
  -v "${PWD}:/workspace" \
  -v "/path/to/data:/data" \
  -w /workspace \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  /bin/bash
```

Start the interactive shell container and install the DeSTA2.5-Audio package
```bash
cd /workspace
pip install -e .
```

Note: This might not the standard usage of Docker. We don't build the DeSTA2.5-Audio image ourselves just because we want to keep the development simple, you can build your own image if you need in production.


## Offline Inference

see also:
- [examples/vllm/offline_inference.py](../../examples/vllm/offline_inference.py)
- [examples/evaluation/MMAU-v05.15.25/inference_desta25_audio_vllm.py](../../examples/evaluation/MMAU-v05.15.25/inference_desta25_audio_vllm.py)

```python
# Register DeSTA25 model with vLLM
import desta.vllm
from vllm import LLM, SamplingParams
import librosa

AUDIO_PLACEHOLDER = "<|AUDIO|>"

# Initialize vLLM
llm = LLM(model="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B", trust_remote_code=True)
tokenizer = llm.get_tokenizer()

# Load audio
audio, sr = librosa.load("/path/to/audio.wav", sr=16000)


messages = [
    {"role": "system", "content": "Focus on the audio clips and instructions."},
    {"role": "user", "content": f"{AUDIO_PLACEHOLDER}\n\nDescribe the audio."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Generate text
outputs = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"audio": (audio, sr)},
    }],
)

print(outputs[0].outputs[0].text)
```



## Online Serving

```bash
python3 examples/vllm/start_server.py
```

Sending requests to the server

```python
import base64
from openai import OpenAI

client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require API key
        base_url="http://localhost:8000/v1",
    )

AUDIO_PLACEHOLDER = "<|AUDIO|>"


def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')


audio_base64 = encode_audio("/path/to/audio.wav")

response = client.chat.completions.create(
    model="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B",
    messages=[
        {
            "role": "system",
            "content": f"Focus on the audio clip and instruction.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""{AUDIO_PLACEHOLDER}\n\nDescribe the audio.""",
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav",
                    },
                },
            
            ],
        },
    ],
    max_completion_tokens=1024,
    temperature=0.0,
)

print(response.choices[0].message.content)
```