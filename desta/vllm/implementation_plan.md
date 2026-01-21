# DeSTA2.5-Audio Huggingface -> vLLM 遷移技術規格書

## Overview

- 將 DeSTA2.5-Audio 模型遷移至 vLLM 框架，實現高吞吐量的多模態推理，支援：
- VLLM 原生多模態支援
- 多輪對話 (Multi-turn)
- 單輪多音訊 (Multi-Audio)：支援同時傳入多段音訊進行比較或分析。
- OpenAI 相容介面：提供標準的多模態 API。

Reference: https://docs.vllm.ai/en/stable/features/multimodal_inputs/
Support VLLM version: 0.14.0
Use vllm native function instead of too many custom implementations.


## Modules

Multimodal User input --> Perception Module(Qformer output) --> LLM Module --> Response

### Perception

- Whisper Encoder
- Qformer Connector (Fixed 64xdim output)

- Input: Audio features (files from user input), can be multiple audio files
- Output: Qformer output(64xdim)


### LLM

- Taking Qformer output, ASR output, and User prompt, generate response text
- Input: Qformer output(64xdim), ASR output(string), User prompt(string) --> Prepare LLM input (ref: our implementation in models/modeling_desta25.py)
- Output: Response text(string)


## Supported input and output

Design:
- The input is text based, user can place a special token `<|AUDIO|>` to indicate the place for audio input. Then we handle the audio input and insert the actual audio embeddings at the appropriate position.
- User can provide multiple audio files, and the model will process them all. (For example, `Compare two voice notes <|AUDIO|> and <|AUDIO|>`). We need to make sure audio count matches the `<|AUDIO|>` count.


### Offline inference

```python
import torchaudio
from vllm import LLM

audio, sr = torchaudio.load("/root/lab/DeSTA2.5-Audio-vllm/assets/audios/dog10.wav")
print(f"Original shape: {audio.shape}")  # e.g., torch.Size([2, 16000])

# vLLM automatically converts to mono for Whisper-based models
llm = LLM(model="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")

outputs = llm.generate({
    "prompt": "What do you hear in this audio? <|AUDIO|>",
    "multi_modal_data": {"audio": (audio.numpy(), sr)},
})

print(outputs)
```


### Online serving

```bash
vllm serve DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B
```

```python
import base64
import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset

def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Any format supported by librosa is supported
audio_url = AudioAsset("/root/lab/DeSTA2.5-Audio-vllm/assets/audios/dog10.wav").url
audio_base64 = encode_base64_content_from_url(audio_url)

chat_completion_from_base64 = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you hear in this audio? <|AUDIO|>",
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav",
                    },
                    "uuid": audio_url,  # Optional
                },
            ],
        },
    ],
    model=model,
    max_completion_tokens=64,
)

result = chat_completion_from_base64.choices[0].message.content
print("Chat completion output from input audio:", result)
```


Note: In DeSTA2.5-Audio:
  - Config: https://huggingface.co/DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B/blob/main/config.json
  The model weights are **separated** into three parts to save storage space:
  - LLM: DeSTA-ntu/Llama-3.1-8B-Instruct (text)
  - Whisper: openai/whisper-large-v3 (audio)
  - Qformer: DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B (only qformer weights)

We need to register the model with vLLM, and use the native function to prepare the input for the LLM.