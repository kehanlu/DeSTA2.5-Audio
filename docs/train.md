# DeSTA Model Training

This guide walks you through running an example script to train a DeSTA-Audio model.

---

## ▶️ Quick Start: Example Script

This example script launches a lightweight training job using a small model and a sample dataset.

### 1. Download the example audio files

```sh
wget https://huggingface.co/datasets/kehanlu/example/resolve/main/LibriTTS_R.tar
tar -xvf LibriTTS_R.tar

mv LibriTTS_R /path/to/data_root/
```

### 2. Run the example training script

Set your environment variables in `train_example.sh`, then execute:

```sh
bash ./examples/train/train_example.sh
```

> 🔧 You can modify the configuration or dataset path in the script for your own data or experimental setup.


#### DeSTA2.5-Audio Training Configs

- Config: `examples/train/config/desta25_llama31-8B_Qformer6L.yaml`
- Dataset: `examples/train/config/dataset/desta-AQA5M.yaml`


---

## 📁 Prepare Your Dataset

### 🔊 Audio File Structure

Organize your audio files in the following directory structure:

```
/path/to/data_root/
├── LibriTTS_R/
│   └── train-clean-360/
│       └── 93/123172/
│           ├── 93_123172_000001_000000.wav
│           ├── 93_123172_000002_000000.wav
│           └── ...
```

---

### 📝 Training Data Format: JSONL

Each line in your dataset should be a JSON object with:

- `messages`: a list of dialog messages (input to the model)
- `response`: the assistant’s reply (target output)

Each `message` contains:

- `role`: either `"system"` or `"user"`
- `content`: the message text (use the `<|AUDIO|>` token to indicate an audio clip)
- If `<|AUDIO|>` is included, also add:
  - `audios`: a list of:
    - `audio`: relative path to the audio file
    - `text`: transcription of the audio

#### ✅ Example Entry

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant. The audio clips appear between <start_audio> and <end_audio> tags."
    },
    {
      "role": "user",
      "content": "Describe the audio file. <start_audio><|AUDIO|><end_audio>",
      "audios": [
        {
          "audio": "LibriTTS_R/train-clean-360/93/123172/93_123172_000001_000000.wav",
          "text": "Hello, how are you?"
        }
      ]
    }
  ],
  "response": "The audio is a recording of a person speaking. The person is saying 'Hello, how are you?'"
}
```


## 🚧 Coming soon

- [ ] Add more configs
- [ ] Tutorial for finetuning DeSTA2.5-Audio with LoRA and your own dataset
