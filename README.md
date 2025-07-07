# DeSTA2.5-Audio

[📑 Paper](https://arxiv.org/abs/2507.02768) | [👩‍💻 Github](https://github.com/kehanlu/DeSTA2.5-Audio) | [🤗 Model](https://huggingface.co/collections/DeSTA-ntu/desta25-audio-686a6b9e71afd92e1dd87486) 

**DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment**
> **Self-generated data is what you need for developing general-purpose LALMs!**

- 🧪 **A new training framework** ([read the paper](https://arxiv.org/abs/2507.02768))  
  - Highly scalable and efficient without task-specific instruction-tuning data  
  - Preserves language ability and avoids catastrophic forgetting  
  - Comprehensive studies on data quality in LALM development  
- 📦 **Open resources for the community**  
  - Model checkpoints and Training scripts
  - DeSTA-AQA5M dataset (5M audio-text pairs from 7,000 hours of audio)  



![](https://github.com/user-attachments/assets/f89dce86-2942-4644-aee5-a40ab4129328)




## ✨ News
- 🚧 Coming soon! Dataset(DeSTA-AQA5M), training and data construction script.
- 2025/07/03 DeSTA2.5-Audio paper is now available on arXiv. [📑 Paper](https://arxiv.org/abs/2507.02768)




## 🚀Quickstart

### Installation
```shell
git clone https://github.com/kehanlu/DeSTA2.5-Audio.git
cd DeSTA2.5-Audio
pip install -e .
```

### Inference
```python
from desta import DeSTA25AudioModel

# Load the model from Hugging Face
model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
model.to("cuda")

# Run inference with audio input
messages = [
    {
        "role": "system",
        "content": "Focus on the audio clips and instructions."
    },
    {
        "role": "user",
        "content": "<|AUDIO|>\nDescribe this audio.",
        "audios": [{
            "audio": "/path/to/audio.wav",  # Path to your audio file
            "text": None
        }]
    }
]

outputs = model.generate(
    messages=messages,
    do_sample=False,
    top_p=1.0,
    temperature=1.0,
    max_new_tokens=512
)

print(outputs.text)
```



## Training & Finetuning

🚧 Coming soon!
- We will release DeSTA-AQA5M.
- We will release the data construction script and training script.


## 📚 Citation
```bibtex
@article{lu2025desta25Audio,
  title={DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment},
  author={Lu, Ke-Han and Chen, Zhehuai and Fu, Szu-Wei and Yang, Chao-Han Huck and Huang, Sung-Feng and Yang, Chih-Kai and Yu, Chee-En and Chen, Chun-Wei and Chen, Wei-Chih and Huang, Chien-yu and others},
  journal={arXiv preprint arXiv:2507.02768},
  year={2025}
}

@inproceedings{lu2025developing,
  title={Developing instruction-following speech language model without speech instruction-tuning data},
  author={Lu, Ke-Han and Chen, Zhehuai and Fu, Szu-Wei and Yang, Chao-Han Huck and Balam, Jagadeesh and Ginsburg, Boris and Wang, Yu-Chiang Frank and Lee, Hung-yi},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}

@inproceedings{lu24c_interspeech,
  title     = {DeSTA: Enhancing Speech Language Models through Descriptive Speech-Text Alignment},
  author    = {Ke-Han Lu and Zhehuai Chen and Szu-Wei Fu and He Huang and Boris Ginsburg and Yu-Chiang Frank Wang and Hung-yi Lee},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4159--4163},
  doi       = {10.21437/Interspeech.2024-457},
  issn      = {2958-1796},
}
```



## 👥 Contributors
Ke-Han Lu, Zhehuai Chen, Szu-Wei Fu, Chao-Han Huck Yang, Sung-Feng Huang, Chih-Kai Yang, Chee-En Yu, Chun-Wei Chen, Wei-Chih Chen, Chien-yu Huang, Yi-Cheng Lin, Yu-Xiang Lin, Chi-An Fu, Chun-Yi Kuan, Wenze Ren, Xuanjun Chen, Wei-Ping Huang, En-Pei Hu, Tzu-Quan Lin, Yuan-Kuei Wu, Kuan-Po Huang, Hsiao-Ying Huang, Huang-Cheng Chou, Kai-Wei Chang, Cheng-Han Chiang, Boris Ginsburg, Yu-Chiang Frank Wang, Hung-yi Lee
