# MMAU-v05.15.25 evaluation 

|                                                                             | Speech    | Speech | Sound     | Sound | Music     | Music | Avg       |       |
| --------------------------------------------------------------------------- | --------- | ------ | --------- | ----- | --------- | ----- | --------- | ----- |
| Name                                                                        | Test-mini | Test   | Test-mini | Test  | Test-mini | Test  | Test-mini | Test  |
|                                                                             |           |        |           |       |           |       |           |       |
| [Audio Flamingo 3](https://www.arxiv.org/abs/2507.08128)                 | 66.37     | 66.97  | 79.58     | 75.83 | 66.77     | 74.47 | 73.30     | 72.42 |
| [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)                         | 70.60     | 68.90  | 78.10     | 76.77 | 65.90     | 67.33 | 71.50     | 71.00 |
| [Gemini 2.5 Pro](https://ai.google.dev/gemini-api/docs/models)           | 71.47     | 72.67  | 75.08     | 70.63 | 68.26     | 64.77 | 71.60     | 69.36 |
|[DeSTA2.5-Audio](https://arxiv.org/abs/2507.02768) ❤️❤️❤️ |**71.17**    |**71.94** |**70.57**    |**66.83**|**56.29**    |**57.09**|**66.00**       |**65.21**|
| [Kimi-Audio](https://arxiv.org/abs/2504.18425)                              | 62.16     | 56.57  | 75.68     | 70.70 | 66.77     | 65.93 | 68.20     | 64.40 |
| [Audio Reasoner](https://huggingface.co/zhifeixie/Audio-Reasoner/tree/main) | 66.07     | 62.53  | 67.87     | 67.27 | 69.16     | 61.53 | 67.70     | 63.78 |
| [Phi-4-multimodal](https://huggingface.co/microsoft/phi-4)                  | 67.27     | 63.80  | 65.47     | 62.67 | 64.37     | 61.97 | 65.70     | 62.81 |
| [Gemini 2.5 Flash Lite](https://ai.google.dev/gemini-api/docs/models)       | 72.07     | 67.47  | 63.06     | 62.50 | 63.47     | 54.87 | 66.20     | 61.61 |
| [Audio Flamingo 2](https://huggingface.co/nvidia/audio-flamingo)            | 44.74     | 44.87  | 71.47     | 68.13 | 70.96     | 70.20 | 62.40     | 61.06 |
| [GPT-4o Audio](https://arxiv.org/abs/2410.21276)                            | 66.67     | 69.33  | 64.56     | 63.20 | 56.29     | 49.93 | 62.50     | 60.82 |
| [Qwen2-Audio-Instruct](https://arxiv.org/abs/2407.10759)                    | 55.26     | 55.37  | 67.27     | 61.17 | 56.29     | 55.67 | 59.60     | 57.40 |
| [Gemma 3n](https://deepmind.google/models/gemma/gemma-3n/)                  | 61.26     | 62.13  | 55.86     | 50.27 | 56.89     | 53.20 | 58.00     | 55.20 |
| [GPT-4o mini Audio](https://arxiv.org/abs/2410.21276)                       | 69.07     | 67.47  | 50.75     | 49.67 | 39.22     | 35.97 | 53.00     | 51.03 |
| [SALMONN](https://arxiv.org/pdf/2310.13289)                                 | 26.43     | 28.77  | 41.14     | 42.10 | 37.13     | 37.83 | 34.90     | 36.23 |
| [GAMA-IT](https://huggingface.co/spaces/sonalkum/GAMA-IT)                   | 10.81     | 11.57  | 30.93     | 32.73 | 26.74     | 22.37 | 22.83     | 22.22 |
| [LTU](https://openreview.net/pdf?id=nBZBPXdJlC)                             | 15.92     | 15.33  | 20.42     | 20.67 | 15.97     | 15.68 | 17.44     | 17.23 |
| [Audio Flamingo Chat](https://huggingface.co/nvidia/audio-flamingo)         | 6.91      | 7.67   | 25.23     | 23.33 | 17.66     | 15.77 | 16.60     | 15.59 |



## Inference model

```shell
# MMAU test-mini
CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/.cache python inference_desta25_audio.py --data_root /lab/DeSTA2.5-Audio/my_data -i ./MMAU-051525/data/mmau-test-mini.json --model_id desta25

# MMAU test
CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/.cache python inference_desta25_audio.py --data_root /lab/DeSTA2.5-Audio/my_data -i ./MMAU-051525/data/mmau-test.json --model_id desta25
```

**Evaluate test-mini with official evaluation scripts** from [evaluate.py](https://github.com/Sakshi113/MMAU/blob/main/evaluation.py)
```shell
python mmau_evaluate.py --input /lab/DeSTA2.5-Audio/examples/evaluation/MMAU/results/mmau-test-mini_results.json
```

test-mini results:
```
******************************
Task-wise Accuracy:
sound : 70.57% over 333 samples
music : 56.29% over 334 samples
speech : 71.17% over 333 samples
******************************
```
