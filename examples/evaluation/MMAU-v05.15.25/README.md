## MMAU-v05.15.25 evaluation 


| Model                                | Speech | Sound | Music | Avg   |
|--------------------------------------|--------|-------|--------|-------|
| **OLD MMAU**                         |        |       |        |       |
| R1-AQA (Qwen2-Audio-7B-Instruct + GRPO) | 63.66  | 68.77 | 64.37  | 65.60 |
| ==DeSTA2.5-Audio==                       | 59.16  | 60.66 | 52.69  | 57.50 |
| Audio-flamingo 2                    | 30.93  | 61.56 | 73.95  | 55.48 |
| Qwen2-Audio-Instruct                | 42.04  | 54.95 | 50.98  | 49.20 |
| **MMAU-05-15-25 (non-parsed)**       |        |       |        |       |
| Qwen2.5 omni                        | 61.26  | 76.58 | 65.87  | 67.90 |
| ==DeSTA2.5-Audio==                       | 70.57  | 64.56 | 53.89  | 63.00 |
| Phi-4                               | 62.46  | 60.96 | 62.87  | 62.10 |
| Qwen2-Audio-Instruct                | 41.14  | 50.75 | 49.70  | 47.20 |



## Run

```shell
CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/.cache python inference_desta25_audio.py --data_root /lab/DeSTA2.5-Audio/my_data -i ./MMAU-051525/data/mmau-test-mini.json --model_id desta25
```

```shell
python mmau_evaluate.py --input /lab/DeSTA2.5-Audio/examples/evaluation/MMAU/results/results@desta25.json
```


## Results
```
******************************
Task-wise Accuracy:
sound : 64.56% over 333 samples
music : 53.89% over 334 samples
speech : 70.57% over 333 samples
******************************
Difficulty-wise Accuracy:
easy : 62.05% over 224 samples
hard : 58.47% over 236 samples
medium : 65.37% over 540 samples
******************************
Total Accuracy: 63.00% over 1000 samples
******************************
```
