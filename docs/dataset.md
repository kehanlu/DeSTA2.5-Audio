
## DeSTA-AQA5M

| Name        | Response Generated From | HuggingFace ID                                                                                                                     | Preview                                                                                      |
| ----------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| DeSTA-AQA5M | Llama3.1-8B-Instruct    | [DeSTA-ntu/DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct](https://huggingface.co/datasets/DeSTA-ntu/DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct) | [🔍](https://huggingface.co/datasets/DeSTA-ntu/DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct/viewer) |




### Load Dataset

```python
from datasets import load_dataset

dataset = load_dataset("DeSTA-ntu/DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct")

# Load from chunked data files
# dataset = load_dataset("DeSTA-ntu/DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct", data_files=["audio.0.jsonl", "audio.1.jsonl", "speech.0.jsonl", "speech.1.jsonl"])
```

```
DatasetDict({
    train: Dataset({
        features: ['id', 'dataset', 'seed_description', 'prompt', 'response', 'messages'],
        num_rows: 4963845
    })
})
```

- **Core fields (used for dataset generation and training):**
  - `messages`: The input messages used for data generation and model training.
  - `response`: The model-generated response (used as the training target).

- **Auxiliary fields (for display or metadata purposes):**
  - `id`: Audio file ID(relative audio filepath)
  - `dataset`: The source dataset.
  - `seed_description`: The textual description constructed from the audio metadata.
  - `prompt`: The sampled prompt from the instruction pool.


> **Note:** We do not hold the license to redistribute the original audio files. Please download the audio files directly from the original dataset sources.



## Dataset Details

| Dataset | Example seed description | Paper | Dataset Link(TBA) | 
|---|----------------------------------|---|---|
| IEMOCAP | [00:00-00:02] Excuse me? (Emotion: Neutral, Gender: Female, Pitch: Very high, Volume: Low, Speaking speed: Very slow, Duration: 2s) | [link](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf) |  | 
| DailyTalk | [00:00-00:02] I'm figuring out my budget. (Emotion: No emotion, Act: Inform, Gender: Male, Duration: 2s) | [link](https://arxiv.org/abs/2207.01063) |  | 
| GLOBE | [00:00-00:04] the belts needed periodic cleaning and conditioning to keep them in good condition. (Accent: United States English, Age: teens, Gender: male, Duration: 4s) | [link](https://arxiv.org/abs/2406.14875) |  | 
| VCTK-corpus | [00:00-00:07] She can scoop these things into three red bags, and we will go meet her Wednesday at the train station. (Gender: Female, Pitch:low, Accent: newzealand, Age: 23, Emotion: neutral) | [link](https://datashare.ed.ac.uk/handle/10283/2950) |  | 
| MELD | [00:00-00:02] Take it. (Emotion:Neutral, Sentiment:Neutral, Gender: Female, Duration: 2s) | [link](https://arxiv.org/abs/1810.02508) |  | 
| PromptTTS | [00:00-00:05] Deeply engrossed in congenial work (Speaking speed: Slow, Volume: Normal, Pitch: Low, Gender: Female, Emotion: cheerful, Duration: 5s) | [link](https://arxiv.org/abs/2211.12171) |  | 
| Expresso | [00:00-00:01] Karen's in Switzerland? (Style: confused, Gender: Male, Duration: 1s) | [link](https://arxiv.org/abs/2308.05725) |  | 
| AccentDB | [00:00-00:04] The hogs were fed chopped corn and garbage. (Accent: American, Gender: Female, Emotion: Neutral, Duration: 4s) | [link](https://aclanthology.org/2020.lrec-1.659.pdf) |  | 
| VoxCeleb1 | [00:00-00:09] and it's all what side of the coin you're looking at... (Gender: Male, Emotion: Neutral, Duration: 9s) | [link](https://arxiv.org/abs/1706.08612) |  | 
| Anispeech | [00:00-00:06] You have no idea how heartless those snuffs are... (Emotion: angry, Speaking speed: Fast, Pitch: normal, Gender: Male, Duration: 6s) | [link](https://huggingface.co/datasets/ShoukanLabs/AniSpeech) |  | 
| MSP-IMPROV | [00:00-00:05] Yeah, well I'm going to go to class... (Gender: Female, Emotion: Angry, Activation: 3.0/5.0, Valence: 2.2/5.0, Dominance: 3.2/5.0, Naturalness: 4.4/5.0) | [link](https://ieeexplore.ieee.org/document/7374697) |  | 
| Fair-speech | [00:00-00:05] hi there is something i would like you to see (Gender: male, Age: 31 - 45, First Language: English, Socioeconomic Background: Medium, Ethnicity: Black or African American) | [link](https://www.isca-archive.org/interspeech_2024/veliche24_interspeech.pdf) |  | 
| CREMA-D | [00:00-00:02] It's eleven o'clock (Emotion: Anger, Gender: Male, Age: 51, Race: Caucasian, Ethnicity: Not Hispanic) | [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/) |  | 
| CAFE | [00:00-00:04] Trois cygnes aveugles au bord du lac (Emotion: Anger, Gender: Male, Intensity: Low intensity, Age: 46, English Translation: Three blind swans by the lake) | [link](https://dl.acm.org/doi/10.1145/3204949.3208121) |  | 
| EMOVO | [00:00-00:05] Vorrei il numero telefonico del Signor Piatti. (Emotion: Disgust, Gender: Female, Age: 28) | [link](https://aclanthology.org/L14-1478/) |  | 
| Speech accent archive | [00:00-00:22] Please call Stella... (Gender: male, Age: 40, Native Language: afrikaans, Country: south africa) | [link](https://brill.com/downloadpdf/display/book/9789401206884/B9789401206884-s014.pdf) |  | 
| EMNS | [00:00-00:05] He was a plucked instrument instructor... (Emotion: Happy, Gender: Female, Speaker: 3, Age: 20s) | [link](https://arxiv.org/pdf/2305.13137) |  | 
| KeSpeech | [00:00-00:04] 看重成长质量和竞争优势的成长型基金 (Speaker: 1000048, Dialect: Mandarin) | [link](https://openreview.net/pdf?id=b3Zoeq2sCLq) |  | 
| ESD | [00:00-00:02] 我每个月打一次电话。 (Emotion: Angry, Gender: Female) | [link](https://ieeexplore.ieee.org/document/9413391) |  | 
| LibriSpeech-c | [00:00-00:05] (Number of Speakers: 0, Duration: 5s) | [link](https://arxiv.org/abs/2406.11064) |  | 
| L2Arctic | [00:00-00:03] Lord but I'm glad to see you again Phil (Accent: Arabic, Speaker: ABA) | [link](https://www.isca-archive.org/interspeech_2018/zhao18b_interspeech.pdf) |  | 
| CommonVoice (EN and CN) | [00:00-00:04] The boy swore that... (Gender: male_masculine, Age: thirties, Accent: West Indies and Bermuda, Duration: 4s) | [link](https://arxiv.org/abs/1806.09514) |  | 
| EmoV-DB | [00:00-00:05] And you always want to see it in the superlative degree. (Emotion: Amused, Gender: Female) | [link](https://arxiv.org/abs/1806.09514) |  | 
| LibriTTS-R | [00:00-00:05] About artists and their work mr Quilter... (Gender: male, Speaker: John Rose, Pitch Type: moderate pitch, Noise Type: balanced in clarity, etc.) | [link](https://research.google/pubs/libritts-r-restoration-of-a-large-scale-multi-speaker-tts-corpus/) |  | 
| Dusha | [00:00-00:06] афина поприкольней было чем джой джой дура (Emotion: angry, Speaker: ..., Speaker_Emotion: angry) | [link](https://arxiv.org/abs/2212.12266) |  | 
| MSP-PODCAST | [00:00-00:05] yeah. so we're going to end this... (Emotion: N, Gender: Unknown, Arousal: 3.0, Valence: 3.6, Dominance: 4.0) | [link](https://ieeexplore.ieee.org/document/8003425) |  | 
| AliMeeting | Multi-speaker Mandarin conversation samples (multiple timestamps and speakers) | [link](https://arxiv.org/abs/2110.07393) |  | 
| CSZS | [00:00-00:03] Juan de Talavera belonged to the so-called escuela toledana. (Gender: male, Language: Spanish-English code-switching) | [link](https://arxiv.org/abs/2310.03018) |  | 
| NTUML2021 | [00:00-00:02] 好那我們就開始上課吧 (Gender: male) | [link](https://arxiv.org/abs/2401.00273) |  | 
| Speech Command | [00:00-00:01] (Command: silence, Duration: 1s) | [link](https://arxiv.org/abs/1804.03209) |  | 
| Libricoount | [00:00-00:05] (Number of Speakers: 0, Duration: 5s) | [link](https://zenodo.org/records/1216072) |  | 
| Voxlingual | [00:00-00:19] بس عم يلزق بالمدخنين... (Language: Arabic, Duration: 19s) | [link](https://arxiv.org/abs/2203.03022) |  | 
| ASVspoof | [00:00-00:02] I'm not worried about the critics. (Gender: Male, Source: Real human) | [link](https://hal.science/hal-04615766/) |  | 
| BIIC-Podcast | [00:00-00:03] 這個連結只有在現場才感受的到 所以大 (Emotion: Happy, Gender: Female, Sentiment: somewhat positive) | [link](https://biic.ee.nthu.edu.tw/archive/doc/research/An_Intelligent_Infrastructure_Toward_Large_Scale_Naturalistic_Affective_Speech_Corpora_Collection.pdf) |  | 
| CodecFake | [00:00-00:06] There is , according to legend, a boiling pot of gold at one end. (Gender: Female, Accent: england, Source: Synthesis speech) | [link](https://arxiv.org/abs/2406.07237) [link](https://arxiv.org/abs/2501.08238) |  | 
| Paraspeechcaps | [00:00-00:05] So as a person who doesn't live in a bubble... (Gender: male, emotion: disgusted, etc.) | [link](https://arxiv.org/pdf/2503.04713) |  | 
| VCTK+MUSAN | [00:00-00:03] (4 speakers talking) (Noise Level: Noisy, Signal-to-Noise Ratio: 10db) | [link](https://arxiv.org/abs/1510.08484) |  | 
| Dynamic-SUPERB-Train-noise-reverb | [00:00:00 - 00:00:04]Alan_Alda: "I pull the covers up just enough so the next time I look at them, it'll be a little gift to myself." (Gender:Male, Noise Level: Moderate(Signal-to-Noise Ratio: 15db), Reverberation(C50): 60ms, Duration: 4s) | [link](https://arxiv.org/abs/2309.09510) |  | 
| Audioset | [00:00-00:10] (speech, gush) | [link](https://ieeexplore.ieee.org/abstract/document/7952261) |  | 
| AudioCaps | [00:00-00:10] (Plastic crinkling... people talk) | [link](https://aclanthology.org/N19-1011/) |  | 
| Wavcaps | [00:00-00:10] (There is the sound of a truck.) | [link](https://ieeexplore.ieee.org/abstract/document/10572302) |  | 
| Clotho | [00:00-00:26] (Someone opening and closing a door...) | [link](https://ieeexplore.ieee.org/document/9052990) |  | 
| VocalSound | [00:00-00:11] (Sneeze) | [link](https://ieeexplore.ieee.org/abstract/document/9746828) |  | 
| ESC50 | [00:00-00:05] (Audio category: door_wood_knock, Duration: 5.0) | [link](http://dx.doi.org/10.1145/2733373.2806390) |  | 
| FSD50K | [00:00-00:18] (Type: ['water', 'gurgling', 'toilet flush']...) | [link](https://ieeexplore.ieee.org/abstract/document/9645159) |  | 
| THMINT-QI | [00:00-00:04] 妈亲来解妄语的关系哦 (gender: male, mos_score: 2, Speech_quality: 2/5(Poor)) | [link](https://arxiv.org/abs/2309.12766) |  | 
| Nsynth | [00:00-00:04] (Family: bass, Source: electronic, MIDI Note: 022...) | [link](https://arxiv.org/abs/1704.01279) |  | 
| OpenSinger | [00:00-00:06] 多少凉薄世态可动荡 (Gender: Male, Song: 一如年少模样) | [link](https://arxiv.org/abs/2112.10358) |  | 
| FMA | [00:00-00:30] You can watch the show... (Genre: Rock) | [link](https://arxiv.org/abs/1612.01840) |  | 
| GTZAN | [00:00-00:30] (Genre: reggae, Duration: 30s) | [link](https://ieeexplore.ieee.org/document/1021072) |  | 
| Mridangam | [00:00-00:01] (Stroke: cha, Tonic: b, Duration: 1s) | [link](https://ieeexplore.ieee.org/abstract/document/6637633) |  | 
