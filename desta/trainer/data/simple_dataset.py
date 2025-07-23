import datasets
import logging
import os
import torch

from desta.utils.audio import AudioSegment
from desta.models.modeling_desta25 import _prepare_audio_context_and_start_positions
from lulutils import resolve_filepath

def _resolve_audio_filepath(audio_filepath):
    """
    There might be some ext "name" mismatch in the audio_filepath
    """

    # First check if the original file exists
    if os.path.exists(audio_filepath):
        return audio_filepath
    
    # If not, try to resolve to .wav version
    base, ext = os.path.splitext(audio_filepath)
    wav_filepath = base + ".wav"
    
    if os.path.exists(wav_filepath):
        return wav_filepath
    
    raise FileNotFoundError(f"Audio file {audio_filepath} does not exist")


class BaseCollateFn(object):
    def __init__(self, data_cfg, tokenizer, processor):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = data_cfg.max_seq_length

    def __call__(self, batch):
        """
        Prepare training batch:
        Input:
        """
        # ====================
        # Prepare text inputs
        # ====================

        # audio_text_inputs = audio_context + target
        assert self.tokenizer.padding_side == "left", f"padding_side must be left, but got {self.tokenizer.padding_side}"
        audio_text_inputs = self.tokenizer(
            [item["audio_context"] + item["target"] for item in batch],
            truncation=True,
            padding="longest",
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_length=True,
            add_special_tokens=False,
        )

        audio_context_inputs = self.tokenizer(
            [item["audio_context"] for item in batch],
            truncation=True,
            padding="longest",
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_length=True,
            add_special_tokens=False,
        )
        
        labels = torch.full_like(audio_text_inputs['input_ids'], -100)

        # ====================
        # Prepare text and audio inputs
        # ====================
        
        batch_features = []
        batch_start_positions = [] # list of tuple (batch_idx, start_position)
        batch_transcription_ids = []

        audio_context_batch_start_positions = []

        audio_start_answer_positions = []
        for i, item in enumerate(batch):
            # Prepare labels
            total_length = audio_text_inputs["length"][i]
            audio_context_length = len(self.tokenizer.tokenize(item["audio_context"]))
            pad_length = total_length - audio_text_inputs["attention_mask"][i].sum()
            
            start_answer_position = pad_length + audio_context_length
            labels[i, start_answer_position:] = audio_text_inputs['input_ids'][i, start_answer_position:]
            
            audio_start_answer_positions.append(start_answer_position)
      

            # Prepare audio inputs
            for audio_dict in item["audios"]:
                feature = AudioSegment.from_file(
                    audio_dict["audio"],
                    target_sr=16000,
                    channel_selector="average" # average two channels
                ).samples
                batch_features.append(feature)

            for transcription in item["transcription_list"]:
                batch_transcription_ids.append(
                    self.tokenizer.encode(transcription, add_special_tokens=False, return_tensors="pt").long()
                )
            
            for start_position in item["start_positions"]:
                batch_start_positions.append((i, start_position + pad_length))

            # Prepare context
            total_length = audio_context_inputs["length"][i]
            pad_length = total_length - audio_context_inputs["attention_mask"][i].sum()
            for start_position in item["start_positions"]:
                audio_context_batch_start_positions.append((i, start_position + pad_length))


        batch_features = self.processor(batch_features, sampling_rate=16000, return_tensors="pt").input_features

        assert len(batch_features) == len(batch_start_positions) == len(batch_transcription_ids), f"batch_features({len(batch_features)}), batch_start_positions({len(batch_start_positions)}), batch_transcription_ids({len(batch_transcription_ids)}), must have the same length."

        assert len(batch) == len(audio_text_inputs["input_ids"]) == len(audio_text_inputs["attention_mask"]) == len(labels), f"batch, audio_text_inputs, labels must have the same length."

        # for inp, lab, att in zip(context_inputs["input_ids"][0], labels[0], context_inputs["attention_mask"][0]):
        #     print(0 if lab == -100 else self.tokenizer.decode(lab), "    ", self.tokenizer.decode(inp), f"({inp.item()})", "    ", att.item())
        
        new_batch = {
            # audio text sequence
            "input_ids": audio_text_inputs['input_ids'],
            "attention_mask": audio_text_inputs['attention_mask'],
            "labels": labels,
            "audio_start_answer_positions": audio_start_answer_positions,

            # audio batch features
            "batch_features": batch_features, # tensor of shape (batch_size, audio_seq)
            "batch_transcription_ids": batch_transcription_ids, # list of list of transcription ids
            "batch_start_positions": batch_start_positions, # list of tuple (batch_idx, start_position)

            # context inputs (for evaluation)
            "context_input_ids": audio_context_inputs['input_ids'],
            "context_attention_mask": audio_context_inputs['attention_mask'],
            "context_batch_start_positions": audio_context_batch_start_positions, # list of tuple (batch_idx, start_position)

            "metadata": [item for item in batch]
        }

        return new_batch
        

class BaseAudioTextDataset(object):
    """
    Dataset format(jsonl):
    {
        "id": "123",
        "messages": [
            {
                "role": "user",
                "content": "Describe the following audio. <|AUDIO|>",
                "audios": [
                    {
                        "audio": "path/to/audio/file.wav",
                        "text": "Hello world", # leave ' ' if no transcription
                    }
                ]
            }
        ],
        "response": "The first audio is more similar to the user's query."
    }
    """
    def __init__(self, 
                 cfg,
                 data_cfg,
                 tokenizer,
                 processor
                ):
        """
        audio_locator: tokens in dataset that will be tokenized as a token.
        placeholder_token: token in the LLM embedding table that will be replaced with audio in forward pass. Use a least used or reserved token to avoid conflicts.

        Here is an audio: <|AUDIO|>. bla bla bla
        -->
        Here is an audio: <|placeholder_token|><|placeholder_token|><|placeholder_token|><|placeholder_token|> bla bla bla <|eos_token|>
        """
        self.audio_locator = cfg.model.audio_locator
        self.placeholder_token = cfg.model.placeholder_token

        self.data_root = data_cfg.data_root
        self.manifest_filepaths = [data_cfg.manifest_filepaths] if isinstance(data_cfg.manifest_filepaths, str) else data_cfg.manifest_filepaths # convert to list if single file

        for manifest_filepath in self.manifest_filepaths:
            logging.info(f"manifest_filepath: {manifest_filepath}")
        
        self.prompt_size = cfg.model.connector.prompt_size
        self.tokenizer = tokenizer
        self.processor = processor

        self.dataset = datasets.load_dataset(
            "json", data_files=[resolve_filepath(manifest_filepath) for manifest_filepath in self.manifest_filepaths]
        )["train"]

        self.dataset = self.dataset.map(
            self._preprocess_function,
            batched=True,
            batch_size=256,
            num_proc=8
        )

        logging.info(f"Dataset length: {len(self.dataset)}")

        self.collate_fn = BaseCollateFn(data_cfg=data_cfg, tokenizer=self.tokenizer, processor=self.processor)



    def _preprocess_function(self, examples):
        audio_context_list = []
        start_positions_list = []
        audio_list = []
        transcription_list = []

        # merge colums: response -> target
        # if "response" in examples:
        #     if examples.get("target") is None: 
        #         # only reponse
        #         examples["target"] = examples["response"]
        #     else:
        #         for i in range(len(examples["response"])):
        #             if examples["response"][i] is not None:
        #                 examples["target"][i] = examples["response"][i]

        # gather all audios from all messages
        batch_audios = []
        for messages in examples["messages"]:
            _a = []
            for message in messages:
                if message.get("audios"):
                    _a.extend(message["audios"])
            batch_audios.append(_a)
            
        examples["audios"] = batch_audios

        for messages, audios in zip(examples["messages"], examples["audios"]):
            audio_context = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ) # for audio_locator
            
            assert len(audios) == audio_context.count(self.audio_locator), f"Number of audios {len(audios)} does not match number of audio locators {audio_context.count(self.audio_locator)}, audios: {audios}"


            # modify audio_filepath to be absolute path
            new_audios = []
            for audio_dict in audios:
                audio_dict["audio"] = _resolve_audio_filepath(
                    os.path.join(self.data_root, audio_dict["audio"])
                )
                new_audios.append(audio_dict)

            # Replace audio_locator with [placeholder_token] * (prompt_size + transcription_size) in context and get start positions
            # this serves as placeholder for features and transcription
            # Transcript the speech <features+transcription>
            # Now assume all audios have the same size

            audio_size_list = [self.prompt_size] * len(audios)

            # truncate transcriptions to 100 tokens
            transcriptions = [
                self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.tokenize(audio["text"], add_special_tokens=False)[:100]
                ) for audio in audios
            ]
            transcription_size_list = [
                len(self.tokenizer.tokenize(text, add_special_tokens=False)) for text in transcriptions
            ]
            transcription_list.append(transcriptions)

            audio_context, start_positions = _prepare_audio_context_and_start_positions(
                token_list=self.tokenizer.tokenize(audio_context), 
                audio_locator=self.audio_locator,
                audio_size_list=audio_size_list,
                transcription_size_list=transcription_size_list,
                placeholder_token=self.placeholder_token
            )

            audio_context = self.tokenizer.convert_tokens_to_string(audio_context)
            audio_context_list.append(audio_context)
            start_positions_list.append(start_positions)
            audio_list.append(new_audios)
        
        examples["audio_context"] = audio_context_list
        examples["start_positions"] = start_positions_list # list of list of start positions
        examples["transcription_list"] = transcription_list # list of list of transcription ids
        examples["audios"] = audio_list # list of list of audio dicts

        examples["target"] = [target + self.tokenizer.eos_token for target in examples["response"]]
        examples["length"] = [len(self.tokenizer.tokenize(audio_context + target)) for audio_context, target in zip(examples["audio_context"], examples["response"])]

        return examples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
        
