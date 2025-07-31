
import os
import torch
import torch.nn as nn
from collections import OrderedDict
import logging

from dataclasses import dataclass
from desta.utils.audio import AudioSegment

from transformers import AutoTokenizer, AutoProcessor
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import WhisperForConditionalGeneration, BertConfig
from safetensors.torch import load_file


def _prepare_audio_context_and_start_positions(
                                             token_list,
                                             audio_locator,
                                             audio_size_list,
                                             transcription_size_list,
                                             placeholder_token
        ):
        assert len(audio_size_list) == len(transcription_size_list), f"audio_size_list and transcription_size_list must have the same length, audio_size_list: {audio_size_list}, transcription_size_list: {transcription_size_list}"

        result = []
        start_positions = []
        for x in token_list:
            if x == audio_locator:
                # start_positions.append(len(result))
                transcription_size = transcription_size_list.pop(0)
                audio_size = audio_size_list.pop(0)

                # result.extend(transcription)
                start_positions.append(len(result))
                result.extend([placeholder_token] * (audio_size))
                result.extend([placeholder_token] * (transcription_size))
            else:
                result.append(x)
                
        return result, start_positions


class QformerConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.encoder_model_id == "openai/whisper-medium":
            self.config.target_layer_ids = [5, 11, 17, 23]
        elif self.config.encoder_model_id == "openai/whisper-small":
            self.config.target_layer_ids = [2, 5, 8, 11]
        elif self.config.encoder_model_id == "openai/whisper-tiny":
            self.config.target_layer_ids = [0, 1, 2, 3]
        elif self.config.encoder_model_id == "openai/whisper-large-v3":
            self.config.target_layer_ids = [7, 15, 23, 31]
        else:
            raise NotImplementedError(f"model_id {self.config.encoder_model_id} not implemented")


        self.layer_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.config.prompt_size, self.config.encoder_config.d_model)) for _ in range(len(self.config.target_layer_ids))]
        )

        self.layer_weights = nn.Parameter(torch.zeros(self.config.prompt_size, len(self.config.target_layer_ids), dtype=torch.float))

        if self.config.connector_mode == "qformer_1":
            # init Qformerblock
            qformer_config = BertConfig()
            qformer_config.num_hidden_layers = self.config.qformer_num_hidden_layers
            qformer_config.num_attention_heads = self.config.encoder_config.encoder_attention_heads
            qformer_config.hidden_size = self.config.encoder_config.d_model
            qformer_config.add_cross_attention = True
            qformer_config.is_decoder = True
            qformer_config._attn_implementation = "eager"

            self.qformer = BertEncoder(qformer_config)
            self.proj = nn.Sequential(
                    nn.LayerNorm(self.config.encoder_config.d_model),
                    nn.Linear(self.config.encoder_config.d_model, self.config.llm_config.hidden_size) # project to llm hidden size
                )
        else:
            raise NotImplementedError(f"connector_mode {self.config.connector_mode} not implemented")
        

    def forward(self, encoder_hidden_states):
        """
        input: 
            encoder_hidden_states: layerwise hidden states from the encoder
        """
        layer_prompt_outputs = []
        for idx, encoder_hidden_state in enumerate(encoder_hidden_states):
            if idx in self.config.target_layer_ids:
                layer_prompt = self.layer_prompts[self.config.target_layer_ids.index(idx)].expand(encoder_hidden_state.size(0), -1, -1)
                qformer_output = self.qformer(
                    hidden_states=layer_prompt,
                    encoder_hidden_states=encoder_hidden_state,
                )
                layer_prompt_output = qformer_output.last_hidden_state
                layer_prompt_outputs.append(layer_prompt_output)
        
        layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0)
        layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3)
        self.norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=-1).unsqueeze(-1)
        output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_llm)
        output = self.proj(output)
        
        return output

@dataclass
class GenerationOutput():
    audios: list[str]
    generated_ids: list[torch.Tensor]
    text: list[str]

class WhisperPerception(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            self.config.encoder_model_id, cache_dir=os.getenv("HF_HOME"))

        self.connector = QformerConnector(config)


    def forward(self, input_features, attention_mask=None, transcription_embeddings_list=None, **kwargs):
        bs = input_features.size(0)

        audio_features = self.forward_whisper(input_features=input_features, transcription_embeddings_list=transcription_embeddings_list)
        speech_feature_lengths = [self.config.prompt_size] * audio_features.size(0) # (b, )
        
        return audio_features, speech_feature_lengths


    def forward_whisper(self, input_features, attention_mask=None, transcription_embeddings_list=None, **kwargs):
        """
        
        """
        bs = input_features.size(0)
        
        expected_seq_length = self.whisper.model.encoder.config.max_source_positions * self.whisper.model.encoder.conv1.stride[0] * self.whisper.model.encoder.conv2.stride[0]

        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )
        

        inputs_embeds = nn.functional.gelu(self.whisper.model.encoder.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.whisper.model.encoder.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.whisper.model.encoder.embed_positions.weight[:self.whisper.model.encoder.config.max_source_positions, :] # @kehan

        hidden_states = inputs_embeds + embed_pos
        features_length = hidden_states.size(1)

        if self.config.connector_mode == "qformer_1":
            layer_prompt_outputs = []
            for idx, encoder_layer in enumerate(self.whisper.model.encoder.layers):
                
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=None,
                    layer_head_mask=None,
                    output_attentions=None,
                )
                hidden_states = layer_outputs[0]

                if idx in self.connector.config.target_layer_ids:
                    # use different prompt for different layers
                    layer_prompt = self.connector.layer_prompts[self.connector.config.target_layer_ids.index(idx)].expand(bs, -1, -1)
                    
                    # Qformer is a BERTEncoder(but set to decoder) from huggingface Transformers
                    qformer_output = self.connector.qformer(
                        layer_prompt,
                        encoder_hidden_states=hidden_states,
                    )
                    
                    layer_prompt_output = qformer_output.last_hidden_state[:, :self.config.prompt_size, :] # (b, prompt_size, d_model)
                    layer_prompt_outputs.append(layer_prompt_output) # list of (b, prompt_size, d_model)

            layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0) # (layer, b, prompt_size, d_model)
            layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3) # (b, prompt_size, layer, d_model)
            
            self.norm_weights = torch.nn.functional.softmax(self.connector.layer_weights, dim=-1).unsqueeze(-1) # (prompt_size, layer, 1)
            prompt_output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_model)
            assert prompt_output.size(1) == self.config.prompt_size, prompt_output.size()
            prompt_output = self.connector.proj(prompt_output)
            
            return prompt_output

        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")
    
    



class DeSTA25Config(PretrainedConfig):
    model_type = "desta25"

    def __init__(self, 
                 llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
                 encoder_model_id="openai/whisper-large-v3",
                 connector_mode="qformer_1", 
                 qformer_num_hidden_layers=2, 
                 prompt_size=64, 
                 use_lora=False,
                 audio_locator="<|AUDIO|>",
                 placeholder_token="<|reserved_special_token_87|>",
                 **kwargs):
        
        super().__init__(**kwargs)

        self.llm_model_id = llm_model_id
        self.encoder_model_id = encoder_model_id
        self.connector_mode = connector_mode
        self.qformer_num_hidden_layers = qformer_num_hidden_layers
        self.prompt_size = prompt_size

        self.audio_locator = audio_locator
        self.placeholder_token = placeholder_token

        self.llm_config = AutoConfig.from_pretrained(self.llm_model_id)
        self.encoder_config = AutoConfig.from_pretrained(self.encoder_model_id)

        self.use_lora = use_lora

        self.info = "Ｄｅｓｔａ２。５ Ａｕｄｉｏ"



class DeSTA25AudioModel(PreTrainedModel):
    config_class = DeSTA25Config

    def __init__(self, config, cache_dir=None, token=None, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config

        token = token if token else os.getenv("HF_TOKEN")
        cache_dir = cache_dir if cache_dir else os.getenv("HF_HOME")

        self.audio_locator = config.audio_locator
        self.placeholder_token = config.placeholder_token

        print(f"Loading LLM model from {self.config.llm_model_id}")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            token=token,
        )

        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj"],
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config).base_model.model
        
        print(f"Loading Audio model from {self.config.encoder_model_id}")
        self.perception = WhisperPerception(self.config)

        self.configure_trainable_parameters()

    def forward(self, input_ids,
                attention_mask, 
                batch_features, 
                batch_transcription_ids,
                batch_start_positions,
                labels=None,
                **kwargs):
        
        inputs_embeds = self._prepare_inputs_for_llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            batch_features=batch_features,
            batch_transcription_ids=batch_transcription_ids, 
            batch_start_positions=batch_start_positions
        )


        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs 

    def _prepare_inputs_for_llm(self, 
                               input_ids,
                               attention_mask,
                               batch_features,
                               batch_transcription_ids,
                               batch_start_positions
        ):
        """
        Prepare the embeddings input for the LLM.
        Batch_features: list of audio features
        Batch_transcription_ids: list of transcription ids
        Batch_start_positions: list of start positions
        """

        N_audio = len(batch_start_positions)
        
        # Get list of transcription embeddings
        transcription_embeddings_list = []
        with torch.no_grad():
            for audio_batch_idx in range(N_audio):
                transcription_embeddings = self.llm_model.model.embed_tokens(
                    batch_transcription_ids[audio_batch_idx].squeeze(0)
                ) # (length, dim)
                transcription_embeddings_list.append(transcription_embeddings)

        # Forward speech encoder and connector
        # Get audio features from Qformer
        batch_audio_features, batch_audio_feature_lengths = self.perception(
            input_features=batch_features, transcription_embeddings_list=transcription_embeddings_list
        )

        assert len(batch_start_positions) == len(batch_transcription_ids) == batch_audio_features.size(0) == len(batch_audio_feature_lengths), "batch_start_positions, batch_transcription_ids, audio_features, speech_feature_lengths must have the same length."


        # [---- Other text embeddings ----][---- placeholder embeddings ----][---- Other text embeddings ----]
        inputs_embeds = self.llm_model.model.embed_tokens(input_ids)
        
        
        for audio_batch_idx in range(N_audio):
            start_position = batch_start_positions[audio_batch_idx] # tuple (text_idx, audio_start_position)
            text_batch_idx = start_position[0]
            audio_start_position = start_position[1]

            # get the speech features   
            audio_features = batch_audio_features[audio_batch_idx]
            speech_feature_length = batch_audio_feature_lengths[audio_batch_idx]

            # get transcription embeddings
            transcription_embeddings = transcription_embeddings_list[audio_batch_idx] # (length, dim)

            # # concat the speech features and transcription embeddings
            audio_embeddings = torch.cat([audio_features, transcription_embeddings], dim=0)

            assert audio_embeddings.size(0) == (speech_feature_length + transcription_embeddings.size(0))

            # # replace the input_embeds with the audio features
            # # [---- Other text embeddings ----][---- audio features + transcription embeddings ----][---- Other text embeddings ----]
            target_slice = slice(audio_start_position, audio_start_position + audio_embeddings.size(0))
            inputs_embeds[text_batch_idx, target_slice] = audio_embeddings
            


            if input_ids[text_batch_idx, audio_start_position-1] == 128096:
                # for debugging
                logging.warning(input_ids[text_batch_idx, audio_start_position-1: audio_start_position + audio_embeddings.size(0)+1])

            # clean GPU memory
            del audio_features, speech_feature_length, transcription_embeddings, audio_embeddings

        return inputs_embeds
        
    def state_dict(self):
        """
        Only return "trainable" parameters, since most of the parameters are frozen
        """
        trainable_state_dict = OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data.clone().detach()
        return trainable_state_dict


    def _generate_step(self, inputs, pad_token_id, temperature=0.7, top_p=0.9, max_new_tokens=512, do_sample=True):
        input_ids = inputs["context_input_ids"] # only context inputs
        attention_mask = inputs["context_attention_mask"] # only context attention mask
        batch_start_positions = inputs["context_batch_start_positions"]

        batch_transcription_ids = inputs["batch_transcription_ids"]
        # batch_audio_features, batch_audio_feature_lengths = self.perception()

        # get the generated text
        inputs_embeds = self._prepare_inputs_for_llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            batch_features=inputs["batch_features"],
            batch_transcription_ids=batch_transcription_ids, 
            batch_start_positions=batch_start_positions
        )

        if do_sample is False:
            top_p = None
            temperature = None
        
        generated_ids = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

        return generated_ids


    def configure_trainable_parameters(self):
        """
        for training, log the trainable parameters
        """

        known_parameters = []
        # Freeze LLM parameters
        for name, params in self.llm_model.named_parameters():
            params.requires_grad = False
            known_parameters.append(f"llm_model.{name}")

        # Freeze encoder parameters
        for name, params in self.perception.whisper.named_parameters():
            params.requires_grad = False
            known_parameters.append(f"perception.whisper.{name}")


        # Make other parameters or lora parameters trainable
        self.trainable_parameter_names = []
        trainable_parameters = []
        for name, params in self.named_parameters():
            if name not in known_parameters or "lora" in name:
                params.requires_grad = True
                self.trainable_parameter_names.append(name)
                trainable_parameters.append(params)



    def _setup_generation(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_id, cache_dir=os.getenv("HF_HOME"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        self.tokenizer.add_tokens([self.audio_locator])
        self.processor = AutoProcessor.from_pretrained(self.config.encoder_model_id, cache_dir=os.getenv("HF_HOME"))

        assert len(self.tokenizer.tokenize(self.audio_locator)) == 1, "audio_locator must be a single token"
        assert len(self.tokenizer.tokenize(self.placeholder_token)) == 1, "placeholder_token must be a single token in the tokenizer"

        # VAD
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (self.get_speech_timestamps, _, _, _, _) = utils


    def generate(self, messages,
        # LLM generation args
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        max_new_tokens=512,
        ):
        """
        messages = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions.",
            },
            {
                "role": "user",
                "content": "Hello! this is my audio <|AUDIO|>. Help me transcribe."
                "audios": [
                    "audio": "/path/to/filepath", # path to audio file
                    "text": None # Optional, None or provide text
                ]
            },
        ]
        """
        if not hasattr(self, "tokenizer"):
            self._setup_generation()

        if isinstance(messages, list):
            if isinstance(messages[0], dict):
                messages_list = [messages]
            else: 
                messages_list = messages
        else:
            raise ValueError("messages should be a list of dictionaries or a list of lists.")

        all_audios = []
        all_transcriptions = []
        for messages in messages_list:
            for message in messages:
                content = message["content"]
                audios = message.get("audios", [])
                assert len(audios) == content.count(self.audio_locator), "audio count does not match (<|AUDIO|>) count"

                for audio in audios:
                    all_audios.append(audio["audio"])
                    all_transcriptions.append(audio.get("text"))

        if len(all_audios) > 0:
            """
            If audios are provided, run:
            1. get features and transcription
            2. prepare LLM inputs
            3. run generation
            """

            batch_features = []
            asr_features = []
            asr_indices = []
            for i, (audio, trans) in enumerate(zip(all_audios, all_transcriptions)):
                if not os.path.exists(audio):
                    raise ValueError(f"Audio file {audio} does not exist.")

                # Extract audio features
                feature = AudioSegment.from_file(
                    audio,
                    target_sr=16000,
                    channel_selector="average"
                ).samples

                batch_features.append(feature)

                # Run VAD detect if there is speech in the audio
                is_speech = self.get_speech_timestamps(feature, self.vad_model)
                if is_speech and trans is None:
                    asr_features.append(feature)
                    asr_indices.append(i)
                if not is_speech:
                    all_transcriptions[i] = " "
            
            batch_features = self.processor(batch_features, sampling_rate=16000, return_tensors="pt").input_features
            batch_features = batch_features.to(self.device)
            audio_size_list = [self.config.prompt_size] * len(batch_features)


            # RUN ASR
            if asr_features:
                asr_features = self.processor(asr_features, sampling_rate=16000, return_tensors="pt").input_features
                asr_features = asr_features.to(self.device)

                transcriptions = self.perception.whisper.generate(
                    input_features=asr_features,
                    attention_mask=None,
                    max_new_tokens=128
                )
                transcriptions = self.processor.batch_decode(
                    transcriptions,
                    skip_special_tokens=True,
                )
            else:
                # no audio needs ASR result
                transcriptions = []

            
            for i, transcription in zip(asr_indices, transcriptions):
                all_transcriptions[i] = transcription.strip()
                    
            transcription_size_list = [
                len(self.tokenizer.tokenize(text, add_special_tokens=False)) for text in all_transcriptions
            ]


            audio_context_list = []
            start_positions_list = []
            for messages in messages_list:
                audio_context = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # <start_audio><|AUDIO|><end_audio> is a indicator used in the training stage
                # We replace <|AUDIO|> with <start_audio><|AUDIO|><end_audio> here
                audio_context = audio_context.replace(self.audio_locator, f"<start_audio>{self.audio_locator}<end_audio>")

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


            audio_context_inputs = self.tokenizer(
                audio_context_list,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_length=True,
                add_special_tokens=False,
            )

            audio_context_batch_start_positions = []
            for i in range(audio_context_inputs["length"].size(0)):
                total_length = audio_context_inputs["length"][i]
                pad_length = total_length - audio_context_inputs["attention_mask"][i].sum()

                for start_position in start_positions_list[i]:
                    audio_context_batch_start_positions.append((i, start_position + pad_length))

            batch_transcription_ids = []
            for transcription in all_transcriptions:
                batch_transcription_ids.append(
                    self.tokenizer.encode(transcription, add_special_tokens=False, return_tensors="pt").long().to(self.device)
                )

            inputs = {
                "batch_features": batch_features,
                "batch_transcription_ids": batch_transcription_ids,

                "context_input_ids": audio_context_inputs["input_ids"],
                "context_attention_mask": audio_context_inputs['attention_mask'],
                "context_batch_start_positions": audio_context_batch_start_positions,
            }
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            generated_ids = self._generate_step(
                inputs, 
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample)

            return GenerationOutput(
                text=self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
                audios=[(a, t) for a,t in zip(all_audios, all_transcriptions)],
                generated_ids=generated_ids.tolist()
            )

        else:
            """
            if no audios are provided, it's identical to the original LLM generation
            """

            inputs = self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            generated_ids = self.llm_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                eos_token_id=terminators,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
            )

            generated_ids_list = []
            for i in range(len(generated_ids)):
                generated_ids_list.append(generated_ids[i][inputs["input_ids"].shape[1]:].tolist())

            return GenerationOutput(
                text=self.tokenizer.batch_decode(generated_ids_list, skip_special_tokens=True),
                audios=[],
                generated_ids=generated_ids_list
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Custom from_pretrained method to load pretrained LLM and Whisper model.
        model.safetensors only contains trainable parameters from DeSTA2.5-Audio.
        """
        
        cache_dir = kwargs.get("cache_dir", os.getenv("HF_HOME"))

        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
        model = cls(config)
        
        if os.path.isdir(pretrained_model_name_or_path):
            model.load_state_dict(
                load_file(os.path.join(pretrained_model_name_or_path, "model.safetensors")), strict=False
            )
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="model.safetensors", cache_dir=cache_dir)
            model.load_state_dict(
                load_file(path), strict=False
            )

        return model