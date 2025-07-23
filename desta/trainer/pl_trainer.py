import pytorch_lightning as pl
import torch
from desta.utils.utils import run
from apex.optimizers import FusedAdam
from desta.models.modeling_desta25 import DeSTA25AudioModel, DeSTA25Config
from transformers import get_cosine_schedule_with_warmup
import logging
from desta.trainer.data.simple_dataset import BaseAudioTextDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoFeatureExtractor
import os
import json
from omegaconf import OmegaConf
from collections import OrderedDict, defaultdict
from pathlib import Path
from lulutils import get_unique_filepath
from desta.utils.metrics import ConsecutiveWordsAccuracyMetric

class DeSTA25AudioPLModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        
        # PTL
        self.automatic_optimization = True

        model_config = DeSTA25Config(
            llm_model_id=self.cfg.model.llm.model_id,
            encoder_model_id=self.cfg.model.encoder.model_id,
            connector_mode=self.cfg.model.connector.mode,
            qformer_num_hidden_layers=self.cfg.model.connector.num_hidden_layers,
            prompt_size=self.cfg.model.connector.prompt_size,
            use_lora=self.cfg.model.llm.use_lora if hasattr(self.cfg.model.llm, "use_lora") else False,
            audio_locator=self.cfg.model.audio_locator,
            placeholder_token=self.cfg.model.placeholder_token,
        )

        print("="*100)
        self.model = DeSTA25AudioModel(model_config)
        self.model.config.train_id = 30678
        self.model.config.trainer_version = "ea4ad585d7d50d1bbd191d64d194c4bc5eabd537b6bcf97e6510f784e4cb2f0f"

        # remove whisper decoder during PTL training (we only use Whisper decoder during inference)
        del self.model.perception.whisper.model.decoder
        del self.model.perception.whisper.proj_out

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.llm.model_id, cache_dir=os.getenv("HF_HOME"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_tokens([self.cfg.model.audio_locator])
        
        self.processor = AutoFeatureExtractor.from_pretrained(self.cfg.model.encoder.model_id, cache_dir=os.getenv("HF_HOME"))

        self.metrics = ConsecutiveWordsAccuracyMetric()
        
        self.prediction_step_outputs = []

        self.model.config.commit = run("git rev-parse HEAD")

    def forward(self, batch):
        return self.model(**batch)

    
    def on_train_epoch_begin(self):
        logging.info(self.model.dtype)

    # Training / Validation / Prediction
    def training_step(self, batch, batch_idx):
        self.model.train()
        
        outputs = self(batch)
        loss = outputs.loss
        
        perplexity = torch.exp(loss)
        batch_size = batch["input_ids"].size(0)
        self.log("train/loss", loss, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)
        self.log("train/ppl", perplexity, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)
        
        return loss


    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss = 0
        perplexity = 0
        predictions = []

        outputs = self(batch)
        loss = outputs.loss
        perplexity = torch.exp(loss)

        predictions = self.predict_step(batch, batch_idx)

        batch_size = batch["input_ids"].size(0)
        
        self.log("val/loss", loss.item(), sync_dist=True, batch_size=batch_size)
        self.log("val/ppl", perplexity.item(), sync_dist=True, batch_size=batch_size)


        return {"val/loss": loss, "val/ppl": perplexity, "predictions": predictions}


    def predict_step(self, batch, batch_idx):
        self.model.eval()
        generated_ids = self.model._generate_step(batch, 
                                                  pad_token_id=self.tokenizer.eos_token_id,
                                                  temperature=self.cfg.model.generation_kwargs.temperature,
                                                  top_p=self.cfg.model.generation_kwargs.top_p,
                                                  max_new_tokens=self.cfg.model.generation_kwargs.max_new_tokens,
                                                  do_sample=self.cfg.model.generation_kwargs.do_sample,
                                                  ) # batched


        batch["context_input_ids"][batch["context_input_ids"] == -100] = self.tokenizer.eos_token_id
        batch["labels"][batch["labels"] == -100] = self.tokenizer.eos_token_id
        generated_ids[generated_ids == -100] = self.tokenizer.eos_token_id

        contexts = self.tokenizer.batch_decode(batch["context_input_ids"], skip_special_tokens=False)
        labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Record predictions
        for context, label, pred, metadata in zip(contexts, labels, preds, batch["metadata"]):
            metadata.update({
                "context": context,
                "prediction": pred,
                "label": label,
            })
            self.prediction_step_outputs.append(metadata)

        return {"loss": 0}
    
    def on_validation_epoch_begin(self):
        pass

    def on_validation_epoch_end(self):
        dataset_name = "val"
        os.makedirs(f"{self.cfg.exp_dir}/results/{dataset_name}", exist_ok=True)
        output_path = f"{self.cfg.exp_dir}/results/{dataset_name}/val@ep={self.trainer.current_epoch}-{self.trainer.global_step}-rank={self.trainer.global_rank}.jsonl"

        report = self.write_to_file(
            results=self.prediction_step_outputs,
            filepath=output_path, 
            cfg=self.cfg,
            ckpt=f"ep={self.trainer.current_epoch}-{self.trainer.global_step}",
            write_report=True
        )

        self.log("val/accuracy", report["accuracy_by_sample"], sync_dist=True)
        self.model.config.validation_id = 128000
        self.prediction_step_outputs.clear()



    # Dataloader
    def _build_dataloader(self, data_cfg):
        dataset = BaseAudioTextDataset(
            cfg=self.cfg,
            data_cfg=data_cfg,
            tokenizer=self.tokenizer,
            processor=self.processor
        )
        logging.info(dataset[0])
        
        dataloader = DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=data_cfg.shuffle,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            drop_last=data_cfg.drop_last,
        )
        return dataloader


    def train_dataloader(self):
        data_cfg = self.cfg.dataset.train_ds
        logging.info("\n********************* Training dataset *********************\n")
        
        dataloader = self._build_dataloader(data_cfg)
        
        logging.info("\n***************** End of Training dataset *****************\n")
        return dataloader

    def val_dataloader(self):
        data_cfg = self.cfg.dataset.validation_ds
        logging.info("\n******************** Validation dataset ********************\n")
        dataloader = self._build_dataloader(data_cfg)
        logging.info("\n**************** End of Validation dataset ****************\n")
        return dataloader

        

    # Optimizer and scheduler
    def configure_optimizers(self):
        
        trainable_parameters = []
        for name, params in self.model.named_parameters():
            if name in self.model.trainable_parameter_names:
                trainable_parameters.append(params)

        optimizer = FusedAdam(trainable_parameters, 
                              lr=self.cfg.optim.lr,
                              betas=(self.cfg.optim.betas),
                              weight_decay=self.cfg.optim.weight_decay,
                              )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.optim.sched.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        for name in self.model.trainable_parameter_names:
            logging.info(f"Training parameter: {name}")
        
        logging.info(f"Optimizer: {optimizer}")
        logging.info(f"Scheduler: {scheduler}")
        logging.info(f"{run('git rev-parse HEAD')}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val/loss",
            },
        }
    
    def state_dict(self):
        """
        save only trainable parameters. 
        Note: There are two types of state_dict()
        - This state_dict() will be called by PTL Trainer.save_checkpoint() for saving checkpoints
        - DeSTA3Model.state_dict() will be called by huggingface PreTrainedModel.save_pretrained() for saving model weights and can be easily loaded by DeSTA3Model.from_pretrained()
        
        The only difference is the prefix(model.) in the state_dict keys.
        """
        logging.info(f"{run('git rev-parse HEAD')}")
        trainable_state_dict = OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data.clone().detach()

        return trainable_state_dict
    

    def write_to_file(self, results, filepath, cfg=None, ckpt=None, write_report=True):
        filepath = Path(filepath)
        
        categories_accuracy = defaultdict(list)
        
        jsonl_path = Path(get_unique_filepath(filepath.parent / "preds" / filepath.name))
        os.makedirs(jsonl_path.parent, exist_ok=True)

        with open(jsonl_path, "w") as f:
            for i, result in enumerate(results):
                result["correct"] = self.metrics(result["prediction"], result["label"])
                result["index"] = i
                f.write(json.dumps(result) + "\n")
                categories_accuracy[result.get("category", "all")].append(result["correct"])

        if write_report:
            # Report
            report_path = jsonl_path.parent.parent / jsonl_path.name.replace(".jsonl", "-report.json")

            reported_results = []
            for i, result in enumerate(results):
                # remove context and audio_context for better readability (too long!)
                del result["context"]
                del result["audio_context"]
                reported_results.append(result)

            report = {
                "metric": self.metrics.metric_name,
                "preds_path": str(jsonl_path),
                "accuracy_by_sample": sum([reported_results[i]["correct"] for i in range(len(reported_results))]) / len(reported_results),
                "avg_accuracy_by_category": sum([sum(v) / len(v) for v in categories_accuracy.values()]),
                "categories_accuracy": dict([ (k, sum(v) / len(v)) for k, v in categories_accuracy.items()]),
                "ckpt": str(ckpt),
                "results": reported_results,
                "exp_dir": self.cfg.exp_dir,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "commit": run("git rev-parse HEAD"),
                "name": "DeSTA2.5-Audio",
            }
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logging.info(f"Report saved to\n{report_path}\n")
            print(f"Report saved to\n{report_path}\n")

            return report
