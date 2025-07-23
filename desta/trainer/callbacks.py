from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
from desta.utils.utils import run
import logging
class HuggingfaceCallback(Callback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.exp_dir = cfg.exp_dir

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        pass

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        try:
            pl_module.model.config.commit = run("git rev-parse HEAD")
            pl_module.model.save_pretrained(f"{self.exp_dir}/hf_models/epoch-{pl_module.current_epoch}")
            pl_module.model.config.save_pretrained(f"{self.exp_dir}/hf_models/epoch-{pl_module.current_epoch}")
            pl_module.tokenizer.save_pretrained(f"{self.exp_dir}/hf_models/epoch-{pl_module.current_epoch}")

            self.on_train_epoch_end(trainer, pl_module)
        except Exception as e:
            logging.error(f"Error saving to Hugging Face: {e}")
        
        