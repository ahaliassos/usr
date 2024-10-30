import os
import tempfile

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.memory import ModelSummary
import torch.distributed
import torch

from .utils import write_to_txt


class StartEndCallBack(Callback):
    def __init__(self, wandb=True):
        super(StartEndCallBack, self).__init__()
        self.wandb = wandb

    def on_train_start(self, trainer, pl_module):
        if torch.cuda.device_count() <= 1 or torch.torch.distributed.get_rank() == 0:
            pl_module.print("Training has started!")
            model_name = "model_summary_" + next(tempfile._get_candidate_names()) + ".txt"
            write_to_txt([ModelSummary(pl_module, mode="full")], os.path.join("/tmp", model_name))
            if self.wandb:
                pl_module.print(
                    "Initialized wandb. Run can be tracked on " + pl_module.logger.experiment.get_url() + "."
                )
                pl_module.logger.experiment.save(os.path.join("/tmp", model_name), base_path="/tmp")

    def on_train_end(self, trainer, pl_module):
        pl_module.print("Training is done!")


class SwitchStagesCallBack(Callback):
    def __init__(self, patience=10, max_epochs=150, rollback=False):
        super(SwitchStagesCallBack, self).__init__()

        self.patience, self.max_epochs = patience, max_epochs
        self.min_val, self.count_consecutive_best = float("inf"), 0

        self.rollback, self.best_state = rollback, None

    def on_validation_epoch_end(self, trainer, pl_module):
        loss_v2a = trainer.callback_metrics["loss_v2a_val_pre"]
        loss_a2v = trainer.callback_metrics["loss_a2v_val_pre"]

        cur_loss = (loss_v2a + loss_a2v) / 2
        if cur_loss < self.min_val:
            self.min_val = cur_loss
            self.count_consecutive_best = 0
            self.best_state = pl_module.state_dict()
        else:
            if not self.rollback:
                self.best_state = pl_module.state_dict()
            self.count_consecutive_best += 1
            if self.count_consecutive_best == self.patience:
                trainer.should_stop = True  # stop training

        if trainer.current_epoch >= self.max_epochs:
            trainer.should_stop = True
