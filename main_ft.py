import logging
import os

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from data.data_module_ft import DataModule
from learner_ft import SSLLearner
from utils.utils import average_checkpoints

# static vars
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("lightning").propagate = False
# __spec__ = None


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    if cfg.fix_seed:
        seed_everything(42, workers=True)

    print("The SLURM job ID for this run is {}".format(os.environ["SLURM_JOB_ID"]))
    cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]

    cfg.gpus = torch.cuda.device_count()
    print('num gpus:', cfg.gpus)

    wandb_logger = None
    if cfg.log_wandb:
        wandb_logger = instantiate(cfg.logger)
    
    torch.set_float32_matmul_precision(precision=cfg.matmul_precision)

    data_module = DataModule(cfg)
    learner = SSLLearner(cfg)

    ckpt_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        dirpath=os.path.join(cfg.checkpoint.dirpath, cfg.experiment_name) if cfg.checkpoint.dirpath else None,
        save_last=True,
        filename=f'{{epoch}}',
        save_top_k=cfg.checkpoint.save_top_k,
    )
    callbacks = []
    if cfg.log_wandb:
        callbacks = [
            ckpt_callback,
            LearningRateMonitor(logging_interval=cfg.logging.logging_interval),
        ]
    trainer = Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    if cfg.test:
        trainer.test(learner, datamodule=data_module)
    else:
        if not cfg.test_avg:
            trainer.fit(learner, data_module, ckpt_path=cfg.ckpt_path)

            torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            last = [
                os.path.join(
                    cfg.checkpoint.dirpath, cfg.experiment_name, f"epoch={n}.ckpt"
                ) for n in range(trainer.max_epochs - cfg.model.avg_ckpts, trainer.max_epochs)
            ]
            avg = average_checkpoints(last)

            model_path = os.path.join(
                cfg.checkpoint.dirpath, cfg.experiment_name, f"model_avg_{cfg.model.avg_ckpts}.pth"
            )
            torch.save(avg, model_path)

            # compute WER
            cfg.gpus = cfg.trainer.devices = cfg.trainer.num_nodes = 1
            cfg.model.pretrained_model_path = model_path
            cfg.model.transfer_only_encoder = False
            data_module = DataModule(cfg)
            learner = SSLLearner(cfg)
            trainer = Trainer(**cfg.trainer, logger=wandb_logger)
            trainer.test(learner, datamodule=data_module)

    
if __name__ == "__main__":
    main()
