import logging
import time
import wandb

import torch

from torch_geometric.graphgym.checkpoint import (
    clean_ckpt,
    load_ckpt,
    save_ckpt,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.train import GraphGymDataModule
from typing import Optional, Dict, Any, List, Tuple
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.imports import pl
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def distill_train(
    model: GraphGymModule,
    datamodule: GraphGymDataModule,
    trainer_config: Optional[Dict[str, Any]] = None,
):
    r"""Trains a GraphGym model using PyTorch Lightning.

    Args:
        model (GraphGymModule): The GraphGym model.
        datamodule (GraphGymDataModule): The GraphGym data module.
    """
    callbacks = []
    callbacks.append(LoggerCallback())
    if cfg.train.early_stopping:
        callbacks.append(EarlyStopping(monitor='val/accuracy', patience=50, mode='max'))
    if cfg.train.enable_ckpt:
        ckpt_cbk = ModelCheckpoint(
            dirpath=get_ckpt_dir(), monitor='val/accuracy', mode='max')
        callbacks.append(ckpt_cbk)


    trainer_config = trainer_config if trainer_config is not None else {}
    trainer = pl.Trainer(
        **trainer_config,
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices='auto' if not torch.cuda.is_available() else cfg.devices
    )
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)