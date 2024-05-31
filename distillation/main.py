import os

import custom_graphgym
import logging
from regex import P
import wandb
  # noqa, register custom modules
import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.graphgym.config import (
    cfg,
    set_cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from custom_graphgym.config.config import optim_cf
from custom_graphgym.train.distill_train import distill_train
from torch_geometric.graphgym.logger import set_printing
from man_model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
import json


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    optim_cf(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        set_printing()
        # Set configurations for each run
        seed_everything(cfg.seed)

        auto_select_device()
        # Set machine learning pipeline
        datamodule = GraphGymDataModule()
        
        model = create_model()
        if cfg.train.from_pretrain:
            # Load pre-trained model
            sd1 = model.state_dict()
            sd2 = torch.load(cfg.train.from_pretrain)['state_dict']
            for k,v in sd2.items():
                if ('model.mp' in k or ('model.layers' in k and not 'model.layers.0' in k)) and (k in sd1):
                    sd1[k] = v
            model.load_state_dict(sd1)
            print()
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        if cfg.wandb:
            wandb_logger = WandbLogger(
                project=cfg.project, name=f'{cfg.out_dir}_{cfg.seed}', config=cfg, resume="auto", id=f'{cfg.now}_{cfg.seed}')
        else:
            wandb_logger = None
        trainer_config = {
            'logger': wandb_logger,
            'log_every_n_steps': 1
        }
        distill_train(model, datamodule, trainer_config=trainer_config)
        cfg.seed = cfg.seed + 1
        if wandb_logger is not None:
            wandb_logger.experiment.log(
                {f'final/{k}': v for k, v in model.best_metrics.items()})
            wandb_logger.experiment.finish()

    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    