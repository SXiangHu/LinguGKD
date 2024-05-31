from os import path
import os
import time

from torch_geometric.graphgym.config import cfg, get_fname, makedirs_rm_exist
from torch_geometric.graphgym.register import register_config


def optim_cf(out_dir, fname):
    fname = get_fname(fname)
    cfg.out_dir = os.path.join(
        out_dir, f'{fname}/{cfg.gnn.layer_type}/hops_{cfg.gnn.layers_mp}/{cfg.now}')
    if cfg.train.auto_resume:
        os.makedirs(cfg.out_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.out_dir)

    if not cfg.distill.mode:
        cfg.gnn.batchnorm = ~cfg.gnn.batchnorm
        cfg.gnn.l2norm = ~cfg.gnn.l2norm
        

@register_config('general_config')
def set_general_cfg(cfg):
    cfg.now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    cfg.project = 'GKD_LLM'
    cfg.remark = ''
    
    cfg.train.early_stopping = False
    cfg.wandb = False
    cfg.seed = 42
    cfg.accelerator = 'cuda'
    cfg.dataset.format = 'llmPyG'
    cfg.dataset.task = 'node'
    cfg.dataset.task_type = 'classification'
    cfg.dataset.transductive = True
    cfg.dataset.split = [0.6, 0.2, 0.2]
    cfg.dataset.to_undirected = True
    cfg.dataset.transform = None
    cfg.gnn.graph_pooling = 'add'
    cfg.gnn.edge_decoding = 'dot'
    cfg.gnn.stage_type = 'stack'
    cfg.gnn.act = 'prelu'
    cfg.gnn.dropout = 0.2
    cfg.gnn.agg = 'add'
