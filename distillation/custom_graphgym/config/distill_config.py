from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('distill')
def set_cfg_distill(cfg):
    r"""This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    """
    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument
    # cfg.example_arg = 'example'

    # example argument group
    cfg.distill = CN()

    # then argument can be specified within the group
    cfg.distill.mode = True
    cfg.distill.llm_dir = ''
    
    cfg.model.distill_temp = 0.07
    cfg.train.from_pretrain = ''
    cfg.gnn.filter_layer = 1
    cfg.gnn.projector_layer = 1
    cfg.gnn.dim_llm = 4096
    cfg.gnn.batchnorm = True
    cfg.gnn.l2norm = True
    cfg.gnn.normalize_adj = True
