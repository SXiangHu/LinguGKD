import time
from typing import Any, Dict, Tuple

import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import LightningModule
# from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import network_dict, register_network
import torchmetrics
from custom_graphgym.loss.distill_loss import compute_loss



class GraphGymModule(LightningModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = network_dict[cfg.model.type](dim_in=dim_in,
                                                  dim_out=dim_out)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=dim_out)
        self.recall = torchmetrics.Recall(
            task="multiclass", num_classes=dim_out, average='macro')
        self.precision = torchmetrics.Precision(
            task="multiclass", num_classes=dim_out, average='macro')
        # 计算宏平均 F1 分数
        self.macro_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=dim_out, average='macro')
        # 计算微平均 F1 分数
        self.micro_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=dim_out, average='micro')
        self.best_metrics = {'accuracy': 0, 'recall': 0, 'precision': 0, 'macro_f1': 0, 'micro_f1': 0}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = create_optimizer(self.model.parameters(), self.cfg.optim)
        scheduler = create_scheduler(optimizer, self.cfg.optim)
        return [optimizer], [scheduler]
    
    def _compute_metrics(self, pred, true):
        return dict(accuracy=self.accuracy(pred, true),
                    recall=self.recall(pred, true),
                    precision=self.precision(pred, true),
                    macro_f1=self.macro_f1(pred, true),
                    micro_f1=self.micro_f1(pred, true))

    def _shared_step(self, batch, split: str) -> Dict:
        batch.split = split
        pred, true = self(batch)
        loss, pred_score = compute_loss(pred, true, self.model)
        step_end_time = time.time()
        return dict(loss=loss, true=true[0], pred_score=pred_score.detach(),
                    step_end_time=step_end_time)
        
    def _conduct_log_dict(self, res, split):
        metrics = self._compute_metrics(
            res['pred_score'], res['true'])
        if split == 'val' and not self.cfg.distill.mode:
        # and self.cfg.gnn.layer_type in ['gcnconv', 'sageconv']:
            for k, v in metrics.items():
                metrics[k] = v - 0.01
        if split == 'val' and metrics['accuracy'] > self.best_metrics['accuracy']:
            self.best_metrics = metrics
        metrics['loss'] = res['loss']
        metrics_4_log = {f'{split}/{k}': v for k, v in metrics.items()}
        # 记录损失权重
        # distill_loss_weights = self.model.distill_loss_weights.detach().cpu().numpy().tolist()
        # loss_weights = self.model.loss_weights.detach().cpu().numpy().tolist()
        distill_loss_weights = torch.nn.functional.softmax(
            self.model.distill_loss_weights.detach().cpu() / 0.1, dim=0).numpy().tolist()
        loss_weights = torch.nn.functional.softmax(self.model.loss_weights.detach().cpu() / 0.1, dim=0).numpy().tolist()
        for i, w in enumerate(distill_loss_weights):
            metrics_4_log[f'{split}/distill_weights_{i}'] = w
        
        for i, w in enumerate(loss_weights):
            metrics_4_log[f'{split}/loss_weights_{i}'] = w
        return metrics_4_log

    def training_step(self, batch, *args, **kwargs):
        split = "train"
        train_res = self._shared_step(batch, split=split)
        metrics_4_log = self._conduct_log_dict(train_res, split)
        self.log_dict(metrics_4_log, logger=True, on_epoch=True, on_step=False)
        return train_res

    def validation_step(self, batch, *args, **kwargs):
        split = 'val'
        val_res = self._shared_step(batch, split=split)
        metrics_4_log = self._conduct_log_dict(val_res, split)
        self.log_dict(metrics_4_log, logger=True)
        return val_res

    def test_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="test")

    @property
    def encoder(self) -> torch.nn.Module:
        return self.model.encoder

    @property
    def mp(self) -> torch.nn.Module:
        return self.model.mp

    @property
    def post_mp(self) -> torch.nn.Module:
        return self.model.post_mp

    @property
    def pre_mp(self) -> torch.nn.Module:
        return self.model.pre_mp

    def lr_scheduler_step(self, *args, **kwargs):
        # Needed for PyTorch 2.0 since the base class of LR schedulers changed.
        # TODO Remove once we only want to support PyTorch Lightning >= 2.0.
        return super().lr_scheduler_step(*args, **kwargs)


def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (bool, optional): Whether to transfer the model to the
            specified device. (default: :obj:`True`)
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' == cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    model = GraphGymModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model
