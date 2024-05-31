import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.gnn import GNN, GNNPreMP
from torch_geometric.graphgym.models.layer import GeneralMultiLayer, new_layer_config

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.PReLU())
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.Dropout(cfg.gnn.dropout))
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

@register_network('dis_gnn') # type: ignore
class DistillGNN(GNN):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """
        Knowledge Distillation GNN model.
        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        GNNStage = register.stage_dict[cfg.gnn.stage_type]

        self.layer_norm = nn.LayerNorm(cfg.gnn.dim_inner)
        self.layers = nn.ModuleList()
        self.low_dim_projection = nn.Sequential(
            nn.Linear(cfg.gnn.dim_llm, input_dim),
            nn.PReLU()
        )

        if cfg.distill.mode:
            self.distill_loss_weights = nn.Parameter(
                torch.ones(cfg.gnn.layers_mp + 1), requires_grad=True)
            self.loss_weights = nn.Parameter(torch.ones(2), requires_grad=True)

        input_dim = self.encoder.dim_in
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                input_dim, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            self.layers.append(self.pre_mp)

        if cfg.gnn.layers_post_mp > 0:
            self.fc = GeneralMultiLayer(
                'linear',
                layer_config=new_layer_config(
                    cfg.gnn.dim_inner,
                    cfg.gnn.dim_inner,
                    cfg.gnn.layers_post_mp,
                    has_act=True,
                    has_bias=True,
                    cfg=cfg,
                ),
            )

        if cfg.gnn.layers_mp > 0:
            for _ in range(cfg.gnn.layers_mp):
                self.layers.append(
                    GNNStage(dim_in=cfg.gnn.dim_inner, dim_out=cfg.gnn.dim_inner, num_layers=1))

        self.projectors = None
        if cfg.distill.mode:
            self.projectors_ = nn.ModuleList()
            for i in range(cfg.gnn.layers_mp + 1):
                self.projectors_.append(MLP(
                    cfg.gnn.dim_llm, [cfg.gnn.dim_inner] * cfg.gnn.filter_layer, cfg.gnn.dim_inner))
            self.llm_transform = nn.Sequential()
            for i in range(cfg.gnn.projector_layer):
                self.llm_transform.add_module(f'linear_{i}', nn.Linear(
                    cfg.gnn.dim_inner, cfg.gnn.dim_inner, bias=False))
        self.apply(init_weights)


    def _index_split(self, batch, hop):
        """
        Splits the index for knowledge distillation.
        Args:
            batch (Batch): Batch of data.
            hop (int): Hop index.
        Returns:
            Tuple[Tensor, Tensor]: Predicted and true hidden states.
        """
        hidden_state_pred = batch.x
        if 'llm_feature' not in batch:
            raise Exception('llm_feature not in batch')
        if self.projectors_ is not None:
            hidden_state_true = self.llm_transform(
                self.projectors_[hop](batch.llm_feature[hop]))
        else:
            hidden_state_true = self.llm_transform(batch.llm_feature[hop])
        if 'split' not in batch:
            return hidden_state_pred, hidden_state_true
        mask = batch[f'{batch.split}_mask']
        return hidden_state_pred[mask].unsqueeze(dim=0), hidden_state_true[mask].unsqueeze(dim=0)

    def forward(self, batch):
        """
        Forward pass through the DistillGNN model.
        Args:
            batch (Batch): Batch of data.
        Returns:
            Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]: Predicted and true hidden states, and labels.
        """
        hidden_state_pred, hidden_state_true = None, None
        for i, mp in enumerate(self.layers):
            batch = mp(batch)
            if cfg.distill.mode:
                latent_pred_layer, latent_true_layer = self._index_split(
                    batch, i)
                if hidden_state_pred is None:
                    hidden_state_pred = latent_pred_layer
                    hidden_state_true = latent_true_layer
                else:
                    hidden_state_pred = torch.cat(
                        [hidden_state_pred, latent_pred_layer], dim=0)
                    hidden_state_true = torch.cat(
                        [hidden_state_true, latent_true_layer], dim=0) # type: ignore
        batch = self.fc(batch)
        pred, label = self.post_mp(batch)
        return (pred, hidden_state_pred), (label, hidden_state_true)
