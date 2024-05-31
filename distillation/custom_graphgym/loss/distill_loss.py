import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_loss


def compute_loss(pred, true, model=None):
    """
    Compute loss and prediction score

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Ground truth values
        model: The model being trained (optional)

    Returns:
        tuple: Loss and normalized prediction score
    """
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    mse_loss = torch.nn.MSELoss(reduction=cfg.model.size_average)

    if cfg.model.loss_fun == 'cross_entropy' or cfg.model.loss_fun == 'mse':
        pred = pred.squeeze(-1) if pred.ndim > 1 else pred
        true = true.squeeze(-1) if true.ndim > 1 else true
    else:
        for func in register.loss_dict.values():
            value = func(pred, true, model)
            if value is not None:
                return value

    if cfg.model.loss_fun == 'cross_entropy':
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        true = true.float()
        return mse_loss(pred, true), pred
    else:
        raise ValueError(f"Loss function '{cfg.model.loss_fun}' not supported")


def infoNCE_loss(hidden_states, llm_features, labels, temperature=0.1, model=None):
    """
    Compute the InfoNCE loss for knowledge distillation.

    Args:
        hidden_states (torch.tensor): Hidden states from the student model
        llm_features (torch.tensor): Features from the teacher model
        labels (torch.tensor): Ground truth labels
        temperature (float): Temperature scaling for the logits
        model: The model being trained (optional)

    Returns:
        torch.tensor: Computed InfoNCE loss
    """
    # Normalize the feature vectors
    hidden_states = F.normalize(hidden_states, dim=-1)
    llm_features = F.normalize(llm_features, dim=-1)

    # Compute similarity matrix
    similarity_matrix = torch.bmm(
        hidden_states, llm_features.transpose(-1, -2)) / temperature

    mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    positives_mask = torch.zeros_like(mask)
    positives_mask.fill_diagonal_(1)
    negatives_mask = ~mask
    invalid_mask = ~(positives_mask + negatives_mask)
    logits = similarity_matrix.masked_fill(invalid_mask, float('-inf'))
    label = torch.arange(logits.size(1)).to(logits.device)
    loss = 0
    loss_weights = F.softmax(model.distill_loss_weights, dim=0)
    for i in range(similarity_matrix.size(0)):
        loss += loss_weights[i] * F.cross_entropy(logits[i], label)

    return loss


@register_loss('distill_loss')
def distill_loss(pred, true, model=None):
    """
    Compute the distillation loss combining classification loss and mapping loss.

    Args:
        pred (tuple): Tuple containing predicted logits and hidden states
        true (tuple): Tuple containing true labels and hidden states from the teacher model
        model: The model being trained (optional)

    Returns:
        tuple: Total loss and logits prediction
    """
    logits_pred = pred[0]
    hidden_state_pred = pred[1]
    labels = true[0]
    llm_hidden_state = true[1]

    if logits_pred.ndim > 1 and labels.ndim == 1:
        logits_pred = F.log_softmax(logits_pred, dim=-1)
        classify_loss = F.nll_loss(logits_pred, labels)
    else:
        labels = labels.float()
        classify_loss = F.binary_cross_entropy_with_logits(logits_pred, labels)

    if cfg.distill.mode:
        map_loss = infoNCE_loss(hidden_state_pred, llm_hidden_state,
                                labels, temperature=cfg.model.distill_temp, model=model)
    else:
        map_loss = 0

    if cfg.distill.mode:
        weights = F.softmax(model.loss_weights, dim=0) # type: ignore
        total_loss = weights[0] * classify_loss + weights[1] * map_loss
    else:
        total_loss = classify_loss

    return total_loss, logits_pred
