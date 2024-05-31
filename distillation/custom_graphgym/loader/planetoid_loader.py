from torch_geometric.datasets import Planetoid
from torch_geometric.graphgym.register import register_loader
from custom_graphgym.transform.preprocess import get_preprocess_func
from custom_graphgym.loader.dataset.distill_dataset import DistillData
from torch_geometric.graphgym.config import cfg


@register_loader('planetoid_loader')
def load_dataset_example(format, name, dataset_dir):
    if name in ['Cora', 'PubMed', 'Arxiv'] and format == 'llmPyG':
        dataset_dir = f'{dataset_dir}'
        dataset = DistillData(root=dataset_dir, name=name)
        pre_transform = get_preprocess_func(
            f'{dataset_dir}/{name}/processed', f'{cfg.distill.llm_dir}/results')
        dataset = pre_transform(dataset)
        return dataset
