import argparse
from numpy import save
from world import configs
import time
import json
import pickle
import os
from os import path
import logging
import numpy as np
import math
import wandb
import jsonlines

# Load arguments from the command line


class Timer():
    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        if duration < 60:
            logging.info(f'Time cost: {duration:.2f} s')
        elif duration < 3600:
            logging.info(f'Time cost: {duration/60:.2f} min')
        else:
            logging.info(f'Time cost: {duration/3600:.2f} h')


def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(data)

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = jsonlines.Reader(f)
        data = list(reader)
        return data


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=True)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_logger(args):
    logging_path = path.join(args.logging_dir, 'log')

    logging.basicConfig(filename=logging_path,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger('llm_logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 清除已存在的处理器
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler and set level to debug
    # fh = logging.FileHandler(path.join(args.logging_dir, 'log'))
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    return logger

def init_wandb(args):
    wandb.init(project=args.project, name=args.run_name, notes=args.remark,
               config=args, settings=wandb.Settings(start_method="fork"), id=args.now, resume="allow")

def load_args():
    parser = argparse.ArgumentParser()
    for section in configs:
        if configs[section]:
            for key in configs[section]:
                parser.add_argument(f'--{key}', **configs[section][key])

    args = parser.parse_args()
    if args.now is None:
        args.now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # args.now = '2023-12-27-16-02-15'
    args.postfix = 'with_id' if args.has_id else 'without_id'
    args.postfix = f"{'old_' if args.is_old else ''}{args.postfix}"

    if args.is_dpo:
        assert bool(args.lora_id) and args.dataset_name == args.lora_dataset
        args.logging_dir = f"{args.log_root}/llm/{args.dataset_name}/{args.llm_model.split('/')[-1]}/{args.lora_id}/dpo"
        args.output_path = f"{args.output_dir}/{args.dataset_name}/{args.llm_model.split('/')[-1]}/{args.lora_id}/dpo"
        args.dataset_root = f"{args.output_dir}/{args.dataset_name}/{args.llm_model.split('/')[-1]}/{args.lora_id}/results"
    else:
        args.logging_dir = f"{args.log_root}/llm/{args.dataset_name}/{args.llm_model.split('/')[-1]}/{args.now}"
        args.output_path = f"{args.output_dir}/{args.dataset_name}/{args.llm_model.split('/')[-1]}/{args.now}"
    args.run_name = f"{args.dataset_name}_{args.llm_model}_{args.now}{'_dpo' if args.is_dpo else ''}"
    if not path.exists(args.logging_dir):
        os.makedirs(args.logging_dir, exist_ok=True)

    save_json(vars(args), path.join(args.logging_dir, 'args.json'))
    args.logger = create_logger(args)
    save_pkl(args, path.join(args.logging_dir, 'args.pkl'))
    return args


args = load_args()


def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_max_length(dataset, tokenizer, filed=None, percentage=100) -> int:
    """ 获取数据集中最大长度
    Args:
        dataset: 数据集
        tokenizer: 分词器
        filed: 数据集中的字段
        percentage: 百分位，默认为 100, 即获取所有数据中的最大长度
    Returns:
        max_length: 最大长度
    """
    tokenized_inputs = dataset.map(lambda x: tokenizer(x if filed is None else x[filed], truncation=True), batched=True)
    input_lengths = [len(x) for x in tokenized_inputs['input_ids']]
    max_length = math.ceil(np.percentile(input_lengths, percentage))
    return max_length