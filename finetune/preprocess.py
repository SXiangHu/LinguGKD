import token
import numpy as np
from datasets import concatenate_datasets
from man_utils import load_json, save_json, load_pkl, save_pkl, args, get_max_length, save_jsonl, load_jsonl
from os import path
from tqdm import tqdm
import re
import random
from collections import defaultdict
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import swifter
import json


def _per_func(instruction, input, target, center_node, task_id, mode='train', tokenizer=None, **kwargs):
    assert tokenizer is not None
    if mode == 'train':
        return f'{tokenizer.bos_token}### Instruction: {instruction}\n### Input: {input}\n### Response: {target}{tokenizer.eos_token}'
    elif mode == 'test':
        return {'instruction': f'{tokenizer.bos_token}### Instruction: {instruction}\n### Input: {input}\n### Response: ',
                'label': f'{target}',
                'center_node': center_node,
                'task_id': task_id}


def get_format_instruction(tokenizer, mode='train', enhance=[]):
    def format_instruction(samples, mode=mode, enhance=enhance, tokenizer=tokenizer):
        if isinstance(samples, list):
            samples_df = pd.DataFrame(samples)
            instructions = samples_df.swifter.progress_bar(
                False).apply(lambda x: _per_func(**x, tokenizer=tokenizer), axis=1).tolist()
            return instructions
        if isinstance(samples, dict): 
            return _per_func(samples['instruction'], samples['input'], samples['target'], samples['center_node'], samples['task_id'], mode=mode, tokenizer=tokenizer)
        
        instructions = []
        print('how are you epoch?')
        for inst, inputs, target, task_type, center_node, task_id in zip(samples['instruction'], samples['input'], samples['target'], samples['task_type'], samples['center_node'], samples['task_id']):
            if mode == 'test' and task_type == 'link':
                continue

            if len(enhance) > 0:
                if task_id.split('-')[2] not in enhance:
                    continue

            instructions.append(
                _per_func(inst, inputs, target, center_node, task_id, mode=mode, tokenizer=tokenizer))
        return instructions
    return format_instruction

def get_format_instruction_all(mode='train', enhance=[]):
    def format_instruction(samples, mode=mode, enhance=enhance):
        instructions = []
        for inst, inputs, target, task_type, center_node, task_id in zip(samples['instruction'], samples['input'], samples['target'], samples['task_type'], samples['center_node'], samples['task_id']):
            if mode == 'test' and task_type == 'link':
                continue
            if len(enhance) > 0:
                if task_id.split('-')[2] not in enhance:
                    continue
            instructions.append(_per_func(inst, inputs, target, center_node, task_id, mode=mode))
        return instructions
    return format_instruction

def format_instruction(samples, tokenizer, mode='train', enhance=[]):
    instructions = []
    for inst, inputs, target, task_type, center_node, task_id in zip(samples['instruction'], samples['input'], samples['target'], samples['task_type'], samples['center_node'], samples['task_id']):
        if mode == 'test' and task_type == 'link':
            continue
        if len(enhance) > 0:
            if task_id.split('-')[2] not in enhance:
                continue
        instructions.append(_per_func(inst, inputs, target, center_node, task_id, mode=mode, tokenizer=tokenizer))
    return instructions


class LLMPreprocessor(object):
    def __init__(self, tokenizer, dataset) -> None:
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_s_length, self.max_t_length = self._get_max_length(
            dataset, tokenizer)

    def _get_max_length(self, dataset, tokenizer):
        max_source_length = get_max_length(dataset, tokenizer, 'input')
        print(f"Max source length: {max_source_length}")

        max_target_length = get_max_length(dataset, tokenizer, 'target')
        print(f"Max target length: {max_target_length}")
        return max_source_length, max_target_length

    def preprocess_function(self, sample, padding="max_length"):
        model_inputs = self.tokenizer(
            text_target=sample['input'], max_length=self.max_s_length, padding=padding, truncation=True)

        labels = self.tokenizer(
            text_target=sample["target"], max_length=self.max_t_length, padding=padding, truncation=True)

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
