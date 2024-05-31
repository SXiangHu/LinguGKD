from os import path
import os

from man_utils import save_json, load_json, args
from tqdm import tqdm
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, LlamaForCausalLM
from model import MLP
from preprocess import LLMPreprocessor
from datasets import load_from_disk
from sklearn.decomposition import PCA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from typing import List, Dict, Optional, Union
import torch
from peft import PeftModel, AutoPeftModelForCausalLM



def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    # if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer, ''


def prepare_llm_model(llm_model, tokenizer=None, new_tokens: list = None, mode='train'):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_model, device_map='auto', attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, quantization_config=bnb_config)
    model.config.use_cache = False
    return model


def prepare_inputs(dataset, tokenizer):
    corpus_root = path.join(args.language_data_dir, args.dataset_name)
    tokenized_data_path = path.join(corpus_root, 'tokenized_data')
    llm_prepocessor = LLMPreprocessor(tokenizer, dataset)
    if not args.overwrite and path.exists(tokenized_data_path):
        tokenized_dataset = load_from_disk(tokenized_data_path)
    else:

        tokenized_dataset = dataset.map(
            llm_prepocessor.preprocess_function, batched=True, remove_columns=["task_type", "task_id", "center_node"])
        print(
            f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
        # save datasets to disk for later easy loading
        tokenized_dataset.save_to_disk(tokenized_data_path)
    tokenized_dataset = tokenized_dataset.shuffle(seed=args.seed)
    return tokenized_dataset, llm_prepocessor

# 合并 hugging face 预训练模型和微调后的 lora 权重
def merge_model(lora_path, is_trainable=False, is_merge=True, model_name=None, tokenizer=None, adapter_name='train'):
    if is_merge:
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_path, device_map={'': 'cpu'}, torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
    else:
        # 加载预训练模型
        model = prepare_llm_model(model_name, tokenizer)
        # 加载 lora 权重
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=is_trainable)
    return model
    

def prepare_peft_model(model, target_modules: list = None, r=64, lora_alpha=32, lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM, config_only=False):
    # Define LoRA Config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type
    )
    # prepare int-8 model for training
    if not config_only:
        # add LoRA adaptor
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model = prepare_model_for_kbit_training(model)

    return model, lora_config

    
