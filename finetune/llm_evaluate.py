import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
from cgi import test
from man_utils import load_pkl, get_max_length, save_json, init_wandb, save_pkl, save_jsonl
from transformers import AutoTokenizer, GenerationConfig
from datasets import load_dataset
from peft import PeftModel
from loader.llm_loader import prepare_llm_model, prepare_tokenizer
from os import path
from preprocess import format_instruction, get_dpo_line
import numpy as np
import time
from tqdm import tqdm
from typing import List
import wandb
from datasets import concatenate_datasets
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_first_not_empty_list(lists):
    for item in lists:
        if len(item) != 0:
            return item
    return []


def cast_feature(map_count, mode='full'):
    node_2_del = []
    final_key = 'llm_features_full' if mode == 'full' else 'llm_features_correct'
    
    for node, item in map_count.items():
        feature_list = [np.array(item['features'][str(hop)]).squeeze() if mode == 'full' else np.array(item['features'][str(hop)])[item['corrects'][str(hop)]].squeeze() for hop in range(4)]
        for hop in range(4):
            features = feature_list[hop]
            hop = str(hop)
            if len(features) == 0:
                if hop == '0':
                    feature_to_cast = get_first_not_empty_list(feature_list[1:])
                    if len(feature_to_cast) > 0:
                        features = feature_to_cast
                    else:
                        if mode == 'full':
                            node_2_del.append(node)
                        break
                else:
                    map_count[node][final_key][hop] = map_count[node][final_key][str(
                        int(hop)-1)]
                    continue
            if len(features.shape) > 1:
                map_count[node][final_key][hop] = torch.mean(
                    torch.tensor(features), dim=0)
            else:
                map_count[node][final_key][hop] = features
    
    if len(node_2_del) > 0:
        print('deleting nodes: ', node_2_del)
        for node in node_2_del:
            del map_count[node]
    return map_count


def cal_acc_hop(map_count):
    acc_hop = {0: 0, 1: 0, 2: 0, 3: 0}
    for node, item in map_count.items():
        for hop in range(4):
            hop = str(hop)
            if sum(item['corrects'][hop]) >= 1:
                acc_hop[int(hop)] += 1
    acc_hop = {key: f'{value / len(map_count):.4f}' for key, value in acc_hop.items()}
    return acc_hop

def cal_acc_macro(map_count):
    correct = [1 if item['count'] > 0 else 0 for item in map_count.values()]
    accuracy = np.mean(correct)
    return '{:4f}'.format(accuracy)

def cal_acc_micro(map_count):
    correct, total = 0, 0.0001
    for item in map_count.values():
        for v in item['corrects'].values():
            correct += sum(v)
            total += len(v)
    return '{:.4f}'.format(correct / total)

def readout(predictions, references):
    preds = defaultdict(dict)
    labels = {}
    temps  = defaultdict(lambda: defaultdict(list))
    for pred_dic, item in zip(predictions, references):
        reference = item['label']
        node = item['center_node']
        task_id = item['task_id']
        hop = task_id.split('-')[2]
        pred = pred_dic['text']
        
        labels[node] = reference
        temps[node]['pred'].append(pred)
        temps[node]['hop'].append(int(hop))

    
    for node, item in temps.items():
        for hop in set(item['hop']):
            pred_vals = [pred for pred, hop_ in zip(
                item['pred'], item['hop']) if hop_ <= hop] 
            pred_vals = sorted(pred_vals, key=pred_vals.count,
                               reverse=True) 
            pred_label = pred_vals[0] 
            preds[hop][node] = pred_label
    
    return preds, labels

def cal_multi_acc(predictions:dict, labels: dict):
    accs = {}
    for hop, preds in predictions.items():
        correct = 0
        for node, pred in preds.items():
            if labels[node].lower() in pred.lower():
                correct += 1
        # print(f'hop: {hop}, acc: {correct / len(labels):.4f}')
        accs[hop] = '{:4f}'.format(correct / len(labels))
    return accs


def compute_metrics(predictions: List[str], references: List[str], llm_features, map_count):

    dpo_data = []
    preds, labels = readout(predictions, references)
    acc_readout = cal_multi_acc(preds, labels)
    
    inference_times = []

    task_count = defaultdict(lambda: {'total': 0.0001, 'correct': 0})
    
    for prediction, item, llm_feature in tqdm(zip(predictions, references, llm_features), desc='Computing metrics'):
        reference = item['label']
        node = item['center_node']
        task_id = item['task_id']
        hop = task_id.split('-')[2]
        task_count[hop]['total'] += 1

        is_correct = False
        if reference.lower() in prediction['text'].lower():
            is_correct = True
            map_count[node]['count'] += 1
            map_count[node]['instruction'][hop].append(item)
            task_count[hop]['correct'] += 1
        else:
            # print(f"reference: {reference}, prediction: {prediction['text']}")
            dpo_data.append(get_dpo_line(item['instruction'], reference, prediction['text']))

        map_count[node]['features'][hop].append(llm_feature)
        map_count[node]['corrects'][hop].append(is_correct)

        inference_times.append(prediction['time'])
    
    average_time = sum(inference_times) / len(inference_times)
    correct = [1 if item['count'] > 0 else 0 for item in map_count.values()]
    accuracy = np.mean(correct)
    accuracy_micro = np.sum([count['correct'] for count in task_count.values()]) / len(references)
    accuracy_hop_micro = {
        hop: f"{count['correct'] / count['total']:.4f}" for hop, count in task_count.items()}
    acc_hop = cal_acc_hop(map_count)
    
    return {"accuracy": accuracy, "accuracy_micro": accuracy_micro, "accuracy_readout": acc_readout, "accuracy_hop": acc_hop, "accuracy_hop_micro": accuracy_hop_micro, "average_inference_time": average_time}, task_count, dpo_data

def evaluate(args, split, splits, model=None):
    # init_wandb(args)
    args.logger.info(f"Starting evaluation, run name: {args.run_name}, data name: {args.dataset_name}, split: {split}, splits: {splits}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        f'{args.output_path}/results')
    tokenizer.padding_side = 'left'
    
    if model is None:
        model = prepare_llm_model(args.llm_model, tokenizer, mode='test')
        model = PeftModel.from_pretrained(model, f"{args.output_path}/results")

    data_files = {'train': 'train_gen.jsonl',
                  'test': 'test_gen.jsonl'}
    # Load dataset
    datasets = load_dataset(
        path.join(args.language_data_dir, args.dataset_name), data_files=data_files, split=split)

    all_dataset = format_instruction(datasets, tokenizer, mode='test')

    dataset_len = len(all_dataset)
    split_num = dataset_len // splits


    new_max_tokens = get_max_length(datasets, tokenizer, 'target')

    with torch.no_grad():
        task_count = defaultdict(lambda: {'total': 0.0001, 'correct': 0})
        map_count = defaultdict(
            lambda: {'count': 0, 'features': defaultdict(list), 'instruction': defaultdict(list), 'corrects': defaultdict(list), 'llm_features_full': {}, 'llm_features_correct': {}})
        
        for i in range(splits):
            test_dataset = all_dataset[i*split_num: (i+1)*split_num]
            test_dataset = MyDataset(test_dataset)
            dataloader = DataLoader(
                test_dataset, batch_size=1)
            predictions = []
            llm_features = []
            references = []
            
            for item in tqdm(dataloader, desc=f'Generating {i+1}/{splits}'): 
                start_time = time.time()
                
                inputs = tokenizer.batch_encode_plus(
                    item['instruction'], return_tensors='pt', padding=True, truncation=False).to(model.device)
                outputs = model.generate(
                    **inputs, max_new_tokens=new_max_tokens+10, output_hidden_states=True, output_attentions=False, return_dict_in_generate=True)
                    # **inputs, max_new_tokens=new_max_tokens+10, output_hidden_states=True, output_attentions=False, return_dict_in_generate=True, temperature=0.1, do_sample=True)
                hidden_states = outputs.hidden_states[0][-1][:, -1, :].cpu().detach().float()
                llm_features.extend(list(hidden_states))

                end_time = time.time()
                output_sentence = outputs['sequences']
                generated_texts = tokenizer.batch_decode(
                    output_sentence[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                inference_time = end_time - start_time

                predictions.extend([{'text': generated_text, 'time': inference_time} for generated_text in generated_texts])
                references.extend([{key: value[i] if key != 'center_node' else int(value[i])  for key, value in item.items()} for i in range(len(item['instruction']))])

            results, task_count_batch, dpo_data = compute_metrics(predictions, references, llm_features, map_count)
            args.logger.info(f'round: {i+1}/{splits}, results: {results}')
            
            for key, value in task_count_batch.items():
                task_count[key]['total'] += value['total']
                task_count[key]['correct'] += value['correct']
            save_pkl(
                dict(map_count), f'{args.output_path}/results/map_count_{i+1}_{splits}_{split}.pkl')
            if path.exists(f'{args.output_path}/results/map_count_{i}_{splits}_{split}.pkl'):
                os.remove(f'{args.output_path}/results/map_count_{i}_{splits}_{split}.pkl')
                
        map_count_full = cast_feature(map_count, mode='full')
        map_count_correct = cast_feature(map_count_full, mode='correct') 
        save_pkl(dict(map_count_correct), f'{args.output_path}/results/llm_features_{split}.pkl')
        args.logger.info(f'final map_count saved to {args.output_path}/results/llm_features_{split}.pkl')
        save_jsonl(dpo_data, f'{args.output_path}/results/{split}_{args.postfix}_dpo.jsonl')
        return model


if __name__ == '__main__':
    run_name = ''
    data_name = ''
    model_name = ''
    split = 'test'
    splits = 1
    is_dpo = False
    args = load_pkl(
        f'./log/llm/{data_name}/{model_name}/{"dpo/" if is_dpo else ""}{run_name}/args.pkl')
    model = evaluate(args, split, splits)

