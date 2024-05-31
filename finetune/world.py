import ast


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

configs = {
    'general': {
        'project': {
            'type': str,
            'default': 'LLM4GKD',
            'help': 'The name of the project.'
        },
        'is_dpo': {
            'type': str2bool,
            'default': False,
            'help': 'Whether to use DPO.'
        },
        'remark': {
            'type': str,
            'default': '',
            'help': 'The remark of the experiment.'
        },
        'seed': {
            'type': int,
            'default': 42,
            'help': 'The random seed to use.'
        },
        'reproducibility': {
            'type': bool,
            'default': True,
            'help': 'Whether to make the experiment reproducible.'
        },
        'output_dir': {
            'type': str,
            'default': './saved',
            'help': 'The directory to store the output.'
        },
        'log_root': {
            'type': str,
            'default': './log',
            'help': 'The root directory to store the log.'
        },
        'device': {
            'type': str,
            'default': '1',
            'help': 'The device to use.'
        },
        'overwrite': {
            'type': bool,
            'default': True,
            'help': 'Whether to overwrite the output directory.'
        },
        'now': {
            'type': str,
            'default': None,
            'help': 'The current time.'
        },
    },
    'dataset': {
        'graph_data_dir': {
            'type': str,
            'default': './dataset/graph',
            'help': 'The directory to store the graph data.'
        },
        'has_id': {
            'type': str2bool,
            'default': True,
            'help': 'Whether the dataset has node id.'
        },
        'is_old': {
            'type': str2bool,
            'default': True,
        },
        'language_data_dir': {
            'type': str,
            'default': './dataset',
            'help': 'The directory to store the graph-language data.'
        },
        'template_dir': {
            'type': str,
            'default': './instruction_template',
            'help': 'The directory to store the instruction templates.'
        },
        'instruction_list': {
            'type': list,
            'default': ['node_classification'],
            'help': 'The list of instruction types.'
        },
        'dataset_name': {
            'type': str,
            'default': 'Cora',
            'help': 'The name of the dataset to use (via the datasets library).'
        },
        'split_map': {
            'type': list,
            'default': [0.6, 0.2, 0.2],
            'help': 'The ratio of dataset split.'
        },
        'hops': {
            'type': int,
            'default': 2,
            'help': 'The number of hops to use for graph sampling.'
        }
    },
    'training': {
        'epochs': {
            'type': int,
            'default': 3,
            'help': 'The number of training epochs.'
        },
        'lora_id': {
            'type': str,
            'default': None,
            'help': 'The id of the LoRA adaptor.'
        },
        'enhance': {
            'type': ast.literal_eval,
            'default': [],
            'help': 'The list of enhanced hops.'
        },
        'lora_dataset': {
            'type': str,
            'default': 'Cora',
            'help': 'The dataset of the LoRA adaptor.'
        },
        'is_complete_only': {
            # 'type': bool,
            # 'default': True,
            'help': 'Whether to only use completations for training.',
            'action': 'store_true'
        },
        'early_stopping_patience': {
            'type': int,
            'default': 10,
            'help': 'The number of epochs to wait before early stopping.'
        },
        'resume': {
            # 'type': bool,
            # 'default': False,
            'action': 'store_true',
            'help': 'Whether to resume training.'
        },
        'max_steps': {
            'type': int,
            'default': 500,
            'help': 'The maximum number of training steps.'
        },
        'train_batch_size': {
            'type': int,
            'default': 4,
            'help': 'The batch size.'
        },
        'learning_rate': {
            'type': float,
            'default': 0.0001,
            'help': 'The learning rate.'
        },
        'stopping_step': {
            'type': int,
            'default': 10,
            'help': 'The number of epochs to wait before early stopping.'
        },
        'weight_decay': {
            'type': float,
            'default': 0.0,
            'help': 'The weight decay.'
        },
        'learner': {
            'type': str,
            'default': 'adam',
            'help': 'The learner to use.'
        },
        'max_seq_length': {
            'type': int,
            'default': 2100,
            'help': 'The maximum sequence length.'
        },
        'grad_steps': {
            'type': int,
            'default': 1,
            'help': 'The number of gradient accumulation steps.'
        },
    },
    'model': {
        'llm_model': {
            'type': str,
            'default': 'meta-llama/Llama-2-7b-hf',
            'help': 'The model to use.'
        },
        'is_new_token': {
            'action': 'store_true',
            'help': 'Whether to add new tokens.'
        }
    },
    'evaluation': {
        'eval_args': {
            'type': dict,
            'default': {
                'group_by': 'user',
                'order': 'RO',
                'split': {'RS': [0.8, 0.1, 0.1]},
            },
            'help': 'The arguments to pass to the evaluation function.'
        },
        'metrics': {
            'type': list,
            'default': ['Recall', 'Precision'],
            'help': 'The metrics to use for evaluation.'
        },
        'valid_metric': {
            'type': str,
            'default': 'Recall',
            'help': 'The metric to use for early stopping.'
        },
    }
}
