import subprocess
from man_utils import args, init_wandb, load_jsonl
import os
# Load arguments
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from datasets import load_dataset
from os import path
from loader.llm_loader import prepare_tokenizer, prepare_llm_model, prepare_peft_model, merge_model
import numpy as np
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from preprocess import get_format_instruction_all, get_format_instruction
from trainer import MySFTTrainer, DynamicTrainDataset
import sys
from llm_evaluate import evaluate
import torch

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def compute_metrics_text_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        predictions = np.where(predictions != -100,
                               predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True)
        preds = list(map(lambda x: x[-40:].strip(), decoded_preds))
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        targets = list(map(lambda x: x.split('\n')[-1].strip(), decoded_labels))
        
        acc_list = [targets[i].lower() in preds[i].lower() and targets[i] != '' for i in range(len(targets))]
        acc = np.mean(acc_list)

        return {'accuracy': acc}

    return compute_metrics


if __name__ == '__main__':
    os.chdir(path.dirname(__file__))
    logger = args.logger
    init_wandb(args)
    
    logger.info(args)
    logger.info(f"Start loading dataset {args.dataset_name}...")
    
    logger.info(f"Loading tokenizer {args.llm_model}...")
    tokenizer, new_tokens = prepare_tokenizer()
    logger.info(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
    
    formatting_func = get_format_instruction(enhance=args.enhance, tokenizer=tokenizer)
    data_files = {'test': f'test_gen_{args.postfix}.jsonl'}
    
    # Load dataset
    train_dataset = load_jsonl(path.join(args.language_data_dir, args.dataset_name, 'train.jsonl'))
    train_dataset = DynamicTrainDataset(train_dataset, tokenizer, args.max_seq_length, require_process=False)
    
    test_dataset = load_jsonl(path.join(args.language_data_dir, args.dataset_name, 'test.jsonl'))
    test_dataset = DynamicTrainDataset(test_dataset, tokenizer, args.max_seq_length, require_process=False)
    streaming = False

    logger.info("Dataset loaded successfully!")
    if not streaming:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        small_train_dataset = train_dataset.select(10)
        logger.info(f"Small train dataset size: {len(small_train_dataset)}")
    else:
        small_train_dataset = train_dataset.shuffle(42).take(10)
        small_eval_dataset = test_dataset.shuffle(42).take(10000)

    # Load model and tokenier    
    logger.info(f"Loading LLM model {args.llm_model}...")
    model = prepare_llm_model(args.llm_model, tokenizer, new_tokens)
    logger.info("Model loaded successfully!")
    
    logger.info("Adding LoRA adaptor...")

    target_modules = ["q_proj", "k_proj", "v_proj"
                      "o_proj", "lm_head"]

    _, lora_config = prepare_peft_model(model, target_modules=target_modules, config_only=True)

    # Define training args
    eval_strategy = "steps"
    print(len(train_dataset))
    eval_steps = max(3000, (len(train_dataset) // 3 // args.train_batch_size // args.grad_steps // 1000) * 1000)
    logger.info(f'eval_steps: {eval_steps}')
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_eval_batch_size=args.train_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate, 
        gradient_accumulation_steps=args.grad_steps,
        num_train_epochs=args.epochs,
        logging_dir=args.logging_dir,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy=eval_strategy,
        save_total_limit=5,
        report_to=["tensorboard", "wandb"],
        evaluation_strategy=eval_strategy ,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        seed=args.seed,
        bf16=True,
        fp16=False, 
        run_name=args.run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy" if args.dataset_name != 'All' else "eval_loss",
    )

    callbacks = [
        EarlyStoppingCallback(args.early_stopping_patience),
    ] if args.dataset_name != 'All' and args.early_stopping_patience > 0 else []
    # Create Trainer instance
    instruction_template = "### Instruction:"
    # We added context here: "\n". This is enough for this tokenizer
    response_template_with_context = "\n### Response:"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template, response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
    
    trainer = MySFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        peft_config=lora_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # train_dataset=small_train_dataset,
        formatting_func=formatting_func,
        data_collator=collator if args.is_complete_only else None,
        max_seq_length=args.max_seq_length,
        callbacks=callbacks,
        neftune_noise_alpha=5,  # “NEFTune: Noisy Embeddings Improve Instruction Finetuning”
        compute_metrics=compute_metrics_text_aux(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    logger.info("Start training...")
    # train model
    trainer.train(resume_from_checkpoint=args.resume)

    # Save our LoRA model & tokenizer results
    peft_model_id = path.join(args.output_path, "results")
    logger.info(f"Saving model to {peft_model_id}")
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    
    # Free memory
    # del model
    torch.cuda.empty_cache()
    
    # evalute model
    model = evaluate(args, 'test', splits=1)
    evaluate(args, 'train', splits=1, model=model)
