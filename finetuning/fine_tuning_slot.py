import os
import sys
from utils.slot_acc import ValueRouge,Acc
import evaluate
import torch
import argparse
import json
import logging
import re
import math
import os
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq, 
    get_scheduler,
    SchedulerType,
    AutoConfig,
    T5Tokenizer,
    T5ForConditionalGeneration
)
import wandb
wandb.login(key='')
#1.25e-3
lr_rate = 1.25e-3
wandb.init(
    project="slot",
    config={
        'evaluation_strategy':'epoch',
        'lr':lr_rate,
    },
    name='0602-1.25e-3'
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='data/construct_for_hm_label_only_slot_12.json',
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default='data/hm/hm-1000_slot_with_prompt.jsonl',
    )
    parser.add_argument("--output_dir",
                         type=str, default='models/t5-ft/0602')
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=192
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=192
        )
    
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default='models/t5-cn'
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=lr_rate ,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_train_epochs", 
                        type=int, 
                        default=15, 
                        )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None
    )

    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
   
    args = parser.parse_args()
    # Sanity checks
    if args.dataset_path is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    return args

def main():
    class myTokenizer():
        #用于rouge的分词,默认的nltk分词在中文上表现不佳
        def __init__(self):
            pass
        def __call__(self, text):
            return self.tokenize(text)
        def tokenize(self, text):
            result = []
            for i in text.split(' '):
                result.append(i)
            return result
        
    rouge_tokenizer = myTokenizer()

    args = parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        
    log_file_name = os.path.join(args.output_dir, "fine_tuning.log")
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
        filemode='a',
        filename=log_file_name,
        #encoding='utf-8
        )
    if args.seed is not None:
        set_seed(args.seed)

    # load dataset
    if args.dataset_path is not None:     
        raw_datasets = load_dataset(args.dataset_path.split(".")[-1], data_files=args.dataset_path)
        raw_datasets = raw_datasets['train'].train_test_split(train_size=0.8,test_size=0.2,seed=42)
        raw_datasets['validation'] = raw_datasets['test']
        del raw_datasets['test']
    test_dataset = load_dataset('json', data_files={'test':args.test_dataset_path})
    raw_datasets['test'] = test_dataset['test']
       
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path,config=config)
    model.to('cuda')
    
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    text_column = 'sentence'
    summary_column = 'label'
    column_names = ['sentence','label','sentence_id']
    max_target_length = args.max_target_length
    padding = 'longest'

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = [ str(i) for i in   examples[summary_column] ]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    train_dataset = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    # Temporarily set max_target_length for validation.
    max_target_length = args.val_max_target_length
    eval_dataset = raw_datasets["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    test_dataset = raw_datasets["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=['sentence','label',],
        desc="Running tokenizer on dataset",
    )    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logging.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if  args.mixed_precision=='fp16'  else None,
    )

    def postprocess_text(preds, labels):
        # 处理以便于使用指标计算,便于评估.
        post_preds = [' '.join(tokenizer.tokenize(pred.strip())  ) for pred in preds]
        post_labels = [' '.join(tokenizer.tokenize(label.strip())  ) for label in labels]
        return post_preds, post_labels
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # 测试集没有11.医学检测
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    num_update_steps_per_epoch = len(train_dataloader) 
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    lr_scheduler = get_scheduler(
        num_training_steps=args.max_train_steps,
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0
    )

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Metric
    metric = evaluate.load("hf-evaluate/rouge")
    
    metric_value_rouge = ValueRouge(tokenizer=tokenizer,use_num=True)
    metric_acc = Acc(use_num=True)
    
    # Train!
    total_batch_size = args.per_device_train_batch_size 

    logging.info("***** Running training *****")
    logging.info(f"  Num train examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {args.num_train_epochs}")

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            batch.to('cuda')
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            progress_bar.update(1)
            completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

        print(' epoch:{}, total:{}，loss:{}'.format(epoch,args.num_train_epochs,total_loss))
        model.eval()
        gen_kwargs = {
            "max_length": args.val_max_target_length,
            "num_beams": args.num_beams,
        }
        eval_progress_bar = tqdm(range(len(eval_dataloader)))
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch.to('cuda')
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                labels = batch["labels"]
                # If we did not pad to max length, we need to pad the labels too
                # 实际上用的是longest模式
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                # 接受分词前的结果.
                metric_acc.add_batch(decoded_preds,decoded_labels)
                metric_value_rouge.add_batch(predicts=decoded_preds,references=decoded_labels)

                # 先用t5的分词器分词,然后再用自定义的分词器分词.
                post_process_decoded_preds, post_process_decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(
                    predictions=post_process_decoded_preds,
                    references=post_process_decoded_labels,
                )
                eval_progress_bar.update(1)
                
        result = metric.compute(use_stemmer=False,
                                tokenizer=rouge_tokenizer,)
        result_acc = metric_acc.compute()
        result_value_rouge = metric_value_rouge.compute()
        
        result = {k: round(v , 4) for k, v in result.items()}
        result["train_loss"] = total_loss.item() / len(train_dataloader)
        result["epoch"] = epoch
        result["step"] = completed_steps
        for k,v in result_acc.items():
            result['acc_'+k] = v
        for k,v in result_value_rouge.items():
            result['rouge_'+k] = v
        print(result)
        logging.info(result)
        wandb.log(result)
        # 测试集
        test_progress_bar = tqdm(range(len(test_dataloader)))
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                batch.to('cuda')
                # 为什么这里使用generate方法
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                labels = batch["labels"]
                # If we did not pad to max length, we need to pad the labels too
                # 实际上用的是longest模式
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                # 接受分词前的结果.
                # 这里处理一下,因为测试集中没有11.医学检测
                ptn = re.compile(r'11:*?\s')
                decoded_labels = [re.sub(ptn,'',label) for label in decoded_labels]
                metric_acc.add_batch(decoded_preds,decoded_labels)
                metric_value_rouge.add_batch(predicts=decoded_preds,references=decoded_labels)

                # 先用t5的分词器分词,然后再用自定义的分词器分词.
                post_process_decoded_preds, post_process_decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(
                    predictions=post_process_decoded_preds,
                    references=post_process_decoded_labels,
                )
                test_progress_bar.update(1)
        result = metric.compute(use_stemmer=False,
                                tokenizer=rouge_tokenizer,)
        result_acc = metric_acc.compute()
        result_value_rouge = metric_value_rouge.compute()
        for k,v in result_acc.items():
            result['acc_'+k] = v
        for k,v in result_value_rouge.items():
            result['rouge_'+k] = v
        result = { 'test_'+ k : v for k,v in result.items()}
        
        result["epoch"] = epoch
        result["step"] = completed_steps    
            
        logging.info(result)
        wandb.log(result)
        print(result)
        
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)    
                
    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        all_results = {f"eval_{k}": v for k, v in result.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)
    wandb.finish()

if __name__ == "__main__":
    main()

