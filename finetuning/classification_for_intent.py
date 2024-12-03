import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys

from torch.utils.data import DataLoader
import torch 
import datasets
import transformers
import evaluate
from tqdm import tqdm
import numpy as np
from utils.slot_acc import IntentMetric
import copy
datasets.disable_caching()
import wandb
wandb.login(key='')
model_path = ''
project_path = ''
if not os.path.exists(project_path):
    os.makedirs(project_path)
dataset_path = 'data/intent_construct_for_hm_label_12_2_2_add2k_no_question.json'
test_dataset_path = 'data/hm/hm-1000_intent.jsonl'
args = {
        'output_dir':project_path,
        'evaluation_strategy':'epoch',
        'num_train_epochs':25,
        'learning_rate':1e-5,
        'save_strategy':'no',
        'do_train':True,
        'do_eval':True,
    }
wandb.init(
    project="intent-classification-DeBERTa-v2-97M-Chinese-singel-label",
    config=args,
    name='0602-cly-add2k'
)


metric = evaluate.combine(["hf-evaluate/metrics/f1", 'hf-evaluate/metrics/recall','hf-evaluate/metrics/precision','hf-evaluate/metrics/accuracy'])
dataset = datasets.load_dataset('json', data_files=dataset_path)
dataset = dataset['train'].train_test_split(test_size=0.2,shuffle=True,seed=21)
dataset['val'] = dataset['test']
del dataset['test']
test_dataset = datasets.load_dataset('json', data_files={'test':test_dataset_path})
dataset['test'] = test_dataset['test']

id2label_str = {
        "0": "ask_symptom_duration",
        "1": "ask_symptom_description_color",
        "2": "ask_symptom_description_smell",
        "3": "ask_symptom_description_degree",
        "4": "ask_symptom_description_position",
        "5": "ask_symptom_description_shape",
        "6": "ask_symptom_description_time",
        "7": "ask_coitus",
        "8": "ask_tobacco",
        "9": "ask_alcohol",
        "10": "ask_drug",
        "11": "ask_receiving_treatment",
        "12": "ask_age",
        "13": "ask_weight",
        "14": "ask_gender",
        "15": "ask_symptom",
        "16": "ask_medical_examination",
        "17": "ask_living_habit",
        "18": "ask_past_medication",
        "19": "ask_past_diagonsis",
        "20": "ask_physical_change",
        "21": "ask_trauma_surgery",
        "22": "ask_preventive",
        "23": "ask_allergic_history",
        "24": "ask_contact_history",
        "25": "ask_family_history"
    }
id2label = {int(k):v for k,v in id2label_str.items()}
label2id = {v:k for k,v in id2label.items()}
num_labels = len(label2id)
intent_metric = IntentMetric(id2label)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def token_function(example):
        result = tokenizer(example['sentence'],max_length=256,truncation=True,padding='max_length')
        labels = [0.0]*num_labels
        for label in example['label']:
            labels[label2id[label]] = 1.0
        result['labels'] = labels
        return result
def test_datset_function(example): 
    result = tokenizer(example['sentence'],max_length=256,truncation=True,padding='max_length')
        # 处理一下test的label,这里使用ask_symptom_description_color视为详情
    labels = [0.0]*num_labels
    for label in example['label']:
        labels[label2id[label]] = 1.0
    label_temp = labels[1:7]
    if 1.0 in label_temp:
        labels[1] = 1.0
        labels[2:7] = [0.0]*5
    else:
        labels[1:7] = [0.0]*6
    result['labels'] = labels
    return result
    
    

def compute_metrics(eval_pred,is_test=False):
    if not is_test:
        predictions, labels = eval_pred
        predictions = sigmoid(predictions)
        predictions = (predictions > 0.5).astype(int)
        labels = labels.astype(int)
        intent_metric.add_batch(predictions, labels)
        result = intent_metric.compute()
        # result = metric.compute(predictions=predictions, references=labels.astype(int).reshape(-1))
        return result
    else:
        predictions, labels = eval_pred
        predictions = sigmoid(predictions)
        predictions = (predictions > 0.5).astype(int)
        labels = labels.astype(int)
        # 这里处理一下,不比较详情了, 将第一个视为详情试试
        new_ps = []
        for p in predictions:
            new_p = copy.deepcopy(p)
            if 1.0 in p[1:7]:
                new_p[1] = 1.0
                new_p[2:7] = [0]*5
            else:
                new_p[1:7] = [0]*6
            new_ps.append(new_p)
        predictions = np.array(new_ps)
        intent_metric.add_batch(predictions, labels)
        result = intent_metric.compute()
        return result
    
if __name__ == '__main__':
    tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
    config = transformers.AutoConfig.from_pretrained(model_path,
                                            num_labels=num_labels,
                                            finetuning_task="text-classification",
                                            problem_type="multi_label_classification",
                                            id2label=id2label,
                                            label2id=label2id,
                                            )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                    config=config,
                                                                    )
    train_dataset = dataset['train'].map(token_function,batched=False,remove_columns=['sentence','label'])
    val_dataset = dataset['val'].map(token_function,batched=False,remove_columns=['sentence','label'])
    
    test_dataset = dataset['test'].map(test_datset_function,batched=False,remove_columns=['sentence','label','symptom_description'])
    data_collator = transformers.DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False,collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False,collate_fn=data_collator)
    model.to('cuda')
    epoch = args['num_train_epochs']
    optimizer = transformers.AdamW(model.parameters(),lr=args['learning_rate'])
    for i in range(epoch):
        model.train()
        train_total_loss = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')
            outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
            loss = outputs.loss
            train_total_loss += loss.item()
            loss.backward()
            optimizer.step()
        wandb.log({'train_loss':train_total_loss, 'epoch':i+1})
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            val_predictions, val_labels = [], []
            for batch in tqdm(val_dataloader):
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                labels = batch['labels'].to('cuda')
                outputs = model(input_ids,attention_mask=attention_mask,labels = labels)
                loss = outputs.loss
                val_total_loss += loss.item()
                predict = outputs.logits.cpu().numpy()
                refer = batch['labels'].cpu().numpy()
                val_predictions.extend(predict)
                val_labels.extend(refer)
            eval_result = compute_metrics((np.array(val_predictions),np.array(val_labels)))
            eval_result['train_loss'] = train_total_loss
            eval_result['val_loss'] = val_total_loss
            eval_result['epoch'] = i+1
            wandb.log(eval_result)
        # 测试集
        with torch.no_grad():
            test_total_loss = 0
            test_predictions, test_labels = [], []
            for batch in tqdm(test_dataloader):
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                labels = batch['labels'].to('cuda')
                outputs = model(input_ids,attention_mask=attention_mask,labels = labels)
                loss = outputs.loss
                test_total_loss += loss.item()
                predict = outputs.logits.cpu().numpy()
                refer = batch['labels'].cpu().numpy()
                test_predictions.extend(predict)
                test_labels.extend(refer)
            test_result = compute_metrics((np.array(test_predictions),np.array(test_labels)),is_test=True)
            new_result = {}
            for k,v in test_result.items():
                new_result['test_'+k] = v
            new_result['test_loss'] = test_total_loss
            new_result['epoch'] = i+1
            wandb.log(new_result)

        # 每个epoch保存一下
        epoch_dir = os.path.join(project_path,'epoch_'+str(i))
        os.makedirs(epoch_dir,exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
            
    # 保存模型
    model.save_pretrained(project_path)
    tokenizer.save_pretrained(project_path)
    wandb.finish()


















