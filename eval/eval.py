import os
import sys
from rouge_score import rouge_scorer
from transformers import T5Tokenizer
t5tokenizer = T5Tokenizer.from_pretrained('models/slot_0602_epoch3')
import evaluate
metric = evaluate.load("hf-evaluate/rouge")
from utils.is_same import BertScore
bert_score = BertScore('models/bert-base-chinese',8)
from tqdm import tqdm

def cal_bert_score(preds,refs):
    score = 0
    for pred,ref in tqdm(zip(preds,refs)):
        score += bert_score.is_same_func(pred,ref)
    return score/len(preds)

class myTokenizer():
    def __init__(self):
        pass
    def __call__(self, text):
        return self.tokenize(text)
    def tokenize(self, text):
        result = t5tokenizer.tokenize(text)
        result = result[1:]
        
        return result
        
rouge_tokenizer = myTokenizer()

path_list = [
    'eval/0604/Baichuan2-7B-Chat/main_out.json',
    'eval/0604/bianque2/main_out.json',
    'eval/0604/chatglm3-6b/main_out.json',
    'eval/0604/HuatuoGPT2-7b/main_out.json',
    'eval/0604/ming-7b/main_out.json',
    'eval/0604/pulse-7b/main_out.json',
    'eval/0604/Qwen1.5-7B-Chat/main_out.json',
    'eval/0604/gpt35/main_out.json',
    'eval/0604/HuatuoGPT2-13b/main_out.json'
    ]
standard_path = 'data/case/select7-0604-final-case_en.json'
import json
with open(standard_path, 'r') as f:
    standard_data = json.load(f)

model_content = {}

for file in path_list:
    standard_reason = []
    standard_advice = []
    standard_diagnosis = []
    reason = []
    advice = []
    diagnosis = []    
    
    with open(file, 'r') as f:
        data = json.load(f)
    
    for k,v  in tqdm(data.items()):
        k_s = k
        v_s = standard_data[k]

        v = v['messages']
        d_resason = v[-1]['content']
        d_advice = v[-3]['content']
        d_diagnosis = v[-5]['content']
        d_resason_standard = v_s['reason']
        d_advice_standard = v_s['final-treatment']
        d_diagnosis_standard = v_s['final-diagnosis']
        
        reason.append(d_resason)
        advice.append(d_advice)
        diagnosis.append(d_diagnosis)
        standard_reason.append(d_resason_standard)
        standard_advice.append(d_advice_standard)
        standard_diagnosis.append(d_diagnosis_standard)
    metric.add_batch(predictions=reason, references=standard_reason)
    rouge_score_reason = metric.compute(tokenizer=rouge_tokenizer)
    bert_score_reason = cal_bert_score(reason,standard_reason)
    
    metric.add_batch(predictions=advice, references=standard_advice)
    rouge_score_advice = metric.compute(tokenizer=rouge_tokenizer)
    bert_score_advice = cal_bert_score(advice,standard_advice)
    
    metric.add_batch(predictions=diagnosis, references=standard_diagnosis)
    rouge_score_diagnosis = metric.compute(tokenizer=rouge_tokenizer)
    bert_score_diagnosis = cal_bert_score(diagnosis,standard_diagnosis)
    # 乘100保留2位小数
    rouge_score_reason = {k:round(v*100,2) for k,v in rouge_score_reason.items()}
    rouge_score_advice = {k:round(v*100,2) for k,v in rouge_score_advice.items()}
    rouge_score_diagnosis = {k:round(v*100,2) for k,v in rouge_score_diagnosis.items()}
    bert_score_reason = round(bert_score_reason*100,2)
    bert_score_advice = round(bert_score_advice*100,2)
    bert_score_diagnosis = round(bert_score_diagnosis*100,2)
    
    print(file)
    print('reason:',rouge_score_reason)
    print('advice:',rouge_score_advice)
    print('diagnosis:',rouge_score_diagnosis)
    print('reason bert:',bert_score_reason)
    print('advice bert:',bert_score_advice)
    print('diagnosis bert:',bert_score_diagnosis)
    model_name = file
    model_content[model_name] = {'reason':rouge_score_reason,'advice':rouge_score_advice,'diagnosis':rouge_score_diagnosis,
                                 'reason_bert':bert_score_reason,'advice_bert':bert_score_advice,'diagnosis_bert':bert_score_diagnosis}
new_dict = {}
for k,v in model_content.items():
    new_dict[k] = v
    new_dict[k]['avg'] = {}
    for k1,v1 in v['reason'].items():
        new_dict[k]['avg'][k1] = (v['reason'][k1]+v['advice'][k1]+v['diagnosis'][k1])/3
    new_dict[k]['avg']['bert'] = (v['reason_bert']+v['advice_bert']+v['diagnosis_bert'])/3
    new_dict[k]['avg'] = {k:round(v,2) for k,v in new_dict[k]['avg'].items()}
with open('eval/eval.json','w',encoding='utf-8') as f:
    json.dump(new_dict,f,ensure_ascii=False,indent=4)
    json.dump(model_content,f,ensure_ascii=False,indent=4)
        
        
        




