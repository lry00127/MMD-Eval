import evaluate
import re
class myTokenizer():
        def __init__(self):
            pass
        def __call__(self, text):
            return self.tokenize(text)
        def tokenize(self, text):
            result = []
            for i in text.split(' '):
                result.append(i)
            return result
class Acc():
    def __init__(self,use_num=False):
        self.num2slot = {
            "1": "症状名称",
            "2": "过往诊断",
            "3": "过往用药",
            "4": "身体变化",
            "5": "外伤手术",
            "6": "预防注射",
            "7": "过敏史",
            "8": "接触史",
            "9": "家庭史",
            "10": "生活习惯",
            "11": "医学检查"
            }
        self.predicts = []
        self.refers = []
        self.all_slot_list = list(self.num2slot.values())
        self.num=len(self.all_slot_list)
        self.use_num=use_num
    def add_batch(self,predict,reference):
        self.predicts.extend(predict)
        self.refers.extend(reference)
    
    def compute(self):
        format_acc = 0
        tp,fp,tn,fn = 0,0,0,0
        esp = 1e-3
        if not self.use_num:
            for predict,reference in zip(self.predicts,self.refers):
                
                reference = eval(reference)
                try:
                    predict_dict = eval(predict)
                    if type(predict_dict)==dict:
                        format_acc += 1
                except:
                    pass
                
                for k,v in reference.items():
                    if k in predict:
                        tp += 1
                    else:
                        fn += 1
                for k in self.all_slot_list:
                    if k not in reference:
                        if k in predict:
                            fp += 1
                        else:
                            tn += 1
        else:

            slot_single_dict = {}
            for predict,reference in zip(self.predicts,self.refers):
                reference_key = [r.split(':')[0] for r in reference.split(' ') if r != '' and r.split(':')[0].isdigit()]
                predict_key=[]
                try:
                    if len(predict)!=0:
                        predict = predict.split(' ')
                        for i in range(len(predict)):
                            if predict[i].split(':')[0].isdigit():
                                num = int(predict[i].split(':')[0])
                                if num>=1 and num<=self.num:
                                    predict_key.append(str(num))
                    format_acc += 1
                except:
                    pass
                tn += self.num
                
                for p in predict_key:
               
                    tn -= 1
                    if p in reference_key:
                        tp += 1
                        
                        if slot_single_dict.get('tp_'+str(p))==None:
                            slot_single_dict['tp_'+str(p)] = 0
                        slot_single_dict['tp_'+str(p)]+=1
                        
                    else:
                        fp += 1
                        if slot_single_dict.get('fp_'+str(p))==None:
                            slot_single_dict['fp_'+str(p)] = 0
                        slot_single_dict['fp_'+str(p)]+=1
                for r in reference_key:
                    if r not in predict_key:
                        if slot_single_dict.get('fn_'+str(r))==None:
                            slot_single_dict['fn_'+str(r)] = 0
                        slot_single_dict['fn_'+str(r)]+=1
                        fn += 1
                        tn -= 1
                
   
        format_acc = round(format_acc/len(self.predicts), 3)
        precision = tp/(tp+fp+esp)
        recall = tp/(tp+fn+esp)
        acc = (tp+tn)/(tp+tn+fp+fn+esp)
        f1 = 2*precision*recall/(precision+recall+esp)
        precision = round(precision,3)
        recall = round(recall,3)
        acc = round(acc,3)
        f1 = round(f1,3)
        slot_p_r = {}
        for i in range(1,self.num+1):
            name = self.num2slot[str(i)]
            slot_p_r['precision_'+name] = round(slot_single_dict.get('tp_'+str(i),0)/(slot_single_dict.get('tp_'+str(i),0)+slot_single_dict.get('fp_'+str(i),0)+esp),3)
            slot_p_r['recall_'+name] = round(slot_single_dict.get('tp_'+str(i),0)/(slot_single_dict.get('tp_'+str(i),0)+slot_single_dict.get('fn_'+str(i),0)+esp),3)
        self.predicts = []
        self.refers = []
        result = {"precision":precision,"recall":recall,"acc":acc,"f1":f1,"format_acc":format_acc,
                "tp":tp,"fp":fp,"tn":tn,"fn":fn}
        result.update(slot_p_r)
        return result
        
   
            
class ValueRouge():
    def __init__(self,tokenizer=None,use_num=False):
        self.num2slot = {
        "1": "症状名称",
        "2": "过往诊断",
        "3": "过往用药",
        "4": "身体变化",
        "5": "外伤手术",
        "6": "预防注射",
        "7": "过敏史",
        "8": "接触史",
        "9": "家庭史",
        "10": "生活习惯",
        "11": "医学检查"
        }
        self.all_slot_list = list(self.num2slot.values())
        self.num=len(self.all_slot_list)
        self.predicts = []
        self.refers = []
        self.metric = evaluate.load("hf-evaluate/rouge")
        self.tokenizer = tokenizer
        self.rouge_tokenizer = myTokenizer()
        self.use_num=use_num
    def add_batch(self,predicts,references):
        if not self.use_num:
            for predict,reference in zip(predicts,references):
                reference = eval(reference)
                reference_value = ""
                for k,v in reference.items():
                    reference_value += ' '
                    reference_value += v

                predict_value = ""
                try:
                    predict_dict = eval(predict)
                    assert type(predict_dict)==dict
                    for k,v in predict_dict.items():
                        predict_value += ' '
                        predict_value += v
                except:
                    predict_value = str(predict)
                    predict_value = predict_value.replace("{","")
                    predict_value = predict_value.replace("}","")
                    predict_value = predict_value.replace("\"","")
                    predict_value = predict_value.replace("'","")
                    for i in self.all_slot_list:
                        predict_value = predict_value.replace(i,"")
                self.predicts.append(predict_value)
                self.refers.append(reference_value)
        else:
            for predict,reference in zip(predicts,references):
                reference_value = [r[2:] for r in reference.split(' ') if r != '']
                reference_value = ' '.join(reference_value)
                
                try:
                    predict_value=""
                    predict_temp = predict.split(' ')
                    for i in range(len(predict_temp)):
                        if predict_temp[i][0].isdigit():
                            num = int(predict_temp[i][0])
                            
                            if num>=1 and num<=self.num:
                                
                                predict_value+=predict_temp[i][2:]
                except:
                    predict_value=""
                    predict_temp = predict.replace(":","")
                    predict_temp = re.sub(r'\d+','',predict_temp)
                    predict_value = predict_temp
                    
                self.predicts.append(predict_value)
                self.refers.append(reference_value)
            
    def compute(self):
        self.predicts,self.refers = self.postprocess_text(self.predicts,self.refers,self.tokenizer)
        
        self.metric.add_batch(predictions=self.predicts,
                              references=self.refers)
        score = self.metric.compute(use_stemmer=False,
                                tokenizer=self.rouge_tokenizer)
        self.predicts=[]
        self.refers=[]
        return score
        
    def postprocess_text(self, preds, labels,tokenizer):
        post_preds = [' '.join(tokenizer.tokenize(pred.strip())  ) for pred in preds]
        post_labels = [' '.join(tokenizer.tokenize(label.strip())  ) for label in labels]
        return post_preds, post_labels
                
        
        
class KeyAcc():
    def __init__(self):
        self.num2slot = {
        "1": "症状名称",
        "2": "过往诊断",
        "3": "过往用药",
        "4": "身体变化",
        "5": "外伤手术",
        "6": "预防注射",
        "7": "过敏史",
        "8": "接触史",
        "9": "家庭史",
        "10": "生活习惯",
        "11": "医学检查"
        }
        self.predicts = []
        self.refers = []
        self.num=len(self.num2slot.keys())
    
    def add_batch(self,predict,reference):
        self.predicts.extend(predict)
        self.refers.extend(reference)
    
    def compute(self):
        format_acc = 0
        tp,fp,tn,fn = 0,0,0,0
        esp = 1e-3
        
        for predict,reference in zip(self.predicts,self.refers):
            
            reference = reference.split(' ')
            if len(reference)==1 and reference[0]=='':
                reference = []
            try:
                predict_list = predict.split(' ')
                if len(predict_list)==1 and predict_list[0]=='':
                    predict_list = []
                pos = True
                for i in predict_list:
                    if i.isdigit() and int(i)>=1 and int(i)<=self.num:
                        pass
                    else:
                        pos = False
                        break
                if pos:
                    format_acc += 1
                    
            except:
                pass
            
            for k in reference:
                if k in predict_list :
                    tp += 1
                else:
                    fn += 1
            for k in range(1,self.num+1):
                k = str(k)
                if k not in reference:
                    if k in predict_list:
                        fp += 1
                    else:
                        tn += 1
        
   
        format_acc = round(format_acc/len(self.predicts), 3)
        precision = tp/(tp+fp+esp)
        recall = tp/(tp+fn+esp)
        acc = (tp+tn)/(tp+tn+fp+fn+esp)
        f1 = 2*precision*recall/(precision+recall+esp)
        precision = round(precision,3)
        recall = round(recall,3)
        acc = round(acc,3)
        f1 = round(f1,3)
        slot_p_r = {}
        self.predicts = []
        self.refers = []
        result = {"precision":precision,"recall":recall,"acc":acc,"f1":f1,"format_acc":format_acc,
                "tp":tp,"fp":fp,"tn":tn,"fn":fn}
        result.update(slot_p_r)
        return result
    
class IntentMetric():

    def __init__(self,id2label):
        self.id2label = id2label
        self.label2id = {v:k for k,v in id2label.items()}
        self.predicts = []
        self.refers = []
        self.num=len(self.id2label.keys())
        self.sentence_ids = []
    def add_batch(self,predict,reference,sent_id=[]):
        self.sentence_ids.extend(sent_id)
        self.predicts.extend(predict)
        self.refers.extend(reference)
    def compute(self):
        tp,fp,tn,fn = 0,0,0,0
        esp = 1e-4
        intent_p_r_data = {}
        for intent in self.label2id.keys():
            intent_p_r_data[intent] = {'tp':0,'fp':0,'tn':0,'fn':0}
        if self.sentence_ids!=[]:
            sent_pos = True
        else:
            sent_pos = False
            self.sentence_ids = [-1]*len(self.predicts)
        sentence_f = {}
        
        for predict,reference,s_id in zip(self.predicts,self.refers,self.sentence_ids):
            if type(s_id)!=int:
                s_id = s_id.item()
            for p ,r ,i in zip(predict,reference,list(range(self.num))):
                if p==1 and r==1:
                    intent_p_r_data[self.id2label[i]]['tp']+=1
                    tp+=1
                elif p==1 and r==0:
                    intent_p_r_data[self.id2label[i]]['fp']+=1
                    fp+=1
                    sentence_f.update({s_id:'fp'})
                elif p==0 and r==1:
                    intent_p_r_data[self.id2label[i]]['fn']+=1
                    fn+=1
                    sentence_f.update({s_id:'fn'})
                elif p==0 and r==0:
                    intent_p_r_data[self.id2label[i]]['tn']+=1
                    tn+=1
                else:
                    raise ValueError("error")
                

        precision = tp/(tp+fp+esp)
        recall = tp/(tp+fn+esp)
        acc = (tp+tn)/(tp+tn+fp+fn+esp)
        f1 = 2*precision*recall/(precision+recall+esp)
        precision = round(precision,3)
        recall = round(recall,3)
        acc = round(acc,3)
        f1 = round(f1,3)
        intent_p_r = {}
        for intent, v in intent_p_r_data.items():
            tp1 = v['tp']
            fp1 = v['fp']
            fn1 = v['fn']
            tn1 = v['tn']
            intent_p_r[intent+'_precision'] = round(tp1/(tp1+fp1+esp),3)
            intent_p_r[intent+'_recall'] = round(tp1/(tp1+fn1+esp),3)
            intent_p_r[intent+'_acc'] = round((tp1+tn1)/(tp1+tn1+fp1+fn1+esp),3)
            intent_p_r[intent+'_f1'] = round(2*tp1/(2*tp1+fp1+fn1+esp),3)
        
        self.predicts = []
        self.refers = []
        self.sentence_ids = []
        result = {"precision":precision,"recall":recall,"acc":acc,"f1":f1,
                  'tp':tp,'fp':fp,'tn':tn,'fn':fn}
        result.update(intent_p_r)
        if sent_pos:
            result['sentence_f'] = sentence_f
        return result        
        
class SlotMetric():
    def __init__(self,tokenizer=None,use_num=False):
        self.num2slot = {
        "1": "症状名称",
        "2": "过往诊断",
        "3": "过往用药",
        "4": "身体变化",
        "5": "外伤手术",
        "6": "预防注射",
        "7": "过敏史",
        "8": "接触史",
        "9": "家庭史",
        "10": "生活习惯",
        "11": "医学检查"
        }

        self.all_slot_list = self.num2slot.values()
        self.slot2num = {v:k for k,v in self.num2slot.items()}
        self.num=len(self.all_slot_list)
        self.predicts = []
        self.refers = []
        
        self.metric = evaluate.load("hf-evaluate/rouge")
        self.tokenizer = tokenizer
        self.rouge_tokenizer = myTokenizer()
        self.use_num=use_num
    def add_batch(self,predicts,references):
        if not self.use_num:
            for predict,reference in zip(predicts,references):
                
                reference = eval(reference)
                
                
                reference_value = ""
                for k,v in reference.items():
                    reference_value += ' '
                    reference_value += v
                predict_value = ""
                try:
                    predict_dict = eval(predict)
                    assert type(predict_dict)==dict
                    for k,v in predict_dict.items():
                        predict_value += ' '
                        predict_value += v
                except:
                    predict_value = str(predict)
                    predict_value = predict_value.replace("{","")
                    predict_value = predict_value.replace("}","")
                    predict_value = predict_value.replace("\"","")
                    predict_value = predict_value.replace("'","")
                    for i in self.all_slot_list:
                        predict_value = predict_value.replace(i,"")
                self.predicts.append(predict_value)
                self.refers.append(reference_value)
        else:
            for predict,reference in zip(predicts,references):
                reference_value = [r[2:] for r in reference.split(' ') if r != '']
                reference_value = ' '.join(reference_value)
                
                try:
                    predict_value=""
                    predict_temp = predict.split(' ')
                    for i in range(len(predict_temp)):
                        if predict_temp[i][0].isdigit():
                            num = int(predict_temp[i][0])
                            
                            if num>=1 and num<=self.num:
                                
                                predict_value+=predict_temp[i][2:]
                except:
                    predict_value=""
                    predict_temp = predict.replace(":","")
                    predict_temp = re.sub(r'\d+','',predict_temp)
                    predict_value = predict_temp
                    
                self.predicts.append(predict_value)
                self.refers.append(reference_value)
            
    def compute(self):
        self.predicts,self.refers = self.postprocess_text(self.predicts,self.refers,self.tokenizer)
        
        self.metric.add_batch(predictions=self.predicts,
                              references=self.refers)
        score = self.metric.compute(use_stemmer=False,
                                tokenizer=self.rouge_tokenizer)
        self.predicts=[]
        self.refers=[]
        return score
        
    def postprocess_text(self, preds, labels,tokenizer):
        post_preds = [' '.join(tokenizer.tokenize(pred.strip())  ) for pred in preds]
        post_labels = [' '.join(tokenizer.tokenize(label.strip())  ) for label in labels]
        return post_preds, post_labels
    
    
        
