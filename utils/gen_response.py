import json
from rouge_score import rouge_scorer
import copy

class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, sentence):
        # 将字符串分割开,用于计算BLEU, ROUGE等指标
        return list(sentence)

class GenResponse:

    def __init__(self, file_path,intent_slot_path,is_same_func):

        self.response_template = self.load_response_template(file_path)
        
        self.tokenizer = Tokenizer()
        with open(intent_slot_path, 'r', encoding='utf-8') as f:
            f_dict = json.load(f)
        self.is_same_func = is_same_func
        self.nun2intent = f_dict['intent']
        self.num2slot = f_dict['slot']
        self.intent2num = {v: k for k, v in self.nun2intent.items()}
        self.slot2num = {v: k for k, v in self.num2slot.items()}


    def __call__(self, slot, intent, case, case_sheet):
        return self.gen_response(slot, intent, case, case_sheet)
    @staticmethod
    def bigram_accuracy(text1, text2,threshold=0.2):
        words1 = list(text1)
        words2 = list(text2)

        bigrams1 = [' '.join(words1[i:i+2]) for i in range(len(words1)-1)]

        bigrams2 = [' '.join(words2[i:i+2]) for i in range(len(words2)-1)]

        matched_bigrams = set(bigrams1).intersection(bigrams2)

        accuracy = len(matched_bigrams) / len(bigrams1) if bigrams1 else 0
        if accuracy > threshold:
            return True
        else:
            return False
        
    
    @staticmethod
    def updata_sheet(case_sheet:dict,case_item,detail=None,case=None):
        new_case_sheet = copy.deepcopy(case_sheet)
        if case_item in case_sheet.keys():
            if case_sheet.get(case_item) == 'Y' :
                new_case_sheet[case_item]='N'
            elif case_sheet.get(case_item)=='O':
                new_case_sheet[case_item]='N'
            elif case_sheet.get(case_item)=='N':
                raise ValueError('数据错误,不应该为N,N已经回复过了，不应该被更新')
            else:
                raise ValueError('数据错误')
        else:
            if detail == None or case==None:
                raise ValueError('数据错误,没有detail')
            else:
                new_list = []
                
                for i in range(len(case_sheet.get('symptom-description-detail'))):
                    case_sheet_cont = case_sheet.get('symptom-description-detail')[i]
                    case_cont = case.get('symptom-description-detail')[i]
                    new_dict = copy.deepcopy(case_sheet_cont)
                    if case_cont['symptom'] == detail:
                        if case_sheet_cont[case_item]=='Y':
                            new_dict[case_item] = 'N'
                        elif case_sheet_cont[case_item]=='O':
                            new_dict[case_item] = 'N'
                        else:
                            raise ValueError('数据错误,不应该为O,O为空项，不应该被更新')
                        new_list.append(new_dict)
                    else:
                        new_list.append(new_dict)
                new_case_sheet['symptom-description-detail'] = new_list
        return new_case_sheet      
            
    @staticmethod
    def check_sheet(case_sheet:dict,case_item,detail=None,case=None):

        if case_item in case_sheet.keys():
            if case_sheet.get(case_item) == 'Y' :
                return 'Y'
            elif case_sheet.get(case_item)=='O':
                return 'O'
            elif case_sheet.get(case_item)=='N':
                return 'N'
            else:
                raise ValueError('数据错误')
        else:
            if detail == None or case==None:
                raise ValueError('数据错误,没有detail')
            else:
                for i in range(len(case_sheet.get('symptom-description-detail'))):
                    case_sheet_cont = case_sheet.get('symptom-description-detail')[i]
                    case_cont = case.get('symptom-description-detail')[i]
                    if case_cont['symptom'] == detail:
                        if case_sheet_cont.get(case_item)=='Y':
                            return 'Y'
                        elif case_sheet_cont.get(case_item)=='O':
                            return 'O'
                        elif case_sheet_cont.get(case_item)=='N':
                            return 'N'
                        else :
                            return 'N'
                raise ValueError('数据错误,没有找到detail')
    
    def detail_search(self,case,slot_value=None):
        if slot_value is None:
            return None
        
        for i in case.get('symptom-description-detail'):
            if self.is_same_func(i.get('symptom','') ,slot_value)>=0.8:
                return i.get('symptom','')
        return None
    @staticmethod
    def case_search(case,case_item,case_detail_item=None):

        if case_item in case.keys():
            return case.get(case_item)
        else:
            if case_detail_item == None:
                raise ValueError('数据错误,没有case_detail_item')
            else:
                for i in case.get('symptom-description-detail'):
                    if i['symptom'] == case_detail_item:
                        return i[case_item]
                    
    
    def gen_response(self, slot, intent, case, case_sheet):

        if len(intent)>0 and not intent[0].isdigit():
            intent = [self.intent2num[i] for i in intent]
        response = ""
        for single_intent in intent:
            if self.response_template.get(single_intent):
                if self.response_template[single_intent].get('action')=='none':
                    case_item = self.response_template[single_intent].get('case_item')
                    if self.check_sheet(case_sheet,case_item)=='Y':

                        item_value = self.case_search(case,case_item)
                        response_tmp = self.response_template[single_intent].get('template')['1']
                        response_tmp  = response_tmp.format(item_value)
                        
                        if response_tmp.endswith('。'):
                            response += response_tmp
                        else:
                            response += response_tmp + '。'
                        
                        case_sheet = self.updata_sheet(case_sheet,case_item)
                        pass
                    elif self.check_sheet(case_sheet,case_item)=='O':
                        response_tmp = self.response_template[single_intent].get('template')['0']
                        if response_tmp.endswith('。'):
                            response += response_tmp
                        else:
                            response += response_tmp + '。'
                        case_sheet = self.updata_sheet(case_sheet,case_item)
                        pass
                    else:
                        continue
                else:
                    slot_key = self.response_template[single_intent].get('action')
                    slot_value = slot.get(slot_key)
                    case_item = self.response_template[single_intent].get('case_item')

                    if case_item in ["symptom_description_color","symptom_description_smell",
                                     "symptom_description_degree","symptom_description_position",
                                     "symptom_description_shape","symptom_description_time"]:
                        case_detail_item = self.detail_search(case,slot_value)
                        if case_detail_item is None:
                            if self.check_sheet(case_sheet,'symptom-description')=='Y':
                                item_value = self.case_search(case,'symptom-description')
                                response_tmp = self.response_template[single_intent].get('template')['1']   
                                response_tmp  = response_tmp.format(item_value)
                                if response_tmp.endswith('。'):
                                    response += response_tmp
                                else:
                                    response += response_tmp + '。'
                                case_sheet = self.updata_sheet(case_sheet=case_sheet,
                                                            case_item='symptom-description',
                                                            detail=case_detail_item,
                                                            case=case)
                            elif self.check_sheet(case_sheet,'symptom-description')=='O':
                                response_tmp = self.response_template[single_intent].get('template')['0']  
                                if response_tmp.endswith('。'):
                                    response += response_tmp
                                else:
                                    response += response_tmp + '。'
                                case_sheet = self.updata_sheet(case_sheet=case_sheet,
                                                            case_item='symptom-description',
                                                            detail=case_detail_item,
                                                            case=case)
                        else:
                            if self.check_sheet(case_sheet,case_item,detail=case_detail_item,case=case)=='Y':    
                                search_res = self.case_search(case,case_item,case_detail_item)
                                response_tmp = self.response_template[single_intent].get('template')['1']
                                response_tmp  = response_tmp.format(search_res)
                                if response_tmp.endswith('。'):
                                    response += response_tmp
                                else:
                                    response += response_tmp + '。'  
                                case_sheet = self.updata_sheet(case_sheet=case_sheet,
                                                                case_item=case_item,
                                                                detail=case_detail_item,
                                                                case=case) 
                            elif self.check_sheet(case_sheet,case_item,detail=case_detail_item,case=case)=='O':
                                    response_tmp = self.response_template[single_intent].get('template')['0']  
                                    if response_tmp.endswith('。'):
                                        response += response_tmp
                                    else:
                                        response += response_tmp + '。'
                                    case_sheet = self.updata_sheet(case_sheet=case_sheet,
                                                                case_item=case_item,
                                                                detail=case_detail_item,
                                                                case=case) 
                    else:
                        if self.check_sheet(case_sheet,case_item)=='Y':

                            item_value = self.case_search(case,case_item)
                            response_pos = -1
                            if slot_value is not None:
                                if self.bigram_accuracy(slot_value,item_value):
                                    response_tmp = self.response_template[single_intent].get('template')['2']
                                    response_tmp  = response_tmp.format(item_value)
                                    if response_tmp.endswith('。'):
                                        response += response_tmp
                                    else:
                                        response += response_tmp + '。'
                                    response_pos = 2
                            if response_pos == -1:
                                response_tmp = self.response_template[single_intent].get('template')['1']
                                response_tmp  = response_tmp.format(item_value)
                                if response_tmp.endswith('。'):
                                    response += response_tmp
                                else:
                                    response += response_tmp + '。'
                            case_sheet = self.updata_sheet(case_sheet,case_item)
                        elif self.check_sheet(case_sheet,case_item)=='O':
                            response_tmp = self.response_template[single_intent].get('template')['0']
                            if response_tmp.endswith('。'):
                                response += response_tmp
                            else:
                                response += response_tmp + '。'
                            case_sheet = self.updata_sheet(case_sheet,case_item)
                            pass
                        else:
                            continue
            else:
                continue
        return response,case_sheet
    @staticmethod
    def load_response_template(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            response_template = json.load(f)
        return response_template
