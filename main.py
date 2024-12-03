import os
import sys
os.environ['OPENAI_API_BASE'] = ""
os.environ['OPENAI_API_KEY'] = ''
import argparse

from utils.load_model import ActionModel, SlotModel, DoctorModel,PatientModel
from utils.gen_response import GenResponse
import evaluate
import json
from tqdm import tqdm
import re
import copy
from utils.tokenizer import TokenizerForRouge
from utils.is_same import BertScore
import logging

class Dialogues:
    def __init__(self,
                 action_model_path,
                 slot_model_path,
                 prefix_path='',
                 prefix_index='',
                 data_path='',
                 response_template_path='',
                 slot_intent_path='',
                 d_model_path=None,
                 d_api_base_url=None,
                 d_api_key=None,
                 d_prompt=None,
                 save_path=None,
                 summary_diagnosis=None,
                 summary_advice=None,
                 summary_reason=None,
                 p_model=None,
                 p_prompt=None,
                 p_api_base_url=None,
                 p_api_key=None,
                 bertscore_model=None,
                 num_layers=None,
                 is_polish=False,
                 is_socket=False
                 ):
        self.p_model = p_model
        self.BertScore = BertScore(bertscore_model,num_layers)
        self.is_polish = is_polish
        self.is_socket = is_socket
        if is_polish:
            if not self.is_socket:
                with open('utils/polish_prompt.json', 'r', encoding='utf-8') as f:
                    self.polish_prompt = json.load(f)['0']
                self.polish_model = DoctorModel(self.polish_prompt, 'models/Qwen1.5-4B-Chat')
            else:
                import socket
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                server_address = './uds_socket'
                self.sock.connect(server_address)
            
        self.gen_response = GenResponse(response_template_path,slot_intent_path,is_same_func=self.BertScore.is_same_func)
        if not p_model:
            self.action_model = ActionModel(action_model_path,slot_intent_path)
            self.slot_model = SlotModel(slot_model_path, prefix_path, prefix_index, slot_intent_path)
            self.doctor_model = DoctorModel(d_prompt, d_model_path, d_api_base_url, d_api_key)
            self.data = self.get_data(data_path)
            if self.is_polish:
                with open('utils/random_select.json','r',encoding='utf-8') as f:
                    select_ids = json.load(f)['select_500']
            self.data = {k:v for k,v in self.data.items() if k in select_ids}    
            self.save_path = save_path
            self.summary_diagnosis = summary_diagnosis
            self.summary_advice = summary_advice
            self.summary_reason = summary_reason
            
        else:
            self.p_model = PatientModel(p_model, p_prompt, p_api_base_url, p_api_key)
          
    @staticmethod
    def get_data(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def get_case_sheet(case:dict):
        case_sheet = copy.deepcopy(case)
        for k,v in case.items():
            if type(v) == str:
                if v=='' or v=='null':
                    case_sheet[k] = 'O'
                else:
                    case_sheet[k] = 'Y'
            elif type(v) == int or type(v) == float:
                case_sheet[k] = 'Y'
            elif type(v) == list:
                new_list = []
                for i in v:
                    new_dict = {}
                    if type(i) == dict:
                        for key,value in i.items():
                            if value == '' or value == 'null':
                                new_dict[key] = 'O'
                            else:
                                new_dict[key] = 'Y'
                    new_list.append(new_dict)
                case_sheet[k] = new_list 
            elif type(v)==dict:
                case_sheet[k] = 'Y'
            else:
                raise ValueError('数据类型错误')
        return case_sheet
    def gen_dialogue(self,start_index=None,end_index=None):

        if not self.p_model:
            count = 0
            result = {}
            data_length = len(self.data)
            p_bar = tqdm(range(data_length), desc='生成对话')
            for k, v in self.data.items():
                count+=1
                if start_index is not None:
                    if count<int(start_index):
                        p_bar.update(1)
                        continue
                if end_index is not None:
                    if count>int(end_index):
                        break
                case_sheet = self.get_case_sheet(v)
                stop = False
                first_dialogue = v['first-dialogue']
                messages = [{"role": "user", "content": first_dialogue}]
                summary_truns = 0
                while True:
                    d_response = self.doctor_model.gen_response(messages)
                    messages.append({"role": "system", "content": d_response})
                    
                    if stop:
                        break
                    if summary_truns == 0:

                        p_sencence_intent = self.action_model.gen_action(d_response)
                        p_sentence_slot = self.slot_model.gen_slots(p_sencence_intent)
                        p_response = ''
                        for sentence,intents,slots in zip(p_sencence_intent.keys(),p_sencence_intent.values(),p_sentence_slot.values()):
                            response, case_sheet = self.gen_response(slot=slots, 
                                                         intent = intents,
                                                         case = v,
                                                         case_sheet = case_sheet)
                            p_response += response
                    if len(messages) >= 20:
                        p_response = ''        
                            
                    if len(p_response.strip().replace('.', '')) != 0 and self.dialogue_analysis(messages,p_response):
                        if self.is_polish:
                            messages_polish = [{"role": "user", "content": p_response}]
                            if self.is_socket:
                                self.sock.sendall(str(messages_polish).encode())
                                p_response_polish = self.sock.recv(4000).decode()
                            else:
                                p_response_polish = self.polish_model.gen_response(messages_polish)
                            logging.info('润色前后的回答为:{},{}'.format(p_response,p_response_polish))
                        messages.append({"role": "user", "content": p_response_polish})
                    
                    else:
                        if summary_truns == 0:
                            messages.append({"role": "user", "content": self.summary_diagnosis})
                            summary_truns += 1
                        elif summary_truns == 1:
                            messages.append({"role": "user", "content": self.summary_advice})
                            summary_truns += 1
                        elif summary_truns == 2:
                            messages.append({"role": "user", "content": self.summary_reason})
                            summary_truns += 1
                            stop = True
                        else:
                            raise ValueError('对话生成出错')
                single_result = {k:{'messages':messages,'sheet':case_sheet}}
                logging.info(str(single_result))
                result.update(single_result)
                p_bar.update(1)
            return result
        
        else:
            count = 0
            result = {}
            for k, v in self.data.items():
                count += 1
                if count > 3:
                    break
                p_info=""
                for singel_slot_key,singel_slot_value in v:
                    if singel_slot_value !='null':
                        if singel_slot_value == 'true':
                            singel_slot_value = '是'
                        single_slot_key = self.gen_response.en2cn.get(singel_slot_key)
                        temp = single_slot_key + ':' + singel_slot_value + ','
                        p_info += temp
                # 管理对话历史,第一句由病人发问, 注意系统级提示, 两边的模型都是在本地生成的, 这里不保存系统级提示.
                messages = []
                while True:
                    p_response = self.p_model.gen_response(messages,p_info)
                    if p_response.strip() == '':
                        break
                    messages.append({"role": "user", "content": p_response})
                    d_response = self.doctor_model.gen_response(messages)
                    if d_response.strip() == '':
                        break
                    messages.append({"role": "system", "content": d_response})
                result.update({k: messages})
                p_bar.update(1)
            return result
            
        
    def dialogue_analysis(self,messages:list,p_response:str)->bool:
        p_response_list=[i['content'] for i in messages if i['role']=='user']
        for i in p_response_list:
            if p_response in i:
                return False
        return True
class Eval():
    def __init__(self,data:dict, data_path:str,model_path:str):
        # 对话数据. 对话数据是通过列表传进来的, 而非文件
        self.data=data
        self.case = self.get_data(data_path)
        self.response_result = {}
        self.tokenizer = TokenizerForRouge(model_path)
        self.metric = evaluate.load("hf-evaluate/rouge")
        self.extract_result()
        self.extract_result()
        
    @staticmethod
    def get_data(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    def extract_result(self):
        for k,v in self.data.items():
            d_resason = v[-1]['content']
            d_advice = v[-3]['content']
            d_diagnosis = v[-5]['content'] 

            self.response_result.update({k:{"诊断结果":d_diagnosis,
                                            "治疗建议":d_advice,
                                            "诊断原因":d_resason}}
                                        )

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--p_model_path', type=str, default='')
    parse.add_argument('--start_index', type=str, default=None)
    parse.add_argument('--end_index', type=str, default=None)
    args = parse.parse_args()

    d_model_path = args.p_model_path
    d_model_name = d_model_path.split('/')[-1]
    path_save = 'eval/0604-polish/'+d_model_name
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    file_save = path_save + '/main_out.json'

    dialogues = Dialogues(                                                   
        action_model_path='models/intent_0602_epoch7_add2k_no_question',
        slot_model_path='models/slot_0602_epoch3',
        prefix_path='utils/slot_prefix.json',
        prefix_index='3',
        slot_intent_path='utils/slot_intent.json',
        data_path='data/case/select7-0604-final-case_en.json',
        response_template_path='utils/response_template_0509.json',
        d_model_path=d_model_path,
        d_api_base_url="",
        d_api_key=  '',
        d_prompt="你是一个医生, 下面会有病人向你咨询病情, 请认真分析病人情况, 并在必要时主动向患者提问, 以便更好的了解病人情况.",
        save_path='main_out.json',
        summary_advice='针对我的情况, 请总结一下你的治疗建议.',
        summary_diagnosis='针对我的情况, 请总结一下你的诊断结果.',
        summary_reason='针对我的情况, 请总结一下你的诊断原因.',
        bertscore_model='models/bert-base-chinese',
        num_layers = 8,
        is_polish=False,
        is_socket=False
        )
    messages = dialogues.gen_dialogue(start_index=args.start_index,end_index=args.end_index)
 
    with open(file_save, 'w', encoding='utf-8') as f:
        f.write(json.dumps(messages, ensure_ascii=False, indent=2))      

    
    
