from transformers import (
    AutoConfig,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BertTokenizer,
    AutoModelForSequenceClassification
)
import numpy as np

from typing import List,  Any
import torch
import dataclasses
import torch
import re
import json
import os
import openai


@dataclasses.dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep: str = "###"
    sep2: str = None
    skip_next: bool = False
    conv_id: Any = None
    def get_prompt(self):
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                if i == 0 and "我" in message and "你好" not in message and "您好" not in message:
                    ret += role + ": 你好，" + message + seps[i % 2]
                else:
                    ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":" 
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)
        


def generate_stream(model, 
                    tokenizer, 
                    params, 
                    beam_size,
                    context_len=4096, ):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.2))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    if stop_str == tokenizer.eos_token:
        stop_str = None

    input_ids = tokenizer(prompt).input_ids

    max_src_len = context_len - max_new_tokens - 8
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids = torch.tensor(input_ids[-max_src_len:]).unsqueeze(0).to(device)

    outputs = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=beam_size,
        temperature=temperature,
    )

    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return output


class SlotModel:

    def __init__(self,
                 model_path,
                 prefix_path='utils/slot_prefix.json',
                 prefix_index='3',
                 slot_intent_path=None
                 ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, config=self.config)
        if self.device.type == "cuda":
            self.model.to(self.device)
        self.model.eval()
        self.prefix = self.get_prefix(prefix_path, prefix_index)
        self.slot = self.get_slot(slot_intent_path)
        self.num_slot = self.slot
        self.slot_num = {v: k for k, v in self.slot.items()}

    @staticmethod
    def get_prefix(prefix_path, prefix_index):
        with open(prefix_path, 'r', encoding='utf-8') as f:
            prefix = json.load(f)
        index = prefix_index
        return prefix[index]

    @staticmethod
    def get_slot(slot_intent_path):
        with open(slot_intent_path, 'r', encoding='utf-8') as f:
            slot = json.load(f)['slot']
        return slot

    def gen_single_slot(self, sentence):

        assert isinstance(sentence, str), "Input must be a string"
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt", max_length=192, truncation=True)
        if self.device.type == "cuda":
            input_ids = input_ids.to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids,num_beams=3)
        slot = self.tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            slot_temp = slot.split(' ')
            slot_dict = {i[0]:i[2:] for i in slot_temp if len(i) > 2 and i[0].isdigit()} 
            slot = slot_dict
            assert type(slot) == dict, "slot type error"
        except:
            slot = {}
        
        for k, v in list(slot.items()):
            if int(k)<=0 or int(k)>len(self.slot) or len(v) == 0:
                slot.pop(k)
        return slot

    def gen_slots(self, statement):
        assert isinstance(statement, dict), "Input must be a dict"
        slots = {}
        for k, v in statement.items():
            result = {}
            if v :
                result = self.gen_single_slot(self.prefix + k)
            slots[k] = result
        return slots

class ActionModel:

    def __init__(self, 
                 model_path,
                 action_path ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        if self.device.type == "cuda":
            self.model.to(self.device)
        with open(action_path, 'r', encoding='utf-8') as f:
            self.action_map = json.load(f)['intent']
        self.model.eval()
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def gen_action(self, statement):
        assert isinstance(statement, str), "Input must be a string"
        statement_list = self.gen_single_sentence(statement)
        action_list = []
        for sentence in statement_list:
            input_ids = self.tokenizer.encode(sentence, return_tensors="pt", max_length=192, truncation=True)
            if self.device.type == "cuda":
                input_ids = input_ids.to(self.device)
            with torch.no_grad():
                output = self.model(input_ids)
                predict = output.logits.cpu().numpy()
                predict = self.sigmoid(predict)
                predict = (predict > 0.5).astype(int)
                

                predict_num = np.where(predict[0] == 1)[0]
                predict_num = [str(i) for i in predict_num]
            action_temp = []
            for n in predict_num:
                if n in self.action_map.keys():
                    action_temp.append(self.action_map.get(n))
                
            action_list.append(action_temp)
        return dict(zip(statement_list, action_list))

    @staticmethod
    def gen_single_sentence(statement):
        sentence_list = re.split(r'(?<=[。？?\n])', statement)
        sentence_list = [i for i in sentence_list if len(i.strip()) > 1]
        return sentence_list


class DoctorModel:

    def __init__(self, prompt, model_path=None, api_base_url=None, api_key=None):
        self.prompt = prompt
        self.model_path = model_path
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if api_base_url is None:
            self.pos = 'local'
            if 'baichuan' in model_path.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", 
                              trust_remote_code=True, torch_dtype=torch.float16)
                self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", 
                              trust_remote_code=True, torch_dtype=torch.float16)
                
                self.sys_prompt = prompt
                self.model.generation_config = GenerationConfig.from_pretrained(model_path)
            elif 'bianque' in model_path.lower():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half()
                self.model.to(device)
                self.sys_prompt = prompt
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            elif 'ming' in model_path.lower() or 'pulse' in model_path.lower():
                model_path = self.model_path
                device = "cuda"
                self.temperature=1.2
                self.max_new_tokens=512
                self.beam_size=3
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="balanced"
                    )
                
                self.model.config.use_cache = True
                self.model.eval()
                self.conv_bloom = Conversation(
                    system=self.prompt,
                    roles=("USER", "ASSISTANT"),
                    messages=(),
                    offset=0,
                    sep=" ",
                    sep2="</s>",
                )
                self.sys_prompt = prompt
            elif 'huatuogpt' in model_path.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
                self.model.generation_config = GenerationConfig.from_pretrained(model_path)

            elif 'qwen' in model_path.lower():
                self.model = AutoModelForCausalLM.from_pretrained(
                     model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.config = AutoConfig.from_pretrained(model_path,trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_path, config=self.config,
                                                       device_map="auto",
                                                       trust_remote_code=True)
                self.model.eval()
                self.sys_prompt = prompt
                
            
        elif api_base_url and api_key:
            self.pos = 'remote'
            self.api_base_url = api_base_url
            self.api_key = api_key
            openai.api_key = api_key
            openai.api_base = api_base_url

        else:
            raise ValueError("model_path and api_base_url, api_key must have one")

    def gen_response(self, statement: list) -> str:

        if self.pos == 'local':
            if 'chatglm' in  self.model_path.lower():

                history = [{"role": "assistant", "content": self.sys_prompt}]
                last_statement=statement[-1].get('content')
                statement = statement[:-1]
                for i in statement:
                    if i.get('role') == 'user':
                        history.append(i)
                    else:
                        history.append({"role": "assistant", "content": i.get('content')})
                response, history = self.model.chat(self.tokenizer, 
                                                    last_statement, 
                                                    history)
                return response
            elif 'qwen' in self.model_path.lower():
                messages = [{"role": "system", "content": self.prompt}]
                messages.extend(statement)
                text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                    )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response 
            elif 'baichuan' in self.model_path.lower().lower():
                messages = [{"role": "system", "content": self.prompt}]
                for i in statement:
                    if i.get('role') == 'user':
                        messages.append({"role": "user", "content": i.get('content')})
                    elif i.get('role') == 'system':
                        messages.append({"role": "assistent", "content": i.get('content')})
                response = self.model.chat(self.tokenizer, messages)
                return response
            elif 'bianque' in self.model_path.lower():
                
                user_history = []
                bot_history = ['我是利用人工智能技术，结合大数据训练得到的智能医疗问答模型扁鹊，你可以向我提问。']
                for i in range(len(statement)):
                    if statement[i]['role'] == 'user':
                        user_history.append(statement[i]['content'])
                    else:
                        bot_history.append(statement[i]['content'])
                
                context = "\n".join([f"病人：{user_history[i]}\n医生：{bot_history[i]}" for i in range(len(bot_history))])
                input_text = context + "\n病人：" + user_history[-1] + "\n医生："

                response, history = self.model.chat(self.tokenizer, query=input_text, history=None, max_length=2048, num_beams=1, do_sample=True, top_p=0.75, temperature=0.95, logits_processor=None)
                return response   
            elif 'ming' in self.model_path.lower() or 'pulse' in self.model_path.lower() or 'huozi' in self.model_path.lower():
                last_statement = statement[-1].get('content')
                conv = self.conv_bloom.copy()
                for i in statement:
                    if i.get('role') == 'USER':
                        conv.append_message(conv.roles[0], i.get('content'))
                    else:
                        conv.append_message(conv.roles[1], i.get('content'))
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                skip_echo_len = len(prompt) - prompt.count("</s>") * 4

                params = {
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_new_tokens":self.max_new_tokens,
                    "stop": "</s>",
                }
                context_len = len(prompt)  + self.max_new_tokens + 8
                
                output_stream = generate_stream(self.model, self.tokenizer, params, self.beam_size,context_len=context_len)
                output_stream = output_stream[-1][skip_echo_len:].strip()
                conv.messages[-1][-1] = output_stream
                return output_stream
            elif 'huatuogpt' in self.model_path.lower():
                messages = []
                for i in statement:
                    if i.get('role') == 'user':
                        messages.append({"role": "user", "content": i.get('content')})
                    else:
                        messages.append({"role": "assistant", "content": i.get('content')})
                response = self.model.HuatuoChat(self.tokenizer, messages)
                return response
                 
        elif self.pos == 'remote':
            openai.api_key = self.api_key
            openai.api_base = self.api_base_url
            sys_message = self.prompt
            messages = [{"role": "system", "content": sys_message}]
            messages.extend(statement)
            # 需要多尝试几遍
            is_retry = True
            while(is_retry):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0125",
                        messages=messages,
                        stream=False,
                        temperature=0.5,
                        api_base=self.api_base_url
                    )
                    assert response['choices'][0].message.role == 'assistant'
                    assert response['choices'][0].message.content != ''
                    is_retry = False
                    response = response.choices[0].message.content
                    return response
                except:
                    continue
        else:
            raise ValueError("model_path and api_base_url, api_key must have one")

class PatientModel():
    def __init__(self, model_path, prompt, api_base_url=None, api_key=None):
        if 'chatglm' or 'baichuan' in  model_path.lower():
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.config = AutoConfig.from_pretrained(model_path,trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, config=self.config,trust_remote_code=True)
            self.model.eval()
            self.model.to(self.device)
            self.sys_prompt = prompt
        elif 'gpt' in model_path.lower():
            assert api_base_url and api_key, "api_base_url and api_key must have"
            self.api_base_url = api_base_url
            self.api_key = api_key
            openai.api_key = api_key
            openai.api_base = api_base_url
            
    def gen_response(self, statement: list, p_info:str) -> str:
        if 'chatglm' in  self.model_path.lower():

                sys_prompt = self.sys_prompt+p_info
                history = [{"role": "assistant", "content": sys_prompt}]
                last_statement=statement[-1].get('content')
                statement = statement[:-1]
                for i in statement:
                    if i.get('role') == 'user':
                        history.append({"role": "assistant", "content": i.get('content')})
                    else:
                        history.append({"role": "user", "content": i.get('content')})
                response, history = self.model.chat(self.tokenizer, 
                                                    last_statement, 
                                                    history)
                return response
        elif 'baichuan' in self.model_path.lower():
                sys_prompt = self.sys_prompt+p_info
                
                messages = [{"role": "system", "content": sys_prompt}]
                for i in statement:
                    if i.get('role') == 'user':
                        messages.append({"role": "assistant", "content": i.get('content')})
                    else:
                        messages.append({"role": "user", "content": i.get('content')})
 
                response = self.model.chat(self.tokenizer, messages)
                return response
            
        elif 'qwen' in self.model_path.lower():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sys_prompt = self.sys_prompt+p_info
            messages = [{"role": "system", "content": sys_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                pad_token_id=151645,
                do_sample=True,
                temperature=0.1,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
                    
        elif 'gpt' in self.model_path.lower():
            sys_prompt = self.sys_prompt+p_info
            
            messages = [{"role": "system", "content": sys_prompt}]
            for i in statement:
                if i.get('role') == 'user':
                    messages.append({"role": "system", "content": i.get('content')})
                else:
                    messages.append({"role": "user", "content": i.get('content')})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=False,
                temperature=0.5,
                api_key=self.api_key,
                api_base=self.api_base_url
            )
            response = response.choices[0].message.content
            return response
            
        else:
            raise ValueError("model_path and api_base_url, api_key must have one")

            