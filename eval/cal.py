import json
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-4-0613")

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

def cal_sentence_length(data):
    total_length = 0
    total_turn = 0
    for k,v in data.items():
        messages = v['messages']
        for message in messages:
            if message['role'] == 'system':
                total_length += len(message['content'])
                total_turn += 1
    
    return round(total_length/total_turn,2)
    
def cal_sentence_num(data):
    total_turn = 0
    for k,v in data.items():
        messages = v['messages']
        for message in messages:
            if message['role'] == 'system':
                total_turn += 1
        total_turn -= 3
    avg_turn = round(total_turn/len(data),2)
    
    return avg_turn  
    
def cal_question_num(data):
    total_turn = 0
    for k,v in data.items():
        sheet = v['sheet']
        for k1,v1 in sheet.items():
            if isinstance(v1,dict):
                for k2,v2 in v1.items():
                    if v2 == 'N':
                        total_turn += 1
            elif isinstance(v1,list):
                for item in v1:
                    for k2,v2 in item.items():
                        if v2 == 'N':
                            total_turn += 1
            elif isinstance(v1,str):
                if v1 == 'N':
                    total_turn += 1
    avg_turn = round(total_turn/len(data),2)
    return avg_turn

def cal_examination(data):
    total_turn = 0
    for k,v in data.items():
        sheet = v['sheet']
        if sheet.get('medical-examination') == 'N':
            total_turn += 1
    avg_turn = round(total_turn/len(data),2)
    
    return avg_turn

def cal_cost(data):
    input_length = 0
    output_length = 0
    for k,v in data.items():
        messages = v['messages']
        turns =  len(messages)/2
        for message in messages:
            if message['role'] == 'system':
                output_length += len(encoding.encode(message['content']))
                turns -= 1
                input_length += len(encoding.encode(message['content']))*turns
            else:
                input_length += len(encoding.encode(message['content']))*turns
    total_cost = input_length*0.03/1000 + output_length*0.06/1000
    total_cost = total_cost
    return round(total_cost,2)
        
def cal_other(data,label=None):
    if label==None:
        label = [
            'symptom-name',
            'symptom-duration',
            'symptom-description',
            'past-diagnosis',
            'past-medication',
            'physical-change',
            'trauma-surgery',
            'preventive',
            'allergic-history',
            'contact-history',
            'habit-tobacco',
            'habit-wine',
            'habit-drug',
            'habit-living',
            'coitus-history',
            'family-history',
            'personal-age',
            'personal-weight',
            'personal-sex',
            'medical-examination',
            'extended-information',
            'receiving_treatment'
            ]
    result_dict = {}
    for singel_label in label:
        total_turn = 0
        for k,v in data.items():
            sheet = v['sheet']
            if sheet.get(singel_label) == 'N':
                total_turn += 1
        avg_turn = round(total_turn/len(data),2)
        result_dict[singel_label] = avg_turn
    return result_dict
def cal_other_deep(data,label=None):
    if label==None:
        label = [
            "symptom_description_color",
            "symptom_description_smell",
            "symptom_texture",
            "symptom_description_degree",
            "symptom_description_position",
            "symptom_description_time",
            "symptom_description_shape"
            ]
    result_dict = {}
    for singel_label in label:
        total_turn = 0
        for k,v in data.items():
            sheet = v['sheet']
            detail = sheet.get('symptom-description-detail')
            for item in detail:
                if item.get(singel_label) == 'N':
                    total_turn += 1
        avg_turn = round(total_turn/len(data),2)
        result_dict[singel_label] = avg_turn
    return result_dict
        





      
        
        
    
    
    
    
    
    
    
    
    
sum_temp = 0    
for single_path in path_list:
    
    with open(single_path,'r',encoding='utf-8') as f:
        data = json.load(f)
        # 执行上述检测
        avg_sentence_length = cal_sentence_length(data)
        avg_sentence_num = cal_sentence_num(data)
        avg_question_num = cal_question_num(data)
        avg_examination = cal_examination(data)
        common_ask = cal_other(data)
        deep_ask = cal_other_deep(data)
        total_cost = cal_cost(data)
        # 打印结果
        sum_temp += total_cost
        print(f"模型路径：{single_path}")
        print(f"平均句子长度：{avg_sentence_length}")
        print(f"平均句子数量：{avg_sentence_num}")
        print(f"平均问题数量：{avg_question_num}")
        print(f"平均检查项数量：{avg_examination}")
        print(f"token费用：{total_cost}")
        print(f"常见问题：{common_ask}")
        print(f"深层问题：{deep_ask}")
sum_temp = sum_temp/len(path_list)
print(f"平均token费用：{sum_temp}")
        
        















