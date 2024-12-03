from typing import Any
from transformers import T5Tokenizer
class TokenizerForRouge():
    def __init__(self,model_path:str):
        self.t5tokenizer=T5Tokenizer.from_pretrained(model_path)    
    
    def __call__(self,text:str)->list:
        return self.tokenize(text)
    
    def tokenize(self,text:str)->list:
        t5_tokenized=self.t5tokenizer.tokenize(text)
        t5_tokenized=t5_tokenized[1:]
        return t5_tokenized