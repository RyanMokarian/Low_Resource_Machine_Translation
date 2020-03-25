import re

import numpy as np



class ExtraProcessing():
    
    def __init__(self):
        self.BOS,self.EOS,self.TK_MAJ,self.TK_UP,self.TK_NUM = 'xxstart', 'xxend', 'xxmaj','xxup', 'xxnum'
        
    def regx(self,x):
        # Remove space, comma or point  between numbers
        x = re.sub(r"(\d+)[\s+|,|.](\d+)",r"\1\2",x)            
        # Replace numbers with TK_NUM
        x = re.sub("\d+",self.TK_NUM,x) 
        # Remove some punctuation
        PUNCT = '#&\()*+/<=>@[\\]^_{|}~'
        table = str.maketrans("","",PUNCT)
        x = x.translate(table)
        x = re.sub("\.\.+","",x)
        # Add space between '-' and words
        x = re.sub(r"(-)(\w)",r"\1 \2",x)
        x = re.sub(r"(\w)(-)",r"\1 \2",x)
        # Remove unecessary space
        x = re.sub("\s\s+"," ",x)
        return x
        
    def start_end(self,x):
        "Put in start and end tokens"
        x.insert(0,self.BOS)
        x.append(self.EOS)
        return x
        
    def replace_all_caps(self,x):
        "Replace tokens in ALL CAPS in `x` by their lower version and add `TK_UP` before."
        res = []
        for t in x:
            if t.isupper() and len(t) > 1: res.append(self.TK_UP); res.append(t.lower())
            else: res.append(t)
        return res

    def deal_caps(self,x):
        "Replace all Capitalized tokens in `x` by their lower version and add `TK_MAJ` before."
        res = []
        for t in x:
            if t == '': continue
            if t[0].isupper() and len(t) > 1 and (t[1:].islower() or (t[1] == "’" or t[1] == "'" )): 
                res.append(self.TK_MAJ)
            res.append(t.lower())
        return res
    
    def process(self,x):
        x = self.regx(x.strip())
        x = x.split()
        post = [self.deal_caps,self.replace_all_caps,self.start_end]
        for p in post:
            x = p(x)
        return ' '.join(x)