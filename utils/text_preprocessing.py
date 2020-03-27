import re

import numpy as np

BOS, EOS, TK_MAJ, TK_UP, TK_NUM = '<start>', '<end>', '<maj>', '<up>', '<num>'

def regx(x):
    """Some character level processing"""
    # Remove space, comma or point  between numbers
    x = re.sub(r"(\d+)[\s+|,|.](\d+)",r"\1\2",x)
    # Replace ’ with ' 
    x = re.sub("’","'",x)
    # Replace numbers with TK_NUM
    x = re.sub("\d+",TK_NUM,x)
    # Remove some punctuation
    PUNCT = '#&\()*+/<=>@[\\]^_{|}~'
    table = str.maketrans("","",PUNCT)
    x = x.translate(table)
    x = re.sub("\.\.+","",x)
    # Add space between '-' and words
    x = re.sub(r"(-)(\w)",r"\1 \2",x)
    x = re.sub(r"(\w)(-)",r"\1 \2",x)
    # Remove unnecessary space
    x = re.sub("\s\s+"," ",x)
    return x

def start_end(x):
    """Puts in start and end tokens"""
    x.insert(0,BOS)
    x.append(EOS)
    return x

def replace_all_caps(x):
    """Replace tokens in ALL CAPS in `x` by their lower version and add `TK_UP` before."""
    res = []
    for t in x:
        if t.isupper() and len(t) > 1: 
            res.append(TK_UP)
            res.append(t.lower())
        else: 
            res.append(t)
    return res

def deal_caps(x):
    """Replace all Capitalized tokens in `x` by their lower version and add `TK_MAJ` before."""
    res = []
    for t in x:
        if t == '': continue
        if t[0].isupper() and len(t) > 1 and (t[1:].islower() or (t[1] == "’" or t[1] == "'" )): 
            res.append(TK_MAJ)
        res.append(t.lower())
    return res

def process(x):
    x = regx(x.strip())
    x = x.split()
    post = [deal_caps, replace_all_caps, start_end]
    for p in post:
        x = p(x)
    return ' '.join(x)