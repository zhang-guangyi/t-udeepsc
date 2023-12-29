import os
import torch
import pandas as pd
import numpy as np
import pytreebank
import torch.utils.data as data

from loguru import logger
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
#####################################################

sst = pytreebank.load_sst()
def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)

def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label < 2:
        return 0
    if label > 2:
        return 1
    raise ValueError("Invalid label")


class SST_CR(Dataset):
    def __init__(self, root=True, train=True, binary=True, if_class: bool = True):
        logger.info("Loading the tokenizer")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        logger.info("Loading SST")
        
        if train:
            self.sst = sst["train"]
        else:
            self.sst = sst["test"]
        self.if_class = if_class
        if root and binary:
            self.data = [(rpad(tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66),
                          get_binary_label(tree.label),) 
                    for tree in self.sst if tree.label != 2]
        elif root and not binary:
            self.data = [(rpad(tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66),
                           tree.label,) 
                    for tree in self.sst]
        elif not root and not binary:
            self.data = [(rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66),
                           label)
                for tree in self.sst 
                for label, line in tree.to_labeled_lines()]
        else:
            self.data = [(rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66),
                    get_binary_label(label),)
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
                if label != 2]
        # for tree in self.sst:
        #     for label, line in tree.to_labeled_lines():
        #         if label != 2:
        #             print(line)
        #             print(get_binary_label(label))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index]
        sentence = torch.tensor(sentence)
        if self.if_class:
            return sentence, target
        else:
            return sentence, sentence