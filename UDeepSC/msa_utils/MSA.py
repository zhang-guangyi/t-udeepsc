import os
import re
import sys
# import mmsdk
import pickle
import torch
import pprint 
import numpy as np
import torch.nn as nn

from pathlib import Path
# from transformers import *
from tqdm import tqdm_notebook
# from mmsdk import mmdatasdk as md
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from subprocess import check_call, CalledProcessError



def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK

def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
    f = open(path_to_embedding, 'r')
    found = 0
    for line in tqdm_notebook(f, total=embedding_vocab):
        content = line.strip().split()
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
            found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()


class MOSI():
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))
        
        data_path = str(config.dataset_dir)
        cache_path = data_path + '/embedding_and_mapping.pt'

        try:
            self.train = load_pickle(data_path + '/train.pkl')
            self.dev = load_pickle(data_path + '/dev.pkl')
            self.test = load_pickle(data_path + '/test.pkl')
     
        except:
            print('error')
            pass

    def get_data(self, is_train):
        if is_train:
            return self.train
        else:              
            return self.test


class MSA(Dataset):
    def __init__(self, config, train=True):
        dataset = MOSI(config)
 
        self.data = dataset.get_data(train)
        self.len = len(self.data)

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
    


class Config_MSA(object):
    def __init__(self,):
        project_dir = Path(__file__).resolve().parent.parent
        sdk_dir = project_dir.joinpath('/home/hqyyqh888/SemanRes2/MSA/CMU-MultimodalSDK/')
        data_dir = project_dir.joinpath('data/msadata')
        data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
            'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}
        word_emb_path = '/home/hqyyqh888/SemanRes2/MSA/MISA/glove/glove.840B.300d.txt'
        assert(word_emb_path is not None)
        
        self.dataset_dir = data_dict['mosei']
        self.sdk_dir = sdk_dir
        self.word_emb_path = word_emb_path
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str