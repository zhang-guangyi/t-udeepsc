import numpy as np
import glob, json, torch, time
import os, torch, random

from .vqa_config import path
from types import MethodType
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
from .data_utils import proc_img_feat, proc_ans, ans_stat, rpad
from .data_utils import img_feat_path_load, img_feat_load, ques_load


class VQA2(Dataset):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    def __init__(self, configs, train):
        self.configs = configs
        if train:
            self.configs.run_mode='train'
        else:
            self.configs.run_mode='val'
        self.img_feat_path_list = []
        
        split_list = self.configs.split[self.configs.run_mode].split('+')
        print('split_list', split_list)
        for split in split_list:
            if split in ['train', 'val', 'test']:
                self.img_feat_path_list += glob.glob(configs.img_feat_path[split] + '*.npz')

        # Loading question word list
        self.stat_ques_list = \
            json.load(open(self.configs.question_path['train'], 'r'))['questions'] + \
            json.load(open(self.configs.question_path['val'], 'r'))['questions'] + \
            json.load(open(self.configs.question_path['test'], 'r'))['questions'] + \
            json.load(open(self.configs.question_path['vg'], 'r'))['questions']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        split_list = self.configs.split[configs.run_mode].split('+')
        for split in split_list:
            self.ques_list += json.load(open(self.configs.question_path[split], 'r'))['questions']
            if self.configs.run_mode in ['train']:
                self.ans_list += json.load(open(self.configs.answer_path[split], 'r'))['annotations']

        # Define run data size
        if self.configs.run_mode in ['train']:
            self.data_size = self.ans_list.__len__()
            self.data = self.ans_list
        else:
            self.data_size = self.ques_list.__len__()
            self.data = self.ques_list

        if self.configs.preload:
            self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
        else:
            self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        self.qid_to_ques = ques_load(self.ques_list)
        self.ans_to_ix, self.ix_to_ans = ans_stat('vqa_utils/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()

    def __getitem__(self, idx):
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)
        # Process ['train'] and ['val', 'test'] respectively
        if self.configs.run_mode in ['train']:
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]
            # Process image feature from (.npz) file
            if self.configs.preload:
                img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = proc_img_feat(img_feat_x, self.configs.img_feat_pad_size)
            ques_ix = np.array(rpad(self.tokenizer.encode("[CLS] " + ques['question'] + " [SEP]"), self.configs.max_token))
            ques_ix_iter = np.zeros(self.configs.max_token, np.int64)
            ques_ix_iter[:ques_ix.shape[0]] = ques_ix[:]
            ans_iter = proc_ans(ans, self.ans_to_ix)

        else:
            ques = self.ques_list[idx]
            if self.configs.preload:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = proc_img_feat(img_feat_x, self.configs.img_feat_pad_size)

            ques_ix = np.array(rpad(self.tokenizer.encode("[CLS] " + ques['question'] + " [SEP]"), self.configs.max_token))
            ques_ix_iter = np.zeros(self.configs.max_token, np.int64)
            ques_ix_iter[:ques_ix.shape[0]] = ques_ix[:]

        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter)
 
               
    def __len__(self):
        return self.data_size

class Config_VQA(path):
    def __init__(self):
        super(Config_VQA, self).__init__()
        self.run_mode = 'train'
        self.version = str(random.randint(0, 999999)) # For saving the result file
        self.eval_every_epoch = True
        self.test_save_pred = False
        self.preload = False
        self.split = {
            'train': '',
            'val': 'val',
            'test': 'test'}

        self.train_split = 'train'
        # Max length of question sentences
        self.max_token = 16
        self.img_feat_pad_size = 100
        self.img_feat_size = 2048

    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)
        return args_dict

    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def proc(self, args):
        assert self.run_mode in ['train', 'val', 'test']
        self.split['train'] = self.train_split
        if 'val' in self.split['train'].split('+') or self.run_mode not in ['train']:
            self.eval_every_epoch = False
        if self.run_mode not in ['test']:
            self.test_save_pred = False
        self.eval_batch_size = args.batch_size

    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))
        return ''


