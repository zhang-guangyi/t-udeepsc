import numpy as np
import random, re, json

from .ans_punct import prep_ans
from pytorch_transformers import BertTokenizer

###################   Initialization


tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

def shuffle_list(ans_list):
    random.shuffle(ans_list)
    
def img_feat_path_load(path_list):
    iid_to_path = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        iid_to_path[iid] = path

    return iid_to_path


def img_feat_load(path_list):
    iid_to_feat = {}
    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        img_feat = np.load(path)
        img_feat_x = img_feat['x'].transpose((1, 0))
        iid_to_feat[iid] = img_feat_x
        print('\rPre-Loading: [{} | {}] '.format(ix, path_list.__len__()), end='          ')
    return iid_to_feat


def ques_load(ques_list):
    qid_to_ques = {}
    for ques in ques_list:
        qid = str(ques['question_id'])
        qid_to_ques[qid] = ques
    return qid_to_ques


def ans_stat(json_file):
    ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))
    return ans_to_ix, ix_to_ans


def proc_img_feat(img_feat, img_feat_pad_size):
    if img_feat.shape[0] > img_feat_pad_size:
        img_feat = img_feat[:img_feat_pad_size]
    img_feat = np.pad(
        img_feat,
        ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0)
    return img_feat


def proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)
    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques['question'].lower()
    ).replace('-', ' ').replace('/', ' ').split()
    print(ques['question'])
    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break
    return ques_ix

def proc_ques_bert(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)
    ques_ix = tokenizer.encode("[CLS] " + ques['question'] + " [SEP]")
    return ques_ix

def get_score(occur):
    if occur == 0:
        return .0
    elif occur == 1:
        return .3
    elif occur == 2:
        return .6
    elif occur == 3:
        return .9
    else:
        return 1.

def proc_ans(ans, ans_to_ix):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    ans_prob_dict = {}
    for ans_ in ans['answers']:
        ans_proc = prep_ans(ans_['answer'])
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1
    for ans_ in ans_prob_dict:
        if ans_ in ans_to_ix:
            ans_score[ans_to_ix[ans_]] = get_score(ans_prob_dict[ans_])
    return ans_score


def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len >= n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)