import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data



from transformers import BertTokenizer
from data import CIFAR_CR,SST_CR
from timm.data import create_transform
from vqa_utils import VQA2, Config_VQA
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms
from msa_utils import PAD, Config_MSA, MSA
# from pytorch_transformers import BertTokenizer
from torch.utils.data.sampler import RandomSampler
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, number_samp=5000):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_samp = number_samp
        self.largest_dataset_size = number_samp

    def __len__(self):
        return self.number_samp

    def __iter__(self):
        sampler = RandomSampler(self.dataset)
        sampler_iterator = sampler.__iter__()
        step = self.batch_size
        samples_to_grab = self.batch_size
        epoch_samples = self.number_samp
        final_samples_list = []  
        ### this is a list of indexes from the combined dataset
        for es in range(0, epoch_samples, step):
            cur_batch_sampler = sampler_iterator
            cur_samples = []
            for eg in range(samples_to_grab):
                try:
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org
                    cur_samples.append(cur_sample)
                except StopIteration:
                    ### got to the end of iterator - restart the iterator and continue to get samples
                    sampler_iterator = sampler.__iter__()
                    cur_batch_sampler = sampler_iterator
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org
                    cur_samples.append(cur_sample)
            final_samples_list.extend(cur_samples)
        # print(final_samples_list)
        return iter(final_samples_list)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def build_dataloader(ta_sel, trainsets, args):
    trainloaders = {}
    for ta in ta_sel:
        trainset = trainsets[ta]
        Collate_fn = collate_fn if ta.startswith('msa') else None 
        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                                sampler=BatchSchedulerSampler(dataset=trainset,batch_size=args.batch_size,
                                                number_samp=10000*len(ta_sel)),
                                                num_workers=args.num_workers, pin_memory=True,
                                                batch_size=args.batch_size, shuffle=False,collate_fn=Collate_fn)
        trainloaders[ta] = trainloader
    return trainloaders



def build_dataset_test(is_train, args):
    if args.ta_perform.startswith('img'):
        transform = build_img_transform(is_train, args)
        print("Transform = ")
        if isinstance(transform, tuple):
            for trans in transform:
                print(" - - - - - - - - - - ")
                for t in trans.transforms:
                    print(t)
        else:
            for t in transform.transforms:
                print(t)
        print("------------------------------------------------------")

    if  args.ta_perform.startswith('imgc'):
        args.input_size = 224
        transform = build_img_transform(is_train, args)
        dataset = CIFAR_CR(args.data_path, train=is_train, transform=transform, 
                                        download=True, if_class=True)
    elif  args.ta_perform.startswith('imgr'):
        args.input_size = 32
        transform = build_img_transform(is_train, args)
        dataset = CIFAR_CR(args.data_path, train=is_train, transform=transform, 
                                        download=True, if_class=False)
    elif args.ta_perform.startswith('textc'):
        dataset = SST_CR(root=False, train=is_train, binary=True, if_class=True)

    elif args.ta_perform.startswith('textr'):
        dataset = SST_CR(root=True, train=is_train, binary=True, if_class=False)

    elif args.ta_perform.startswith('vqa'):
        config_vqa = Config_VQA()
        config_vqa.proc(args)
        dataset = VQA2(config_vqa, train=is_train)
        
    elif args.ta_perform.startswith('msa'):
        config_msa = Config_MSA()
        dataset = MSA(config_msa, train=is_train)
    
    else:
        raise NotImplementedError()

    return dataset

def build_dataset_train(is_train, ta_sel, args):
    # if args.ta_perform.startswith('img'):
    
    # print("Transform = ")
    # if isinstance(transform, tuple):
    #     for trans in transform:
    #         print(" - - - - - - - - - - ")
    #         for t in trans.transforms:
    #             print(t)
    # else:
    #     for t in transform.transforms:
    #         print(t)
    # print("------------------------------------------------------")

    datasets = {}
    for ta in ta_sel:
        if  ta.startswith('imgc'):
            args.input_size = 224
            transform = build_img_transform(is_train, args)
            dataset = CIFAR_CR(args.data_path, train=is_train, transform=transform, 
                                            download=True, if_class=True)
        elif  ta.startswith('imgr'):
            args.input_size = 32
            transform = build_img_transform(is_train, args)
            dataset = CIFAR_CR(args.data_path, train=is_train, transform=transform, 
                                            download=True, if_class=False)
        elif ta.startswith('textc'):
            dataset = SST_CR(root=False, train=is_train, binary=True, if_class=True)

        elif ta.startswith('textr'):
            dataset = SST_CR(root=True, train=is_train, binary=True, if_class=False)

        elif ta.startswith('vqa'):
            config_vqa = Config_VQA()
            config_vqa.proc(args)
            dataset = VQA2(config_vqa, train=is_train)
            
        elif ta.startswith('msa'):
            config_msa = Config_MSA()
            dataset = MSA(config_msa, train=is_train)
        
        else:
            raise NotImplementedError()
        
        datasets[ta] = dataset

    return datasets



def build_img_transform(is_train, args):
    resize_im = args.input_size > 32
    mean = (0.,0.,0.)
    std =  (1.,1.,1.)
    t = []
    if is_train:
        if resize_im:
            crop_pct = 1
            size = int(args.input_size / crop_pct)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)



def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)  
    targets = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    texts = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    images = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    speechs = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    # print(texts.permute(1,0))
    SENT_LEN = texts.size(0)
    # Create bert indices using tokenizer
    bert_details = []
    for sample in batch:
        text = " ".join(sample[0][3])
        encoded_bert_sent = bert_tokenizer.encode_plus(
            text, max_length=SENT_LEN+2, add_special_tokens=True, pad_to_max_length=True,truncation=True)
        bert_details.append(encoded_bert_sent)

    bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
    texts = bert_sentences

    return images.permute(1,0,2), texts, speechs.permute(1,0,2), targets