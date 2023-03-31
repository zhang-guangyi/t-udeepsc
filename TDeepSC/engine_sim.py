import torch
import math
import nltk
import torch.nn as nn
import sys

from utils import *
from tqdm import tqdm
from timm.data import Mixup
from einops import rearrange
from typing import Iterable, Optional
from vqa_utils import VQA_Tool, VQA_Eval
from timm.utils import accuracy, AverageMeter
from nltk.translate.bleu_score import sentence_bleu
####################################

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

@torch.no_grad()
def evaluate(ta_perform: str, net: torch.nn.Module, dataloader: Iterable, 
                  device: torch.device, criterion: torch.nn.Module, print_freq=10):
    net.eval()
    SNRdBs = np.arange(-6, 19, 3)
    Result_Group = np.zeros(len(SNRdBs))
    for l in range(len(SNRdBs)):
        snr = SNRdBs[l]
        print('Test SNR =%d dB' % (snr) )
        if ta_perform.startswith('imgc'):
            acc_meter = AverageMeter()
            loss_meter = AverageMeter()
            with torch.no_grad():
                for batch_idx, (imgs, targets) in enumerate(dataloader):
                    imgs, targets = imgs.to(device), targets.to(device)
                    outputs = net(img=imgs, ta_perform=ta_perform,test_snr=torch.FloatTensor([l]))
                    loss = criterion(outputs, targets)
                    batch_size = targets.size(0)
                    idx, predicted = outputs.max(1)
                    acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)
                    loss_meter.update(loss.item(), 1)
                    if batch_idx % print_freq == 0:
                        print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100]' %(batch_idx*batch_size, 
                                len(dataloader.dataset), loss_meter.avg, acc_meter.avg*100))   
            test_stat = {'loss': loss_meter.avg,
                'acc': acc_meter.avg}  
            Result_Group[l] = acc_meter.avg
            # return test_stat
        
        elif ta_perform.startswith('imgr'):
            psnr_meter = AverageMeter()
            loss_meter = AverageMeter()
            psnr_list = []
            with torch.no_grad():
                for batch_idx, (imgs, targets) in enumerate(dataloader):
                    imgs, targets = imgs.to(device), targets.to(device)
                    outputs = net(img=imgs, ta_perform=ta_perform,test_snr=torch.FloatTensor([l]))
                    outputs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
                    outputs = rearrange(outputs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
                    loss = criterion(outputs, targets)
                    batch_size = targets.shape[0]
                    ######  Predictions  ######
                    predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
                    targets = torch.chunk(imgs, chunks=imgs.size(0), dim=0)
                    psnr_vals = calc_psnr(predictions, targets)
                    psnr_list.extend(psnr_vals)
                    psnr_meter.update(torch.mean(torch.tensor(psnr_vals)).item(), n=batch_size)
                    loss_meter.update(loss.item(), 1)
                    if batch_idx % print_freq == 0:
                        print('Test %d/%d: [loss: %.4f] [psnr: %.3f dB]' %(batch_idx*batch_size, 
                                len(dataloader.dataset), loss_meter.avg, psnr_meter.avg))   
            test_stat = {'loss': loss_meter.avg,
                'psnr': psnr_meter.avg}  
            Result_Group[l] = psnr_meter.avg
            # return test_stat
        
        elif ta_perform.startswith('textc'):
            acc_meter = AverageMeter()
            loss_meter = AverageMeter()
            with torch.no_grad():
                for batch_idx, (texts, targets) in enumerate(dataloader):
                    texts, targets = texts.to(device), targets.to(device)
                    outputs = net(text=texts, ta_perform=ta_perform,test_snr=torch.FloatTensor([l]))
                    loss = criterion(outputs, targets)
                    batch_size = targets.size(0)
                    idx, predicted = outputs.max(1)
                    acc_meter.update(predicted.eq(targets).float().mean().item(), n=batch_size)
                    loss_meter.update(loss.item(), 1)
                    if batch_idx % print_freq == 0:
                        print('Test %d/%d: [loss: %.4f] [acc1: %.3f/100]' %(batch_idx*batch_size, 
                                len(dataloader.dataset), loss_meter.avg, acc_meter.avg*100))   
            test_stat = {'loss': loss_meter.avg,
                'acc': acc_meter.avg}  
            Result_Group[l] = acc_meter.avg
            # return test_stat
        
        elif ta_perform.startswith('textr'):
            bleu_meter = AverageMeter()
            loss_meter = AverageMeter()
            result = []
            with torch.no_grad():
                for batch_idx, (texts, targets) in enumerate(dataloader):
                    loss = 0
                    texts, targets = texts.to(device), targets.to(device)
                    outputs = net(text=texts,ta_perform=ta_perform)
                    batch_size = targets.size(0)
                    preds = torch.zeros_like(targets)
                    for i in range(outputs.shape[1]):
                
                        loss += criterion(outputs, targets)
                        preds[:,i] = outputs[:,i].max(-1)[-1] 
                    preds = tokens2sentence(preds)
                    targets = tokens2sentence(targets)
                    for pred, target in zip(preds, targets):
                        result.append((pred, target))
            
                    bleu_meter.update(computebleu(preds, targets)/batch_size, n=batch_size)
                    loss_meter.update(loss.item(), 1)
                    if batch_idx % print_freq == 0:
                        print('Test %d/%d: [loss: %.4f] [bleu: %.3f]' %(batch_idx*batch_size, 
                                len(dataloader.dataset), loss_meter.avg, bleu_meter.avg))   
            test_stat = {'loss': loss_meter.avg,
                'bleu': bleu_meter.avg}  
            Result_Group[l] = bleu_meter.avg
    print(Result_Group)
    return test_stat

@torch.no_grad()
def evaluate_vqa(ta_perform: str, net: torch.nn.Module, dataloader: Iterable, 
                  device: torch.device, criterion: torch.nn.Module, print_freq=500):
    net.eval()
    dataset = dataloader.dataset
    qid_list = [ques['question_id'] for ques in dataset.ques_list]
    ans_ix_list = []
    i = 0
    for batch_idx, (imgs, texts, targets) in enumerate(dataloader):
        imgs, texts, targets = imgs.to(device), texts.to(device), targets.to(device)
        batch_size = imgs.shape[0]
        i += batch_size
        outputs = net(img=imgs, text=texts, ta_perform=ta_perform)
        pred_np = outputs.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)
        if pred_argmax.shape[0] != dataset.configs.eval_batch_size:
            pred_argmax = np.pad(
                pred_argmax,(0, dataset.configs.eval_batch_size - pred_argmax.shape[0]),
                mode='constant',constant_values=-1)
        ans_ix_list.append(pred_argmax)
        if batch_idx % print_freq == 0:
            print('Test %d/%d:' %(batch_idx*batch_size, 
                        len(dataloader.dataset)))
        
    ans_ix_list = np.array(ans_ix_list).reshape(-1)
    result = [{
        'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
        'question_id': int(qid_list[qix])}for qix in range(qid_list.__len__())]

    result_eval_file = 'vqaeval_result/result_run_' + dataset.configs.version + '.json'
    print('Save the result to file: {}'.format(result_eval_file))
    json.dump(result, open(result_eval_file, 'w'))

    # create vqa object and vqaRes object
    ques_file_path = dataset.configs.question_path['val']
    ans_file_path = dataset.configs.answer_path['val']
    vqa = VQA_Tool(ans_file_path, ques_file_path)
    vqaRes = vqa.loadRes(result_eval_file, ques_file_path)
    vqaEval = VQA_Eval(vqa, vqaRes, n=2)  
    vqaEval.evaluate()

    return vqaEval.accuracy


@torch.no_grad()
def evaluate_msa(ta_perform: str, net: torch.nn.Module, dataloader: Iterable, 
                  device: torch.device, criterion: torch.nn.Module, print_freq=10):
    net.eval()
    loss_meter = AverageMeter()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_idx, (imgs,texts,speechs, targets) in enumerate(dataloader):
            imgs, texts, speechs, targets = imgs.to(device), texts.to(device), speechs.to(device), targets.to(device)
            outputs = net(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform)
            loss = criterion(outputs, targets)
            y_pred.append(outputs.detach().cpu().numpy())
            y_true.append(targets.detach().cpu().numpy())
            loss_meter.update(loss.item(), 1)
    y_true = np.concatenate(y_true, axis=0).squeeze()
    y_pred = np.concatenate(y_pred, axis=0).squeeze()
    acc = calc_metrics(y_true, y_pred)        
    test_stat = {'loss':loss_meter.avg,
                 'acc': acc}
    return test_stat
    

