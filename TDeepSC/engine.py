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
    if ta_perform.startswith('imgc'):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = net(img=imgs, ta_perform=ta_perform)
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
        return test_stat
    
    elif ta_perform.startswith('imgr'):
        psnr_meter = AverageMeter()
        loss_meter = AverageMeter()
        psnr_list = []
        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = net(img=imgs, ta_perform=ta_perform)
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
        return test_stat
    
    elif ta_perform.startswith('textc'):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (texts, targets) in enumerate(dataloader):
                texts, targets = texts.to(device), targets.to(device)
                outputs = net(text=texts, ta_perform=ta_perform)
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
        return test_stat
    
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


def train_class_batch_it(ta_perform, model, samples, targets, criterion):
    if ta_perform.startswith('imgc'):
        outputs = model(img=samples,ta_perform=ta_perform)
        loss = criterion(outputs, targets)
    elif ta_perform.startswith('imgr'):
        outputs = model(img=samples, ta_perform=ta_perform)
        targets = rearrange(targets, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
        loss = criterion(outputs, targets)
    elif ta_perform.startswith('textc'):
        outputs = model(text=samples, ta_perform=ta_perform)
        loss = criterion(outputs, targets)
    elif ta_perform.startswith('textr'):
        outputs = model(text=samples, ta_perform=ta_perform)
        loss = 0
        for i in range(outputs.shape[1]):
            loss += criterion(outputs[:,i], samples[:,i])
    return loss, outputs

def train_epoch_it(model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, ta_perform, max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=50):
    model.train(True)                                                         
    acc_meter = AverageMeter()
    psnr_meter = AverageMeter()
    loss_meter = AverageMeter()

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples ,targets) in enumerate(data_loader):    
        step = data_iter_step // update_freq
        it = start_steps + step  
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]                
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        targets = targets.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        batch_size = samples.size(0)
                                                   
        with torch.cuda.amp.autocast():
            loss, outputs = train_class_batch_it(
                ta_perform, model, samples, targets, criterion)
        loss_value = loss.item()

        ######  Error                              
        if not math.isfinite(loss_value):   
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        ######  Update
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()    

        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])

        if ta_perform.endswith('c'):
            acc_meter.update((outputs.max(-1)[-1] == targets).float().mean().item(), n=batch_size)
            loss_meter.update(loss_value, 1)
        elif ta_perform.startswith('imgr'):
            outputs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
            outputs = rearrange(outputs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
            tr_imgs = torch.tensor((samples*255).detach().cpu().numpy().astype(int).clip(0,255)).float()
            re_imgs = torch.tensor((outputs*255).detach().cpu().numpy().astype(int).clip(0,255)).float()
            psnr_meter.update(10 * math.log10(255.0**2/(criterion(tr_imgs, re_imgs))), n=1)
            loss_meter.update(loss_value, 1)
        elif ta_perform.startswith('textr'):
            loss_meter.update(loss_value, 1)
        
        if data_iter_step % print_freq == 0:
            if ta_perform.startswith('imgc'):
                print('Epoch:[%d] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' 
                    %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                        loss_meter.avg, acc_meter.avg*100, max_lr))
            elif ta_perform.startswith('imgr'):
                print('Epoch:[%d] %d/%d: [loss: %.3f] [psnr: %.3f dB] [lr: %.3e]' 
                    %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                        loss_meter.avg, psnr_meter.avg, max_lr)) 
            elif ta_perform.startswith('textc'):
                print('Epoch:[%d] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' 
                    %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                        loss_meter.avg, acc_meter.avg*100, max_lr)) 
            elif ta_perform.startswith('textr'):
                print('Epoch:[%d] %d/%d: [loss: %.3f] [lr: %.3e]' 
                    %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                        loss_meter.avg, max_lr)) 
              
    train_stat = {'loss': loss_meter.avg,
        'acc': acc_meter.avg}

    return train_stat 


def train_class_batch_vqa(ta_perform, model, imgs, texts, targets, criterion):
    if ta_perform.startswith('vqa'):
        outputs = model(img=imgs, text=texts, ta_perform=ta_perform)
        loss = criterion(outputs, targets)
    return loss, outputs


def train_epoch_vqa(model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, ta_perform, max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=500):
    model.train(True)                                                         
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (imgs, texts, targets) in enumerate(data_loader):    
        step = data_iter_step // update_freq
        it = start_steps + step  
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]                
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        imgs = imgs.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        batch_size = imgs.size(0)        
                           
        # with torch.cuda.amp.autocast():
        loss, outputs = train_class_batch_vqa(
                ta_perform, model, imgs, texts, targets, criterion)
        loss_value = loss.item()

        ######  Error                              
        if not math.isfinite(loss_value):   
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        ######  Update
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()    

        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])

        if ta_perform.startswith('vqa'):
            acc_meter.update((outputs.max(-1)[-1] == targets.max(-1)[-1]).float().mean().item(), n=batch_size)
            loss_meter.update(loss_value, 1)
        
        if data_iter_step % print_freq == 0:
            if ta_perform.startswith('vqa'):
                print('Epoch:[%d] %d/%d: [loss: %.3f] [acc1: %.3f /100] [lr: %.3e]' 
                    %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                        loss_meter.avg, acc_meter.avg*100, max_lr))
              
    train_stat = {'loss': loss_meter.avg,
        'acc': acc_meter.avg}

    return train_stat 





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
    




def train_class_batch_msa(ta_perform, model, imgs, texts, speechs, targets, criterion):
    if ta_perform.startswith('msa'):
        outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform)
        loss = criterion(outputs, targets)
        # pass
    return loss, outputs

def train_epoch_msa(model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader: Iterable, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, ta_perform, max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=5):
    model.train(True)                                                         
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (imgs, texts, speechs, targets) in enumerate(data_loader):    
        step = data_iter_step // update_freq
        it = start_steps + step  
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]                
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        imgs = imgs.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)
        speechs = speechs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = imgs.size(0)        
                           
        with torch.cuda.amp.autocast():
            loss, outputs = train_class_batch_msa(
                ta_perform, model, imgs, texts, speechs, targets, criterion)
        loss_value = loss.item()

        ######  Error                              
        if not math.isfinite(loss_value):   
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        ######  Update
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()    

        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])

        if ta_perform.startswith('msa'):
            # acc_meter.update((outputs.max(-1)[-1] == targets.max(-1)[-1]).float().mean().item(), n=batch_size)
            loss_meter.update(loss_value, 1)
        
        if data_iter_step % print_freq == 0:
            if ta_perform.startswith('msa'):
                print('Epoch:[%d] %d/%d: [loss: %.3f] [lr: %.3e]' 
                    %(epoch, batch_size*data_iter_step, len(data_loader.dataset),
                        loss_meter.avg,  max_lr))
              
    train_stat = {'loss': loss_meter.avg,
        'acc': acc_meter.avg}

    return train_stat 
