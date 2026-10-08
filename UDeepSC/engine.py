import torch
import math
import nltk
import torch.nn as nn
import torch.nn.functional as F
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

def _forward_snr_kwargs(net):
    raw = net.module if hasattr(net, 'module') else net
    return {
        'train_snr': getattr(raw, 'train_snr', 12.0),
        'test_snr': getattr(raw, 'test_snr', 12.0),
    }

def _model_attr(net, name, default=None):
    raw = net.module if hasattr(net, 'module') else net
    return getattr(raw, name, default)

@torch.no_grad()
def evaluate(ta_perform: str, net: torch.nn.Module, dataloader: Iterable, 
                  device: torch.device, criterion: torch.nn.Module, print_freq=10, max_batches=0):
    net.eval()
    snr_kwargs = _forward_snr_kwargs(net)
    if ta_perform.startswith('imgc'):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = net(img=imgs, ta_perform=ta_perform, **snr_kwargs)
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
                if max_batches and batch_idx >= max_batches:
                    break
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = net(img=imgs, ta_perform=ta_perform, **snr_kwargs)
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
                if max_batches and batch_idx >= max_batches:
                    break
                texts, targets = texts.to(device), targets.to(device)
                outputs = net(text=texts, ta_perform=ta_perform, **snr_kwargs)
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
                if max_batches and batch_idx >= max_batches:
                    break
                loss = 0
                texts, targets = texts.to(device), targets.to(device)
                targets = targets[:,1:]
                outputs = net(text=texts, ta_perform=ta_perform, **snr_kwargs)
                batch_size = targets.size(0)
                seq_len = min(outputs.shape[1], targets.shape[1])
                outputs = outputs[:, :seq_len]
                targets = targets[:, :seq_len]
                preds = torch.zeros_like(targets)
                for i in range(seq_len):
                    loss += criterion(outputs[:, i], targets[:, i])
                    preds[:,i] = outputs[:,i].max(-1)[-1] 
                preds = tokens2sentence(preds)
                targets = tokens2sentence(targets)
                for pred, target in zip(preds, targets):
                    # print(pred,target)
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
                  device: torch.device, criterion: torch.nn.Module, print_freq=500, max_batches=0):
    net.eval()
    snr_kwargs = _forward_snr_kwargs(net)
    dataset = dataloader.dataset
    qid_list = [ques['question_id'] for ques in dataset.ques_list]
    eval_qids = []
    ans_ix_list = []
    i = 0
    for batch_idx, (imgs, texts, targets) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        imgs, texts, targets = imgs.to(device), texts.to(device), targets.to(device)
        batch_size = imgs.shape[0]
        eval_qids.extend(qid_list[i:i + batch_size])
        i += batch_size
        outputs = net(img=imgs, text=texts, ta_perform=ta_perform, **snr_kwargs)
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
        'question_id': int(eval_qids[qix])}for qix in range(eval_qids.__len__())]

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


def train_class_batch_uni(task_name, model, batch_inputs, targets, criterion):
    loss = 0
    imgs, texts, speechs = batch_inputs
    snr_kwargs = _forward_snr_kwargs(model)
    if task_name.startswith('imgc'):
        outputs = model(img=imgs, ta_perform=task_name, **snr_kwargs)
        loss = criterion[task_name](outputs, targets)
    elif task_name.startswith('imgr'):
        outputs = model(img=imgs, ta_perform=task_name, **snr_kwargs)
        targets = rearrange(targets, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
        loss = criterion[task_name](outputs, targets)
    elif task_name.startswith('textc'):
        outputs = model(text=texts, ta_perform=task_name, **snr_kwargs)
        loss = criterion[task_name](outputs, targets)
    elif task_name.startswith('textr'):
        outputs = model(text=texts, ta_perform=task_name, **snr_kwargs)
        targets = targets[:,1:]
        seq_len = min(outputs.shape[1], targets.shape[1])
        outputs = outputs[:, :seq_len]
        targets = targets[:, :seq_len]
        for i in range(seq_len):
            loss += criterion[task_name](outputs[:,i], targets[:,i])
    elif task_name.startswith('vqa'):
        outputs = model(img=imgs, text=texts, ta_perform=task_name, **snr_kwargs)
        loss = criterion[task_name](outputs, targets)
    elif task_name.startswith('msa'):
        outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=task_name, **snr_kwargs)
        loss = criterion[task_name](outputs, targets)
    return loss, outputs

def meter(ta_sel):
    acc_meter_dict = {}
    acc_meter_dict['imgc'] = AverageMeter()
    acc_meter_dict['textc'] = AverageMeter()
    acc_meter_dict['vqa'] = AverageMeter()

    loss_meter_dict = {}
    for ta in ta_sel:
        loss_meter_dict[ta] = AverageMeter()
    psnr_meter = AverageMeter()
    return acc_meter_dict, loss_meter_dict, psnr_meter

def _next_task_batch(task_name, data_iters, data_loaders, device):
    try:
        batch = next(data_iters[task_name])
    except StopIteration:
        data_iters[task_name] = iter(data_loaders[task_name])
        batch = next(data_iters[task_name])

    imgs, texts, speechs, targets = None, None, None, None
    if task_name.startswith('img'):
        imgs = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
    elif task_name.startswith('text'):
        texts = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
    elif task_name.startswith('vqa'):
        imgs = batch[0].to(device, non_blocking=True)
        texts = batch[1].to(device, non_blocking=True)
        targets = batch[2].to(device, non_blocking=True)
    elif task_name.startswith('msa'):
        imgs = batch[0].to(device, non_blocking=True)
        texts = batch[1].to(device, non_blocking=True)
        speechs = batch[2].to(device, non_blocking=True)
        targets = batch[3].to(device, non_blocking=True)
    else:
        raise NotImplementedError(task_name)
    return [imgs, texts, speechs], targets

def _sample_train_tasks(task_names, tasks_per_step, task_weights):
    if tasks_per_step <= 0 or tasks_per_step >= len(task_names):
        return list(task_names)
    weights = np.array([task_weights.get(task, 1.0) for task in task_names], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(task_names), dtype=np.float64)
    probs = weights / weights.sum()
    sampled = np.random.choice(len(task_names), size=tasks_per_step, replace=False, p=probs)
    return [task_names[idx] for idx in sampled]

def _shared_trainable_parameters(model):
    raw_model = model.module if hasattr(model, 'module') else model
    params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('head.') or name.startswith('task_dict.'):
            continue
        params.append(param)
    return params

def _flatten_grads(loss, params, retain_graph):
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    flat_grads = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat_grads.append(torch.zeros_like(param).reshape(-1))
        else:
            flat_grads.append(grad.detach().reshape(-1))
    if len(flat_grads) == 0:
        return None
    return torch.cat(flat_grads)

def _print_gradient_conflicts(model, weighted_losses, step):
    if len(weighted_losses) < 2:
        return
    params = _shared_trainable_parameters(model)
    if len(params) == 0:
        return
    task_names = list(weighted_losses.keys())
    grad_vectors = {}
    for idx, task_name in enumerate(task_names):
        grad_vectors[task_name] = _flatten_grads(weighted_losses[task_name], params, retain_graph=True)
    pairs = []
    for left_idx in range(len(task_names)):
        for right_idx in range(left_idx + 1, len(task_names)):
            left_name = task_names[left_idx]
            right_name = task_names[right_idx]
            left_grad = grad_vectors[left_name]
            right_grad = grad_vectors[right_name]
            if left_grad is None or right_grad is None:
                continue
            cosine = F.cosine_similarity(left_grad, right_grad, dim=0).item()
            pairs.append(f"{left_name}/{right_name}: {cosine:.4f}")
    if pairs:
        print(f"Grad cosine step {step}: " + " | ".join(pairs))


def train_epoch_uni(model: torch.nn.Module, criterion: dict,
                data_dict: dict, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, loss_scaler, ta_sel, max_norm: float=0,
                start_steps=None,lr_schedule_values=None, wd_schedule_values=None, 
                update_freq=None, print_freq=10, metrics_writer=None):
    model.train(True)                                                         
    acc_meter_dict, loss_meter_dict, psnr_meter = meter(ta_sel)

    if loss_scaler is None:    
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    num_samples = _model_attr(model, 'num_samples', 5000)
    loss_weights = _model_attr(model, 'loss_weights', {})
    task_weights = _model_attr(model, 'task_weights', {})
    tasks_per_step = _model_attr(model, 'tasks_per_step', 1)
    grad_conflict_freq = _model_attr(model, 'grad_conflict_freq', 0)
    data_iters = {ta: iter(data_dict[ta]) for ta in ta_sel}
    num_batches = min(len(data_dict[ta]) for ta in ta_sel)
    for data_iter_step in range(num_batches):
        step = data_iter_step // update_freq
        it = start_steps + step  
        if (lr_schedule_values is not None or wd_schedule_values is not None) and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]                
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        selected_tasks = _sample_train_tasks(ta_sel, tasks_per_step, task_weights)
        task_results = {}
        weighted_losses = {}
        total_loss = 0
        for task_name in selected_tasks:
            batch_inputs, targets = _next_task_batch(task_name, data_iters, data_dict, device)
            raw_loss, outputs = train_class_batch_uni(
                task_name, model, batch_inputs, targets, criterion)
            loss_weight = loss_weights.get(task_name, 1.0)
            weighted_loss = raw_loss * loss_weight
            total_loss = total_loss + weighted_loss
            task_results[task_name] = {
                'inputs': batch_inputs,
                'targets': targets,
                'outputs': outputs,
                'raw_loss': raw_loss.detach().item(),
                'weighted_loss': weighted_loss.detach().item(),
                'batch_size': targets.shape[0],
            }
            weighted_losses[task_name] = weighted_loss
        loss = total_loss / max(len(selected_tasks), 1)
        loss_value = loss.item()
        ######  Error                              
        if not math.isfinite(loss_value):   
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if grad_conflict_freq > 0 and data_iter_step % grad_conflict_freq == 0:
            _print_gradient_conflicts(model, weighted_losses, data_iter_step)
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        min_lr,max_lr = 10., 0.
        for group in optimizer.param_groups:
            min_lr,max_lr = min(min_lr, group["lr"]),max(max_lr, group["lr"])

        for task_name, result in task_results.items():
            batch_inputs = result['inputs']
            targets = result['targets']
            outputs = result['outputs']
            batch_size = result['batch_size']
            loss_meter_dict[task_name].update(result['weighted_loss'], 1)
            if task_name.endswith('c'):
                acc_meter_dict[task_name].update((outputs.max(-1)[-1] == targets).float().mean().item(), n=batch_size)
            elif task_name.startswith('imgr'):
                imgs = batch_inputs[0]
                outputs = rearrange(outputs, 'b n (p c) -> b n p c', c=3)
                outputs = rearrange(outputs, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
                tr_imgs = torch.tensor((imgs*255).detach().cpu().numpy().astype(int).clip(0,255)).float()
                re_imgs = torch.tensor((outputs*255).detach().cpu().numpy().astype(int).clip(0,255)).float()
                mse_cal = nn.MSELoss()
                psnr_meter.update(10 * math.log10(255.0**2/(mse_cal(tr_imgs, re_imgs))), n=1)
            elif task_name.startswith('vqa'):
                acc_meter_dict[task_name].update((outputs.max(-1)[-1] == targets.max(-1)[-1]).float().mean().item(), n=batch_size)
        
        if print_freq > 0 and data_iter_step % print_freq == 0:
            metrics_record = {
                'event': 'train_step',
                'epoch': epoch,
                'step': data_iter_step,
                'num_steps': num_batches,
                'total_loss': loss_value,
                'lr': max_lr,
                'tasks': {},
            }
            print('Epoch %03d | step %05d/%05d | total_loss %.4f | lr %.3e' %
                  (epoch, data_iter_step, num_batches, loss_value, max_lr))
            for task_name in selected_tasks:
                result = task_results[task_name]
                task_record = {
                    'raw_loss': result['raw_loss'],
                    'weighted_loss': result['weighted_loss'],
                    'avg_weighted_loss': loss_meter_dict[task_name].avg,
                }
                msg = '  %-5s raw %.4f | weighted %.4f | avg %.4f' % (
                    task_name, result['raw_loss'], result['weighted_loss'], loss_meter_dict[task_name].avg)
                if task_name.endswith('c') or task_name.startswith('vqa'):
                    task_record['acc'] = acc_meter_dict[task_name].avg * 100
                    msg += ' | acc %.3f' % task_record['acc']
                elif task_name.startswith('imgr'):
                    task_record['psnr'] = psnr_meter.avg
                    msg += ' | psnr %.3f' % task_record['psnr']
                metrics_record['tasks'][task_name] = task_record
                print(msg)
            if metrics_writer is not None:
                metrics_writer(metrics_record)
              
    train_stat = None

    return train_stat 


def train_class_batch_vqa(ta_perform, model, imgs, texts, targets, criterion):
    if ta_perform.startswith('vqa'):
        outputs = model(img=imgs, text=texts, ta_perform=ta_perform, **_forward_snr_kwargs(model))
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
                  device: torch.device, criterion: torch.nn.Module, print_freq=10, max_batches=0):
    net.eval()
    snr_kwargs = _forward_snr_kwargs(net)
    loss_meter = AverageMeter()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_idx, (imgs,texts,speechs, targets) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            imgs, texts, speechs, targets = imgs.to(device), texts.to(device), speechs.to(device), targets.to(device)
            outputs = net(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform, **snr_kwargs)
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
        outputs = model(img=imgs, text=texts, speech=speechs, ta_perform=ta_perform, **_forward_snr_kwargs(model))
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
