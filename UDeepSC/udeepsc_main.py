import datetime
import json
import numpy as np
import time
import torch
import utils
import model   
import torch.backends.cudnn as cudnn

from engine import *
from pathlib import Path 
from base_args import get_args
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import get_model, sel_criterion_train, sel_criterion_test, load_checkpoint
from datasets import build_dataset_train, build_dataset_test, BatchSchedulerSampler, collate_fn, build_dataloader

############################################################
DEFAULT_LOSS_WEIGHTS = {
    'imgc': 1.0,
    'imgr': 30.0,
    'textc': 0.6,
    'textr': 5.0,
    'vqa': 3.0,
    'msa': 8.0,
}

def parse_weight_overrides(items, defaults=None):
    weights = dict(defaults or {})
    for item in items:
        if ':' not in item:
            raise ValueError(f"Weight override must use task:value format, got {item}")
        task, value = item.split(':', 1)
        weights[task] = float(value)
    return weights

def seed_initial(seed=0):
    seed += utils.get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_model_attr(net, name, value):
    raw = net.module if hasattr(net, 'module') else net
    setattr(raw, name, value)

def get_eval_tasks(args):
    if args.test_tasks:
        return args.test_tasks
    if args.ta_perform:
        return [args.ta_perform]
    return []

def build_eval_loaders(args, device, eval_tasks):
    eval_loaders = {}
    original_ta_perform = args.ta_perform
    original_input_size = args.input_size
    try:
        for task in eval_tasks:
            args.ta_perform = task
            valset = build_dataset_test(is_train=False, args=args)
            sampler_val = torch.utils.data.SequentialSampler(valset)
            Collate_fn = collate_fn if task.startswith('msa') else None
            dataloader_val = torch.utils.data.DataLoader(
                valset, sampler=sampler_val, batch_size=int(1.0 * args.batch_size),
                num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
                collate_fn=Collate_fn)
            criterion_test = sel_criterion_test(args, device)
            eval_loaders[task] = {
                'dataloader': dataloader_val,
                'valset': valset,
                'criterion': criterion_test,
            }
    finally:
        args.ta_perform = original_ta_perform
        args.input_size = original_input_size
    return eval_loaders

def evaluate_task(args, task, model, dataloader_val, valset, device, criterion_test):
    if dataloader_val is None:
        return
    original_ta_perform = args.ta_perform
    args.ta_perform = task
    snr_values = args.test_snr_list if args.test_snr_list else [args.test_snr]
    try:
        for snr in snr_values:
            set_model_attr(model, 'test_snr', snr)
            print(f"Eval task: {task}, SNR: {snr} dB")
            if task.startswith('img') or task.startswith('text'):
                test_stats = evaluate(ta_perform=task,
                                      net=model, dataloader=dataloader_val,
                                      device=device, criterion=criterion_test,
                                      max_batches=args.eval_batches)
                if task.startswith('imgc') or task.startswith('textc'):
                    print(f"Accuracy of the network on the {len(valset)} test samples: {test_stats['acc']*100:.3f}")
                elif task.startswith('imgr'):
                    print(f"Average PSNR on the {len(valset)} test samples: {test_stats['psnr']:.3f}dB")
                elif task.startswith('textr'):
                    print(f"Average BLEU on the {len(valset)} test samples: {test_stats['bleu']:.3f}")
            elif task.startswith('msa'):
                test_stats = evaluate_msa(ta_perform=task,
                                          net=model, dataloader=dataloader_val,
                                          device=device, criterion=criterion_test,
                                          max_batches=args.eval_batches)
                print(f"Accuracy of the network on the {len(valset)} test samples: {test_stats['acc']*100:.3f}")
            elif task.startswith('vqa'):
                test_stats = evaluate_vqa(ta_perform=task,
                                          net=model, dataloader=dataloader_val,
                                          device=device, criterion=criterion_test,
                                          max_batches=args.eval_batches)
                print("Overall Accuracy is: %.02f" % (test_stats['overall']))
                print("Per Answer Type Accuracy is the following:")
                for ansType in test_stats['perAnswerType']:
                    print("%s : %.02f" % (ansType, test_stats['perAnswerType'][ansType]))
    finally:
        args.ta_perform = original_ta_perform

def evaluate_tasks(args, model, eval_loaders, device):
    for task, eval_data in eval_loaders.items():
        evaluate_task(args, task, model, eval_data['dataloader'], eval_data['valset'],
                      device, eval_data['criterion'])

def main(args):
    ### Configuration
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed_initial(seed=args.seed)
    ####################################### Get the model
    model = get_model(args)
    model.train_snr = args.train_snr
    model.test_snr = args.test_snr
    model.loss_weights = parse_weight_overrides(args.loss_weights, DEFAULT_LOSS_WEIGHTS)
    model.task_weights = parse_weight_overrides(args.task_weights, {})
    model.tasks_per_step = args.tasks_per_step
    model.grad_conflict_freq = args.grad_conflict_freq
    model.num_samples = args.num_samples
    if args.resume:
        print(args.resume)
        checkpoint_model = load_checkpoint(model, args)
        
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

        
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module  
    
    print("------------------------------------------------------")
    ############## Get the data and dataloader
    
    ta_sel = args.train_tasks
    print(f"Train tasks: {ta_sel}")
    print(f"Train SNR(dB): {args.train_snr}; Test SNR(dB): {args.test_snr}")
    print(f"Tasks per optimizer step: {'all' if args.tasks_per_step == 0 else args.tasks_per_step}")
    print(f"Loss weights: {model.loss_weights}")
    if model.task_weights:
        print(f"Task sampling weights: {model.task_weights}")
    metrics_file = None
    metrics_path = None
    if args.output_dir and not args.eval and utils.is_main_process():
        metrics_path = Path(args.output_dir) / 'train_metrics.jsonl'
        metrics_file = open(metrics_path, 'w', encoding='utf-8')

        def write_metrics(record):
            metrics_file.write(json.dumps(record, sort_keys=True) + '\n')
            metrics_file.flush()
    else:
        def write_metrics(record):
            return None

    if metrics_path is not None:
        print(f"Structured metrics: {metrics_path}")
        write_metrics({
            'event': 'run_start',
            'epochs': args.epochs,
            'train_tasks': ta_sel,
            'train_snr': args.train_snr,
            'test_snr': args.test_snr,
            'tasks_per_step': args.tasks_per_step,
            'loss_weights': model.loss_weights,
        })
    if args.eval:
        trainloader_group = None
    else:
        trainset_group = build_dataset_train(is_train=True, ta_sel=ta_sel, args=args)
        trainloader_group = build_dataloader(ta_sel,trainset_group, args=args)

    ############################################## Get the test dataloader
    eval_tasks = get_eval_tasks(args)
    build_eval_loader = bool(eval_tasks and (args.eval or args.eval_freq > 0))
    if build_eval_loader:
        print(f"Eval tasks: {eval_tasks}")
        eval_loaders = build_eval_loaders(args, device, eval_tasks)
    else:
        eval_loaders = {}
    
    ################################## Auto load the model in the model record folder
    if args.eval:
        evaluate_tasks(args, model, eval_loaders, device)
        exit(0)

    ############################# Get the optimizer and the other training settings
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = max(args.num_samples // total_batch_size, 1)

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    
    
    ###################################################### Get the criterion
    criterion_train = sel_criterion_train(args,ta_sel, device)

    ################################## Start Training the T-DeepSC
    print(f"Start training for {args.epochs} epochs")
    max_accuracy = 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for trainloader in trainloader_group.values():
                trainloader.sampler.set_epoch(epoch)

        train_stats = train_epoch_uni(
                model, criterion_train, trainloader_group, optimizer, device, epoch, loss_scaler, 
                ta_sel, args.clip_grad,  start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, 
                update_freq=args.update_freq, print_freq=args.print_freq,
                metrics_writer=write_metrics)
   
        
        inter_time = time.time() - start_time
        inter_time_str = str(datetime.timedelta(seconds=int(inter_time)))
        print('Training time {}'.format(inter_time_str))
        write_metrics({
            'event': 'epoch_end',
            'epoch': epoch,
            'elapsed': inter_time_str,
        })

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=None)
        should_eval = args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0
        if eval_loaders and should_eval:
            print(args.output_dir)
            evaluate_tasks(args, model, eval_loaders, device)
       
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    write_metrics({
        'event': 'run_end',
        'elapsed': total_time_str,
    })
    if metrics_file is not None:
        metrics_file.close()


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
