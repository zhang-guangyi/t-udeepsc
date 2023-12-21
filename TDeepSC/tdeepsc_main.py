import datetime
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
from utils import get_model, sel_criterion, load_checkpoint
from utils import NativeScalerWithGradNormCount as NativeScaler
from datasets import build_dataset, BatchSchedulerSampler, collate_fn

############################################################
def seed_initial(seed=0):
    seed += utils.get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    ### Configuration
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed_initial(seed=args.seed)
    ####################################### Get the model
    model = get_model(args)
    if args.resume:
        checkpoint_model = load_checkpoint(model, args)
        
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    # import timm
    # model = timm.create_model("vit_small_patch32_224", pretrained=True)
    # model.head = nn.Linear(model.head.in_features, 10)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=> Number of params: {} M'.format(n_parameters / 1e6))
    print('')
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module  
    
    ############## Get the data and dataloader
    
    # for name, param in model.named_parameters():
    #     print(name)
    #     if name.startswith('text'):
    #         param.requires_grad=False
    #     else:
    #         param.requires_grad=True
        
    
    
    trainset = build_dataset(is_train=True, args=args)

    Collate_fn = collate_fn if args.ta_perform.startswith('msa') else None 
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                            sampler=torch.utils.data.RandomSampler(trainset),
                                            num_workers=args.num_workers, pin_memory=True,
                                            batch_size=args.batch_size, shuffle=False,collate_fn=Collate_fn)
    
    ############################################## Get the test dataloader
    valset = None
    if args.ta_perform:
        valset = build_dataset(is_train=False, args=args)
        sampler_val = torch.utils.data.SequentialSampler(valset)
    else:
        valset = None

    if valset is not None:
        dataloader_val = torch.utils.data.DataLoader(
            valset, sampler=sampler_val, batch_size=int(1.0 * args.batch_size),
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False, collate_fn=Collate_fn)
    else:
        dataloader_val = None
    
    ############################# Get the optimizer and the other training settings
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = args.num_samples // total_batch_size

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
    criterion = sel_criterion(args).to(device)
    
    ################################## Auto load the model in the model record folder
    
    
    if args.eval:
        if args.ta_perform.startswith('img') or args.ta_perform.startswith('text'):
            test_stats = evaluate(ta_perform=args.ta_perform, 
                                net=model, dataloader=dataloader_val, 
                                device=device, criterion=criterion)
            if args.ta_perform.startswith('imgc') or args.ta_perform.startswith('textc'):
                print(f"Accuracy of the network on the {len(valset)} test samples: {test_stats['acc']*100:.3f}")
            elif args.ta_perform.startswith('imgr'):
                print(f"Average PSNR on the {len(valset)} test samples: {test_stats['psnr']:.3f}dB")
            elif args.ta_perform.startswith('textr'):
                print(f"Average BLEU on the {len(valset)} test samples: {test_stats['bleu']:.3f}")
        elif args.ta_perform.startswith('msa'):
            test_stats = evaluate_msa(ta_perform=args.ta_perform, 
                                net=model, dataloader=dataloader_val, 
                                device=device, criterion=criterion)
            print(f"Accuracy of the network on the {len(valset)} test samples: {test_stats['acc']*100:.3f}")
        
        elif args.ta_perform.startswith('vqa'):
            test_stats = evaluate_vqa(ta_perform=args.ta_perform, 
                                net=model, dataloader=dataloader_val, 
                                device=device, criterion=criterion)
            print("Overall Accuracy is: %.02f" % (test_stats['overall']))
            print("Per Answer Type Accuracy is the following:")
            for ansType in test_stats['perAnswerType']:
                print("%s : %.02f" % (ansType, test_stats['perAnswerType'][ansType]))
        exit(0)
    # utils.save_model(
    #                 args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
    #                 loss_scaler=loss_scaler, epoch=10, model_ema=None)
    ################################## Start Training the T-DeepSC
    print(f"Start training for {args.epochs} epochs")
    max_accuracy = 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)

        if args.ta_perform.startswith('img') or args.ta_perform.startswith('text'):
            train_stats = train_epoch_it(
                    model, criterion, trainloader, optimizer, device, epoch, loss_scaler, 
                    args.ta_perform, args.clip_grad,  start_steps=epoch * num_training_steps_per_epoch,
                    lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, 
                    update_freq=args.update_freq)
        elif args.ta_perform.startswith('vqa'):
            train_stats = train_epoch_vqa(
                    model, criterion, trainloader, optimizer, device, epoch, loss_scaler, 
                    args.ta_perform, args.clip_grad,  start_steps=epoch * num_training_steps_per_epoch,
                    lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, 
                    update_freq=args.update_freq)
        elif args.ta_perform.startswith('msa'):
            train_stats = train_epoch_msa(
                    model, criterion, trainloader, optimizer, device, epoch, loss_scaler, 
                    args.ta_perform, args.clip_grad,  start_steps=epoch * num_training_steps_per_epoch,
                    lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, 
                    update_freq=args.update_freq)
      
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=None)
        if dataloader_val is not None:
            print(args.output_dir)
            if args.ta_perform.startswith('img') or args.ta_perform.startswith('text'):
                test_stats = evaluate(ta_perform=args.ta_perform, 
                                    net=model, dataloader=dataloader_val, 
                                    device=device, criterion=criterion)
            elif args.ta_perform.startswith('vqa'):
                test_stats = evaluate_vqa(ta_perform=args.ta_perform, 
                                    net=model, dataloader=dataloader_val, 
                                    device=device, criterion=criterion)
            else:
                test_stats = evaluate_msa(ta_perform=args.ta_perform, 
                                    net=model, dataloader=dataloader_val, 
                                    device=device, criterion=criterion)
            if args.ta_perform.startswith('imgc') or args.ta_perform.startswith('textc'):
                print(f"Accuracy of the network on the {len(valset)} test images: {test_stats['acc']*100:.3f}")
            elif args.ta_perform.startswith('imgr'):
                print(f"Average PSNR on the {len(valset)} test images: {test_stats['psnr']:.3f} dB")
            elif args.ta_perform.startswith('textr'):
                print(f"Average BLEU on the {len(valset)} test samples: {test_stats['bleu']:.3f}")
            elif args.ta_perform.startswith('msa'):
                print(f"Accuracy of the network on the {len(valset)} test samples: {test_stats['acc']*100:.3f}")
            elif args.ta_perform.startswith('vqa'):
                print("Overall Accuracy is: %.02f" % (test_stats['overall']))
                print("Per Answer Type Accuracy is the following:")
                for ansType in test_stats['perAnswerType']:
                    print("%s : %.02f" % (ansType, test_stats['perAnswerType'][ansType]))
       
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
