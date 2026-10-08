import argparse

IMGC_NUMCLASS = 10   # CIFAR Data
IMGR_LENGTH = 48   # CIFAR Data patch4/48   patch2/12
TEXTC_NUMCLASS = 2   # SST Data
TEXTR_NUMCLASS = 34000  # Size of vacab
VQA_NUMCLASS = 3129  # number of VQA class
MSA_NUMCLASS = 1     # number of MSA class
def get_args():
    parser = argparse.ArgumentParser('U-DeepSC training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_freq', default=15, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--chep', default='', type=str,
                        help='chceckpint path')
    
    # Dataset parameters
    parser.add_argument('--num_samples', default=50000, type=int,
                        help='number of data samples per epoch')
    parser.add_argument('--data_path', default='data/', type=str,
                        help='dataset path')
    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size for data')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')  
            
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.02)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--model', default='UDeepSC_Model', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)    
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--init_ckpt', default='',
                        help='initialization checkpoint, e.g. ViT .npz for UDeepSC_new_model image branch')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')

    # Augmentation parameters 
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    
    parser.add_argument('--save_ckpt', action='store_true')
    parser.set_defaults(save_ckpt=True)

    task_choices = ['imgc', 'textc', 'vqa', 'imgr', 'textr', 'msa']
    parser.add_argument('--ta_perform', default='', choices=task_choices,
                        type=str, help='Task used for evaluation during/after training')
    parser.add_argument('--test_tasks', nargs='+', default=[], choices=task_choices,
                        help='Tasks used for evaluation. Overrides ta_perform when provided.')
    parser.add_argument('--train_tasks', nargs='+', default=['msa', 'textr'], choices=task_choices,
                        help='Tasks used for multi-task training')
    parser.add_argument('--tasks_per_step', default=0, type=int,
                        help='Number of tasks aggregated in each optimizer step. 0 means all train_tasks.')
    parser.add_argument('--task_weights', nargs='*', default=[],
                        help='Task sampling weights, e.g. textr:0.4 msa:0.6. Used when tasks_per_step < number of train_tasks.')
    parser.add_argument('--loss_weights', nargs='*', default=[],
                        help='Loss weights, e.g. textr:5 msa:8. Defaults keep the original scaling.')
    parser.add_argument('--train_snr', nargs='+', default=[12.0], type=float,
                        help='Training SNR in dB. Pass one value for fixed SNR or multiple values to sample per batch.')
    parser.add_argument('--test_snr', default=12.0, type=float,
                        help='Evaluation SNR in dB')
    parser.add_argument('--test_snr_list', nargs='*', default=None, type=float,
                        help='Optional list of evaluation SNRs in dB. Overrides single test_snr for evaluation.')
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='Evaluate every N epochs during training. 0 disables periodic evaluation.')
    parser.add_argument('--eval_batches', default=0, type=int,
                        help='Maximum validation batches per evaluation. 0 means full validation set.')
    parser.add_argument('--grad_conflict_freq', default=0, type=int,
                        help='Print pairwise task gradient cosine similarity every N train steps. 0 disables diagnostics.')
    parser.add_argument('--print_freq', default=50, type=int,
                        help='Print training progress every N steps.')


    return parser.parse_args()
