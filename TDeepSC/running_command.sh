CUDA_VISIBLE_DEVICES=2  python3  tdeepsc_main.py \
    --model  TDeepSC_imgr_model  \
    --output_dir ckpt_record   \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 3   \
    --ta_perform imgr \
    # --eval 

TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=3  python3  tdeepsc_main.py \
    --model  TDeepSC_imgc_model  \
    --output_dir ckpt_record   \
    --batch_size 100 \
    --input_size 32 \
    --lr  3e-4 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 2   \
    --ta_perform imgc \
    --resume ckpt_record/ckpt_imgc/checkpoint-119-use.pth \


CUDA_VISIBLE_DEVICES=2  python3  tdeepsc_main.py \
    --model  TDeepSC_textc_model  \
    --output_dir ckpt_record   \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 3   \
    --ta_perform textc \
    # --eval 

1\12dB
2\-2dB


CUDA_VISIBLE_DEVICES=1 python3  tdeepsc_main.py \
    --model  TDeepSC_textr_model  \
    --output_dir ckpt_record   \
    --batch_size 30 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 2   \
    --ta_perform textr \
    --resume  ckpt_saved/ckpt_textr/checkpoint-textr-AWGN-T18dB.pth\
    --eval

CUDA_VISIBLE_DEVICES=2  python3  tdeepsc_main.py \
    --model  TDeepSC_vqa_model  \
    --output_dir ckpt_record   \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 5   \
    --ta_perform vqa \
    # --eval 

CUDA_VISIBLE_DEVICES=2  python3  tdeepsc_main.py \
    --model  TDeepSC_msa_model  \
    --output_dir ckpt_record   \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 3   \
    --ta_perform msa \
    # --eval 


CUDA_VISIBLE_DEVICES=0  python3  sim_main.py \
    --model  TDeepSC_vqa_model  \
    --output_dir ckpt_record   \
    --batch_size 30 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 5   \
    --ta_perform vqa \
    --resume ckpt_saved/ckpt_vqa/checkpoint-vqa-AWGN-T12dB.pth \
    --eval \
