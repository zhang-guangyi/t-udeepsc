CUDA_VISIBLE_DEVICES=2  python3  tdeepsc_main.py \
    --model  TDeepSC_imgr_model  \
    --output_dir ckpt_record   \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 5   \
    --ta_perform imgr \
    # --eval 

CUDA_VISIBLE_DEVICES=2  python3  tdeepsc_main.py \
    --model  TDeepSC_imgc_model  \
    --output_dir ckpt_record   \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 5   \
    --ta_perform imgc \
    # --eval 


CUDA_VISIBLE_DEVICES=2  python3  tdeepsc_main.py \
    --model  TDeepSC_textc_model  \
    --output_dir ckpt_record   \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 5   \
    --ta_perform textc \
    # --eval 


CUDA_VISIBLE_DEVICES=2  python3  tdeepsc_main.py \
    --model  TDeepSC_textr_model  \
    --output_dir ckpt_record   \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 5   \
    --ta_perform textr \
    # --eval 

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
    --model  TDeepSC_vqa_model  \
    --output_dir ckpt_record   \
    --batch_size 50 \
    --input_size 32 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 5   \
    --ta_perform msa \
    # --eval 
