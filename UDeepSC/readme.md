# Simulation illustration
CUDA_VISIBLE_DEVICES=0  python3  udeepsc_main.py \
    --model  TDeepSC_imgr_model   \
    --output_dir ckpt_record   \
    --batch_size 60 \
    --epochs 50  \
    --opt_betas 0.9 0.95  \
    --save_freq 1   \
    --ta_eval imgr \
    #--eval
