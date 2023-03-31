TF_ENABLE_ONEDNN_OPTS=2 CUDA_VISIBLE_DEVICES=3  python3  tdeepsc_main.py \
    --model  TDeepSC_textr_model  \
    --output_dir ckpt_record   \
    --batch_size 5 \
    --input_size 32 \
    --lr  3e-4 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 2   \
    --ta_perform textr \
    # --resume /Data1/zhangguangyi/SemanRes2/JSACCode/TDeepSC_Base/ckpt_record/ckpt_textc/checkpoint-44.pth  \
    # --eval 