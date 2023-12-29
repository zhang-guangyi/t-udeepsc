CUDA_VISIBLE_DEVICES=2 python3  tdeepsc_main.py \
    --model  TDeepSC_vqa_model  \
    --output_dir ckpt_record  \
    --batch_size 30 \
    --input_size 224 \
    --lr  1e-4 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 2   \
    --ta_perform vqa  
  
   