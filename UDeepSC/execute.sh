CUDA_VISIBLE_DEVICES=1  python3  udeepsc_main.py \
    --model  UDeepSC_new_model  \
    --output_dir ckpt_record_12dB_single   \
    --batch_size 16 \
    --input_size 32 \
    --lr  6e-6 \
    --epochs 500  \
    --opt_betas 0.95 0.99  \
    --save_freq 2   \
    --ta_perform textr \
    --resume ckpt_record_12dB_single/ckpt_msa/checkpoint-203.pth\
    --eval
   
  

   

   
 
 
  
  