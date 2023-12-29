CUDA_VISIBLE_DEVICES=1  python3  udeepsc_main.py \
    --model  UDeepSC_model  \
    --output_dir ckpt_record  \
    --batch_size 40 \
    --input_size 224 \
    --lr  6e-6 \
    --epochs 480  \
    --opt_betas 0.95 0.99  \
    --save_freq 2   \
    --ta_perform vqa \
    --eval
   
  
 
 

    

   