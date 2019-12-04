export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

python THUMT/thumt/bin/compare_ckpt.py \
    --model transformer \
    --metrics d1_norm \
    --output d1_norm.csv \
    --checkpoints ckpt/de-ae/model.ckpt-200000 de-decoder-model/5m-val-copy/eval/model.ckpt-199000  ckpt/en-de-trans/model.ckpt-161001 ckpt/de-en-trans/model.ckpt-187000
