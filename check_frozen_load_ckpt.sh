export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=6

# 检查frozen的模型是否正确load了两个ckpt

python THUMT/thumt/bin/compare_ckpt.py \
    --model transformer \
    --metrics d1_norm \
    --output d1_norm.csv \
    --checkpoints en-de-concat-model/small/eval/model.ckpt-168000 ckpt/en-de-trans/model.ckpt-161001 ckpt/de-frozen-ae/model.ckpt-186001
