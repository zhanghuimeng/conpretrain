export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
# ckpt/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt

python THUMT/thumt/bin/compare_ckpt.py \
    --model transformer \
    --metrics d1_norm \
    --output d1_norm_0-40000.csv \
    --checkpoints /data/disk6/private/zhm/en-de-bert-model/decoder/20200201/model.ckpt-0 /data/disk6/private/zhm/en-de-bert-model/decoder/20200201/model.ckpt-40000
