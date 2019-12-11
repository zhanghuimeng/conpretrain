export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=6

python THUMT/thumt/bin/translator.py \
    --input data/corrupted/bad.de \
    --output reconstructed.de \
    --checkpoints ckpt/de-frozen-ae/model.ckpt-186001 \
    --model transformer \
    --vocabulary data/vocab.32k.de.txt data/vocab.32k.de.txt \
    --parameters=device_list=[0],beam_size=4,decode_alpha=1.4
