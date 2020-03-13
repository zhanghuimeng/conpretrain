export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/bert:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python THUMT/thumt/bin/translator_bert.py \
    --checkpoints /data/disk6/private/zhm/en-de-bert-model/decoder/model.ckpt-100000 \
    --init_bert_checkpoint ckpt/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
    --vocabulary ckpt/bert/multi_cased_L-12_H-768_A-12/vocab.txt ckpt/bert/multi_cased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file ckpt/bert/multi_cased_L-12_H-768_A-12/bert_config.json \
    --models bert-transformer \
    --verbose \
    --input data/dev/bert/newstest2014.bert.de \
    --output bert-trans/translation-100000.out \
    --parameters=device_list=[4,5,6,7],beam_size=4,decode_alpha=0.6,layer_indexes=[-1,-2,-3,-4]
