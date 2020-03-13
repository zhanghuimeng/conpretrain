export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/bert:$PYTHONPATH
#export PYTHONPATH=/data/disk1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
#export PYTHONPATH=/data/disk1/private/zhm/201910_cotrain/dep/bert:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python THUMT/thumt/bin/train_with_bert.py \
    --input data/bert/corpus.clean.bert.de.shuf.1w data/bert/corpus.clean.bert.de.shuf.1w \
    --output /data/disk6/private/zhm/en-de-bert-model/decoder/20200310 \
    --vocabulary ckpt/bert/multi_cased_L-12_H-768_A-12/vocab.txt ckpt/bert/multi_cased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file ckpt/bert/multi_cased_L-12_H-768_A-12/bert_config.json \
    --init_bert_checkpoint ckpt/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
    --model bert-transformer \
    --validation data/dev/bert/newstest2014.bert.de \
    --references data/dev/bert/newstest2014.clean.de \
    --parameters=batch_size=3125,constant_batch_size=false,device_list=[0,1,2,3,4,5,6,7],update_cycle=1,keep_checkpoint_max=1000,eval_steps=500,beam_size=4,decode_alpha=0.6,train_steps=200000,layer_indexes=[-1,-2,-3,-4]
