export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/bert:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

python THUMT/thumt/bin/train_with_bert.py \
    --input data/corpus.bert.tc.de.shuf data/corpus.bert.tc.de.shuf \
    --output en-de-bert-model/decoder \
    --vocabulary ckpt/bert/multi_cased_L-12_H-768_A-12/vocab.txt ckpt/bert/multi_cased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file ckpt/bert/multi_cased_L-12_H-768_A-12/bert_config.json \
    --init_bert_checkpoint ckpt/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
    --model bert-transformer \
    --validation data/dev/newstest2014.bert.tc.de \
    --references data/dev/newstest2014.tc.de \
    --parameters=batch_size=3125,constant_batch_size=false,device_list=[0,1,2,3],update_cycle=2,keep_checkpoint_max=5,eval_steps=500,beam_size=4,decode_alpha=0.6,train_steps=200000,layer_indexes=[-1,-2,-3,-4]
