export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

# 直接从头开始训练，检查是否能训练出一个正常的模型
python THUMT/thumt/bin/train_enc_and_dec.py \
    --input data/corpus.tc.32k.en.shuf data/corpus.tc.32k.de.shuf \
    --output en-de-concat-model/small-no-pretrain \
    --vocabulary data/vocab.32k.en.txt data/vocab.32k.de.txt \
    --no_frozen \
    --model transformer \
    --validation data/dev/newstest2014.tc.32k.en \
    --references data/dev/newstest2014.tc.de \
    --parameters=batch_size=6250,constant_batch_size=false,device_list=[0,1],update_cycle=2,keep_checkpoint_max=5,eval_steps=1000,beam_size=4,decode_alpha=1.4,train_steps=200000,adapt_mode=small-random
