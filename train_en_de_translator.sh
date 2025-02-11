#!/usr/bin/env bash
export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
# Train En-De model
python THUMT/thumt/bin/trainer.py \
    --input data/corpus.tc.32k.en.shuf data/corpus.tc.32k.de.shuf \
    --output en-de-model/debug \
    --vocabulary data/vocab.32k.en.txt data/vocab.32k.de.txt \
    --model transformer \
    --validation data/dev/newstest2014.tc.32k.en \
    --references data/dev/newstest2014.tc.de \
    --parameters=batch_size=6250,constant_batch_size=false,device_list=[0,1],update_cycle=2,keep_checkpoint_max=5,eval_steps=1000,beam_size=4,decode_alpha=0.6,train_steps=200000
