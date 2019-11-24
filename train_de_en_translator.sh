#!/usr/bin/env bash
export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
# Train De-En model
python THUMT/thumt/bin/trainer.py \
    --input data/corpus.tc.32k.de.shuf data/corpus.tc.32k.en.shuf \
    --output de-en-model/ \
    --vocabulary data/vocab.32k.de.txt data/vocab.32k.en.txt \
    --model transformer \
    --validation data/dev/newstest2014.tc.32k.de \
    --references data/dev/newstest2014.tc.en \
    --parameters=batch_size=6250,constant_batch_size=false,device_list=[0,1],update_cycle=2,keep_checkpoint_max=5,eval_steps=1000,beam_size=4,decode_alpha=1.4,train_steps=200000
