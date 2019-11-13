export PYTHONPATH=/home2/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
python THUMT/thumt/bin/train_decoder.py \
    --input encmodel/enc_output.tfrecords data/corpus.tc.32k.10w.de.shuf \
    --output de-decoder-model \
    --vocabulary data/vocab.32k.de.txt \
    --model transformer \
    --validation encmodel/enc_output_dev.tfrecords \
    --references data/dev/newstest2015.tc.32k.de \
    --parameters=batch_size=32,constant_batch_size=true,device_list=[0,5],update_cycle=2,keep_checkpoint_max=5,eval_steps=1000,train_steps=20000
