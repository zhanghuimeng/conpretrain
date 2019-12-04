export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

# Train De-De model with frozen (loaded) encoder
# The encoder is loaded from an autoencoder.
python THUMT/thumt/bin/train_frozen_autoencoder.py \
    --input data/corpus.tc.32k.de.shuf data/corpus.tc.32k.de.shuf \
    --vocabulary data/vocab.32k.de.txt data/vocab.32k.de.txt \
    --output de-decoder-model/5m-val-copy \
    --model transformer \
    --frozen \
    --encoder_checkpoint ckpt/de-ae/model.ckpt-200000 \
    --validation data/dev/newstest2014.tc.32k.de \
    --references data/dev/newstest2014.tc.de \
    --parameters=batch_size=6250,constant_batch_size=false,device_list=[0,1],update_cycle=2,keep_checkpoint_max=5,eval_steps=1000,beam_size=4,decode_alpha=1.4,train_steps=200000

# --input data/corpus.tc.32k.de.shuf data/corpus.tc.32k.de.shuf \
# --input data/dev/newstest2014.tc.32k.de data/dev/newstest2014.tc.32k.de \
