export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

# Train En-En model with frozen (loaded) encoder
python THUMT/thumt/bin/train_frozen_autoencoder.py \
    --input data/corpus.tc.32k.en.shuf data/corpus.tc.32k.en.shuf \
    --vocabulary data/vocab.32k.en.txt data/vocab.32k.en.txt \
    --output en-decoder-model/5m \
    --model transformer \
    --frozen \
    --encoder_checkpoint ckpt/en-de-trans/model.ckpt-161001 \
    --validation data/dev/newstest2014.tc.32k.en \
    --references data/dev/newstest2014.tc.en \
    --parameters=batch_size=6250,constant_batch_size=false,device_list=[0,1],update_cycle=2,keep_checkpoint_max=5,eval_steps=1000,beam_size=4,decode_alpha=1.4,train_steps=200000
