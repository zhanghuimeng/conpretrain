export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
# 检查拼起来的enc-dec的效果
python THUMT/thumt/bin/train_enc_and_dec.py \
    --input data/corpus.tc.32k.en.shuf data/corpus.tc.32k.de.shuf \
    --output en-de-concat-model/bi-lstm \
    --vocabulary data/vocab.32k.en.txt data/vocab.32k.de.txt \
    --enc_ckpt ckpt/en-de-trans/model.ckpt-161001 \
    --dec_ckpt ckpt/de-frozen-ae/model.ckpt-186001 \
    --model transformer \
    --validation data/dev/newstest2014.tc.32k.en \
    --references data/dev/newstest2014.tc.de \
    --parameters=batch_size=6250,constant_batch_size=false,device_list=[0,1],update_cycle=2,keep_checkpoint_max=5,eval_steps=500,beam_size=4,decode_alpha=1.4,train_steps=200000,adapt_mode=bi-lstm
