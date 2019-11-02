export PYTHONPATH=/home2/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
python THUMT/thumt/bin/encoder_output.py \
    --model transformer \
    --input data/dev/newstest2015.tc.32k.de \
    --output encmodel/enc_output.npy \
    --vocabulary data/vocab.32k.de.txt \
    --checkpoint encmodel/model.ckpt-187000 \
    --parameters=device_list=[0]
