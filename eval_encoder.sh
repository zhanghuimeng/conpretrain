export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/THUMT:$PYTHONPATH
python THUMT/thumt/bin/encoder_output.py \
    --model transformer \
    --input data/corpus.tc.32k.de.shuf \
    --output encmodel/enc_output_5m.tfrecords \
    --vocabulary data/vocab.32k.de.txt \
    --checkpoint encmodel/model.ckpt-187000 \
    --parameters=device_list=[0]

# --input data/corpus.tc.32k.de.shuf \
# --output encmodel/enc_output_5m.tfrecords \
# --input data/corpus.tc.32k.10w.de.shuf \
# --output encmodel/enc_output.tfrecords \
# --input data/dev/newstest2015.tc.32k.de \
# --output encmodel/enc_output_dev.tfrecords \
