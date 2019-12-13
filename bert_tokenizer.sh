export PYTHONPATH=/home1/private/zhm/201910_cotrain/dep/bert:$PYTHONPATH

python bert_tokenizer.py \
    --input data/dev/newstest2014.tc.de \
    --output data/dev/newstest2014.bert.tc.de \
    --vocab_file ckpt/bert/multi_cased_L-12_H-768_A-12/vocab.txt
