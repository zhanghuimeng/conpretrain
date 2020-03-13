# Tokenize a corpus file using BERT tokenizer.

import argparse
import bert.tokenization as tokenization

parser = argparse.ArgumentParser(
    description="Tokenize a corpus file using BERT tokenizer.",
    usage="bert_tokenizerpython.py [<args>] [-h | --help]"
)

parser.add_argument("--input", type=str,
                    help="Path of corpus file")
parser.add_argument("--vocab_file", type=str,
                    help="Path to BERT vocab file")
parser.add_argument("--output", type=str,
                    help="Path of output tokenized file")
parser.add_argument("--do_lower_case", action="store_true",
                    help="Do lower case or not")

args = parser.parse_args()

tokenizer = tokenization.FullTokenizer(
    vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

with open(args.input) as fr:
    with open(args.output, "w") as fw:
        i = 0
        for line in fr:
            fw.write(" ".join(tokenizer.tokenize(line)) + "\n")
            if i % 10000 == 0:
                print("Tokenizing line %d ..." % i)
            i += 1
