# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', dest='data_file', required=True)
    parser.add_argument('--max', dest='max_vocab_num', type=int)

    return parser.parse_args()


def gather_and_count(filename):
    word2count = defaultdict(int)
    with open(filename) as f_in:
        for line in f_in:
            line = line.rstrip()
            if not line:
                continue
            words = line.split(' ')
            for w in words:
                word2count[w] += 1

    return word2count


def main(args):
    data_file = args.data_file
    max_vocab_num = args.max_vocab_num

    word2count = gather_and_count(data_file)

    if max_vocab_num is None:
        for w in word2count.keys():
            print(w)
    else:
        for (i, w) in enumerate(sorted(word2count, key=word2count.get,
                                       reverse=True)):
            if i >= max_vocab_num:
                break
            print(w)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
