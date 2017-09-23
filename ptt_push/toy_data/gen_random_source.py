# -*- coding: utf-8 -*-

import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', dest='vocab_file', required=True)
    parser.add_argument('--num', dest='sentence_num', default=100)
    parser.add_argument('--min_len', default=3)
    parser.add_argument('--max_len', default=10)

    return parser.parse_args()


def get_vocabs(file):
    with open(file) as f_in:
        vocabs = [v.rstrip() for v in f_in.readlines()]
    return vocabs


def gen_sentence(vocabs, min_len, max_len):
    vocab_size = len(vocabs)
    sentence_len = random.randint(min_len, max_len)
    words = [vocabs[random.randint(0, vocab_size-1)]
             for i in range(sentence_len)]

    return ' '.join(words)


def main(args):
    vocabs = get_vocabs(args.vocab_file)
    for i in range(args.sentence_num):
        s = gen_sentence(vocabs, args.min_len, args.max_len)
        print(s)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
