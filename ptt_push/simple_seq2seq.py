# -*- coding: utf-8 -*-

import argparse
from dataset import get_train_dataset
import tensorflow as tf

BATCH_SIZE = 4
UNK_TOKEN = '<unk>'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', dest='source_file', required=True)
    parser.add_argument('-t', '--target', dest='target_file', required=True)
    parser.add_argument('-sv', '--source_vocab', dest='source_vocab_file',
                        required=True)
    parser.add_argument('-tv', '--target_vocab', dest='target_vocab_file',
                        help='Share with source_vocab if empty')

    return parser.parse_args()


def main(args):
    dataset = get_train_dataset(
        args.source_file, args.target_file,
        args.source_vocab_file, args.target_vocab_file, batch_size= BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    (batch_enc_in, batch_dec_in, batch_dec_out) = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        print(sess.run([batch_enc_in, batch_dec_in, batch_dec_out]))
        print(sess.run([batch_enc_in, batch_dec_in, batch_dec_out]))
        #print(sess.run([batch_enc_in, batch_dec_in, batch_dec_out]))
        #print(sess.run([batch_enc_in, batch_dec_in, batch_dec_out]))
    # embedding
    # encoder
    # decoder

    pass


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
