# -*- coding: utf-8 -*-

import argparse
from dataset import TrainDataset
from model import Seq2SeqModel
import tensorflow as tf

BATCH_SIZE = 4
EMBEDDING_SIZE = 1024
HIDDEN_SIZE = 128
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
    train_graph = tf.Graph()

    with train_graph.as_default():
        dataset = TrainDataset(args.source_file, args.target_file,
                               args.source_vocab_file, args.target_vocab_file,
                               batch_size=BATCH_SIZE)

        iterator = dataset.get_tf_dataset().make_initializable_iterator()

        model = Seq2SeqModel(source_vocab_size=dataset.source_vocab_size,
                             target_vocab_size=dataset.target_vocab_size,
                             embedding_size=EMBEDDING_SIZE,
                             hidden_state_size=HIDDEN_SIZE)
        model.build(iterator)

    with tf.Session(graph=train_graph) as sess:
        sess.run(iterator.initializer)
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        print(model)
        print(model.encoder_emb_weight)
        print(model.encoder_state)
        print(model.encoder_output)

    pass


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
