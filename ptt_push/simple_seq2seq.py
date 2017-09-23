# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from dataset import TrainDataset, TestDataset
from model import Seq2SeqModel
from optimize import get_train_step
import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.contrib.learn import ModeKeys

BATCH_SIZE = 32
EMBEDDING_SIZE = 3
HIDDEN_SIZE = 10
LEARNING_RATE = 0.01
MAX_GRADIENT_NORM = 5.0
EPOCH_MAX = 10

UNK_TOKEN = '<unk>'

train_graph = tf.Graph()
infer_graph = tf.Graph()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', dest='source_file', required=True)
    parser.add_argument('-t', '--target', dest='target_file', required=True)
    parser.add_argument('-sv', '--source_vocab', dest='source_vocab_file',
                        required=True)
    parser.add_argument('-tv', '--target_vocab', dest='target_vocab_file',
                        help='Share with source_vocab if empty')
    parser.add_argument('--infer_source', dest='infer_source_file')
    parser.add_argument('--infer_out', dest='infer_out_file',
                        help='Write inference outcomes to a file')

    return parser.parse_args()


def weight_debug(sess, iterator, model):
    sess.run(iterator.initializer)
    print(tf.global_variables())
    output = sess.run([
        model._batch_enc_in,
        model.batch_enc_embed,
        'rnn/basic_lstm_cell/kernel:0',
        'rnn/basic_lstm_cell/bias:0',
        model.encoder_state,
        model._batch_dec_in,
        'decoder/basic_lstm_cell/kernel:0',
        'decoder/basic_lstm_cell/bias:0',
        model.decoder_proj_layer.kernel,
        model.batch_rnn_output,
        model.batch_sample_id,
        model._batch_dec_out,
    ])
    # for o in output:
    #     print(o)
    print(output[0])
    print(output[9])
    print(output[10])
    print(output[11])


def train(args):
    print('=== Building training graph... ===')
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

        train_step = get_train_step(model.loss, LEARNING_RATE,
                                    max_norm=MAX_GRADIENT_NORM)

        model_saver = tf.train.Saver(var_list=tf.global_variables())

    print('=== Run training... ===')
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())

        batch_amount = None
        for epoch_i in range(EPOCH_MAX):
            sess.run(iterator.initializer)

            # XXX: To avoid warning flood from the iterator,
            #      remember how many batches to run in the first epoch.
            #      https://github.com/tensorflow/tensorflow/issues/12414
            batch_i = 0
            while batch_amount is None or batch_i < batch_amount:
                try:
                    _, loss_val = sess.run([train_step, model.loss])
                    batch_i += 1
                except tf.errors.OutOfRangeError:
                    if batch_amount is None:
                        batch_amount = batch_i
                        print('BATCH AMOUNT {}'.format(batch_amount))
                    break

            print('Epoch {}, loss: {}'.format(epoch_i, loss_val))

        model_saver.save(sess, 'tmp_model/model.ckpt')


def infer(args):
    # HACK: get target vocabulary size, just for computation graph creation
    with open(args.target_vocab_file) as f:
        target_vocab_size = len(f.readlines())

    with infer_graph.as_default():
        dataset = TestDataset(args.infer_source_file, args.source_vocab_file, 1)

        iterator = dataset.get_tf_dataset().make_initializable_iterator()

        model = Seq2SeqModel(source_vocab_size=dataset.source_vocab_size,
                             target_vocab_size=target_vocab_size,
                             embedding_size=EMBEDDING_SIZE,
                             hidden_state_size=HIDDEN_SIZE,
                             mode=ModeKeys.INFER)
        model.build(iterator)

        reverse_target_vocab_table = lookup.index_to_string_table_from_file(
            args.target_vocab_file, default_value='<UNK>')
        # XXX: TextFileStringTableInitializer uses int64 keys
        output_words = reverse_target_vocab_table.lookup(
            tf.cast(model.batch_sample_id, tf.int64))

        model_saver = tf.train.Saver(var_list=tf.global_variables())

    infer_f_out = None
    if args.infer_out_file is not None:
        try:
            infer_f_out = open(args.infer_out_file, mode='w')
        except IOError:
            print('Cannot open {} for writing inference.'.format(
                args.infer_out_file))

    with tf.Session(graph=infer_graph) as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)

        model_saver.restore(sess, 'tmp_model/model.ckpt')

        while True:
            try:
                output, out_ids = sess.run([output_words, model.batch_sample_id])
                eos_id = model.target_eos_id
                for sentence, ids in zip(output, out_ids):
                    if eos_id in ids:
                        end_index = ids.tolist().index(eos_id)
                    else:
                        end_index = None
                    translation = ' '.join([
                        b.decode() for b in sentence[:end_index]])
                    print(translation, file=infer_f_out)
            except tf.errors.OutOfRangeError:
                break


def main(args):
    train(args)
    if args.infer_source_file is not None:
        infer(args)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(parsed_args)
