# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.contrib.data import Dataset, TextLineDataset


def get_train_dataset(source_file, target_file,
                      source_vocab_file, target_vocab_file, batch_size,
                      reverse_input=True, input_max=None, output_max=None):
    source_vocab_map = lookup.index_table_from_file(source_vocab_file,
                                                    num_oov_buckets=1)
    if target_vocab_file is None:
        target_vocab_map = source_vocab_map
    else:
        target_vocab_map = lookup.index_table_from_file(target_vocab_file,
                                                        num_oov_buckets=1)

    # HACK: get vocabulary size
    with open(target_vocab_file) as f:
        target_vocab_size = len(f.readlines())
    # target_unknown_id is target_vocab_size
    target_sos_id = tf.constant(target_vocab_size+1, dtype=tf.int64)
    target_eos_id = tf.constant(target_vocab_size+2, dtype=tf.int64)
    with open(source_vocab_file) as f:
        source_vocab_size = len(f.readlines())
    source_eos_id = tf.constant(source_vocab_size+1, dtype=tf.int64)

    # read lines of data files; they should have the same numbers of lines
    src_dataset = TextLineDataset(source_file)
    tgt_dataset = TextLineDataset(target_file)

    # combine two together
    dataset = Dataset.zip((src_dataset, tgt_dataset))

    # line -> words
    dataset = dataset.map(lambda s, t: (tf.string_split([s]).values,
                                        tf.string_split([t]).values))

    # words -> unique ids
    dataset = dataset.map(lambda s, t: (source_vocab_map.lookup(s),
                                        target_vocab_map.lookup(t)))

    # source,target -> source, target (in), target (out)
    dataset = dataset.map(lambda s, t: (s,
                                        tf.concat([[target_sos_id], t], 0),
                                        tf.concat([t, [target_eos_id]], 0)))

    # padded for mini-batch: source, target (in), target (out)
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(tf.TensorShape([None]),
                       tf.TensorShape([None]),
                       tf.TensorShape([None])),
        padding_values=(source_eos_id, target_eos_id, target_eos_id)
    )
    return dataset
