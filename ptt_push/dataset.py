# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.contrib.data import Dataset, TextLineDataset


class TrainDataset:
    def __init__(self, source_file, target_file,
                 source_vocab_file, target_vocab_file, batch_size,
                 reverse_input=True, input_max=None, output_max=None):
        self._source_file = source_file
        self._target_file = target_file
        self._source_vocab_file = source_vocab_file
        self._target_vocab_file = target_vocab_file
        self._batch_size = batch_size
        self._reverse_input = reverse_input
        self._input_max = input_max
        self._output_max = output_max

        # HACK: get vocabulary size
        with open(target_vocab_file) as f:
            self._target_vocab_size = len(f.readlines())
        with open(source_vocab_file) as f:
            self._source_vocab_size = len(f.readlines())

    @property
    def source_vocab_size(self):
        return self._source_vocab_size

    @property
    def target_vocab_size(self):
        return self._target_vocab_size

    def get_tf_dataset(self):
        source_vocab_map = lookup.index_table_from_file(self._source_vocab_file,
                                                        num_oov_buckets=1)
        if self._target_vocab_file is None:
            target_vocab_map = self._source_vocab_map
        else:
            target_vocab_map = lookup.index_table_from_file(
                self._target_vocab_file, num_oov_buckets=1)

        # unknown_id is vocab_size
        target_sos_id = tf.constant(self._target_vocab_size+1, dtype=tf.int64)
        target_eos_id = tf.constant(self._target_vocab_size+2, dtype=tf.int64)
        source_eos_id = tf.constant(self._source_vocab_size+1, dtype=tf.int64)

        # read lines of data files; they should have the same numbers of lines
        src_dataset = TextLineDataset(self._source_file)
        tgt_dataset = TextLineDataset(self._target_file)

        # combine two together
        dataset = Dataset.zip((src_dataset, tgt_dataset))
        #dataset = dataset.shuffle(buffer_size=self._batch_size*1024, seed=1)

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

        # append data sequence length, for seq2seq Helper
        dataset = dataset.map(lambda s, t_i, t_o: (s, t_i, t_o, tf.size(t_i)))

        # padded for mini-batch: source, target (in), target (out), target len
        # where we don't pad for target len
        dataset = dataset.padded_batch(
            batch_size=self._batch_size,
            padded_shapes=(tf.TensorShape([None]),
                           tf.TensorShape([None]),
                           tf.TensorShape([None]),
                           tf.TensorShape([])),
            padding_values=(source_eos_id, target_eos_id, target_eos_id, 0)
        )
        return dataset


class TestDataset:
    def __init__(self, source_file, source_vocab_file, batch_size,
                 reverse_input=True, input_max=None):
        self._source_file = source_file
        self._source_vocab_file = source_vocab_file
        self._batch_size = batch_size
        self._reverse_input = reverse_input
        self._input_max = input_max

        # HACK: get vocabulary size
        with open(source_vocab_file) as f:
            self._source_vocab_size = len(f.readlines())

    @property
    def source_vocab_size(self):
        return self._source_vocab_size

    def get_tf_dataset(self):
        source_vocab_map = lookup.index_table_from_file(self._source_vocab_file,
                                                        num_oov_buckets=1)
        # unknown_id is vocab_size
        source_eos_id = tf.constant(self._source_vocab_size+1, dtype=tf.int64)

        # read lines of data files
        dataset = TextLineDataset(self._source_file)

        # line -> words
        dataset = dataset.map(lambda s: tf.string_split([s]).values)

        # words -> unique ids
        dataset = dataset.map(lambda s: source_vocab_map.lookup(s))

        # Infer 1 sentence at a time
        dataset = dataset.batch(1)

        return dataset
