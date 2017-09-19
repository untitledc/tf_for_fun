# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.rnn import BasicLSTMCell


class Seq2SeqModel:
    def __init__(self, source_vocab_size, target_vocab_size, embedding_size,
                 hidden_state_size,
                 mode=ModeKeys.TRAIN):
        self._mode = mode
        # source eos for padding
        self._source_dim = source_vocab_size + 1
        # target sos/eos for decoder input/output
        self._target_dim = target_vocab_size + 2
        self._embedding_dim = embedding_size
        self._hidden_state_size = hidden_state_size

    @property
    def encoder_emb_weight(self):
        return self._encoder_emb_weight

    @property
    def batch_in_embed(self):
        return self._batch_in_embed

    @property
    def encoder_state(self):
        return self._encoder_state

    @property
    def encoder_output(self):
        return self._encoder_outputs

    def _build_encoder_embedding(self, batch_in_seq):
        """[batch, time] -> [batch, time, embed]"""

        # embedding weight
        encoder_emb_weight = tf.get_variable(
            "encoder_emb_weight", shape=[self._source_dim, self._embedding_dim])
        self._encoder_emb_weight = encoder_emb_weight

        # (batch) one-hot -> (batch) embeddings
        # batch_in_seq  : [batch, time]
        # batch_in_embed: [batch, time, embed]
        # https://www.tensorflow.org/api_docs/python/tf/gather
        # https://www.tensorflow.org/programmers_guide/embedding#training_an_embedding
        self._batch_in_embed = tf.gather(encoder_emb_weight, batch_in_seq)

    def _build_encoder(self, batch_in_embed):
        encoder_cell = BasicLSTMCell(self._hidden_state_size)

        # batch_in_embed: [batch, time, embed]
        # encoder_state : [batch, hidden]
        self._encoder_outputs, self._encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, batch_in_embed, dtype=tf.float32)

    def build(self, data_iterator):
        (batch_enc_in, batch_dec_in, batch_dec_out) = data_iterator.get_next()
        self._build_encoder_embedding(batch_enc_in)
        self._build_encoder(self._batch_in_embed)
