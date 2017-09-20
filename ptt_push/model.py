# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.seq2seq import TrainingHelper


class Seq2SeqModel:
    def __init__(self, source_vocab_size, target_vocab_size, embedding_size,
                 hidden_state_size,
                 mode=ModeKeys.TRAIN):
        self._mode = mode
        # unknown + source eos for padding
        self._source_dim = source_vocab_size + 2
        # unknown + target sos/eos for decoder input/output
        self._target_dim = target_vocab_size + 3
        self._embedding_dim = embedding_size
        self._hidden_state_size = hidden_state_size

    @property
    def encoder_emb_weight(self):
        return self._encoder_emb_weight

    @property
    def batch_enc_embed(self):
        return self._batch_enc_embed

    @property
    def encoder_state(self):
        return self._encoder_state

    @property
    def encoder_output(self):
        return self._encoder_output

    @property
    def decoder_emb_weight(self):
        return self._decoder_emb_weight

    @property
    def batch_dec_embed(self):
        return self._batch_dec_embed

    def _build_encoder_embedding(self, batch_in_seq):
        """[batch, time] -> [batch, time, embed]"""

        # embedding weight
        encoder_emb_weight = tf.get_variable(
            "encoder_emb_weight", shape=[self._source_dim, self._embedding_dim])
        self._encoder_emb_weight = encoder_emb_weight

        # (batch) one-hot -> (batch) embeddings
        # batch_in_seq  : [batch, time]
        # batch_enc_embed: [batch, time, embed]
        # https://www.tensorflow.org/api_docs/python/tf/gather
        # https://www.tensorflow.org/programmers_guide/embedding#training_an_embedding
        batch_enc_embed = tf.gather(encoder_emb_weight, batch_in_seq)
        self._batch_enc_embed = batch_enc_embed

        return batch_enc_embed

    def _build_decoder_embedding(self, batch_in_seq):
        """[batch, time] -> [batch, time, embed]"""

        # embedding weight
        decoder_emb_weight = tf.get_variable(
            "decoder_emb_weight", shape=[self._target_dim, self._embedding_dim])
        self._decoder_emb_weight = decoder_emb_weight

        # (batch) one-hot -> (batch) embeddings
        batch_dec_embed = tf.gather(decoder_emb_weight, batch_in_seq)
        self._batch_dec_embed = batch_dec_embed

        return batch_dec_embed

    def _build_encoder(self, batch_enc_embed):
        encoder_cell = BasicLSTMCell(self._hidden_state_size)

        # batch_enc_embed: [batch, time, embed]
        # encoder_state : [batch, hidden]
        # encoder_output : [batch, time, hidden]
        encoder_output, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, batch_enc_embed, dtype=tf.float32)
        self._encoder_output = encoder_output
        self._encoder_state = encoder_state

        return encoder_output, encoder_state

    def _build_decoder(self, encoder_state, batch_dec_embed):
        decoder_cell = BasicLSTMCell(self._hidden_state_size)

        # https://www.tensorflow.org/api_guides/python/contrib.seq2seq
        if self._mode == ModeKeys.TRAIN:
            # XXX: why do we need to pass sequence length to create the helper?
            # embed shape: [batch, time, embed]
            # expect input length = embed shape[1] + 1 (due to <SOS>)
            dec_seq_length = tf.shape(batch_dec_embed)[1] + 1
            helper = TrainingHelper(batch_dec_embed, dec_seq_length)
            pass

    def build(self, data_iterator):
        (batch_enc_in, batch_dec_in, batch_dec_out) = data_iterator.get_next()
        # XXX: debug
        self._batch_enc_in = batch_enc_in
        self._batch_dec_in = batch_dec_in
        self._batch_dec_out = batch_dec_out

        batch_enc_embed = self._build_encoder_embedding(batch_enc_in)
        encoder_output, encoder_state = self._build_encoder(batch_enc_embed)
        batch_dec_embed = self._build_decoder_embedding(batch_dec_in)
        self._test = tf.shape(batch_dec_embed)[1] + 1
