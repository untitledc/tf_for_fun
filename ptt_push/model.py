# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.seq2seq import BasicDecoder
from tensorflow.contrib.seq2seq import TrainingHelper, GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.python.layers.core import Dense


class Seq2SeqModel:
    def __init__(self, source_vocab_size, target_vocab_size, embedding_size,
                 hidden_state_size, mode=ModeKeys.TRAIN):
        self._mode = mode
        self._target_vocab_size = target_vocab_size
        # unknown + source eos for padding
        self._source_dim = source_vocab_size + 2
        # unknown + target sos/eos for decoder input/output
        self._target_dim = target_vocab_size + 3
        self._embedding_dim = embedding_size
        self._hidden_state_size = hidden_state_size

        self._swap_memory_in_train = False
        self._max_decode_iterations = 20

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
    def decoder_emb_weight(self):
        return self._decoder_emb_weight

    @property
    def batch_dec_embed(self):
        return self._batch_dec_embed

    @property
    def decoder_proj_layer(self):
        return self._decoder_proj_layer

    @property
    def batch_rnn_output(self):
        return self._final_outputs.rnn_output

    @property
    def batch_sample_id(self):
        return self._final_outputs.sample_id

    @property
    def target_eos_id(self):
        return self._target_eos_id

    @property
    def loss(self):
        return self._loss

    def _build_encoder_embedding(self, batch_in_seq):
        """[batch, time] -> [batch, time, embed]"""

        # embedding weight
        encoder_emb_weight = tf.get_variable(
            'encoder_emb_weight', shape=[self._source_dim, self._embedding_dim])
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
            'decoder_emb_weight', shape=[self._target_dim, self._embedding_dim])
        self._decoder_emb_weight = decoder_emb_weight

        # FIXME: not clean enough
        if batch_in_seq is None:
            return decoder_emb_weight, None

        # (batch) one-hot -> (batch) embeddings
        batch_dec_embed = tf.gather(decoder_emb_weight, batch_in_seq)
        self._batch_dec_embed = batch_dec_embed

        return decoder_emb_weight, batch_dec_embed

    def _build_encoder(self, batch_enc_embed):
        encoder_cell = BasicLSTMCell(self._hidden_state_size)

        # batch_enc_embed: [batch, time, embed]
        # encoder_state  : [batch, hidden]
        # encoder_output : [batch, time, hidden]
        encoder_output, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, inputs=batch_enc_embed, dtype=tf.float32)
        self._encoder_output = encoder_output
        self._encoder_state = encoder_state

        return encoder_output, encoder_state

    def _build_decoder_projection(self):
        proj_layer = Dense(self._target_dim, use_bias=False,
                           name='output_projection')
        self._decoder_proj_layer = proj_layer

        return proj_layer

    def _build_decoder(self, encoder_state, batch_dec_embed, batch_dec_length,
                       output_projection, this_batch_size=None,
                       decoder_emb_weight=None):
        decoder_cell = BasicLSTMCell(self._hidden_state_size)

        # https://www.tensorflow.org/api_guides/python/contrib.seq2seq
        if self._mode == ModeKeys.TRAIN:
            helper = TrainingHelper(inputs=batch_dec_embed,
                                    sequence_length=batch_dec_length)
            max_iterations = None
        elif self._mode == ModeKeys.INFER:
            # FIXME: need to be consistent with dataset.py :(
            self._target_eos_id = self._target_vocab_size+2
            tgt_sos_id = tf.constant(self._target_vocab_size+1, dtype=tf.int32)
            tgt_eos_id = tf.constant(self._target_eos_id, dtype=tf.int32)

            helper = GreedyEmbeddingHelper(
                decoder_emb_weight,
                start_tokens=tf.fill([this_batch_size], tgt_sos_id),
                end_token=tgt_eos_id

            )
            max_iterations = self._max_decode_iterations
        else:
            raise NotImplementedError('Unsupported mode '+self._mode)

        decoder = BasicDecoder(decoder_cell, helper,
                               initial_state=encoder_state,
                               output_layer=output_projection)
        # encoder_state  : [batch, hidden]
        # batch_dec_embed: [batch, time, embed]
        # final_outputs.rnn_output: [batch, time, hidden (or proj)]
        # final_outputs.sample_id : [batch, time]
        final_outputs, _, _ = dynamic_decode(
            decoder, maximum_iterations=max_iterations,
            swap_memory=self._swap_memory_in_train)

        self._final_outputs = final_outputs
        return final_outputs.rnn_output, final_outputs.sample_id

    def _build_loss(self, logit, output, batch_output_length):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,
                                                              labels=output)
        # logit : [batch, time, hidden (or proj)]
        # output: [batch, time]
        # output_mask: [batch, time]
        max_time = tf.shape(output)[1]
        output_mask = tf.sequence_mask(batch_output_length, maxlen=max_time,
                                       dtype=tf.float32)
        loss = tf.reduce_sum(tf.reduce_mean(output_mask * xent, axis=0))
        self._loss = loss

        return loss

    def build(self, data_iterator):
        if self._mode == ModeKeys.TRAIN:
            batch_enc_in, batch_dec_in, batch_dec_out, batch_dec_length\
                = data_iterator.get_next()
            # XXX: debug
            self._batch_enc_in = batch_enc_in
            self._batch_dec_in = batch_dec_in
            self._batch_dec_out = batch_dec_out

            batch_enc_embed = self._build_encoder_embedding(batch_enc_in)
            encoder_output, encoder_state = self._build_encoder(batch_enc_embed)
            _, batch_dec_embed = self._build_decoder_embedding(batch_dec_in)
            dec_proj = self._build_decoder_projection()
            rnn_output, sample_id = self._build_decoder(
                encoder_state, batch_dec_embed, batch_dec_length, dec_proj)

            self._build_loss(rnn_output, batch_dec_out, batch_dec_length)
        elif self._mode == ModeKeys.INFER:
            batch_enc_in = data_iterator.get_next()
            # XXX: debug
            self._batch_enc_in = batch_enc_in

            batch_enc_embed = self._build_encoder_embedding(batch_enc_in)
            encoder_output, encoder_state = self._build_encoder(batch_enc_embed)
            decoder_emb_weight, _ = self._build_decoder_embedding(None)
            dec_proj = self._build_decoder_projection()
            self._build_decoder(encoder_state, None, None, dec_proj,
                                this_batch_size=tf.shape(batch_enc_in)[0],
                                decoder_emb_weight=decoder_emb_weight)
        else:
            raise NotImplementedError('Unsupported mode '+self._mode)
