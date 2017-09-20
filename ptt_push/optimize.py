# -*- coding: utf-8 -*-

import tensorflow as tf


def get_train_step(loss, learning_rate, max_norm=None):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    vars = tf.trainable_variables()

    gradients = tf.gradients(loss, vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
    train_step = optimizer.apply_gradients(zip(clipped_gradients, vars))

    return train_step
