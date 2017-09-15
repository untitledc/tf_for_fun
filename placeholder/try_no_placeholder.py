# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset


def do_without_placeholder():
    global data
    global label

    dataset = Dataset.from_tensor_slices((data, label))
    dataset = dataset.batch(3)
    iterator = dataset.make_initializable_iterator()
    (batch_X, batch_y) = iterator.get_next()

    W = tf.Variable([[0], [0]], dtype=tf.float32)
    b = tf.Variable([0], dtype=tf.float32)
    y = tf.matmul(batch_X, W) + b
    loss = tf.losses.mean_squared_error(batch_y, y)
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        print(sess.run(W))

        sess.run(train_step)

        print(sess.run(W))

        sess.run(train_step)

        print(sess.run(W))
        print(sess.run(batch_X))
        print(sess.run(batch_X))


if __name__ == '__main__':
    d1 = np.arange(start=0, stop=10, step=1, dtype=np.float32).reshape(-1, 1)
    d2 = np.arange(start=0, stop=20, step=2, dtype=np.float32).reshape(-1, 1)
    data = np.hstack((d1, d2))
    label = np.sum(data, axis=1, keepdims=True)

    do_without_placeholder()
