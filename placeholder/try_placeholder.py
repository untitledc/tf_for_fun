# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset


def do_with_placeholder():
    global data
    global label

    dataset = Dataset.from_tensor_slices((data, label))
    dataset = dataset.batch(3)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    X = tf.placeholder(tf.float32, shape=[None, 2])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable([[0], [0]], dtype=tf.float32)
    b = tf.Variable([0], dtype=tf.float32)
    y = tf.matmul(X, W) + b
    loss = tf.losses.mean_squared_error(y_, y)
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        print(sess.run(W, feed_dict={X: data, y_: label}))

        batch_X_np, batch_y_np = sess.run(next_batch)
        sess.run(train_step, feed_dict={X: batch_X_np, y_: batch_y_np})

        print(sess.run(W, feed_dict={X: data, y_: label}))

        batch_X_np, batch_y_np = sess.run(next_batch)
        sess.run(train_step, feed_dict={X: batch_X_np, y_: batch_y_np})

        print(sess.run(W, feed_dict={X: data, y_: label}))

        print(sess.run(next_batch))
        print(sess.run(next_batch))


if __name__ == '__main__':
    d1 = np.arange(start=0, stop=10, step=1, dtype=np.float32).reshape(-1, 1)
    d2 = np.arange(start=0, stop=20, step=2, dtype=np.float32).reshape(-1, 1)
    data = np.hstack((d1, d2))
    label = np.sum(data, axis=1, keepdims=True)

    do_with_placeholder()
