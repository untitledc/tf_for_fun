# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


CONV_1_FILTER_SIZE = 5
CONV_1_CHANNEL_N = 32
CONV_2_FILTER_SIZE = 5
CONV_2_CHANNEL_N = 64
CLASS_N = 10
BATCH_SIZE = 50

def main():
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

    X = tf.placeholder(tf.float32, shape=(None, 28*28))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    ### input
    x_image = tf.reshape(X, (-1, 28, 28, 1))

    ### Layer 1: Conv
    # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    W_conv1 = tf.get_variable(
        'W1',
        shape=(CONV_1_FILTER_SIZE, CONV_1_FILTER_SIZE, 1, CONV_1_CHANNEL_N),
        initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b_conv1 = tf.get_variable('b1', shape=(1, CONV_1_CHANNEL_N),
                              initializer=tf.zeros_initializer())
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        + b_conv1
    )
    # https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2 ,1],
                             padding='SAME')

    ### Layer 2: Conv
    W_conv2 = tf.get_variable(
        'W2',
        shape=(CONV_2_FILTER_SIZE, CONV_2_FILTER_SIZE, CONV_1_CHANNEL_N,
               CONV_2_CHANNEL_N),
        initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b_conv2 = tf.get_variable('b2', shape=(1, CONV_2_CHANNEL_N),
                              initializer=tf.zeros_initializer())
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        + b_conv2
    )
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2 ,1],
                             padding='SAME')
    # h_pool2 is Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

    ### Layer 3: Fully connected
    h_pool2_flat = tf.reshape(h_pool2, (-1, 7*7*CONV_2_CHANNEL_N))
    W_fc1 = tf.get_variable(
        'W3',
        shape=(7*7*CONV_2_CHANNEL_N, 1024),
        initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b_fc1 = tf.get_variable('b3', shape=(1, 1024),
                            initializer=tf.zeros_initializer())
    h_fc1 = tf.nn.relu(
        tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ### Layer 4: final layer (for softmax later)
    W_fc2 = tf.get_variable(
        'W4',
        shape=(1024, CLASS_N),
        initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b_fc2 = tf.get_variable('b4', shape=(1, CLASS_N),
                            initializer=tf.zeros_initializer())
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ### metrics
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ### Train
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = mnist_data.train.next_batch(BATCH_SIZE)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    X: batch[0], y: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(
                feed_dict={X: batch[0], y: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            X: mnist_data.test.images, y: mnist_data.test.labels,
            keep_prob: 1.0}))


if __name__ == '__main__':
    main()
