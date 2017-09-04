# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.examples.tutorials.mnist import input_data


CONV_1_FILTER_SIZE = 5
CONV_1_CHANNEL_N = 32
CONV_2_FILTER_SIZE = 5
CONV_2_CHANNEL_N = 64
CLASS_N = 10
BATCH_SIZE = 64
EPOCH_N = 10


def gen_noise_dataset(mnist_dataset):
    """Generate a TF Dataset with additional "noisy data" as the 11th class.

    :param mnist_dataset: mnist.DataSet, which has attributes images/labels
    :return: TF (API) Dataset
    """

    # Create noisy data, which cannot be seen as a valid digit (almost)
    feature_n = mnist_dataset.images.shape[1]
    noisy_n = mnist_dataset.num_examples
    noisy_X = np.random.rand(noisy_n, feature_n)
    noisy_y = np.zeros((noisy_n, 11))
    noisy_y += np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # Expand the original data labels to 11 classes
    orig_y = np.hstack(
        (mnist_dataset.labels,
         np.zeros(mnist_dataset.num_examples).reshape(-1, 1)))

    final_dataset = Dataset.from_tensor_slices((
        np.vstack((mnist_dataset.images, noisy_X)),
        np.vstack((orig_y, noisy_y))
    ))

    assert final_dataset.output_shapes[0] == (784, )
    assert final_dataset.output_shapes[1] == (11, )

    return final_dataset


def create_weight_var(shape, initializer=xavier_initializer(seed=1)):
    return tf.get_variable(name='W', shape=shape, initializer=initializer)


def create_bias_var(shape, initializer=tf.zeros_initializer()):
    return tf.get_variable(name='b', shape=shape, initializer=initializer)


def main():
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_set_10 = Dataset.from_tensor_slices(
        (mnist_data.train.images, mnist_data.train.labels))
    train_set_11 = gen_noise_dataset(mnist_data.train)

    print('Done generating dataset')

    # Input placeholder
    X = tf.placeholder(tf.float32, shape=(None, 28*28))
    X_image = tf.reshape(X, (-1, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Layer 1: Conv
    # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    with tf.variable_scope('conv1'):
        W_conv1 = create_weight_var((CONV_1_FILTER_SIZE, CONV_1_FILTER_SIZE,
                                     1, CONV_1_CHANNEL_N))
        b_conv1 = create_bias_var((1, CONV_1_CHANNEL_N))
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(X_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        + b_conv1
    )
    # https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')

    # Layer 2: Conv
    with tf.variable_scope('conv2'):
        W_conv2 = create_weight_var((CONV_2_FILTER_SIZE, CONV_2_FILTER_SIZE,
                                     CONV_1_CHANNEL_N, CONV_2_CHANNEL_N))
        b_conv2 = create_bias_var((1, CONV_2_CHANNEL_N))
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        + b_conv2
    )
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
    # h_pool2 is Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

    # Layer 3: Fully connected
    h_pool2_flat = tf.reshape(h_pool2, (-1, 7*7*CONV_2_CHANNEL_N))
    with tf.variable_scope('fc1'):
        W_fc1 = create_weight_var((7*7*CONV_2_CHANNEL_N, 1024))
        b_fc1 = create_bias_var((1, 1024))
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Layer 4: final layer (for softmax later)
    with tf.variable_scope('fc2'):
        W_fc2 = create_weight_var((1024, CLASS_N))
        b_fc2 = create_bias_var((1, CLASS_N))
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # metrics
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train OP
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Tensorboard summary
    tf.summary.scalar('cross entropy', cross_entropy)
    tf.summary.scalar('training accuracy', accuracy)
    tf.summary.histogram(
        'L2 activation degree by filters',
        tf.reduce_sum(tf.reduce_mean(h_conv2, axis=0), axis=[0, 1]))
    merged_summary = tf.summary.merge_all()

    # print(list(map(lambda t: t.name, tf.trainable_variables())))

    # RUN!
    dataset = train_set_10
    dataset = dataset.shuffle(buffer_size=8192)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('mnist/logs/', sess.graph)

        step_i = 0
        for epoch_i in range(EPOCH_N):
            sess.run(iterator.initializer)
            while True:
                try:
                    batch_X, batch_y = sess.run(next_batch)
                    sess.run(train_step,
                             feed_dict={X: batch_X, y: batch_y, keep_prob: 0.5})

                    # monitor
                    if step_i % 100 == 0:
                        train_accuracy, summary, w1 = sess.run(
                            [accuracy, merged_summary, W_conv1],
                            feed_dict={X: batch_X, y: batch_y, keep_prob: 1.0})
                        print('step {}, training accuracy {}'.format(
                            step_i, train_accuracy))
                        # print(w1)
                        summary_writer.add_summary(summary, epoch_i)

                    step_i += 1
                except tf.errors.OutOfRangeError:
                    break

        # predict
        print('test accuracy %g' % accuracy.eval(feed_dict={
            X: mnist_data.test.images, y: mnist_data.test.labels,
            keep_prob: 1.0}))


if __name__ == '__main__':
    main()
