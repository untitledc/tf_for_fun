# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.learn import ModeKeys

train_graph = tf.Graph()
infer_graph = tf.Graph()


def build_model(dataset_iterator=None, test_X_ph=None, mode=ModeKeys.TRAIN):
    if mode == ModeKeys.INFER and dataset_iterator is None:
        batch_X = test_X_ph
    else:
        (batch_X, batch_y) = dataset_iterator.get_next()

    W1 = tf.Variable(np.random.randn(2, 5), name='W1', dtype=tf.float32)
    b1 = tf.Variable(np.zeros(5), name='b1', dtype=tf.float32)
    a1 = tf.nn.relu(tf.matmul(batch_X, W1) + b1)
    if mode == ModeKeys.TRAIN:
        a1 = tf.nn.dropout(a1, 0.7)

    W2 = tf.Variable(np.random.randn(5, 2), name='W2', dtype=tf.float32)
    b2 = tf.Variable(np.zeros(2), name='b2', dtype=tf.float32)
    a2 = tf.matmul(a1, W2) + b2

    if mode == ModeKeys.TRAIN:
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=a2, labels=batch_y))
        minimize_op = tf.train.GradientDescentOptimizer(0.1)\
            .minimize(cross_entropy)
        return minimize_op, None
    elif mode == ModeKeys.INFER:
        pred = tf.argmax(a2, 1)
        return None, pred
    else:
        raise NotImplementedError('Unsupported mode '+mode)


def infer():
    global novel_data_X
    global infer_graph

    # 1. Build model structure for inference
    with infer_graph.as_default():
        X_placeholder = tf.placeholder(tf.float32, shape=(None, 2))

        _, pred = build_model(test_X_ph=X_placeholder, mode=ModeKeys.INFER)
        saver = tf.train.Saver(tf.global_variables())

    with tf.Session(graph=infer_graph) as sess:
        # 2. Load model variables (from the last checkpoint)
        sess.run(tf.global_variables_initializer())
        print('Original W1: {}'.format(sess.run('W1:0')))
        saver.restore(sess, '/tmp/model.ckpt')
        print('  Loaded W1: {}'.format(sess.run('W1:0')))

        # 3. predict every data
        print(sess.run(pred, feed_dict={X_placeholder: novel_data_X}))

    print('Inference is DONE')


def train():
    global data_X, data_y
    global train_graph

    # 1. Build model structure for training
    with train_graph.as_default():
        tf.set_random_seed(1)
        dataset = Dataset.from_tensor_slices((data_X, data_y))
        dataset = dataset.shuffle(buffer_size=64, seed=1)
        dataset = dataset.repeat(500)
        dataset = dataset.batch(4)
        iterator = dataset.make_one_shot_iterator()

        minimize_op, _ = build_model(iterator, mode=ModeKeys.TRAIN)
        saver = tf.train.Saver(tf.global_variables())

    with tf.Session(graph=train_graph) as sess:
        # 2. Do training via gradient descent
        sess.run(tf.global_variables_initializer())
        while True:
            try:
                sess.run(minimize_op)
            except tf.errors.OutOfRangeError:
                break
        # 3. Save model (variables)
        saver.save(sess, '/tmp/model.ckpt')
        print(' Trained W1: {}'.format(sess.run('W1:0')))

    print('Training is DONE')


def main():
    train()
    infer()


if __name__ == '__main__':
    np.random.seed(1)
    data_X = np.array([[0, 0], [2, 0], [4, 0], [0, 2], [4, 2],
                       [0, 4], [2, 4], [4, 4],
                       [1, 1], [2, 1], [3, 1], [1, 2], [3, 2],
                       [1, 3], [2, 3], [3, 3]], dtype=np.float32)
    tmp_y = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1])
    data_y = np.zeros((tmp_y.shape[0], 2))
    data_y[np.arange(tmp_y.shape[0]), tmp_y] = 1

    novel_data_X = np.array([[3, 0], [2, 2], [-1, -1]], dtype=np.float32)

    main()
