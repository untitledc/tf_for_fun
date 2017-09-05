# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def check_one_pass(dataset):
    print('=== One epoch, no shuffle ===')

    dataset = dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        while True:
            try:
                batch_X, batch_y = sess.run(next_element)
                print(batch_X)
                print(np.squeeze(batch_y))
                print('---')
            except tf.errors.OutOfRangeError:
                break
    print('done')


def check_batch_in_epochs(dataset):
    print('=== Use dataset.repeat for epochs ===')

    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(3)
    dataset = dataset.repeat(4)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.train.MonitoredTrainingSession() as sess:
        while not sess.should_stop():
            batch_X, batch_y = sess.run(next_element)
            print(batch_X)
            print(np.squeeze(batch_y))
            print('---')
    print('done')


def check_batch_in_epochs2(dataset):
    print('=== Use initializable iterator to know when an epoch ends ===')

    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(3)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for epoch_i in range(4):
            sess.run(iterator.initializer)
            while True:
                try:
                    batch_X, batch_y = sess.run(next_element)
                    print(batch_X)
                    print(np.squeeze(batch_y))
                    print('---')
                except tf.errors.OutOfRangeError:
                    break
            print('EPOCH {} END'.format(epoch_i))
    print('done')


def main():
    X = np.vstack(
        (np.arange(0, 20, step=2.0), np.arange(0, 30, step=3.0))).transpose()
    y = np.arange(10).reshape(10, 1)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((X, y))
    print(dataset)

    check_one_pass(dataset)
    check_batch_in_epochs(dataset)
    check_batch_in_epochs2(dataset)


if __name__ == '__main__':
    main()
