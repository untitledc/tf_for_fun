# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CONV_1_FILTER_SIZE = 5
CONV_1_CHANNEL_N = 1


def build_inversed_graph():
    X = tf.Variable(np.random.rand(1, 28, 28, 1), dtype=tf.float32)
    # X = tf.Variable(np.ones((1, 28, 28, 1))*0.5, dtype=tf.float32)

    #W1 = np.array([
    #    [-0.03087736,  0.01197383, -0.00400480, 0.08230809,  0.00399779],
    #    [ 0.00031647,  0.05443036,  0.11183304, 0.10960845, -0.05192799],
    #    [-0.01184785, -0.04508770, -0.06926913, 0.04023817,  0.07129525],
    #    [ 0.01889761,  0.07342482, -0.03891212, 0.10788606,  0.06181243],
    #    [-0.07107725, -0.00701407,  0.08727332, 0.02233931,  0.00019639]],
    #    dtype=np.float32)
    #W1 = np.array(
    #    [[-0.10523209, -0.06036796, -0.05401075, -0.03102826, -0.07437814],
    #     [-0.0538452, -0.07316288, -0.05655884, -0.10519095, -0.03261073],
    #     [0.07515751, -0.01857454, -0.02234382, 0.02080215, -0.03469184],
    #     [0.08586096, 0.0723625, 0.12675671, 0.11285992, 0.02508893],
    #     [0.10712505, -0.01620406, 0.06248681, 0.02782501, 0.01764546]],
    #    dtype=np.float32)
    W1 = np.array([[0.1, -0.1, -0.1,  0.1,  0.1],
                   [0.1, -0.1, -0.1,  0.1,  0.1],
                   [0.1,  0.1, -0.1, -0.1,  0.1],
                   [0.1,  0.1, -0.1, -0.1,  0.1],
                   [0.1,  0.1,  0.1, -0.1, -0.1]],
                  dtype=np.float32)
    W1_fixed = W1.reshape((CONV_1_FILTER_SIZE, CONV_1_FILTER_SIZE,
                           1, CONV_1_CHANNEL_N))
    b1_fixed = np.zeros((1, CONV_1_CHANNEL_N), dtype=np.float32)

    # Layer 1: Conv
    W_conv1 = tf.Variable(W1_fixed, trainable=False)
    b_conv1 = tf.Variable(b1_fixed, trainable=False)
    z_conv1 = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    h_conv1 = tf.nn.relu(
        z_conv1
        + b_conv1
    )
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')

    return {'W_conv1': W_conv1, 'b_conv1': b_conv1, 'h_conv1': h_conv1,
            'h_pool1': h_pool1, 'X': X, 'z_conv1': z_conv1}


def main():
    model = build_inversed_graph()
    h_conv1 = model['h_conv1']
    z_conv1 = model['z_conv1']
    X = model['X']

    # define high activations...
    degree = tf.reduce_sum(tf.reduce_mean(h_conv1, axis=0), axis=[0, 1])[0]
    optimize_op = tf.train.GradientDescentOptimizer(0.05).minimize(-degree)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3000):
            _, degree_val = sess.run([optimize_op, degree])
            if i % 100 == 0:
                print('iteration {:04d}, degree {:.4f}'.format(i, degree_val))
                X_npy = sess.run(X).reshape(28, 28)
                X_clipped = np.clip(X_npy, 0.0, 1.0)
                plt.imshow(X_clipped.reshape(28, 28), cmap='gray')
                plt.savefig('test{:04d}.png'.format(i))
                plt.close()


if __name__ == '__main__':
    main()
