# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.examples.tutorials.mnist import input_data


CONV_1_FILTER_SIZE = 5
CONV_1_CHANNEL_N = 32
CONV_2_FILTER_SIZE = 5
CONV_2_CHANNEL_N = 64
BATCH_SIZE = 64
EPOCH_N = 1


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
         np.zeros((mnist_dataset.num_examples, 1))))

    final_dataset = Dataset.from_tensor_slices((
        np.vstack((mnist_dataset.images, noisy_X)),
        np.vstack((orig_y, noisy_y))
    ))

    assert final_dataset.output_shapes[0] == (784, )
    assert final_dataset.output_shapes[1] == (11, )

    return final_dataset


def create_weight_var(shape, initializer=xavier_initializer(seed=1)):
    """Create TF Variable for weights

    :param shape: shape of the variable
    :param initializer: (optional) by default, xavier initializer
    :return: the TF trainable variable
    """
    return tf.get_variable(name='W', shape=shape, initializer=initializer)


def create_bias_var(shape, initializer=tf.zeros_initializer()):
    """Create TF Variable for bias

    :param shape: shape of the variable
    :param initializer: (optional) by default, initialize to 0's
    :return: the TF trainable variable
    """
    return tf.get_variable(name='b', shape=shape, initializer=initializer)


def build_cnn_graph(X, keep_prob, class_n):
    """Build a CNN model using placeholders/input

    :param X: TF placeholder
    :param keep_prob: TF placeholder for (1 - dropout rate)
    :param class_n: class number, int.
    :return: A dictionary of TF Variables in this CNN model
    """

    X_image = tf.reshape(X, (-1, 28, 28, 1))

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
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Layer 4: final layer (for softmax later)
    with tf.variable_scope('fc2'):
        W_fc2 = create_weight_var((1024, class_n))
        b_fc2 = create_bias_var((1, class_n))
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # FIXME: is this a good idea?
    return {'y_conv': y_conv, 'h_conv2': h_conv2}


def get_highact_images_bad():
    """Generate an image leading to the highest activation sum

    This function is bad because it is not necessary to build another graph.
    """

    # get model weights
    trainable_vars = tf.trainable_variables()
    model_var_dict = dict(zip(
        map(lambda v: v.name, trainable_vars),
        tf.get_default_session().run(trainable_vars)
    ))
    # print(sorted(model_var_dict.keys()))
    # ['conv1/W:0', 'conv1/b:0', 'conv2/W:0', 'conv2/b:0', 'fc1/W:0', 'fc1/b:0',
    #  'fc2/W:0', 'fc2/b:0']

    X_image = tf.Variable(np.random.rand(1, 28, 28, 1), dtype=tf.float32)
    # X_image = tf.Variable(np.random.rand(1, 28, 28, 1)*0.2+0.4, dtype=tf.float32)
    # X_image = tf.Variable(np.ones((1, 28, 28, 1))*0.5, dtype=tf.float32)
    # X_image = tf.Variable(np.zeros((1, 28, 28, 1)), dtype=tf.float32)
    # X_image = tf.Variable(X_init, dtype=tf.float32)

    # Layer 1: Conv
    W_conv1 = tf.Variable(model_var_dict['conv1/W:0'], trainable=False)
    b_conv1 = tf.Variable(model_var_dict['conv1/b:0'], trainable=False)
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(X_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        + b_conv1
    )
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')

    # Layer 2: Conv
    W_conv2 = tf.Variable(model_var_dict['conv2/W:0'], trainable=False)
    b_conv2 = tf.Variable(model_var_dict['conv2/b:0'], trainable=False)
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        + b_conv2
    )
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')

    # Layer 3: Fully connected
    h_pool2_flat = tf.reshape(h_pool2, (-1, 7*7*CONV_2_CHANNEL_N))
    W_fc1 = tf.Variable(model_var_dict['fc1/W:0'], trainable=False)
    b_fc1 = tf.Variable(model_var_dict['fc1/b:0'], trainable=False)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Layer 4: final layer (for softmax later)
    W_fc2 = tf.Variable(model_var_dict['fc2/W:0'], trainable=False)
    b_fc2 = tf.Variable(model_var_dict['fc2/b:0'], trainable=False)
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    # define high activations...
    lc2_degree = tf.reduce_sum(tf.reduce_mean(h_conv2, axis=0), axis=[0, 1])[0]
    optimize_op = tf.train.GradientDescentOptimizer(1e-3).minimize(-lc2_degree)
    # grads = tf.train.GradientDescentOptimizer(0.1).compute_gradients(
    #     -lc2_degree, var_list=X_image)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(1000):
            sess.run(optimize_op)
            # print(sess.run(grads)[0][0].reshape(784)[:5])
        high_image = sess.run(X_image)

        plt.imshow(high_image.reshape(28, 28), cmap='gray')
        plt.savefig('test.png')
        plt.close()


def get_highact_images(tensor_dict, X):
    h_conv2 = tensor_dict['h_conv2']
    X_image = tf.Variable(np.random.rand(1, 28, 28, 1), dtype=tf.float32)

    X_init = np.random.rand(1, 784)
    lc2_degree = tf.reduce_sum(tf.reduce_mean(h_conv2, axis=0), axis=[0, 1])[0]
    dlc2_dx = tf.gradients(lc2_degree, X)
    for i in range(1000):
        grad_list = tf.get_default_session().run(dlc2_dx, feed_dict={X: X_init})
        # tf.gradients returns a list of size 1 even when there's one tensor
        grad = grad_list[0]
        # gradient 'ascent' for maximizing degree
        X_init += 0.001 * grad
        # make it a realistic image rather than scaling afterwards
        np.clip(X_init, 0.0, 1.0, out=X_init)

    plt.imshow(X_init.reshape(28, 28), cmap='gray')
    plt.savefig('test2.png')
    plt.close()


def predict_data(mnist_dataset, add_negative, accuracy_op):
    if add_negative:
        test_labels = np.hstack(
            (mnist_dataset.labels,
             np.zeros((mnist_dataset.num_examples, 1))))
    else:
        test_labels = mnist_dataset.labels

    acc = accuracy_op.eval(feed_dict={
        'X:0': mnist_dataset.images, 'y:0': test_labels, 'keep_prob:0': 1.0})
    print('test accuracy {:.4f}'.format(acc))


def main(add_negative):
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    if add_negative:
        train_set = gen_noise_dataset(mnist_data.train)
        class_n = 11
    else:
        train_set = Dataset.from_tensor_slices(
            (mnist_data.train.images, mnist_data.train.labels))
        class_n = 10

    print('Done generating dataset')

    # Input placeholder
    X = tf.placeholder(tf.float32, shape=(None, 28*28), name='X')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y = tf.placeholder(tf.float32, shape=(None, class_n), name='y')

    # Model
    model_var_dict = build_cnn_graph(X, keep_prob=keep_prob, class_n=class_n)
    y_conv = model_var_dict['y_conv']
    h_conv2 = model_var_dict['h_conv2']

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

    # RUN!
    dataset = train_set
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
                        train_accuracy, summary = sess.run(
                            [accuracy, merged_summary],
                            feed_dict={X: batch_X, y: batch_y, keep_prob: 1.0})
                        print('step {}, training accuracy {}'.format(
                            step_i, train_accuracy))
                        summary_writer.add_summary(summary, epoch_i)

                    step_i += 1
                except tf.errors.OutOfRangeError:
                    break

        # get_highact_images_bad()
        tensor_dict = model_var_dict
        get_highact_images(tensor_dict, X)

        # predict
        predict_data(mnist_data.test, add_negative, accuracy)


if __name__ == '__main__':
    main(add_negative=False)
