# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.examples.tutorials.mnist import input_data


CONV_1_FILTER_SIZE = 5
CONV_1_CHANNEL_N = 32
CONV_2_FILTER_SIZE = 5
CONV_2_CHANNEL_N = 64
BATCH_SIZE = 64
EPOCH_N = 5


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
        b_conv1 = create_bias_var([CONV_1_CHANNEL_N])
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
        b_conv2 = create_bias_var([CONV_2_CHANNEL_N])
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
        b_fc1 = create_bias_var([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Layer 4: final layer (for softmax later)
    with tf.variable_scope('fc2'):
        W_fc2 = create_weight_var((1024, class_n))
        b_fc2 = create_bias_var([class_n])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # FIXME: is this a good idea?
    return {'y_conv': y_conv,
            'h_conv1': h_conv1,
            'h_conv2': h_conv2,
            'h_fc1': h_fc1}


def get_highact_images(degrees_tensor, X, keep_prob, outname, first_n=None,
                       debug_W_filters=None, debug_b_filters=None,
                       sort_by_degree=True):
    """

    :param degrees_tensor:
    :param X:
    :param keep_prob:
    :param outname:
    :param first_n:
    :param debug_W_filters:
    :param debug_b_filters:
    :return:
    """
    if first_n is None:
        first_n = degrees_tensor.shape.as_list()[0]

    image_and_scores = []
    for filter_i in range(first_n):
        print('Generating image for filter {}'.format(filter_i))
        # X_init = np.ones((1, 784)) * 0.5
        X_init = np.random.rand(1, 784)

        degree_tensor = degrees_tensor[filter_i]
        dlc2_dx = tf.gradients(degree_tensor, X)
        # gradient ascent to find the argmax input image
        # FIXME: fixed iteration number: should've used diff < epsilon
        for i in range(3000):
            degree, grad_list = tf.get_default_session().run(
                [degree_tensor, dlc2_dx], feed_dict={X: X_init, keep_prob: 1.0})
            # tf.gradients returns a list of size 1 even when there's one tensor
            grad = grad_list[0]
            X_init += 0.05 * grad
            # make it a realistic image rather than scaling afterwards
            np.clip(X_init, 0.0, 1.0, out=X_init)
            # if i % 100 == 0:
            #     print('layer activation degree: {}'.format(degree))

        if debug_b_filters is not None and debug_W_filters is not None:
            image_and_scores.append((X_init, degree,
                                     debug_W_filters[:, filter_i].reshape(5, 5),
                                     debug_b_filters[filter_i]))
        else:
            image_and_scores.append((X_init, degree, None, None))

    if sort_by_degree:
        image_and_scores = sorted(image_and_scores, key=lambda t: t[1],
                                  reverse=True)

    for i, (im, sc, w, b) in enumerate(image_and_scores):
        plt.subplot(int(np.ceil(first_n/6)), 6*2, i*2+1)
        plt.axis('off')
        plt.title('d:{:.2f}'.format(sc), fontsize=6)
        plt.imshow(im.reshape(28, 28), cmap='gray', vmin=0, vmax=1)

        if w is not None and b is not None:
            plt.subplot(int(np.ceil(first_n/6)), 6*2, i*2+2)
            plt.axis('off')
            abs_max = np.max(np.abs(w))
            color_norm = matplotlib.colors.Normalize(-abs_max, abs_max)
            plt.title('w:{:.3f}\nb:{:.3f}'.format(np.sum(w), b), fontsize=6)
            plt.imshow(w, cmap=plt.cm.coolwarm, norm=color_norm)

    # plt.tight_layout()
    plt.savefig('{}.png'.format(outname))
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

        # predict
        predict_data(mnist_data.test, add_negative, accuracy)

        # Visualize: what input images lead to high activation?
        debug_W1, debug_b1, debug_W2, debug_b2 = sess.run(
            ['conv1/W:0', 'conv1/b:0', 'conv2/W:0', 'conv2/b:0'])
        fn_prfx = '11_' if add_negative else '10_'

        # reduce mean axis=0 for eliminate mini-batch dimension
        h_conv1 = model_var_dict['h_conv1']
        degrees = tf.reduce_sum(tf.reduce_mean(h_conv1, axis=0), axis=[0, 1])
        get_highact_images(
            degrees, X, keep_prob, fn_prfx+'max_conv1', debug_b_filters=debug_b1,
            debug_W_filters=debug_W1.reshape(-1, CONV_1_CHANNEL_N))

        h_conv2 = model_var_dict['h_conv2']
        degrees = tf.reduce_sum(tf.reduce_mean(h_conv2, axis=0), axis=[0, 1])
        # average L2 filter over input (L1) depth when showing in 2D
        debug_W2 = np.mean(debug_W2, axis=2)
        get_highact_images(
            degrees, X, keep_prob, fn_prfx+'max_conv2', debug_b_filters=debug_b2,
            debug_W_filters=debug_W2.reshape(-1, CONV_2_CHANNEL_N), first_n=30)

        h_fc1 = model_var_dict['h_fc1']
        degrees = tf.reduce_mean(h_fc1, axis=0)
        get_highact_images(degrees, X, keep_prob, fn_prfx+'max_fc1', first_n=30)

        y_conv = model_var_dict['y_conv']
        degrees = tf.reduce_mean(y_conv, axis=0)
        get_highact_images(degrees, X, keep_prob, fn_prfx+'max_y',
                           sort_by_degree=False)


if __name__ == '__main__':
    main(add_negative=False)
