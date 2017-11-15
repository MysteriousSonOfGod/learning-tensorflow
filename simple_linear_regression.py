import numpy as np
import os
from os.path import dirname
import sys
import matplotlib.pyplot as plt
import tensorflow as tf


# hide Tensorflow GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_dummy_data(is_display_graph=False):
    total = 100
    x = np.arange(total)                                  # x is an array of 100 items: [0, 1, 2, ..., 99]
    y = 2.0 * x + np.random.uniform(-30, 30, size=total)  # y = 2 * x + random number -> linear form
    dummy_data = np.stack([x, y], axis=1)

    # draw the graph of input data
    if is_display_graph:
        plt.plot(x, y, 'o', alpha=0.5)
        plt.title('Input data')
        plt.show()

    return dummy_data


# define flags
tf.app.flags.DEFINE_integer('training_epochs', 10000, 'Total training epochs')
tf.app.flags.DEFINE_integer('display_step', 1000, 'Step to display the loss value')
tf.app.flags.DEFINE_float('learning_rate', 1e-9, 'The learning rate')
# tf.app.flags.DEFINE_float('expected_loss_minimum', 1e-10, 'The expected loss minimum value')
FLAGS = tf.app.flags.FLAGS


# init data
data = generate_dummy_data(True)
losses = []


# init the hypothesis model
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
w = tf.Variable(0.0, name='weight')
b = tf.Variable(0.0, name='bias')
hypothesis = w * X + b


# minimize the loss value using gradient descent
loss = tf.reduce_sum(tf.squared_difference(hypothesis, Y)) / 2 * data.shape[0]
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
train = optimizer.minimize(loss)
training_data = {X: data[:, 0], Y: data[:, 1]}


# running the computation in Tensorflow
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # previous_loss = 0.0
    print('\nModel training...')
    for epoch in range(FLAGS.training_epochs):
        _train, _loss = session.run([train, loss], feed_dict=training_data)
        losses.append(_loss)

        if epoch % FLAGS.display_step == 0:
            print('Epoch={}, loss value={}'.format(epoch, _loss))

        # we will quit the training process if we have reached our expected loss minimum value
        # if np.abs(previous_loss - _loss) < FLAGS.expected_loss_minimum:
        #     break
        # previous_loss = _loss

    _w, _b = session.run([w, b], feed_dict=training_data)
    print('\nFinished!!!')
    print('y = {0:.4f} * x + {1:.4f}\n'.format(_w, _b))

    # export model graph to Tensorboard
    # you can run this command to see the graph: tensorboard --logdir="./tensorboard/simple-linear"
    writer = tf.summary.FileWriter(dirname(__file__) + 'tensorboard/simple-linear', session.graph)
    writer.close()

    # display the fitted line
    plt.plot(data[:, 0], data[:, 1], "o", alpha=0.5)
    plt.plot(data[:, 0], _w * data[:, 0] + _b, alpha=0.5, label='Fitted line')
    plt.legend()
    plt.title('After predicted')
    plt.show()

    # draw the loss diagram
    plt.title('The loss diagram after trained')
    plt.plot(losses[:])
    plt.show()

