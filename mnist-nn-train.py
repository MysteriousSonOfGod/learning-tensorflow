import os
from os.path import dirname
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# hide Tensorflow GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


"""
The MNIST dataset:
- is a dataset of handwritten digits
- 55,000 images for training and 10,000 images for testing (include labels for each image)
- each image has the dimension of 28x28 pixels (-> 1D array: 784 pixels)
- each pixel has value from 0 - 255 (0: black, 255: white) to display a specific color -> 256 values can occur

Reference:
- http://www.cs.nyu.edu/~roweis/data/mnist_train2.jpg
- https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAoYAAAAJDI5OGU3NDYwLTEzNWItNGVmZS1hMzVhLWIxYTI0MmU5MDFmYQ.png
- https://www.simplicity.be/articles/recognizing-handwritten-digits/img/matrix.jpg
"""


def show_info():
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    print '\n..::Overview::..'
    print '- Type of mnist data is ', type(mnist)
    print '- Number of train data is ', mnist.train.num_examples
    print '- Number of test data is ', mnist.test.num_examples
    print '- Training images: type {}, shape {}'.format(type(train_images), train_images.shape)
    print '- Training labels: type {}, shape {} '.format(type(train_labels), train_labels.shape)
    print '- Test images: type {}, shape {}'.format(type(test_images), test_images.shape)
    print '- Test labels: type {}, shape {}'.format(type(test_labels), test_labels.shape)

    # take 3 random elements from the training images dataset
    print 'Show random images and labels...'
    random_indexes = np.random.randint(train_images.shape[0], size=3)
    for i in random_indexes:
        img = np.reshape(train_images[i, :], (28, 28))
        label = np.argmax(train_labels[i, :])

        # use matplotlib to show 3 random elements and its label visually
        plt.matshow(img, cmap=plt.get_cmap('gray'))
        plt.title('item ' + str(i) + 'th was labeled ' + str(label))
        plt.show()

    print '\n'


def training():
    learning_rate = 0.001
    training_epochs = 100
    batch_size = 128
    display_step = 10

    # init input layer
    input_nodes = 784  # 28x28 pixels
    X = tf.placeholder(tf.float32, shape=[None, input_nodes])

    # init hidden layer 1
    # hidden layer 1 = weight_1 * X + bias_1
    hidden_1_nodes = 256  # each pixel has value from 0 - 255 (0: black, 255: white) -> 256 values can occur
    weight_1 = tf.Variable(tf.random_normal([input_nodes, hidden_1_nodes]), name='weight_1')
    bias_1 = tf.Variable(tf.random_normal([hidden_1_nodes]), name='bias_1')
    hidden_layer_1 = tf.add(tf.matmul(X, weight_1), bias_1)  # hidden layer 1 = weight * X + bias

    # init hidden layer 2
    # hidden layer 2 = weight_2 * output from hidden layer 1 + bias_2
    hidden_2_nodes = 256  # each pixel has value from 0 - 255 (0: black, 255: white) -> 256 values can occur
    weight_2 = tf.Variable(tf.random_normal([hidden_1_nodes, hidden_2_nodes]), name='weight_2')
    bias_2 = tf.Variable(tf.random_normal([hidden_2_nodes]), name='bias_2')
    hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weight_2), bias_2)

    # init output layer
    # output layer = weight_output * output from hidden layer 2 + bias_output
    output_nodes = 10
    Y = tf.placeholder(tf.float32, shape=[None, output_nodes])
    weight_output = tf.Variable(tf.random_normal([hidden_2_nodes, output_nodes]), name='weight_output')
    bias_output = tf.Variable(tf.random_normal([output_nodes]), name='bias_output')
    output_layer = tf.add(tf.matmul(hidden_layer_2, weight_output), bias_output)

    # reduce loss (back propagation)
    test_dict = {X: mnist.test.images, Y: mnist.test.labels}
    loss_output = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_output = optimizer.minimize(loss_output)
    correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print '..::Model training::..'
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        losses = []

        num_batch = int(mnist.train.num_examples / batch_size)
        for epoch in range(training_epochs):
            for i in range(num_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                train_dict = {X: batch_x, Y: batch_y}
                _train, _loss, _accuracy = session.run([train_output, loss_output, accuracy], feed_dict=train_dict)
                losses.append(_loss)

            if epoch % display_step == 0:
                print 'Epoch ' + str(epoch) + ', minibatch loss= ' + '{:.4f}'.format(_loss) + ', training accuracy= ' + '{:.3f}'.format(_accuracy)

        print 'Finished!!!'
        print 'Learning rate: ', learning_rate
        print 'Batch size: ', batch_size
        print 'Testing accuracy:', session.run(accuracy, feed_dict=test_dict)
        print '\n'

        # export model graph to Tensorboard
        writer = tf.summary.FileWriter(dirname(__file__) + 'tensorboard/mnist-nn', session.graph)
        writer.close()
        print 'The tensorboard graph was saved in path: tensorboard/mnist-nn'
        print 'You can execute this command in terminal to see the graph: tensorboard --logdir="./tensorboard/mnist-nn"'
        print '\n'

        # save the variables to disk
        save_path = saver.save(session, dirname(__file__) + "trained_models/mnist-nn/model")
        print 'Trained model has been saved in path: ', save_path
        print '\n'

        # draw loss diagram
        plt.title('The loss diagram after trained')
        plt.plot(losses[:])
        plt.savefig('diagrams/diagram_of_loss_values_after_predicted')
        plt.show()


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
show_info()
training()

