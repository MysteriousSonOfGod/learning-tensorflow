import os
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


def show_info(dataset):
    train_images = dataset.train.images
    train_labels = dataset.train.labels
    test_images = dataset.test.images
    test_labels = dataset.test.labels
    print '\n..::Overview::..'
    print '- Type of mnist is ', type(dataset)
    print '- Number of train data is ', dataset.train.num_examples
    print '- Number of test data is ', dataset.test.num_examples
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

        # use matplotlib to show 3 random elements and it's label visually
        plt.matshow(img, cmap=plt.get_cmap('gray'))
        plt.title('item ' + str(i) + 'th was labeled ' + str(label))
        plt.show()

    print '\n'


def training(dataset):
    learning_rate = 0.001
    batch_size = 128
    num_steps = 10000
    display_step = 1000

    # init input layer
    input_nodes = 784  # 28x28 pixels
    X = tf.placeholder(tf.float32, shape=[None, input_nodes])

    # init hidden layer 1
    hidden_1_nodes = 256  # each pixel has value from 0 - 255 (0: black, 255: white) -> 256 values can occur
    weight_1 = tf.Variable(tf.random_normal([input_nodes, hidden_1_nodes]), name='weight_1')
    bias_1 = tf.Variable(tf.random_normal([hidden_1_nodes]), name='bias_1')
    hidden_layer_1 = tf.add(tf.matmul(X, weight_1), bias_1)  # hidden layer 1 = weight * X + bias

    # init hidden layer 2
    # hidden layer 2 = weight * X + bias
    hidden_2_nodes = 256  # each pixel has value from 0 - 255 (0: black, 255: white) -> 256 values can occur
    weight_2 = tf.Variable(tf.random_normal([hidden_1_nodes, hidden_2_nodes]), name='weight_2')
    bias_2 = tf.Variable(tf.random_normal([hidden_2_nodes]), name='bias_2')
    hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weight_2), bias_2)

    # init output layer
    # output layer = weight * input from hidden layer 2 + bias
    output_nodes = 10
    Y = tf.placeholder(tf.float32, shape=[None, output_nodes])
    weight_output = tf.Variable(tf.random_normal([hidden_2_nodes, output_nodes]), name='weight_output')
    bias_output = tf.Variable(tf.random_normal([output_nodes]), name='bias_output')
    output_layer = tf.add(tf.matmul(hidden_layer_2, weight_output), bias_output)

    # reduce loss (back propagation)
    test_dict = {X: dataset.test.images, Y: dataset.test.labels}
    loss_output = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_output = optimizer.minimize(loss_output)
    correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print '..::Model training::..'
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for step in range(0, num_steps):
            batch_x, batch_y = dataset.train.next_batch(batch_size)
            train_dict = {X: batch_x, Y: batch_y}

            # run optimization output (backprop)
            session.run(train_output, feed_dict=train_dict)
            if step % display_step == 0:
                # calculate batch loss and accuracy
                loss, acc = session.run([loss_output, accuracy], feed_dict=train_dict)
                print 'Step ' + str(step) + \
                      ', minibatch loss= ' + '{:.4f}'.format(loss) + \
                      ', training accuracy= ' + '{:.3f}'.format(acc)

        print 'Finished!!!'
        print '\n'

        print 'Learning rate: ', learning_rate
        print 'Batch size: ', batch_size
        print 'Testing accuracy:', session.run(accuracy, feed_dict=test_dict)
        print '\n'


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
show_info(mnist)
training(mnist)

