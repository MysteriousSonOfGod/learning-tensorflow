import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = tf.placeholder('float')
b = tf.placeholder('float')
c = tf.multiply(a, b)

with tf.Session() as session:
    print session.run(c, feed_dict={a: 1, b: 2})
