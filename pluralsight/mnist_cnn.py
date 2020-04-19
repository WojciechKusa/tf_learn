import random
import os
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tensorboard_folder = './tb_logs/'
if not os.path.isdir(tensorboard_folder):
    os.mkdir(tensorboard_folder)


sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')


with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1,28,28,1], name="x_image")
    tf.summary.image('input_img', x_image, 5)

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)


with tf.name_scope('conv_1'):
    W_conv1 = weight_variable([5, 5, 1, 32], name='weight')
    b_conv1 = bias_variable([32], name='bias')

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name='relu')

    h_pool1 = max_pool_2x2(h_conv1, name='max_pool')

with tf.name_scope('conv_2'):
    W_conv2 = weight_variable([5, 5, 32, 64], name='weight')
    b_conv2 = bias_variable([64], name='bias')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='relu')
    h_pool2 = max_pool_2x2(h_conv2, name='max_pool')

with tf.name_scope('fully_con'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024], name='weight')
    b_fc1 = bias_variable([1024], name='bias')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='flaten')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='fcon')


keep_prob = tf.placeholder(tf.float32, name='dropout_prob')  # get dropout probability as a training
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='dropout')

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Loss measurement
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

    # loss optimization
with tf.name_scope('loss'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    # What is correct
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # How accurate is it?
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy_scl", cross_entropy)
tf.summary.scalar("train accuracy", accuracy)

# TB - Merge summaries
summarize_all = tf.summary.merge_all()

# Initialize all of the variables
sess.run(tf.global_variables_initializer())

tbWriter = tf.summary.FileWriter(tensorboard_folder, sess.graph)
# Train the model

#  define number of steps and how often we display progress
num_steps = 2000
display_every = 100

# Start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    _, summary = sess.run([train_step, summarize_all], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


    # Periodic status display
    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))
        # write summary to log
        tbWriter.add_summary(summary,i)

# Display summary
#     Time to train
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

#     Accuracy on test data
print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100.0))

sess.close()
