# tensorflow 1.8
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

hello = tf.constant("Hello world")
a = tf.constant(10)
b = tf.constant(11)

with  tf.Session() as sess:
    print(sess.run(hello))
    print(sess.run(a + b))
