import tensorflow as tf
import numpy as np

# elementary operations
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))

print("\n\n\n")

# linear model
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# manually setting values for W and b
# fixW = tf.assign(W, [-1.])
# fixb = tf.assign(b, [1.])
# sess.run([fixW, fixb])
# print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(10):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

# after 10 iterations of gradient descent
print(sess.run([W, b]))
print(sess.run(linear_model, {x:[1,2,3,4]}))
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

for i in range(100):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

# after additional 100
print(sess.run([W, b]))
print(sess.run(linear_model, {x:[1,2,3,4]}))
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
