# solution for https://www.kaggle.com/uciml/iris
# Species' value are changed for simplicity 
# Iris-setosa --> 0
# Iris-versicolor --> 1
# Iris-virginica --> 2

import pandas as pd
import numpy as np
import tensorflow as tf

index  = 150
learning_rate = 0.1
tf.set_random_seed(777)  # reproducibility
training_epochs = 200

csv_data = pd.read_csv('D:/work/personal/python3/iris.csv')

SepalLengthCm = np.array(csv_data.SepalLengthCm.as_matrix()).reshape((index,1))
SepalWidthCm = np.array(csv_data.SepalWidthCm.as_matrix()).reshape((index,1))
PetalLengthCm = np.array(csv_data.PetalLengthCm.as_matrix()).reshape((index,1))
PetalWidthCm = np.array(csv_data.PetalWidthCm.as_matrix()).reshape((index,1))


y_data = np.array(csv_data.Species.as_matrix())


x_data = np.concatenate((SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm), axis = 1)

n_values = np.max(y_data) + 1
y_data = np.eye(n_values)[y_data]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])


W1 = tf.get_variable("W1", shape=[4, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[256, 3],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([3]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    feed_dict = {X: x_data, Y: y_data}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    if epoch % 10 == 0:
        print('Epoch:', '%6d' % (epoch), 'cost =', '{:.9f}'.format(c))
        
print('Learning Finished!')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
