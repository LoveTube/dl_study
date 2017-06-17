import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn import preprocessing


learning_rate = 0.2
tf.set_random_seed(777)  # reproducibility
training_epochs = 3000
neuron = 256
input = 20


csv_data = pd.read_csv('D:/work/personal/python3/voice.csv')
csv_data.sample(frac=0.2)
trainData = csv_data.drop(['label'], axis=1)
train = trainData.as_matrix()
train = preprocessing.normalize(train, norm='l2')
labelData = pd.get_dummies(csv_data['label'])
label = labelData.as_matrix()


x_data = preprocessing.normalize(pd.read_csv('D:/work/personal/python3/voice.csv').drop(['label'], axis=1).as_matrix(), norm='l2')
y_data = pd.get_dummies((pd.read_csv('D:/work/personal/python3/voice.csv'))['label'])


def build_layer(prev, i):
    name = "W" + str(i)
    w = tf.get_variable(str(name), shape=[neuron, neuron],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([neuron]))
    l = tf.nn.relu(tf.matmul(prev, w + b))

    return l

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, input])
Y = tf.placeholder(tf.float32, [None, 2])


W1 = tf.get_variable("W0", shape=[input, neuron],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([neuron]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[neuron, neuron],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([neuron]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[neuron, neuron],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([neuron]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)


W4 = tf.get_variable("W4", shape=[neuron, 2],initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L3, W4) + b4

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    feed_dict = {X: train, Y: label}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    if epoch % 10 == 0:
        print('Epoch:', '%6d' % (epoch), 'cost =', '{:.9f}'.format(c))

    if c < 0.18:
        print('done',  'cost =', '{:.9f}'.format(c))
        break
        
print('Learning Finished!')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))


