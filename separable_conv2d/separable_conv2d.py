import tensorflow as tf
import numpy as np

input_data = tf.Variable(np.random.randint(1,5,size=(1,4,4,3)),dtype=np.float32)
depthwise_filter = tf.Variable(np.random.randint(1,5,size=(3,3,3,3)),dtype = np.float32)
pointwise_filter = tf.Variable(np.random.randint(1,5,size=(1,1,9,1)),dtype=np.float32)
y = tf.nn.separable_conv2d(input_data,depthwise_filter,pointwise_filter,strides=[1,1,1,1],padding='VALID')

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    print("out1=", sess.run(input_data))
    print("out2=",sess.run(depthwise_filter))
    print("out3=",sess.run(pointwise_filter))
    print("out4=",sess.run(y))
    print("out5=",sess.run(tf.shape(y)))

