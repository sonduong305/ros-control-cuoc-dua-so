import tensorflow as tf

class Regression():
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels

    def model(self):
        with tf.variable_scope("regression") as scope:
            #Input: (N,320,320,7)
            flatten = tf.layers.flatten(self.input)
            #flatten: (N, 1, 1, 716800 )
            fc = tf.layers.dense(flatten, 512)
            fc = tf.layers.dense(fc, 128)
            fc = tf.layers.dense(fc, 2)

            return fc

    def loss(self):
        predict = self.model()
        return tf.losses.mean_squared_error(predict, self.labels)