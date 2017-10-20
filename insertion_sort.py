#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
np.random.seed(123)


class InsertionSort():
    def __init__(self, array):
        self.i = tf.constant(1)
        self.j = tf.constant(len(array)-1)
        self.array = tf.Variable(array, trainable=False)
        self.length = len(array)

        cond = lambda i, j, _: tf.less(i-1, self.length-1)
        self.graph = tf.while_loop(cond, self.outer_loop, loop_vars=[self.i, self.j, self.array],
                shape_invariants=[self.i.get_shape(), self.j.get_shape(), tf.TensorShape(self.length)],
                parallel_iterations=1,
                back_prop=False)

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def outer_loop(self, i, j, _):
        j = i
        cond = lambda i, j, array: tf.logical_and(tf.greater(j,0), tf.greater(array[j-1], array[j]))

        loop = tf.while_loop(cond, self.inner_loop, loop_vars=[i, j, self.array],
                    shape_invariants=[i.get_shape(), j.get_shape(), tf.TensorShape(self.length)],
                    parallel_iterations=1,
                    back_prop=False)
        return tf.add(i, 1), loop[1], loop[2]

    def inner_loop(self, i, j, _):
        return i, tf.subtract(j, 1), tf.scatter_nd_update(self.array, [[j-1],[j]], [self.array[j],self.array[j-1]])


with tf.Session() as sess:
    x = np.array([1.,7.,3.,8.])
    print(x)
    print(InsertionSort(x).run()[2])
    y = np.random.rand(10)
    print(y)
    print(InsertionSort(y).run()[2])