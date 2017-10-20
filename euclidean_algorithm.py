#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class EuclideanAlgorithm():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.graph = self.gcd(self.x, self.y)

    def run(self):
        with tf.Session() as sess:
            return sess.run(self.graph[0])

    def gcd(self, x, y):
        cond = lambda _, y: tf.greater(y, 0)
        body = lambda x, y: (y, tf.mod(x, y))
        return tf.while_loop(cond, body, loop_vars=[x, y], back_prop=False)

if __name__ == '__main__':
    print(EuclideanAlgorithm(24, 9).run())
    print(EuclideanAlgorithm(1071, 1029).run())
