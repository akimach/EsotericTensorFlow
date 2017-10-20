#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class LinearSearch():
    def __init__(self, array, x):
        self.x = tf.constant(x)
        self.array = tf.constant(array)
        self.length = len(array)
        self.graph = tf.while_loop(self.cond, self.body, [0, self.x, False],
                            back_prop=False)

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def cond(self, i, _, is_found):
        return tf.logical_and(tf.less(i, self.length), tf.logical_not(is_found))

    def body(self, i, _, is_found):
        return tf.cond(tf.equal(self.array[i], self.x),
                    lambda: (i, self.array[i], True),
                    lambda: (tf.add(i, 1), -1, False))


if __name__ == '__main__':
    array, x = [1, 7, 3, 8], 3
    search = LinearSearch(array, x)
    ix, xx, is_found = ret = search.run()
    print('Array :', array)
    print('Number to search :', x)
    if is_found:
        print('{} is at index {}.'.format(xx, ix))
    else:
        print('Not found.')
