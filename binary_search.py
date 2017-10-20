#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class BinarySearch():
    def __init__(self, array, x):
        self.array = tf.constant(array)
        self.x = tf.constant(x)
        self.loop = tf.while_loop(self.cond, self.body, [-1,False,0,len(array),-1],
                        back_prop=False)

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.loop)

    def cond(self, x, is_found, left, right, mid):
        return tf.logical_and(tf.less_equal(left, right), tf.logical_not(is_found))

    def body(self, x, is_found, left, right, mid):
        mid = tf.to_int32(tf.divide(tf.add(left, right), 2))
        return tf.cond(tf.equal(self.array[mid], self.x),
                    lambda: (self.array[mid], True, left, right, mid),
                    lambda: tf.cond(tf.less(self.array[mid], self.x),
                                lambda: (-1, False, tf.add(mid, 1), right, mid),
                                lambda: (-1, False, left, tf.subtract(mid, 1), mid)))

if __name__ == '__main__':
    array = sorted([1, 7, 3, 8, 5])
    x = 8
    search = BinarySearch(array, x)
    xx, is_found, l, r, m = search.run()

    print('Array :', array)
    print('Number to search :', x)
    if is_found:
        print('{} is at index {}.'.format(xx, m))
    else:
        print('Not found.')
