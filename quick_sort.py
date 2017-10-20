#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
np.random.seed(123)

from stack import push, pop, assign


class QuickSort():
    def __init__(self, arr):
        self.length = len(arr)
        self.arr = tf.Variable(arr, dtype=tf.int32)

        left = tf.constant(0, dtype=tf.int32)
        right = tf.constant(0, dtype=tf.int32)
        last = tf.constant(0, dtype=tf.int32)
        index = tf.constant(0, dtype=tf.int32)

        s0 = tf.constant([-1]*self.length, dtype=tf.int32)
        p0 = tf.constant(0, dtype=tf.int32)

        s1, p1 = push(s0, p0, self.length-1, length=self.length)
        s2, p2 = push(s1, p1, 0, length=self.length)

        self.graph = tf.while_loop(self.cond_outer, self.body_outer, [self.arr, index, s2, p2, left, right, last])

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def cond_outer(self, arr, ix, s, p, left, right, last):
        c = tf.greater_equal(s, 0)
        return tf.greater(tf.reduce_sum(tf.cast(c, tf.int32)), 0)

    def body_outer(self, arr, ix, s, p, left, right, last):
        s1, p1, l = pop(s, p, length=self.length, default_value=-1)
        s2, p2, r = pop(s1, p1, length=self.length, default_value=-1)

        def inner_loop(arr, ix, s, p, left, right, last):
            _, _ix, _s, _p, _left, _right, _last = tf.while_loop(self.cond_inner, self.body_inner, [arr, ix, s, p, left, right, last])

            _arr = tf.scatter_update(self.arr, [_left, _last], [self.arr[_last], self.arr[_left]])

            s1, p1 = push(_s, _p, _last-1, length=self.length)
            s2, p2 = push(s1, p1, _left, length=self.length)
            s3, p3 = push(s2, p2, _right, length=self.length)
            s4, p4 = push(s3, p3, _last+1, length=self.length)
            return _arr, ix, s4, p4, _left, _right, _last

        def fn(arr, ix, s, p, left, right, last):
            last = left
            ix = left + 1
            return inner_loop(arr, ix, s, p, left, right, last)

        return tf.cond(tf.greater_equal(l, r),
                lambda: (arr, ix, s2, p2, l, r, last),
                lambda: fn(arr, ix, s2, p2, l, r, last))

    def cond_inner(self, arr, ix, s, p, left, right, last):
        return tf.less_equal(ix, right)

    def body_inner(self, arr, ix, s, p, left, right, last):
        def fp():
            _last = tf.add(last, 1)
            arr = tf.scatter_update(self.arr, [_last, ix], [self.arr[ix], self.arr[_last]])
            return (arr, tf.add(ix, 1), s, p, left, right, _last)
        return tf.cond(tf.greater(self.arr[left], arr[ix]),
                        lambda: fp(),
                        lambda: (arr, tf.add(ix, 1), s, p, left, right, last))


if __name__ == '__main__':
    arr = np.array([3, 2, 5, 9, 1, 7])
    sorted_arr, ix, s, p, left, right, last =  QuickSort(arr).run()
    print(arr)
    print(sorted_arr)

    arr = np.random.randint(1, 100, 20)
    sorted_arr, _, _, _, _, _, _ =  QuickSort(arr).run()
    print(arr)
    print(sorted_arr)
