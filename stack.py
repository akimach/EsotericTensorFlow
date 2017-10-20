#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Stack():
    def __init__(self, length=10):
        self.length = length
        self.array = tf.constant(np.zeros(length), dtype=tf.int32)
        self.ptr = tf.constant(0, dtype=tf.int32)

    def push(self, x):
        self.ptr = tf.add(self.ptr, 1)
        self.array = tf.concat([
                self.array[:self.ptr-1],
                [x],
                self.array[self.ptr:]], axis=0)
        self.array.set_shape(self.length)
        return self.array

    def pop(self):
        self.ptr = tf.subtract(self.ptr, 1)
        x = self.array[self.ptr]
        self.array = tf.concat([self.array[:self.ptr],[tf.constant(0)],self.array[self.ptr+1:]], axis=0)
        self.array.set_shape(self.length)
        return x, self.array


def push(stack, ptr, x, length=10):
    s = tf.concat([stack[:ptr], [x], stack[ptr+1:]], axis=0)
    s.set_shape(length)
    return s, tf.add(ptr, 1)

def pop(stack, ptr, length=10, default_value=0):
    x = stack[ptr-1]
    s = tf.concat([stack[:ptr-1], [tf.constant(default_value)], stack[ptr:]], axis=0)
    s.set_shape(length)
    return s, tf.subtract(ptr, 1), x

def assign(arr, ptr, x, length=10):
    a = tf.concat([arr[:ptr-1], [x], arr[ptr:]], axis=0)
    a.set_shape(length)
    return a, ptr, x

# if __name__ == '__main__':
#     stack = Stack(length=5)
#     stack.push(tf.constant(10))
#     stack.push(tf.constant(3))
#     stack.push(tf.constant(5))
#     stack.push(tf.constant(8))
#     x0, _ = stack.pop()
#     x1, _ = stack.pop()
#     with tf.Session() as sess:
#         print(sess.run(stack.array))
#         print(sess.run([x0, x1,]))

if __name__ == '__main__':
    length = 10
    p0 = tf.constant(0, dtype=tf.int32)
    s0 = tf.constant([-1]*10, dtype=tf.int32)
    s1, p1 = push(s0, p0, 3)
    s2, p2 = push(s1, p1, 4)
    s3, p3 = push(s2, p2, 7)
    s4, p4, x4 = pop(s3, p3, default_value=-1)
    s5, p5, x5 = pop(s4, p4, default_value=-1)
    s6, p6, x6 = pop(s5, p5, default_value=-1)
    s7, p7, x7 = pop(s6, p6, default_value=-1)
    #s7, p7, x7 = assign(s6, tf.constant(5), tf.constant(9))
    with tf.Session() as sess:
        print(sess.run([s0, p0]))
        print(sess.run([s1, p1]))
        print(sess.run([s2, p2]))
        print(sess.run([s3, p3]))
        print(sess.run([s4, p4, x4]))
        print(sess.run([s5, p5, x5]))
        print(sess.run([s6, p6, x6]))
        #print(sess.run([s7, p7, x7]))
