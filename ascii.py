#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def ascii2char(x):
    default = tf.constant('', dtype=tf.string)
    table = tf.constant([chr(i) for i in range(127)], dtype=tf.string)
    cond = tf.logical_and(tf.greater_equal(x, tf.constant(0)), tf.less(x, tf.constant(127)))
    return tf.cond(cond, lambda: table[x], lambda: default)


if __name__ == '__main__':
    SP = ascii2char(tf.constant(0))
    A = ascii2char(tf.constant(65))
    B = ascii2char(tf.constant(66))
    C = ascii2char(tf.constant(67))
    x = ascii2char(tf.constant(120))
    TILDE = ascii2char(tf.constant(126))
    d1 = ascii2char(tf.constant(127))
    d2 = ascii2char(tf.constant(-1))
    with tf.Session() as sess:
        ret = sess.run([SP,A,B,C,x,TILDE,d1,d2])
    for ch in ret:
        print(ch)