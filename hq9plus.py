#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from bottles_of_bear import BottlesOfBear


class HQ9Plus():
    def __init__(self, tokens):
        self.tokens = tf.constant(tokens)
        self.cnt = tf.Variable(0, trainable=False)
        with open(__file__) as fin:
            self.src = fin.read()
        self.graph = tf.while_loop(self.cond, self.body, [0, self.tokens, ''],
                            back_prop=False)

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def cond(self, i, x, _):
        return tf.less(i, tf.size(self.tokens))

    def body(self, i, _, output):
        def inc():
            return tf.cond(tf.equal(tf.assign(self.cnt, self.cnt +1), 0),
                    lambda: (''),
                    lambda: (''))
        r = tf.cond(tf.equal(self.tokens[i], 'H'),
            lambda: ('Hello, world!\n'),
            lambda: tf.cond(tf.equal(self.tokens[i], 'Q'),
                    lambda: (self.src),
                    lambda: tf.cond(tf.equal(self.tokens[i], '9'),
                            lambda: (BottlesOfBear().graph[1]),
                            lambda: tf.cond(tf.equal(self.tokens[i], '+'),
                                    lambda: inc(),
                                    lambda: ('')))))
        return tf.add(i, 1), self.tokens, output + r


if __name__ == '__main__':
    tokens = list('HQ9+')
    _, _, output = HQ9Plus(tokens).run()
    print(output)
