#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BottlesOfBear():
    def __init__(self, num_bottles=99):
        self.num_bottles = num_bottles
        self.graph = tf.while_loop(self.cond, self.body, [self.num_bottles, ''])

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def cond(self, i, _):
        return tf.greater_equal(i, 0)

    def body(self, i, text):
        before, after, before_uppercase = tf.cond(tf.equal(i, 0),
                lambda: ('no more bottles', '99 bottles', 'No more bottles'),
                lambda: tf.cond(tf.equal(i, 1),
                            lambda: ('1 bottle', 'no more bottles', '1 bottle'),
                            lambda: tf.cond(tf.equal(i, 2),
                                        lambda: ('2 bottles', '1 bottle', '2 bottles'),
                                        lambda: (tf.string_join([tf.as_string(i), ' bottles'], ''),
                                                 tf.string_join([tf.as_string(tf.subtract(i, 1)), ' bottles']),
                                                 tf.string_join([tf.as_string(i), ' bottles'], '')))))
        action = tf.cond(tf.equal(i, 0),
                        lambda: tf.constant('Go to the store and buy some more'),
                        lambda: tf.constant('Take one down and pass it around'))
        return tf.subtract(i, 1), tf.string_join([text, tf.string_join([before_uppercase, ' of beer on the wall, ', before, ' of beer.\n', action, ', ', after, ' of beer on the wall.\n'])])


if __name__ == '__main__':
    _, text = BottlesOfBear(99).run()
    print(text)