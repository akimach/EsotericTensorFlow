#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from stack import push, pop, assign
from ascii import ascii2char


class BrainFuck():
    def __init__(self, src, tape_length=100):
        self.tokens = tf.constant(list(src), dtype=tf.string)
        self.length = len(list(src))

        p0 = tf.constant(0, dtype=tf.int32)
        j0 = tf.constant(np.zeros(self.length, dtype=np.int32))
        p0 = tf.constant(0, dtype=tf.int32)
        s0 = tf.constant(np.zeros(self.length), dtype=tf.int32)
        _, _, _, jumps = tf.while_loop(self.cond_jumps, self.body_jumps, [p0, s0, p0, j0], back_prop=False)

        self.tape = tf.Variable(tf.zeros(tape_length, dtype=tf.int32))
        c0 = tf.constant(0, dtype=tf.int32)
        o0 = tf.constant('', dtype=tf.string)
        self.graph = tf.while_loop(self.cond, self.body, [p0, self.tape, c0, jumps, o0], back_prop=False)

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)

    def cond_jumps(self, pc, s, p, j):
        return tf.less(pc, tf.size(self.tokens))

    def body_jumps(self, pc, s, p, j):
        token = self.tokens[pc]
        inc_pc = tf.add(pc, 1)

        def brackets_begin(i):
            _s, _p = push(s, p, i, length=self.length)
            return (inc_pc, _s, _p, j)

        def brackets_end(i):
            _s, _p, from_ = pop(s, p, length=self.length)
            to_ = i
            j1, _, _ = assign(j, from_+1, to_, length=self.length)
            j2, _, _ = assign(j1, to_+1, from_, length=self.length)
            return (inc_pc, _s, _p, j2)

        return tf.cond(tf.equal(token, '['),
            lambda: (brackets_begin(pc)),
            lambda: tf.cond(tf.equal(token, ']'),
                lambda: (brackets_end(pc)),
                lambda: (inc_pc, s, p, j)))

    def cond(self, pc, tape, cur, jumps, output):
        return tf.less(pc, tf.size(self.tokens))

    def body(self, pc, tape, cur, jumps, output):
        token = self.tokens[pc]
        inc_pc = tf.add(pc, 1)

        def stdin(c):
            #return tf.assign(self.tape[c], input(''))
            return self.tape

        return tf.cond(tf.equal(token, '+'),
            lambda: (inc_pc, tf.assign(self.tape[cur], self.tape[cur]+1), cur, jumps, output),
            lambda: tf.cond(tf.equal(token, '-'),
                lambda: (inc_pc, tf.assign(self.tape[cur], self.tape[cur]-1), cur, jumps, output),
                lambda: tf.cond(tf.equal(token, '>'),
                    lambda: (inc_pc, tape, tf.add(cur, 1), jumps, output),
                    lambda: tf.cond(tf.equal(token, '<'),
                        lambda: (inc_pc, tape, tf.subtract(cur, 1), jumps, output),
                        lambda: tf.cond(tf.equal(token, '.'),
                            lambda: (inc_pc, tape, cur, jumps, tf.string_join([output, ascii2char(tape[cur])])),
                            lambda: tf.cond(tf.equal(token, ','),
                                lambda: (inc_pc, stdin(cur), cur, jumps, output),
                                lambda: tf.cond(tf.equal(token, '['),
                                    lambda: tf.cond(tf.equal(self.tape[cur], 0),
                                        lambda: (jumps[pc], tape, cur, jumps, output),
                                        lambda: (inc_pc, tape, cur, jumps, output)),
                                    lambda: tf.cond(tf.equal(token, ']'),
                                        lambda: tf.cond(tf.not_equal(self.tape[cur], 0),
                                            lambda: (jumps[pc], tape, cur, jumps, output),
                                            lambda: (inc_pc, tape, cur, jumps, output)),
                                        lambda: (inc_pc, tape, cur, jumps, output) ))))))))


if __name__ == '__main__':
    src_A = '++++++[> ++++++++++ < -]> +++++.'
    pc, tape, cur, jumps, output = BrainFuck(src_A).run()
    print(output) #=> A

    src_helloworld ='''
+++++++++[>++++++++>+++++++++++>+++++<<<-]>.>++.+++++++..+++.>-.
------------.<++++++++.--------.+++.------.--------.>+.
'''
    pc, tape, cur, jumps, output = BrainFuck(src_helloworld).run()
    print(output) #=> Hello, world!