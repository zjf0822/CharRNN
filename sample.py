from __future__ import print_function
import tensorflow as tf

import argparse
import os
import pickle

from model import Model

from six import text_type
from tensorflow import flags

flags.DEFINE_string('save_dir', 'save', 'model directory to store checkpointed models')
flags.DEFINE_integer('n', 500, 'number of characters to sample')
flags.DEFINE_string('prime', "u' '", 'prime text')
flags.DEFINE_integer('sample', 1, '0 to use max at each time step, 1 to sample at each time step, 2 to sample on sapces')

FLAGS = flags.FLAGS
FLAGS._parse_flags()

def main():
    sample(FLAGS)


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = pickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, chars, vocab, args.n, args.prime,
                               args.sample).encode('utf-8'))

if __name__ == '__main__':
    main()
