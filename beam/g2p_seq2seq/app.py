# Copyright 2016 AC Technologies LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import tensorflow as tf
import os.path
import argparse
import numpy as np
from marisa_trie import Trie
scriptdir = os.path.dirname(os.path.abspath(__file__))

reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')

def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code) if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret


# local priority
import sys
sys.path.insert(0, os.path.join(scriptdir, ".."))

from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.g2p import TrainingParams

def addonoffarg(parser, arg, dest=None, default=True, help="TODO"):
  ''' add the switches --arg and --no-arg that set parser.arg to true/false, respectively'''
  group = parser.add_mutually_exclusive_group()
  dest = arg if dest is None else dest
  group.add_argument('--%s' % arg, dest=dest, action='store_true', default=default, help=help)
  group.add_argument('--no-%s' % arg, dest=dest, action='store_false', default=default, help="See --%s" % arg)

parser = argparse.ArgumentParser(description="REPLACE WITH DESCRIPTION",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
addonoffarg(parser, 'debug', help="debug mode", default=False)
parser.add_argument("--learning_rate", type=float,default=0.5, help="Learning rate.")
parser.add_argument("--learning_rate_decay_factor", type=float,default=0.99, help="Learning rate decays by this much.")
parser.add_argument("--max_gradient_norm", type=float,default=5.0, help="Clip gradients to this norm.")
parser.add_argument("--dropout_keep_rate", type=float,default=1.0, help="probability of not dropping out.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use during training.")
parser.add_argument("--size", type=int, default=64, help="Size of each model layer.")
parser.add_argument("--display_freq", type=int, default=1000, help="How often to display a demo decode during training.")
parser.add_argument("--display_size", type=int, default=5, help="How many sentences to decode during training display.")
parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model.")
parser.add_argument("--models", nargs='+', required=True, type=str, default=None, help="Training directory.")
parser.add_argument("--steps_per_checkpoint", type=int, default=200, help="How many training steps to do per checkpoint.")
addonoffarg(parser, "interactive", default=False, help="Set to True for interactive decoding.")
addonoffarg(parser, "c2c", default=False, help="Set to True to assume rhs is char based too.")
parser.add_argument("--evaluate", type=str, default="", help="Count word error rate for file.")
parser.add_argument("--decode", type=str, default="", help="Decode file.")
parser.add_argument("--beam", type=int, default=1, help="decoding beam")
parser.add_argument("--beamfactor", type=int, default=1, help="beam*factor = successors to try per context; only useful when vocabulary limiting")
parser.add_argument("--output", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="decoding output file")
parser.add_argument("--train", type=str, default="", help="Train dictionary.")
parser.add_argument("--valid", type=str, default="", help="Development dictionary.")
parser.add_argument("--test", type=str, default="", help="Test dictionary.")
parser.add_argument("--vocab", type=str, default="", help="Limiting vocabulary for decodes.")
addonoffarg(parser, "logcount", default=False, help="use log count instead of plain counts")
addonoffarg(parser, "showscore", default=False, help="show log prob in decodes")
parser.add_argument("--max_steps", type=int, default=0, help="How many training steps to do until stop training (0: no limit).")
addonoffarg(parser, "reinit", default=False, help="Set to True for training from scratch.")
parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer type: sgd, adam, rms-prop. Default: sgd.")

try:
  FLAGS = parser.parse_args()
except IOError as msg:
  parser.error(str(msg))


def main(_=[]):
  """Main function.
  """
  with tf.Graph().as_default():
    g2p_models = [G2PModel(m) for m in FLAGS.models]
    g2p_model = g2p_models[0]
    if len(FLAGS.train) > 0:
      g2p_params = TrainingParams(FLAGS)
      if len(g2p_models) > 1:
        sys.stderr.write("Warning: can only train one model at a time; others will be ignored\n")
      valid_lines = np.array(filter(lambda x: len(x.strip().split('\t')) < 40, codecs.open(FLAGS.valid, "r", "utf-8").readlines()))
      g2p_model.prepare_data(FLAGS.train, FLAGS.valid, FLAGS.test, FLAGS.c2c, FLAGS.logcount)
      if (not os.path.exists(os.path.join(FLAGS.models[0],
                                          "model.data-00000-of-00001"))
          or FLAGS.reinit):
        g2p_model.create_train_model(g2p_params)
      else:
        g2p_model.load_train_model(g2p_params)
      g2p_model.train(c2c=FLAGS.c2c, valid=valid_lines, dispfreq=FLAGS.display_freq, dispsize=FLAGS.display_size)
    else:
      g2p_model.load_decode_model(scope="mainmodel")
      aux_models = []
      for mn, m in enumerate(g2p_models[1:]):
        m.load_decode_model(scope="aux{}".format(mn))
        aux_models.append(m)
      vocab = None
      if FLAGS.vocab:
        vocab = Trie([x.strip() for x in codecs.open(FLAGS.vocab, "r", "utf-8").readlines()])
      if len(FLAGS.decode) > 0:
        decode_lines = codecs.open(FLAGS.decode, "r", "utf-8").readlines()
        output_file = prepfile(FLAGS.output, 'w')
        g2p_model.decode(decode_lines, output_file, c2c=FLAGS.c2c, aux=aux_models, vocab=vocab, beam=FLAGS.beam, beamfactor=FLAGS.beamfactor, showscore=FLAGS.showscore)
      elif FLAGS.interactive:
        g2p_model.interactive(c2c=FLAGS.c2c)
      elif len(FLAGS.evaluate) > 0:
        test_lines = codecs.open(FLAGS.evaluate, "r", "utf-8").readlines()
        g2p_model.evaluate(test_lines, c2c=FLAGS.c2c)

if __name__ == "__main__":
  tf.app.run()
