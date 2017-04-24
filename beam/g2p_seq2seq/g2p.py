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

"""Main class for g2p.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time
import sys
from heapq import heappush, heappop

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2

from g2p_seq2seq import data_utils
from g2p_seq2seq import seq2seq_model
from g2p_seq2seq.seq2seq_model import GO_ID

from six.moves import xrange, input  # pylint: disable=redefined-builtin
from six import text_type

class G2PModel(object):
  """Grapheme-to-Phoneme translation model class.

  Constructor parameters (for training mode only):
    train_lines: Train dictionary;
    valid_lines: Development dictionary;
    test_lines: Test dictionary.

  Attributes:
    gr_vocab: Grapheme vocabulary;
    ph_vocab: Phoneme vocabulary;
    train_set: Training buckets: words and sounds are mapped to ids;
    valid_set: Validation buckets: words and sounds are mapped to ids;
    session: Tensorflow session;
    model: Tensorflow Seq2Seq model for G2PModel object.
    train: Train method.
    interactive: Interactive decode method;
    evaluate: Word-Error-Rate counting method;
    decode: Decode file method.
  """
  # We use a number of buckets and pad to the closest one for efficiency.
  # See seq2seq_model.Seq2SeqModel for details of how they work.
  _BUCKETS = [(5, 15), (10, 20), (40, 50), (70, 80)]

  def __init__(self, model_dir):
    """Initialize model directory."""
    self.model_dir = model_dir

  def load_decode_model(self, scope="foo"):
    """Load G2P model and initialize or load parameters in session."""
    if not os.path.exists(os.path.join(self.model_dir, 'checkpoint')):
      raise RuntimeError("Model not found in %s" % self.model_dir)
    print("loading model with scope {} from {}".format(scope, self.model_dir))
    self.batch_size = 1 # We decode one word at a time.
    #Load model parameters.
    num_layers, size = data_utils.load_params(self.model_dir)
    # Load vocabularies
    self.gr_vocab = data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                            "vocab.grapheme"))
    self.ph_vocab = data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                            "vocab.phoneme"))

    self.rev_ph_vocab =\
      data_utils.load_vocabulary(os.path.join(self.model_dir, "vocab.phoneme"),
                                 reverse=True)
    # progresssive gpu memory allocation
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.session = tf.Session(graph=tf.Graph(), config=config)

    # Restore model.
    print("Creating %d layers of %d units." % (num_layers, size))
    with self.session.graph.as_default():
      with tf.variable_scope(scope):
        self.model = seq2seq_model.Seq2SeqModel(len(self.gr_vocab),
                                                len(self.ph_vocab), self._BUCKETS,
                                                size, num_layers, 0,
                                                self.batch_size, 0, 0,
                                                forward_only=True)
        self.model.saver = tf.train.Saver({'/'.join(v.op.name.split('/')[1:]): v for v in tf.global_variables()}, max_to_keep=1)
        # Check for saved models and restore them.
        print("Reading model parameters from %s" % self.model_dir)
        self.model.saver.restore(self.session, os.path.join(self.model_dir,
                                                            "model"))
    print("done loading {}".format(self.model_dir))


  def __put_into_buckets(self, source, target):
    """Put data from source and target into buckets.

    Args:
      source: data with ids for graphemes;
      target: data with ids for phonemes;
        it must be aligned with the source data: n-th line contains the desired
        output for n-th line from the source.

    Returns:
      data_set: a list of length len(_BUCKETS); data_set[n] contains a list of
        (source, target) pairs read from the provided data that fit
        into the n-th bucket, i.e., such that len(source) < _BUCKETS[n][0] and
        len(target) < _BUCKETS[n][1]; source and target are lists of ids.
    """

    # By default unk to unk
    data_set = [[[[4], [4]]] for _ in self._BUCKETS]

    for source_ids, target_ids in zip(source, target):
      target_ids.append(data_utils.EOS_ID)
      for bucket_id, (source_size, target_size) in enumerate(self._BUCKETS):
        if len(source_ids) < source_size and len(target_ids) < target_size:
          data_set[bucket_id].append([source_ids, target_ids])
          break
    return data_set


  def prepare_data(self, train_path, valid_path, test_path, c2c, logcount):
    """Prepare train/validation/test sets. Create or load vocabularies."""
    # Prepare data.
    print("Preparing G2P data")
    train_gr_ids, train_ph_ids, valid_gr_ids, valid_ph_ids, self.gr_vocab,\
    self.ph_vocab, self.test_lines =\
    data_utils.prepare_g2p_data(self.model_dir, train_path, valid_path,
                                test_path, c2c=c2c, logcount=logcount)
    # Read data into buckets and compute their sizes.
    print ("Reading development and training data.")
    self.valid_set = self.__put_into_buckets(valid_gr_ids, valid_ph_ids)
    self.train_set = self.__put_into_buckets(train_gr_ids, train_ph_ids)

    self.rev_ph_vocab = dict([(x, y) for (y, x) in enumerate(self.ph_vocab)])


  def __prepare_model(self, params):
    """Prepare G2P model for training."""

    self.params = params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.session = tf.Session(config=config)

    # Prepare model.
    print("Creating model with parameters:")
    print(params)
    self.model = seq2seq_model.Seq2SeqModel(len(self.gr_vocab),
                                            len(self.ph_vocab), self._BUCKETS,
                                            self.params.size,
                                            self.params.num_layers,
                                            self.params.max_gradient_norm,
                                            self.params.batch_size,
                                            self.params.learning_rate,
                                            self.params.lr_decay_factor,
                                            forward_only=False,
                                            optimizer=self.params.optimizer,
                                            dropout_keep_rate=self.params.dropout_keep_rate)
    self.model.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


  def load_train_model(self, params):
    """Load G2P model for continuing train."""
    # Check for saved model.
    if not os.path.exists(os.path.join(self.model_dir, 'checkpoint')):
      raise RuntimeError("Model not found in %s" % self.model_dir)

    # Load model parameters.
    params.num_layers, params.size = data_utils.load_params(self.model_dir)

    # Prepare data and G2P Model.
    self.__prepare_model(params)

    # Restore model.
    print("Reading model parameters from %s" % self.model_dir)
    self.model.saver.restore(self.session, os.path.join(self.model_dir,
                                                        "model"))


  def create_train_model(self, params):
    """Create G2P model for train from scratch."""
    # Save model parameters.
    data_utils.save_params(params.num_layers, params.size, self.model_dir)

    # Prepare data and G2P Model
    self.__prepare_model(params)

    print("Created model with fresh parameters.")
    self.session.run(tf.global_variables_initializer())


  def train(self):
    """Train a gr->ph translation model using G2P data."""

    train_bucket_sizes = [len(self.train_set[b])
                          for b in xrange(len(self._BUCKETS))]
    train_total_size = float(sum(train_bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, train_loss = 0.0, 0.0
    current_step, num_iter_wo_improve = 0, 0
    prev_train_losses, prev_valid_losses = [], []
    num_iter_cover_train = int(sum(train_bucket_sizes) /
                               self.params.batch_size /
                               self.params.steps_per_checkpoint)
    saved = False
    while (self.params.max_steps == 0
           or self.model.global_step.eval(self.session)
           <= self.params.max_steps):
      # Get a batch and make a step.
      start_time = time.time()
      step_loss = self.__calc_step_loss(train_buckets_scale)
      step_time += (time.time() - start_time) / self.params.steps_per_checkpoint
      train_loss += step_loss / self.params.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % self.params.steps_per_checkpoint == 0:
        # Print statistics for the previous steps.
        train_ppx = math.exp(train_loss) if train_loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (self.model.global_step.eval(self.session),
                         self.model.learning_rate.eval(self.session),
                         step_time, train_ppx))
        eval_loss = self.__calc_eval_loss()
        eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
        print("  eval: perplexity %.2f" % (eval_ppx))
        # Decrease learning rate if no improvement was seen on train set
        # over last 3 times.
        if (len(prev_train_losses) > 2
            and train_loss > max(prev_train_losses[-3:])):
          self.session.run(self.model.learning_rate_decay_op)

        if (len(prev_valid_losses) > 0
            and eval_loss <= min(prev_valid_losses)):
          print("Saving checkpoint at {}: new eval loss of {} <= {}".format(current_step, eval_loss, min(prev_valid_losses)))
          # Save checkpoint and zero timer and loss.
          self.model.saver.save(self.session,
                                os.path.join(self.model_dir, "model"),
                                write_meta_graph=False)
          saved = True

        if (len(prev_valid_losses) > 0
            and eval_loss >= min(prev_valid_losses)):
          num_iter_wo_improve += 1
        else:
          num_iter_wo_improve = 0

        if self.params.max_steps == 0 and num_iter_wo_improve > num_iter_cover_train * 2:
          print("No improvement over last %d times. Training will stop after %d"
                "iterations if no improvement was seen."
                % (num_iter_wo_improve,
                   (num_iter_cover_train * 3) - num_iter_wo_improve))

        # Stop train if no improvement was seen on validation set
        # over last 3 epochs.
        if self.params.max_steps == 0 and num_iter_wo_improve > num_iter_cover_train * 3:
          break

        prev_train_losses.append(train_loss)
        prev_valid_losses.append(eval_loss)
        step_time, train_loss = 0.0, 0.0

    print('Training done.')
    if not saved:
      print('Never saved before so saving now')
      self.model.saver.save(self.session,
                            os.path.join(self.model_dir, "model"),
                            write_meta_graph=False)

    with tf.Graph().as_default():
      g2p_model_eval = G2PModel(self.model_dir)
      g2p_model_eval.load_decode_model()
      g2p_model_eval.evaluate(self.test_lines)


  def __calc_step_loss(self, train_buckets_scale):
    """Choose a bucket according to data distribution. We pick a random number
    in [0, 1] and use the corresponding interval in train_buckets_scale.
    """
    random_number_01 = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(train_buckets_scale))
                     if train_buckets_scale[i] > random_number_01])

    # Get a batch and make a step.
    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
        self.train_set, bucket_id)
    _, step_loss, _ = self.model.step(self.session, encoder_inputs,
                                      decoder_inputs, target_weights,
                                      bucket_id, False)
    return step_loss


  def __calc_eval_loss(self):
    """Run evals on development set and print their perplexity.
    """
    eval_loss, num_iter_total = 0.0, 0.0
    for bucket_id in xrange(len(self._BUCKETS)):
      num_iter_cover_valid = int(math.ceil(len(self.valid_set[bucket_id])/
                                           self.params.batch_size))
      num_iter_total += num_iter_cover_valid
      for batch_id in xrange(num_iter_cover_valid):
        encoder_inputs, decoder_inputs, target_weights =\
            self.model.get_eval_set_batch(self.valid_set, bucket_id,
                                          batch_id * self.params.batch_size)
        _, eval_batch_loss, _ = self.model.step(self.session, encoder_inputs,
                                                decoder_inputs, target_weights,
                                                bucket_id, True)
        eval_loss += eval_batch_loss
    eval_loss = eval_loss/num_iter_total if num_iter_total > 0 else float('inf')
    return eval_loss



  def decode_word(self, word, c2c=False, aux=[], vocab=None, beam=1, beamfactor=1):
    """Decode input word to sequence of phonemes.

    Args:
      word: input word;
      c2c: is it a character-to-character model?
      aux: other models to ensemble
      vocab: limiting vocabulary in a marisa_trie.Trie
      beam: how many hypotheses to persist simultaneously?

    Returns:
      phonemes: decoded phoneme sequence for input word;
    """
    # Check if all graphemes attended in vocabulary
    gr_absent = [gr for gr in word if gr not in self.gr_vocab]
    if gr_absent:
      print("Symbols '%s' are not in vocabulary" % "','".join(gr_absent).encode('utf-8'))
      return "", 0.0

    for auxm in aux:
      gr2 = [gr for gr in word if gr not in auxm.gr_vocab]
      if gr2 != gr_absent:
        print("Mismatch between model grs: {} vs {}".format(gr_absent, gr2))
    # Get token-ids for the input word.
    token_ids = [self.gr_vocab.get(s, data_utils.UNK_ID) for s in word]
    for auxm in aux:
      ti2 = [auxm.gr_vocab.get(s, data_utils.UNK_ID) for s in word]
      if ti2 != token_ids:
        print("Mismatch between token ids: {} vs {}".format(token_ids, ti2))
    # shrink if too long (and yell)
    if len(token_ids) > self._BUCKETS[-1][0]:
      trim = self._BUCKETS[-1][0]-1
      print("%s is too long (%d); truncating to %d = %s" % (word, len(token_ids), trim, word[:trim]))
      token_ids = token_ids[:trim]
    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(self._BUCKETS))
                     if self._BUCKETS[b][0] > len(token_ids)])
    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id, beam=beam)

    joiner = "" if c2c else " "
    def _allowable(outvec, cand, joiner, vocab):
      # is the candidate allowable according to the vocab?
      if cand == data_utils.EOS_ID:
        return joiner.join(self.rev_ph_vocab[c] for c in outvec) in vocab
      return len(vocab.keys(joiner.join(self.rev_ph_vocab[c] for c in outvec+[cand]))) > 0

    histories = [[GO_ID]]*beam
    weights = [0.]*beam
    #print(len(decoder_inputs))
    for pos in range(len(decoder_inputs)):
      #print("At position {}".format(pos))
      # concatenate all output_logits and get the mean
      logitstack = []
      _, _, output_logits = self.model.step(self.session, encoder_inputs,
                                            decoder_inputs, target_weights,
                                            bucket_id, True, automatic=False)
      logitstack.append(output_logits[pos])
      for auxm in aux:
        _, _, m2_output_logits = auxm.model.step(auxm.session, encoder_inputs,
                                                 decoder_inputs, target_weights,
                                                 bucket_id, True, automatic=False)
        logitstack.append(m2_output_logits[pos])
      #print(logitstack)
      logits = np.mean(np.array(logitstack), axis=0)
      # unsorted argmax_beam*beamfactor for each successor:
      innerbeam = min(beam*beamfactor, logits.shape[1])
      indices = np.argpartition(logits, -innerbeam)[:,-innerbeam:]
      # unsorted vals to pair up with indices
      maxvals = logits[np.matrix(range(indices.shape[0])).transpose(),indices]

      # candidates are previous sequence plus the new index, scoring previous score plus the new (log) score
      def _fillheap(skipvocab): 
        cands = []
        for chist, whist, cgroup, wgroup in zip(histories, weights, indices, maxvals):
          # no exploration from completed sequences allowed
          if chist[-1] == data_utils.EOS_ID:
            heappush(cands, (whist, chist))
            continue
          for c, w in zip(cgroup, wgroup):
            if skipvocab or _allowable(chist[1:], c, joiner, vocab):
              #print("adding {} and {} = {}; concatenating {} and {}".format(whist, -np.log(w), whist+-np.log(w), str(chist), c))
              heappush(cands, (whist+-np.log(w), chist+[c]))
            # else:
            #   print("skipping {} + {} for vocab".format(str(chist), c))
          if pos == 0: # too much duplication first round
            break
        return cands
      # if checking vocab and heap is too small, need to fall back on unchecked
      if vocab is None:
        cands = _fillheap(True)
      else:
        cands = _fillheap(False)
        if len(cands) < beam:
          sys.stderr.write("Could only put {} of {} items into heap; falling back to no vocab check at pos {}\n".format(len(cands), beam, pos))
          cands = _fillheap(True)

      # pop things off the heap and set up for next decoding step
      for i in range(beam):
        weight, cand = heappop(cands)
        #print("{} : {} = {}".format(weight, cand, joiner.join([self.rev_ph_vocab[c] for c in cand[1:]]).encode("utf-8")))
        histories[i] = cand
        weights[i] = weight
        if pos+1 < len(decoder_inputs):
          for j, c in enumerate(cand):
            decoder_inputs[j][i] = c
      #for bn, (cand, weight) in enumerate(zip(histories, weights)):
      #  print("{}:{} {}".format(bn, weight, joiner.join(self.rev_ph_vocab[c] for c in cand[1:])))
      #print()
      if histories[0][-1] == data_utils.EOS_ID:
        break
    outputs = []
    for cand in histories:
      stop = -1 if cand[-1] == data_utils.EOS_ID else None
      outputs.append(joiner.join(self.rev_ph_vocab[c] for c in cand[1:stop]))
    # for on, output in enumerate(outputs):
    #   print("{}:{} {}".format(on, weights[on], output.encode("utf-8")))
    return outputs[0], weights[0]


  def interactive(self, c2c=False):
    """Decode word from standard input.
    """
    while True:
      try:
        word = input("> ")
        if not issubclass(type(word), text_type):
          word = text_type(word, encoding='utf-8', errors='replace')
      except EOFError:
        break
      if not word:
        break
      print(self.decode_word(word, c2c=c2c)[0])


  def calc_error(self, dictionary, c2c=False):
    """Calculate a number of prediction errors.
    """
    errors = 0
    for word, pronunciations in dictionary.items():
      hyp = self.decode_word(word, c2c=c2c)[0]
      if hyp not in pronunciations:
        errors += 1
    return errors


  def evaluate(self, test_lines, c2c=False):
    """Calculate and print out word error rate (WER) and Accuracy
       on test sample.

    Args:
      test_lines: List of test dictionary. Each element of list must be String
                containing word and its pronounciation (e.g., "word W ER D");
    """
    test_dic = data_utils.collect_pronunciations(test_lines)

    if len(test_dic) < 1:
      print("Test dictionary is empty")
      return

    print('Beginning calculation word error rate (WER) on test sample.')
    errors = self.calc_error(test_dic, c2c=c2c)

    print("Words: %d" % len(test_dic))
    print("Errors: %d" % errors)
    print("WER: %.3f" % (float(errors)/len(test_dic)))
    print("Accuracy: %.3f" % float(1-(errors/len(test_dic))))


  def decode(self, decode_lines, output_file=None, c2c=False, aux=[], vocab=None, beam=1, beamfactor=1, showscore=False):
    """Decode words from file.

    Returns:
      if [--output output_file] pointed out, write decoded word sequences in
      this file. Return to calling function, which is helpful for internal scoring
    """
    for word in decode_lines:
      word = word.strip()
      phonemes, score = self.decode_word(word, c2c=c2c, aux=aux, vocab=vocab, beam=beam, beamfactor=beamfactor)
      outdata = [word, phonemes]
      if showscore:
        outdata.append(str(score))
      result = '\t'.join(outdata)+"\n"
      if output_file:
        output_file.write(result)

class TrainingParams(object):
  """Class with training parameters."""
  def __init__(self, flags=None):
    if flags:
      self.learning_rate = flags.learning_rate
      self.lr_decay_factor = flags.learning_rate_decay_factor
      self.max_gradient_norm = flags.max_gradient_norm
      self.batch_size = flags.batch_size
      self.size = flags.size
      self.num_layers = flags.num_layers
      self.steps_per_checkpoint = flags.steps_per_checkpoint
      self.max_steps = flags.max_steps
      self.optimizer = flags.optimizer
      self.dropout_keep_rate = flags.dropout_keep_rate
    else:
      self.learning_rate = 0.5
      self.lr_decay_factor = 0.99
      self.max_gradient_norm = 5.0
      self.batch_size = 64
      self.size = 64
      self.num_layers = 2
      self.steps_per_checkpoint = 200
      self.max_steps = 0
      self.optimizer = "sgd"
      self.dropout_keep_rate = 1.0

  def __str__(self):
    return ("Learning rate:        {}\n"
            "LR decay factor:      {}\n"
            "Dropout keep rate:    {}\n"
            "Max gradient norm:    {}\n"
            "Batch size:           {}\n"
            "Size of layer:        {}\n"
            "Number of layers:     {}\n"
            "Steps per checkpoint: {}\n"
            "Max steps:            {}\n"
            "Optimizer:            {}\n").format(
              self.learning_rate,
              self.lr_decay_factor,
              self.dropout_keep_rate,
              self.max_gradient_norm,
              self.batch_size,
              self.size,
              self.num_layers,
              self.steps_per_checkpoint,
              self.max_steps,
              self.optimizer)
