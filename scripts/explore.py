#!/usr/bin/env python3
# code by Jon May [jonmay@isi.edu]
import argparse
import sys
import codecs
if sys.version_info[0] == 2:
  from itertools import izip
else:
  izip = zip
from collections import defaultdict as dd
import re
import os
import os.path
import gzip
import tempfile
import shutil
import atexit
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

def addonoffarg(parser, arg, dest=None, default=True, help="TODO"):
  ''' add the switches --arg and --no-arg that set parser.arg to true/false, respectively'''
  group = parser.add_mutually_exclusive_group()
  dest = arg if dest is None else dest
  group.add_argument('--%s' % arg, dest=dest, action='store_true', default=default, help=help)
  group.add_argument('--no-%s' % arg, dest=dest, action='store_false', default=default, help="See --%s" % arg)


# need to write handlers for each training type
# need to consume name type vals triples of parameter explore space
# need to define key resource value and its inital value
# need to define rounds and number of initial configurations

class Param:
  def __init__(self, deffile):
    ''' for each line in deffile, store parameter name and set to draw from
    if type is int or float, we are given min and max vals
    if type is cat, we are given all possible vals
    '''
    pass

  def draw(self, configs=1):
    ''' return configs unique configurations along with names for each of them '''
    pass

def orderseq2seq(names, files):
  ''' get lowest perplexity on dev from each log file
  return names sorted by this perplexity, along with the perplexities '''
  pass

def main():
  parser = argparse.ArgumentParser(description="iterative halving/doubling for parameter exploration",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input file")
  parser.add_argument("--deffile", "-d", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="parameter definition file")
  parser.add_argument("--keyparam", "-k", type=str, help="key timing parameter")
  parser.add_argument("--paramstart", "-s", type=int, help="key parameter starting value")
  parser.add_argument("--width", "-w", type=int, default=20, help="initial exploration width")
  parser.add_argument("--fixparam", "-f", type=str, help="fixed parameter values")
  parser.add_argument("--expdir", "-e", type=str, help="experiment directory root")
  parser.add_argument("--binary", type=str, default="/home/nlg-05/jonmay/cnmt/g2p-seq2seq/beam/g2p_seq2seq/app.py", help="experiment binary [TODO: replace with classes]")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")

  workdir = tempfile.mkdtemp(prefix=os.path.basename(__file__), dir=os.getenv('TMPDIR', '/tmp'))

  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))

  def cleanwork():
    shutil.rmtree(workdir, ignore_errors=True)
  if args.debug:
    print(workdir)
  else:
    atexit.register(cleanwork)

  param_selector = Param(args.deffile)

  config_set = param_selector.draw(configs=args.width)
  # create models 
  for cfg_name, cfg_vals in config_set:
    workdir=os.path.join(expdir, cfg_name, "model")
    os.makedirs(workdir, exist_ok=True)
    # TODO: make a class hierarchy that in the leaves has implementations for binary, output, args
    # TODO: allow qsub and checking for when model completes
    expcall="{bin} --{keyparam} {keyval} {fixparam} 
    outfile.write("experiment {} with params {} at {}".format(cfg_name, cfg_vals, workdir))


  infile = prepfile(args.infile, 'r')
  outfile = prepfile(args.outfile, 'w')

  for line in infile:
    pass

if __name__ == '__main__':
  main()
