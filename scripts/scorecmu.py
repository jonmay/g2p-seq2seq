#!/usr/bin/env python
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
import os.path
import gzip
import tempfile
import shutil
import atexit
import numpy as np
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

# adapted from https://martin-thoma.com/word-error-rate-calculation/

# h is a list of strings, rs is a list of list of strings
def per(h, rs):
  cands = []
  for r in rs:
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
     for j in range(len(h)+1):
       if i == 0:
         d[0][j] = j
       elif j == 0:
         d[i][0] = i
    # computation
    for i in range(1, len(r)+1):
      for j in range(1, len(h)+1):
        if r[i-1] == h[j-1]:
          d[i][j] = d[i-1][j-1]
        else:
          substitution = d[i-1][j-1] + 1
          insertion    = d[i][j-1] + 1
          deletion     = d[i-1][j] + 1
          d[i][j] = min(substitution, insertion, deletion)
    cands.append(d[len(r)][len(h)])
  return (min(cands), len(rs[np.argmin(cands)]))

# h is a string, rs is a list of strings
def wer(h, rs):
  for r in rs:
    if h == r:
      return 0
  return 1
    

def main():
  parser = argparse.ArgumentParser(description="calculate phoneme error rate and word error rate",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--srcfile", "-s", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input source file (just srcs); for duplicate management")
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input hypothesis file (just hyps)")
  parser.add_argument("--reffile", "-r", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input reference file (tab separated alternatives)")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
  addonoffarg(parser, 'perline', help="show per line stats", default=False)

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

  srcfile = prepfile(args.srcfile, 'r')
  infile = prepfile(args.infile, 'r')
  reffile = prepfile(args.reffile, 'r')
  outfile = prepfile(args.outfile, 'w')

  wernum = 0.
  werdenom = 0.
  pernum = 0.
  perdenom = 0.
  seen = set()
  for ln, (srcline, inline, refline) in enumerate(zip(srcfile, infile, reffile)):
    srcstring = srcline.strip()
    instring = inline.strip()
    inwords = instring.split()
    refstrings = refline.strip().split('\t')
    refwords = [x.split() for x in refstrings]
    wd = 1.
    wn = wer(instring, refstrings)
    pn, pd = per(inwords, refwords)
    if args.perline:
      outfile.write("{} PER {} WER {}\n".format(ln, pn/pd, wn))
    if srcstring not in seen:
      wernum   += wn
      werdenom += wd
      pernum   += pn
      perdenom += pd
    seen.add(srcstring)
  outfile.write("TOTAL {} PER {} WER {}\n".format(len(seen), pernum/perdenom, wernum/werdenom))
if __name__ == '__main__':
  main()
