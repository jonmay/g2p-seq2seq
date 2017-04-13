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

def addonoffarg(parser, arg, dest=None, default=True, help="tabify spaced output files from g2p"):
  ''' add the switches --arg and --no-arg that set parser.arg to true/false, respectively'''
  group = parser.add_mutually_exclusive_group()
  dest = arg if dest is None else dest
  group.add_argument('--%s' % arg, dest=dest, action='store_true', default=default, help=help)
  group.add_argument('--no-%s' % arg, dest=dest, action='store_false', default=default, help="See --%s" % arg)

def main():
  parser = argparse.ArgumentParser(description="REPLACE WITH DESCRIPTION",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--srcfile", "-s", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input source file (use tab sep lhs)")
  parser.add_argument("--transfile", "-t", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input translation file (expect to look like srcfile, then have a space")
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

  srcfile = prepfile(args.srcfile, 'r')
  transfile = prepfile(args.transfile, 'r')
  outfile = prepfile(args.outfile, 'w')

  for srcline, transline in zip(srcfile, transfile):
    srcside = srcline.strip().split('\t')[0]
    if srcside != transline[:len(srcside)]:
      sys.stderr.write("Problem trying to deal with {} and {}; {} != {}".format(srcline, transline, srcside, transline[:len(srcside)]))
      sys.exit(1)
    if transline[len(srcside)] != " ":
      sys.stderr.write("Problem trying to deal with {} and {}; [{}] is not space".format(srcline, transline, transline[len(srcside)]))
      sys.exit(1)
    outfile.write("{}\t{}".format(srcside, transline[len(srcside)+1:]))

if __name__ == '__main__':
  main()
