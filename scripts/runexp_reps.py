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
from subprocess import check_output
import shlex
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

def main():
  parser = argparse.ArgumentParser(description="launch experiments for a language",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input file of tab-sep modelname, args")
  parser.add_argument("--lang", "-l", type=str, help="iso 639-3 language code")
  parser.add_argument("--qsubargs", "-q", type=str, default="", help="arguments to qsub that override defaults")
  parser.add_argument("--prefixes", "-p", nargs='+', type=str, default=None, help="prefixes for every model")
  parser.add_argument("--script", type=str, default=os.path.join(scriptdir, "g2poov_reps.sh"), help="command we're running")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file with job id paired onto lang, name, model, args")

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

  infile = prepfile(args.infile, 'r')
  outfile = prepfile(args.outfile, 'w')

  for line in infile:
    model, arguments = line.strip().split('\t')
    prefix=" ".join(args.prefixes) if args.prefixes is not None else ".";
    cmd="qsubrun -N \"{1}_{2}\" {4} -- {0} {1} {2} \"{3}\" \"{5}\"".format(args.script, args.lang, model, arguments, args.qsubargs, prefix)
    jobid=check_output(shlex.split(cmd))
    outfile.write("{}\t{}\t{}\t{}\t{}\n".format(jobid, args.lang, model, arguments, cmd))

if __name__ == '__main__':
  main()
