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
import matplotlib as mpl
import numpy as np
from itertools import cycle
mpl.use('Agg') # this is so the code works in non-x11 environments
import matplotlib.pyplot as plt

scriptdir = os.path.dirname(os.path.abspath(__file__))

reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
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
  parser = argparse.ArgumentParser(description="plot files that come from g2ps2s output",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--infile", "-i", nargs='+', type=argparse.FileType('r'), help="input files")
  parser.add_argument("--outfile", "-o", help="output file; type dictated by filename")
  parser.add_argument("--begin", "-b", type=int, default=-1, help="start collecting once x value exceeds this (-1 = start at beginning)")
  parser.add_argument("--stop", "-s", type=int, default=-1, help="stop collecting once x value exceeds this (-1 = don't stop)")
  addonoffarg(parser, "legend", default=True, help="show a legend")
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

  infiles = [prepfile(x, 'r') for x in args.infile]
  #plt.subplot(211)
  colors = cycle(plt.cm.Set1(np.linspace(0,1,9)))
#  colors = cycle(mpl.colors.CSS4_COLORS.keys())
  for infile, color in zip(infiles, colors):
    xs = []
    tys = []
    dys = []
    #print("color for {} is {}".format(infile.name, color))
    skip = False
    for line in infile:
      if line.startswith("global step"):
        try:
          line = line.strip().split()
          nextxs = int(line[2])
          nextty = float(line[9])
          if args.begin >=0 and nextxs < args.begin:
            skip = True
          if args.stop >= 0 and nextxs > args.stop:
            break
          if not skip:
            tys.append(nextty)
            xs.append(nextxs)
        except ValueError:
          sys.stderr.write("Bad line: {}".format(line))
          continue
      elif line.startswith("  eval:"):
        try:
          if not skip:
            nextdy = line.strip().split()[2]
            dys.append(nextdy)
          skip = False
        except ValueError:
          sys.stderr.write("Bad line: {}".format(line))
          continue
    if len(xs) == len(tys):
      plt.plot(xs, tys, color=color, linestyle='-', label=infile.name+" train")
    else:
      sys.stderr.write("Can't plot {}; mismatch\n".format(infile.name+" train"))
    if len(xs) == len(dys):
      plt.plot(xs, dys, color=color, linestyle='--')
    else:
      sys.stderr.write("Can't plot {}; mismatch {} != {}\n".format(infile.name+" dev", len(xs), len(dys)))
  if args.legend:
    plt.legend(loc='upper right', fontsize=10)
  plt.savefig(args.outfile)

if __name__ == '__main__':
  main()
