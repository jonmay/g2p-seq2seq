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

def addonoffarg(parser, arg, dest=None, default=True, help="TODO"):
  ''' add the switches --arg and --no-arg that set parser.arg to true/false, respectively'''
  group = parser.add_mutually_exclusive_group()
  dest = arg if dest is None else dest
  group.add_argument('--%s' % arg, dest=dest, action='store_true', default=default, help=help)
  group.add_argument('--no-%s' % arg, dest=dest, action='store_false', default=default, help="See --%s" % arg)

def main():
  parser = argparse.ArgumentParser(description="print statistics for oov task, comparing to other approaches",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--reffile", "-r", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input reference file (src trg)")
  parser.add_argument("--trainfile", "-t", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input training file (src trg count)")
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input hypothesis file")
  parser.add_argument("--cmpfile", "-c", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="comparison hypothesis file")
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

  infile = prepfile(args.infile, 'r')
  reffile = prepfile(args.reffile, 'r')
  trainfile = prepfile(args.trainfile, 'r')
  cmpfile = prepfile(args.cmpfile, 'r')
  outfile = prepfile(args.outfile, 'w')

  traintrgs = set()
  for line in trainfile:
    traintrgs.add(line.split('\t')[1])

  hypcount = 0.
  rightcount = 0.
  cmprightcount = 0.
  novcount = 0.
  novrightcount = 0.
  overlap = 0.
  rightoverlap = 0.

  refnovel = 0.
  rightensemble = 0.
  for ln, (refline, inline, cmpline) in enumerate(zip(reffile, infile, cmpfile), start=1):
    refline = refline.strip().split('\t')
    inline = inline.strip().split('\t')
    cmpline = cmpline.strip().split('\t')
    if len(refline) < 2:
      sys.stderr.write("Bad line {}: [{}]\n[{}]\n[{}]\n\n".format(ln, refline, inline, cmpline))
      sys.exit(1)      
    if len(inline) < 2:
      inline.append("")
    if len(cmpline) < 2:
      cmpline.append("")
    if refline[0] != inline[0] or inline[0] != cmpline[0]:
      sys.stderr.write("Mismatch at line {}: [{}]\n[{}]\n[{}]\n\n".format(ln, refline, inline, cmpline))
      sys.exit(1)
    hypcount +=1.
    right = (inline[1] == refline[1])
    if right:
      rightcount+=1.
    cmpright = (cmpline[1] == refline[1])
    if cmpright:
      cmprightcount+=1.
    if right or cmpright:
      rightensemble+=1.
    if refline[1] not in traintrgs:
      refnovel+=1.
    if inline[1] not in traintrgs:
      novcount +=1.
      if right:
        novrightcount+=1
        if args.debug:
          outfile.write("NOVEL CORRECT {}: {} -> {} ({})\n".format(ln, inline[0], inline[1], cmpline[1]))
      else:
        if args.debug:
          outfile.write("NOVEL WRONG {}: {} -> {} ({}) [{}]\n".format(ln, inline[0], inline[1], cmpline[1], refline[1]))
    elif args.debug:
      if right:
        if cmpright:
          outfile.write("CORRECT {}: {} -> {} ({})\n".format(ln, inline[0], inline[1], cmpline[1]))
        else:
          outfile.write("BETTER {}: {} -> {} ({})\n".format(ln, inline[0], inline[1], cmpline[1]))
      elif cmpright:
          outfile.write("WORSE {}: {} -> {} ({}) [{}]\n".format(ln, inline[0], inline[1], cmpline[1], refline[1]))
      else:
          outfile.write("WRONG {}: {} -> {} ({}) [{}]\n".format(ln, inline[0], inline[1], cmpline[1], refline[1]))
    if inline[1] == cmpline[1]:
      overlap +=1.
      if right:
        rightoverlap +=1.
  outfile.write("{:.2f} right {:.2f} cmpright {:.2f} rightensemble {:.2f} novel {:.2f} rightnovel {:.2f} refnovel {:.2f} overlap {:.2f} rightoverlap {:.2f} rightoverlapoverall\n".format(rightcount/hypcount, cmprightcount/hypcount, rightensemble/hypcount, novcount/hypcount, novrightcount/rightcount, refnovel/hypcount, overlap/hypcount, rightoverlap/rightcount, rightoverlap/overlap))
    

if __name__ == '__main__':
  main()
