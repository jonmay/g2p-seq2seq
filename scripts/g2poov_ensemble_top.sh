#!/usr/bin/env bash
#PBS -l walltime=100:00:00
#PBS -q isi
#PBS -l nodes=1:gpus=2:shared
#PBS -N g2poov_ensemble
#PBS -j oe
#PBS -o /dev/null
set -e
# g2pseq2seq dev and test decodes and scores for ensembling

LANGDIR=$1
TOP=$2

SCRIPTDIR=`dirname $0`
datadir=$SCRIPTDIR/../$LANGDIR/data
source activate tfgpu
source /usr/usc/cuda/8.0/setup.sh
source /usr/usc/cuDNN/7.5-v5.1/setup.sh

dev=$datadir/dev
test=$datadir/test
APP=$SCRIPTDIR/../beam/g2p_seq2seq/app.py
scorer=$SCRIPTDIR/scorepair

# gather top $TOP systems; assume there are multiple systems with a decode/dev.score and a model directory
# TODO: logcount is bowdlerized below due to vocab misalignment; fix that and re-enable!
for i in `find $LANGDIR -name dev.score`; do echo -n "$i "; tail -1 $i; done | sort -k2nr > $LANGDIR/top.list
echo -n $(for s in $(for i in `find $LANGDIR -name dev.score`; do echo -n "$i "; tail -1 $i; done | grep -v logcount | sort -k2nr | head -$TOP | cut -d' ' -f1); do s=`dirname $s`; s=`dirname $s`; echo -n "$s/model "; done;) > $LANGDIR/top$TOP.systems;

# decode and score dev and test
python -u $APP --c2c --model $(cat $LANGDIR/top$TOP.systems) --decode <(cut -f1 $dev) --output $LANGDIR/top$TOP.dev &> $LANGDIR/top$TOP.dev.log
$scorer $dev $LANGDIR/top$TOP.dev > $LANGDIR/top$TOP.dev.score
python -u $APP --c2c --model $(cat $LANGDIR/top$TOP.systems) --decode <(cut -f1 $test) --output $LANGDIR/top$TOP.test &> $LANGDIR/top$TOP.test.log
$scorer $test $LANGDIR/top$TOP.test > $LANGDIR/top$TOP.test.score
