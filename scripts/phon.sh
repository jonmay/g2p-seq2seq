#!/usr/bin/env bash
#PBS -l walltime=100:00:00
#PBS -q isi
#PBS -l nodes=1:gpus=2:shared
#PBS -N g2p
#PBS -j oe
#PBS -o /dev/null
set -e
# hardcoded running of g2pseq2seq on real g2p data
MODEL=$1
PARAMS=$2

# learning_rate (0.5)
#"learning_rate", 0.5, "Learning rate.")
#"learning_rate_decay_factor", 0.99,
#"Learning rate decays by this much.")
#"max_gradient_norm", 5.0,
#r("batch_size", 64,
#  "Batch size to use during training.")
#r("size", 64, "Size of each model layer.")
#r("num_layers", 2, "Number of layers in the model.")
#r("steps_per_checkpoint", 200,                            "How many training steps to do per checkpoint.")
#("max_steps", 0,                            "How many training steps to do until stop training"                            " (0: no limit).")
#("optimizer", "sgd", "Optimizer type: sgd, adam, rms-prop. Default: sgd.")


SCRIPTDIR=`dirname $0`
datadir=/home/rcf-40/jonmay/projects/cnmt/phonetisaurus-cmudict-split
workdir=$SCRIPTDIR/../$MODEL
modeldir=$workdir/model
train=$datadir/cmudict.dic.train
test=$datadir/cmudict.dic.test
evalsrc=$SCRIPTDIR/../test.src
evalref=$SCRIPTDIR/../test.ref
mkdir -p $workdir
APP=$SCRIPTDIR/../g2p-seq2seq

#python -u $SCRIPTDIR/tabmod/g2p_seq2seq/app.py --c2c --size 512 --train $train --model $modeldir --valid $dev --test $test --max_steps $STEPS &> $LANGUAGE/train.log
$APP --train $train --model $modeldir --test $test $PARAMS &> $workdir/train.log

# decode and score
decodedir=$workdir/decode
mkdir -p $decodedir

$APP --model $modeldir --decode $evalsrc --output $decodedir/test.final &> $workdir/test.log
cut -d' ' -f2- $decodedir/test.final > $decodedir/test.final.justphones
$SCRIPTDIR/scorecmu.py -s $evalsrc -r $evalref -i $decodedir/test.final.justphones -o $decodedir/test.score
