#!/usr/bin/env bash
#PBS -l walltime=100:00:00
#PBS -q isi
#PBS -l nodes=1:gpus=2:shared
#PBS -N g2poov
#PBS -j oe
#PBS -o /dev/null
set -e
# hardcoded running of g2pseq2seq on a language including setup
# run multiple times based on content of reps
LANGUAGE=$1
MODEL=$2
PARAMS=$3
REPS=$4

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
datadir=$SCRIPTDIR/../$LANGUAGE/data
source activate tfgpu
source /usr/usc/cuda/8.0/setup.sh
source /usr/usc/cuDNN/7.5-v5.1/setup.sh
train=$datadir/train
dev=$datadir/dev
test=$datadir/test
APP=$SCRIPTDIR/../beam/g2p_seq2seq/app.py
scorer=$SCRIPTDIR/scorepair

# in case language has path slashes
langasname=$(echo $LANGUAGE | sed 's/\//_/g');

vizstring=""
for prefix in $REPS; do
    workdir=$SCRIPTDIR/../$LANGUAGE/$prefix/$MODEL
    modeldir=$workdir/model
    echo "$modeldir";
    mkdir -p $workdir
    python -u $APP --c2c --train $train --model $modeldir --valid $dev --test $test $PARAMS &> $workdir/train.log

    # decode and score
    decodedir=$workdir/decode
    mkdir -p $decodedir

    python -u $APP --c2c --model $modeldir --decode <(cut -f1 $dev) --output $decodedir/dev.final &> $workdir/dev.log
    $scorer $dev $decodedir/dev.final > $decodedir/dev.score

    python -u $APP --c2c --model $modeldir --decode <(cut -f1 $test) --output $decodedir/test.final &> $workdir/test.log
    $scorer $test $decodedir/test.final > $decodedir/test.score

    $SCRIPTDIR/plotoutput.py -i $workdir/train.log -o $workdir/$langasname.pdf
    vizstring="$vizstring $workdir/train.log";
done
echo "$vizstring";
vizdir=$SCRIPTDIR/../$LANGUAGE/viz
echo $vizdir;
mkdir -p $vizdir
$SCRIPTDIR/plotoutput.py --no-legend -i $vizstring -o $vizdir/$langasname.$MODEL.pdf
