#!/usr/bin/env bash
#PBS -l walltime=100:00:00
#PBS -q isi
#PBS -l nodes=1:gpus=2:shared
#PBS -N g2poov
#PBS -j oe
#PBS -o /dev/null
set -e
# hardcoded running of g2pseq2seq on a language including setup
LANGUAGE=$1
MODEL=$2
PARAMS=$3

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
workdir=$SCRIPTDIR/../$LANGUAGE/$MODEL
modeldir=$workdir/model
source activate tfgpu
source /usr/usc/cuda/8.0/setup.sh
source /usr/usc/cuDNN/7.5-v5.1/setup.sh
train=$datadir/train
dev=$datadir/dev
test=$datadir/test
mkdir -p $workdir
APP=$SCRIPTDIR/../tabmod/g2p_seq2seq/app.py

#python -u $SCRIPTDIR/tabmod/g2p_seq2seq/app.py --c2c --size 512 --train $train --model $modeldir --valid $dev --test $test --max_steps $STEPS &> $LANGUAGE/train.log
python -u $APP --c2c --train $train --model $modeldir --valid $dev --test $test $PARAMS &> $workdir/train.log

# decode and score
decodedir=$workdir/decode
mkdir -p $decodedir

python -u $APP --c2c --model $modeldir --decode <(cut -f1 $dev) --output $decodedir/dev.final &> $workdir/dev.log
num=`paste <(cut -f2 $dev) <(cut -f2 $decodedir/dev.final) | awk -F'\t' '$1==$2{print $1}' | wc -l`;
denom=`cat $dev | wc -l`;
echo "$num / $denom" > $decodedir/dev.score
echo "100.0 * $num.0 / $denom.0" | bc -l >> $decodedir/dev.score

python -u $APP --c2c --model $modeldir --decode <(cut -f1 $test) --output $decodedir/test.final &> $workdir/test.log
num=`paste <(cut -f2 $test) <(cut -f2 $decodedir/test.final) | awk -F'\t' '$1==$2{print $1}' | wc -l`;
denom=`cat $test | wc -l`;
echo "$num / $denom" > $decodedir/test.score
echo "100.0 * $num.0 / $denom.0" | bc -l >> $decodedir/test.score

$SCRIPTDIR/plotoutput.py -i $workdir/train.log -o $workdir/$LANGUAGE.pdf
