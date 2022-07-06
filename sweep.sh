#!/usr/bin/zsh
NUM=$1
SWEEPID=$2
wandb agent --count $NUM $SWEEPID
