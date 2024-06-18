#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

workspace_dir=$1
category=$2
start_time=$3

cd "$workspace_dir" || exit

python src/train.py trainer=gpu experiment="$category" start_time="$start_time" logger=csv
