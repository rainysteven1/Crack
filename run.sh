#!/bin/sh
workspace_dir=$1
category="resunet++"

cd $workspace_dir
python main.py --state train --category "$category"

result_path="result"
latest_folder=$(ls -td "$result_path"/*/ | head -n 1)
latest_folder_name=$(basename "$latest_folder")

python main.py --state predict --category "$category" --load_model_dir "$latest_folder_name"
