#!/bin/sh

CONFIG_FOLDER="resources/config"
CRACK500_PATH="resources/data/crack500"

workspace_dir=$1
category=$2
config_path="$CONFIG_FOLDER/$category.json"

cd $workspace_dir

jq --arg new_path "$workspace_dir/$CRACK500_PATH" '.["data path"]["data_path"] = $new_path' "$config_path" >tmp.json
mv tmp.json "$config_path"

python main.py --state train --category "$category"

result_path="result"
latest_folder=$(ls -td "$result_path"/*/ | head -n 1)
latest_folder_name=$(basename "$latest_folder")

python main.py --state predict --category "$category" --load_model_dir "$latest_folder_name"
