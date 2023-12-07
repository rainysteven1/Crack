#!/bin/sh

CONFIG_FOLDER="resources/config"
CONFIG_PATH="$CONFIG_FOLDER/configuration.json"
CRACK500_PATH="resources/data/crack500"

workspace_dir=$1
category=$2

cd $workspace_dir

jq '.' "$CONFIG_FOLDER/$category.json" >"$CONFIG_PATH"
jq --arg new_path "$workspace_dir/$CRACK500_PATH" '.["data path"]["data_path"] = $new_path' "$CONFIG_PATH" >tmp.json
mv tmp.json "$CONFIG_PATH"

python main.py --state train --category "$category"

result_path="result"
latest_folder=$(ls -td "$result_path"/*/ | head -n 1)
latest_folder_name=$(basename "$latest_folder")

python main.py --state predict --category "$category" --load_model_dir "$latest_folder_name"
