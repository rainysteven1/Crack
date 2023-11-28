#!/bin/sh

cd /workspace/RETINA_UNET
python main.py --state train

result_path="result"
latest_folder=$(ls -td "$result_path"/*/ | head -n 1)
latest_folder_name=$(basename "$latest_folder")

python main.py --state predict --load_model_dir "$latest_folder_name"
