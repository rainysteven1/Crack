#!/bin/bash

make clean
make clean-logs

zip -r Crack.zip . -x "data/*" -x "resources/*" -x ".git/*"

zip -d Crack.zip "scripts/compress.sh"
zip -d Crack.zip "*.ipynb"

scp Crack.zip user@5.tcp.vip.cpolar.cn:/home/user/rainy/deep-learning/Segmentations/Crack
rm -rf Crack.zip
