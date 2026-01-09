#!/bin/bash
# 自动生成的轻量级下载脚本 (仅调用文本清单)
mkdir -p data/crawls/wet

wget -c -P "data/crawls/wet" -i "scripts/download_cc_batch_100_to_1000.txt"
