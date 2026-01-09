#!/bin/bash
mkdir -p data/crawls/wet

cat "scripts/download_cc_batch_0_to_100.txt" | xargs -n 1 -P 8 wget -c -P "data/crawls/wet"
