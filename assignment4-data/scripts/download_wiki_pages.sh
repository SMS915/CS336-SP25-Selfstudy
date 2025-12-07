#!/bin/bash
# 自动生成的维基百科引用页面下载脚本
# 日志将输出到 wget_wiki.log 文件中
wget \
    --timeout=15 \
    --tries=2 \
    --max-redirect=5 \
    --quota=30G \
    --reject-regex '\.(pdf|zip|gz|rar|exe|iso|mp3|mp4|avi|mov)$' \
    -i "data/wiki/sampled_wiki_urls.txt" \
    --warc-file="data/wiki/wiki_cited_pages" \
    -O /dev/null
