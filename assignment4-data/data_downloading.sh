#!/bin/bash

set -e # 如果任何命令失败，脚本将立即退出

DATA_BASE_DIR="data"

echo "确保目录 '$DATA_BASE_DIR' 存在..."
mkdir -p $DATA_BASE_DIR

echo "创建存放cc文件，fasttext分类器文件和wiki外部链接的文件目录"
CRAWLS_DIR="data/crawls"
CLASSIFIERS_DIR="data/classifiers"
WIKI_DATA_DIR="data/wiki"

mkdir -p $CRAWLS_DIR
mkdir -p $CLASSIFIERS_DIR
mkdir -p $WIKI_DATA_DIR

echo "开始下载示例cc文件"
wget -c -P $CRAWLS_DIR https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/warc/CC-MAIN-20250417135010-20250417165010-00065.warc.gz
wget -c -P $CRAWLS_DIR https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/wet/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz

echo "示例cc文件下载完成！"


echo "开始下载分类器模型"

echo "正在下载语言识别模型"
wget -c -P $CLASSIFIERS_DIR https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

echo "正在下载NSFW内容分类器"
wget -c -P $CLASSIFIERS_DIR https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin

echo "正在下载有害言论分类器"
wget -c -P $CLASSIFIERS_DIR https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin

echo "所有分类器模型下载完成！"

echo "正在下载维基百科外部链接URL列表 (enwiki-20240420-extracted_urls.txt.gz)..."
wget -c -P $WIKI_DATA_DIR https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz

echo " URL列表下载完成！"

echo -e "所有必需的源数据都已成功下载到 'data/' 目录中"