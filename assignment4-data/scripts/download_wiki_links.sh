CRAWL_NAME="CC-MAIN-2025-18"
MANIFEST_URL="https://data.commoncrawl.org/crawl-data/${CRAWL_NAME}/wet.paths.gz"
TARGET_DIR="data/cc_path" 

echo "正在下载清单文件: wet.paths.gz..."    
wget -c -P $TARGET_DIR $MANIFEST_URL
echo "清单文件下载完成！"