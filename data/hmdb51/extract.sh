#!/bin/bash

# 移动RAR文件到videos目录
mv ../hmdb51_org.rar .

# 解压主RAR文件
echo "Extracting main RAR file..."
7z x -bsp1 hmdb51_org.rar

# 解压所有生成的子RAR文件
echo "Extracting all sub-RAR files..."
for file in *.rar; do
    if [ "$file" != "hmdb51_org.rar" ]; then
        echo "Extracting $file..."
        7z x -bsp1 "$file"
    fi
done

# 清理RAR文件
echo "Cleaning up RAR files..."
rm *.rar

echo "HMDB-51 dataset extraction completed."