# 批量进行GRPO数据集处理

# 逐个解压/cluster/home1/wzx/EgoReasoner/data/data/images下的所有zip文件到/cluster/home1/wzx/EgoReasoner/data/egoreasoner/data/images下并删除/cluster/home1/wzx/EgoReasoner/data/data/images下的该zip文件

#!/bin/bash

# --- 配置 ---
# 源目录，包含zip文件
SOURCE_DIR="/nfs/home1/wzx/EgoReasoner/data/data/images"
# 目标目录，用于存放解压后的文件
DEST_DIR="/nfs/home1/wzx/EgoReasoner/data/egoreasoner/data/images"
# --- 配置结束 ---

# 检查unzip命令是否存在
if ! command -v unzip &> /dev/null; then
    echo "错误: 'unzip' 命令未找到。请先安装unzip。"
    exit 1
fi

# 确保目标目录存在，如果不存在则创建
if [ ! -d "$DEST_DIR" ]; then
    echo "目标目录 '$DEST_DIR' 不存在，正在创建..."
    mkdir -p "$DEST_DIR"
    if [ $? -ne 0 ]; then
        echo "错误: 无法创建目标目录 '$DEST_DIR'。请检查权限。"
        exit 1
    else
        echo "目标目录 '$DEST_DIR' 创建成功。"
    fi
else
    echo "目标目录 '$DEST_DIR' 已存在。"
fi

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源目录 '$SOURCE_DIR' 不存在。请检查路径是否正确。"
    exit 1
fi

echo ""
echo "开始批量处理 '$SOURCE_DIR' 中的ZIP文件..."
echo "解压到: '$DEST_DIR'"
echo "=================================================="

# 启用nullglob，这样如果没有匹配的.zip文件，循环不会执行
shopt -s nullglob

processed_count=0
skipped_count=0
error_count=0
found_zip_files=0

# 遍历源目录下的所有.zip文件
for zip_file_path in "$SOURCE_DIR"/*.zip; do
    # nullglob 确保 $zip_file_path 是一个实际的文件名
    # （如果*.zip没有匹配，循环不会运行）
    # 但为保险起见，仍可添加 -f 检查，尽管在nullglob下通常不必要
    # if [ ! -f "$zip_file_path" ]; then
    #   continue # Should not happen with nullglob
    # fi

    found_zip_files=$((found_zip_files + 1))
    zip_file_name=$(basename "$zip_file_path")
    echo ""
    echo "处理文件: $zip_file_name"

    # 解压文件到目标目录
    # -o: 覆盖已存在的文件而不提示
    # -q: 安静模式，减少输出 (如果需要详细解压过程，可以移除 -q)
    unzip -o "$zip_file_path" -d "$DEST_DIR"
    unzip_status=$?

    if [ $unzip_status -eq 0 ]; then
        echo "  解压成功: $zip_file_name -> $DEST_DIR"
        
        # 删除原始的zip文件
        rm "$zip_file_path"
        rm_status=$?
        if [ $rm_status -eq 0 ]; then
            echo "  删除成功: $zip_file_path"
            processed_count=$((processed_count + 1))
        else
            echo "  错误: 删除原始ZIP文件 '$zip_file_path' 失败 (错误码: $rm_status)。"
            error_count=$((error_count + 1))
        fi
    elif [ $unzip_status -eq 1 ]; then
        # unzip 状态码 1 通常表示警告，例如文件已存在但未覆盖（如果未使用-o）
        # 或者是一些非致命的警告。这里我们将其视为部分成功但提示用户。
        # 由于使用了 -o，这个状态码可能不常见，除非zip文件本身有问题。
        echo "  警告: 解压文件 '$zip_file_name' 时产生警告 (unzip 错误码: $unzip_status)。文件可能已解压。"
        echo "  请检查 '$DEST_DIR' 中的内容以及 '$zip_file_name' 的完整性。"
        # 尝试删除原始文件，因为内容可能已部分或全部解压
        rm "$zip_file_path"
        rm_status=$?
        if [ $rm_status -eq 0 ]; then
            echo "  原始ZIP文件 '$zip_file_path' 已删除 (基于unzip警告后的操作)。"
            # 也可以将此计入processed_count或单独计数
        else
            echo "  错误: 基于unzip警告后，删除原始ZIP文件 '$zip_file_path' 失败 (错误码: $rm_status)。"
            error_count=$((error_count + 1))
        fi
    else
        echo "  错误: 解压文件 '$zip_file_name' 失败 (unzip 错误码: $unzip_status)。"
        echo "  原始ZIP文件 '$zip_file_path' 将不会被删除。"
        error_count=$((error_count + 1))
    fi
done

# 禁用nullglob
shopt -u nullglob

echo ""
echo "=================================================="
echo "批量处理完成。"
if [ $found_zip_files -eq 0 ]; then
    echo "在源目录 '$SOURCE_DIR' 中没有找到 .zip 文件。"
else
    echo "总共找到ZIP文件: $found_zip_files"
    echo "成功解压并删除的ZIP文件数量: $processed_count"
    if [ $skipped_count -gt 0 ]; then # skipped_count 逻辑目前未实现，保留以备将来扩展
        echo "跳过的文件数量: $skipped_count"
    fi
    if [ $error_count -gt 0 ]; then
        echo "处理失败或出现错误的ZIP文件数量: $error_count"
        echo "请检查上面的日志以获取详细错误信息。"
    fi
fi
echo "=================================================="

# 检查源目录是否还有残留的zip文件（以防万一rm失败）
remaining_zips=("$SOURCE_DIR"/*.zip)
if [ ${#remaining_zips[@]} -gt 0 ] && [ -f "${remaining_zips[0]}" ]; then # 检查数组是否非空且第一个元素是文件
    echo ""
    echo "警告: 源目录 '$SOURCE_DIR' 中仍存在以下ZIP文件 (可能是由于删除失败):"
    for leftover_zip in "${remaining_zips[@]}"; do
        if [ -f "$leftover_zip" ]; then # 再次确认是文件
            echo "  - $(basename "$leftover_zip")"
        fi
    done
    echo "请手动检查并删除。"
fi

exit 0
