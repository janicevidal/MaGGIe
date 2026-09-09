#!/bin/bash

# ========== 配置区 ==========
FOLDER_A="/data/xiaoshuai/human_matting/dichotomous_dataset/_train/image/come/images/"   # 改成你的 A 文件夹路径
FOLDER_B="/data/xiaoshuai/human_matting/dataset/OpenDataLab___COME15K/raw/all_images/"   # 改成你的 B 文件夹路径
OUTPUT_FILE="come_diff_images.txt"      # 结果输出文件
# ============================

# 检测系统使用 md5 还是 md5sum
if command -v md5sum >/dev/null 2>&1; then
    MD5_CMD="md5sum"
elif command -v md5 >/dev/null 2>&1; then
    MD5_CMD="md5 -r"   # macOS 的 md5 -r 输出格式与 md5sum 一致（先哈希后文件名）
else
    echo "错误：未找到 md5sum 或 md5 命令" >&2
    exit 1
fi

# 临时文件（使用 mktemp 保证唯一性）
TMP_A=$(mktemp)
TMP_B=$(mktemp)
trap 'rm -f "$TMP_A" "$TMP_B"' EXIT   # 脚本退出时自动删除临时文件

echo "⏳ 正在索引文件夹 A（计算 MD5）..."
# 进入 A 目录，find 所有文件（包括子目录），计算 MD5，提取哈希值并排序去重
cd "$FOLDER_A" || { echo "无法进入目录 $FOLDER_A"; exit 1; }
find . -type f -print0 | while IFS= read -r -d '' file; do
    $MD5_CMD "$file" | awk '{print $1}'
done | sort -u > "$TMP_A"

count_a=$(wc -l < "$TMP_A")
echo "✅ 文件夹 A 共有 $count_a 个唯一 MD5（文件数量）"

echo "⏳ 正在扫描文件夹 B（计算 MD5 并与 A 比对）..."
cd "$FOLDER_B" || { echo "无法进入目录 $FOLDER_B"; exit 1; }

# 清空输出文件
> "$OUTPUT_FILE"

# 遍历 B 所有文件，只输出不在 A 中的文件路径（相对路径）
find . -type f -print0 | while IFS= read -r -d '' file; do
    # 计算当前文件的 MD5
    md5=$($MD5_CMD "$file" | awk '{print $1}')
    # 检查哈希是否在 A 的集合中（-F 固定字符串，-x 整行匹配，-q 静默）
    if ! grep -Fxq "$md5" "$TMP_A"; then
        # 输出相对路径（去除开头的 "./"）
        echo "${file#./}" >> "$OUTPUT_FILE"
    fi
done

# 统计结果数量
result_count=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
echo "✅ 完成！B 中有但 A 中没有的图像共有 $result_count 张"
echo "📄 文件名已保存到：$(realpath "$OUTPUT_FILE")"