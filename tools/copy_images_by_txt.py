import os
import shutil
import argparse
import sys
from collections import defaultdict

def build_file_map(src_dir):
    """
    递归遍历源目录，建立 文件名 -> 文件完整路径 的映射。
    如果存在同名文件，则保留第一个遇到的路径，并记录冲突信息。
    
    Returns:
        dict: {文件名: 路径}
        list: 冲突文件名列表
    """
    file_map = {}
    conflicts = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file in file_map:
                # 发现同名文件，记录冲突（保留第一个）
                conflicts.append(file)
            else:
                file_map[file] = os.path.join(root, file)
    return file_map, conflicts

def copy_images_from_txt(txt_path, src_dir, dst_dir, verbose=True, skip_conflicts=False):
    """
    根据 txt 文件中的图像名称，从源目录（含子文件夹）复制对应图像到目标目录。
    
    Args:
        txt_path (str): 包含图像名称列表的文本文件路径，每行一个名称。
        src_dir (str): 源图像所在文件夹路径（会递归搜索子文件夹）。
        dst_dir (str): 目标文件夹路径（会自动创建）。
        verbose (bool): 是否打印详细信息。
        skip_conflicts (bool): 若为True，遇到同名冲突时直接跳过该文件；否则复制第一个并警告。
    """
    # 检查源目录
    if not os.path.isdir(src_dir):
        print(f"错误：源目录 '{src_dir}' 不存在。", file=sys.stderr)
        sys.exit(1)

    # 创建目标目录
    os.makedirs(dst_dir, exist_ok=True)

    # 读取 txt 文件
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：文本文件 '{txt_path}' 未找到。", file=sys.stderr)
        sys.exit(1)

    # 构建文件映射（递归搜索）
    if verbose:
        print("正在扫描源目录及子文件夹...")
    file_map, conflicts = build_file_map(src_dir)
    if verbose and conflicts:
        print(f"警告：发现 {len(conflicts)} 个同名文件（仅使用第一个找到的）")
        if verbose > 1:  # 若详细级别高则列出具体冲突名
            for name in conflicts[:10]:  # 只显示前10个
                print(f"  冲突文件: {name}")
            if len(conflicts) > 10:
                print(f"  ... 还有 {len(conflicts)-10} 个")

    # 统计
    total = 0
    copied = 0
    skipped = 0
    not_found = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        total += 1

        # 直接使用文件名在映射中查找
        if line not in file_map:
            if verbose:
                print(f"警告：文件 '{line}' 在源目录中未找到，已跳过")
            not_found += 1
            continue

        src_file = file_map[line]
        dst_file = os.path.join(dst_dir, line)

        # 如果目标文件已存在，可选择覆盖（默认覆盖，无需额外操作）
        try:
            shutil.copy2(src_file, dst_file)
            copied += 1
            if verbose:
                print(f"已复制: {line}")
        except Exception as e:
            print(f"错误：复制 {line} 失败 - {e}", file=sys.stderr)
            skipped += 1

    # 输出汇总
    print(f"\n完成！总计 {total} 个条目，成功复制 {copied} 个，"
          f"未找到 {not_found} 个，其他错误跳过 {skipped} 个。")

def main():
    parser = argparse.ArgumentParser(
        description="根据文本文件中的图像名称列表，从源文件夹（含子文件夹）复制图像到目标文件夹。"
    )
    parser.add_argument(
        'txt_file',
        help='包含图像名称的文本文件路径（每行一个文件名）'
    )
    parser.add_argument(
        'src_dir',
        help='源图像文件夹路径（会递归搜索子文件夹）'
    )
    parser.add_argument(
        'dst_dir',
        help='目标图像文件夹路径（会自动创建）'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，不显示详细信息'
    )
    parser.add_argument(
        '--skip-conflicts',
        action='store_true',
        help='遇到同名文件时直接跳过（默认会复制第一个并给出警告）'
    )
    args = parser.parse_args()

    copy_images_from_txt(
        args.txt_file,
        args.src_dir,
        args.dst_dir,
        verbose=not args.quiet,
        skip_conflicts=args.skip_conflicts
    )

if __name__ == '__main__':
    main()