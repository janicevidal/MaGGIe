import os
import shutil
from pathlib import Path

def build_file_map(source_dir):
    """扫描源文件夹，建立 {文件名: 文件路径} 的映射（如果有同名文件，保留第一个，打印警告）"""
    src_path = Path(source_dir)
    if not src_path.exists():
        raise ValueError(f"源文件夹不存在：{source_dir}")
    
    file_map = {}
    duplicate_warns = 0
    for entry in src_path.rglob('*'):
        if entry.is_file():
            name = entry.name
            if name in file_map:
                # 同名文件，保留第一个，打印警告
                duplicate_warns += 1
                if duplicate_warns <= 5:  # 最多打印前5个警告
                    print(f"⚠️ 发现同名文件：{name}，已存在于 {file_map[name]}，将忽略 {entry}")
                elif duplicate_warns == 6:
                    print("⚠️ 更多同名文件警告将被省略...")
            else:
                file_map[name] = entry
    print(f"📋 源文件夹扫描完成：共 {len(file_map)} 个唯一文件名（忽略 {duplicate_warns} 个重复）")
    return file_map

def build_exclude_set(exclude_dir):
    """扫描排除文件夹，构建文件名集合（仅文件名）"""
    if not exclude_dir:
        return set()
    exclude_path = Path(exclude_dir)
    if not exclude_path.exists():
        print(f"⚠️ 排除文件夹不存在：{exclude_dir}，将不排除任何文件")
        return set()
    names = set()
    for entry in exclude_path.rglob('*'):
        if entry.is_file():
            names.add(entry.name)
    print(f"📋 排除文件夹中共有 {len(names)} 个文件名（含子目录）")
    return names

def copy_images_from_name_list(name_list_file, source_dir, dst_dir, exclude_dir=None):
    """
    根据文件名列表，从源文件夹中查找并拷贝到目标文件夹
    :param name_list_file: 每行一个文件名的文本文件
    :param source_dir: 源文件夹（即文件实际所在的 B 文件夹）
    :param dst_dir: 目标文件夹
    :param exclude_dir: 排除文件夹（同名文件跳过）
    """
    # 创建目标文件夹
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # 扫描源文件夹，建立文件名到路径的映射
    try:
        file_map = build_file_map(source_dir)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # 构建排除文件名集合
    exclude_names = build_exclude_set(exclude_dir)

    # 读取文件名列表
    try:
        with open(name_list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ 错误：文件 {name_list_file} 不存在")
        return

    names = [line.strip() for line in lines if line.strip()]
    total = len(names)
    if total == 0:
        print("⚠️ 列表文件为空，没有可拷贝的图像")
        return

    print(f"📋 共发现 {total} 个文件名需要拷贝")

    success = 0
    skipped_exclude = 0
    skipped_not_found = 0
    conflicts = 0

    for idx, name in enumerate(names, 1):
        # 检查是否在排除列表中
        if name in exclude_names:
            print(f"⏭️  [{idx}/{total}] 跳过（在排除文件夹中存在同名）: {name}")
            skipped_exclude += 1
            continue

        # 从映射中查找源文件
        src_file = file_map.get(name)
        if src_file is None:
            print(f"⚠️ [{idx}/{total}] 跳过（在源文件夹中未找到）: {name}")
            skipped_not_found += 1
            continue

        # 处理目标文件名冲突
        dst_file = dst_path / src_file.name
        if dst_file.exists():
            stem = src_file.stem
            suffix = src_file.suffix
            counter = 1
            while True:
                new_name = f"{stem}_{counter}{suffix}"
                candidate = dst_path / new_name
                if not candidate.exists():
                    dst_file = candidate
                    break
                counter += 1
            conflicts += 1

        # 执行拷贝
        try:
            shutil.copy2(src_file, dst_file)
            print(f"✅ [{idx}/{total}] 已拷贝: {name} -> {dst_file.name}")
            success += 1
        except Exception as e:
            print(f"❌ [{idx}/{total}] 拷贝失败: {name} - {e}")
            skipped_not_found += 1

    # 输出汇总
    print("\n" + "="*50)
    print(f"📊 汇总：成功 {success} 个，因排除跳过 {skipped_exclude} 个，未找到 {skipped_not_found} 个")
    if conflicts:
        print(f"📌 处理文件名冲突 {conflicts} 次（自动添加后缀）")
    print(f"📁 目标文件夹：{dst_path.absolute()}")

if __name__ == "__main__":
    NAME_LIST = "/data/xiaoshuai/human_matting/dataset/OpenDataLab___COME15K/raw/come_diff_images.txt"        # 只有文件名的列表（每行一个）
    SOURCE_DIR = "/data/xiaoshuai/human_matting/dataset/OpenDataLab___COME15K/raw/all_images/"            # 文件实际所在的源文件夹（即B文件夹）
    DEST_DIR = "/data/xiaoshuai/human_matting/dataset/OpenDataLab___COME15K/raw/non_person_images/"       # 目标文件夹
    EXCLUDE_DIR = "/home/zhangxiaoshuai/Data/sod/"       # 排除文件夹（留空则不排除）

    copy_images_from_name_list(NAME_LIST, SOURCE_DIR, DEST_DIR, EXCLUDE_DIR)