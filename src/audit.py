"""
第一阶段：数据审计脚本
全面检查数据集质量、分布和评测规则
"""
import os
import sys
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
import json


def audit_environment():
    """审计环境"""
    print("=" * 60)
    print("环境审计")
    print("=" * 60)
    
    # GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,cuda_version', 
                               '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"GPU: {result.stdout.strip()}")
    except:
        print("GPU: 无法检测")
    
    # PyTorch
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Disk
    result = subprocess.run(['df', '-h', '/root'], capture_output=True, text=True)
    print(f"\nDisk:\n{result.stdout}")
    
    # Python packages
    try:
        import segmentation_models_pytorch as smp
        print(f"segmentation_models_pytorch: {smp.__version__}")
    except:
        print("segmentation_models_pytorch: 未安装")
    
    try:
        import albumentations as A
        print(f"albumentations: {A.__version__}")
    except:
        print("albumentations: 未安装")


def audit_dataset_structure(root_dir):
    """审计数据集目录结构"""
    print("\n" + "=" * 60)
    print("数据集目录结构审计")
    print("=" * 60)
    
    dirs_to_check = [
        'train/image/wheat_rape',
        'train/image/rice',
        'train/label/wheat',
        'train/label/rape',
        'train/label/rice',
        'val/image/wheat_rape',
        'val/image/rice',
        'val/label/wheat',
        'val/label/rape',
        'val/label/rice',
        'testA/image/wheat_rape',
        'testA/image/rice',
    ]
    
    for d in dirs_to_check:
        full_path = os.path.join(root_dir, d)
        if os.path.exists(full_path):
            files = [f for f in os.listdir(full_path) if f.endswith('.png')]
            print(f"  {d}: {len(files)} files")
        else:
            print(f"  {d}: NOT FOUND")


def audit_image_properties(root_dir, sample_size=100):
    """审计图像属性"""
    print("\n" + "=" * 60)
    print("图像属性审计")
    print("=" * 60)
    
    image_dirs = [
        ('train/image/wheat_rape', '小麦油菜影像'),
        ('train/image/rice', '水稻影像'),
    ]
    
    for img_dir, desc in image_dirs:
        full_path = os.path.join(root_dir, img_dir)
        if not os.path.exists(full_path):
            continue
        
        files = sorted([f for f in os.listdir(full_path) if f.endswith('.png')])
        print(f"\n{desc} ({img_dir}): {len(files)} files")
        
        # 采样检查
        sample_files = files[:sample_size]
        sizes = []
        modes = []
        corrupted = []
        
        for f in sample_files:
            try:
                img = Image.open(os.path.join(full_path, f))
                sizes.append(img.size)
                modes.append(img.mode)
            except Exception as e:
                corrupted.append((f, str(e)))
        
        if sizes:
            unique_sizes = Counter(sizes)
            print(f"  采样 {len(sample_files)} 张:")
            print(f"    尺寸分布: {dict(unique_sizes)}")
            print(f"    模式分布: {dict(Counter(modes))}")
        
        if corrupted:
            print(f"    损坏文件: {corrupted[:5]}")


def audit_labels(root_dir):
    """审计标签分布"""
    print("\n" + "=" * 60)
    print("标签分布审计")
    print("=" * 60)
    
    label_configs = [
        ('train/label/wheat', 'train/image/wheat_rape', '小麦'),
        ('train/label/rape', 'train/image/wheat_rape', '油菜'),
        ('train/label/rice', 'train/image/rice', '水稻'),
        ('val/label/wheat', 'val/image/wheat_rape', '小麦(验证)'),
        ('val/label/rape', 'val/image/wheat_rape', '油菜(验证)'),
        ('val/label/rice', 'val/image/rice', '水稻(验证)'),
    ]
    
    for label_dir, img_dir, desc in label_configs:
        label_path = os.path.join(root_dir, label_dir)
        img_path = os.path.join(root_dir, img_dir)
        
        if not os.path.exists(label_path):
            print(f"\n{desc}: 目录不存在")
            continue
        
        label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.png')])
        img_files = sorted([f for f in os.listdir(img_path) if f.endswith('.png')])
        
        print(f"\n{desc}:")
        print(f"  标签文件数: {len(label_files)}")
        print(f"  影像文件数: {len(img_files)}")
        
        # 检查文件名一致性
        label_set = set(label_files)
        img_set = set(img_files)
        common = label_set & img_set
        label_only = label_set - img_set
        img_only = img_set - label_set
        
        print(f"  共同文件: {len(common)}")
        if label_only:
            print(f"  仅标签有: {len(label_only)} (前5: {list(label_only)[:5]})")
        if img_only:
            print(f"  仅影像有: {len(img_only)} (前5: {list(img_only)[:5]})")
        
        # 标签像素值统计
        foreground_pixels = []
        total_pixels = []
        unique_values = set()
        empty_count = 0
        
        for f in label_files[:200]:  # 采样检查
            mask = np.array(Image.open(os.path.join(label_path, f)))
            unique_values.update(np.unique(mask).tolist())
            fg = np.sum(mask > 0)
            foreground_pixels.append(fg)
            total_pixels.append(mask.size)
            if fg == 0:
                empty_count += 1
        
        if foreground_pixels:
            fg_ratio = np.sum(foreground_pixels) / np.sum(total_pixels)
            print(f"  前景像素比例: {fg_ratio:.4f} ({fg_ratio*100:.2f}%)")
            print(f"  空标签比例: {empty_count}/{len(foreground_pixels)} ({empty_count/len(foreground_pixels)*100:.1f}%)")
            print(f"  唯一像素值: {sorted(unique_values)}")


def audit_wheat_rape_overlap(root_dir):
    """检查小麦和油菜标签是否重叠"""
    print("\n" + "=" * 60)
    print("小麦-油菜标签重叠检查")
    print("=" * 60)
    
    wheat_dir = os.path.join(root_dir, 'train/label/wheat')
    rape_dir = os.path.join(root_dir, 'train/label/rape')
    
    if not os.path.exists(wheat_dir) or not os.path.exists(rape_dir):
        print("目录不存在，跳过")
        return
    
    wheat_files = sorted([f for f in os.listdir(wheat_dir) if f.endswith('.png')])
    
    overlap_count = 0
    overlap_pixels = []
    
    for f in wheat_files[:200]:
        wheat_mask = np.array(Image.open(os.path.join(wheat_dir, f))) > 0
        rape_mask = np.array(Image.open(os.path.join(rape_dir, f))) > 0
        
        overlap = wheat_mask & rape_mask
        if overlap.any():
            overlap_count += 1
            overlap_pixels.append(overlap.sum())
    
    print(f"  检查文件数: {len(wheat_files[:200])}")
    print(f"  重叠切片数: {overlap_count}")
    if overlap_pixels:
        print(f"  平均重叠像素: {np.mean(overlap_pixels):.1f}")
        print(f"  最大重叠像素: {np.max(overlap_pixels)}")


def audit_connected_components(root_dir):
    """分析连通域大小分布"""
    print("\n" + "=" * 60)
    print("连通域分析")
    print("=" * 60)
    
    from scipy import ndimage
    
    label_configs = [
        ('train/label/wheat', '小麦'),
        ('train/label/rape', '油菜'),
        ('train/label/rice', '水稻'),
    ]
    
    for label_dir, desc in label_configs:
        full_path = os.path.join(root_dir, label_dir)
        if not os.path.exists(full_path):
            continue
        
        files = sorted([f for f in os.listdir(full_path) if f.endswith('.png')])
        
        component_sizes = []
        component_counts = []
        
        for f in files[:100]:
            mask = np.array(Image.open(os.path.join(full_path, f))) > 0
            if mask.any():
                labeled, num_features = ndimage.label(mask)
                component_counts.append(num_features)
                for i in range(1, num_features + 1):
                    size = np.sum(labeled == i)
                    component_sizes.append(size)
        
        if component_sizes:
            print(f"\n{desc}:")
            print(f"  有目标的切片: {len(component_counts)}/{len(files[:100])}")
            print(f"  连通域数量/切片: {np.mean(component_counts):.1f} ± {np.std(component_counts):.1f}")
            print(f"  连通域大小: min={np.min(component_sizes)}, median={np.median(component_sizes):.0f}, "
                  f"mean={np.mean(component_sizes):.0f}, max={np.max(component_sizes)}")
            print(f"  大小<10像素的连通域: {sum(1 for s in component_sizes if s < 10)}")
            print(f"  大小<100像素的连通域: {sum(1 for s in component_sizes if s < 100)}")


def audit_evaluation_script(root_dir):
    """分析评测脚本"""
    print("\n" + "=" * 60)
    print("评测脚本分析")
    print("=" * 60)
    
    # 查找评测脚本
    eval_script = None
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if 'evaluate' in f.lower() and f.endswith('.py'):
                eval_script = os.path.join(root, f)
                break
        if eval_script:
            break
    
    if eval_script:
        print(f"找到评测脚本: {eval_script}")
        with open(eval_script, 'r') as f:
            content = f.read()
        
        # 检查关键逻辑
        if 'IoU' in content or 'iou' in content:
            print("  包含IoU计算")
        if 'TP' in content and 'FP' in content and 'FN' in content:
            print("  包含TP/FP/FN统计")
        if 'mean' in content or 'average' in content:
            print("  包含平均计算")
        
        # 打印脚本内容摘要
        print("\n  脚本内容摘要:")
        lines = content.split('\n')
        for i, line in enumerate(lines[:50]):
            if line.strip():
                print(f"    {i+1}: {line.rstrip()}")
    else:
        print("未找到评测脚本")


def audit_sample_submission(root_dir):
    """分析提交样例"""
    print("\n" + "=" * 60)
    print("提交样例分析")
    print("=" * 60)
    
    sample_dir = os.path.join(root_dir, 'sample_submission_testA')
    if not os.path.exists(sample_dir):
        print("未找到提交样例目录")
        return
    
    print(f"提交样例目录: {sample_dir}")
    
    # 列出结构
    for item in sorted(os.listdir(sample_dir)):
        item_path = os.path.join(sample_dir, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            print(f"  {item}/: {len(files)} files")
            if files:
                print(f"    示例: {files[0]}")
                # 检查示例文件
                sample_file = os.path.join(item_path, files[0])
                try:
                    img = Image.open(sample_file)
                    arr = np.array(img)
                    print(f"    尺寸: {img.size}, 模式: {img.mode}, 唯一值: {np.unique(arr)}")
                except Exception as e:
                    print(f"    读取失败: {e}")


def audit_filenames(root_dir):
    """分析文件名模式"""
    print("\n" + "=" * 60)
    print("文件名模式分析")
    print("=" * 60)
    
    image_dir = os.path.join(root_dir, 'train/image/wheat_rape')
    if not os.path.exists(image_dir):
        return
    
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    print(f"总文件数: {len(files)}")
    print(f"前10个: {files[:10]}")
    print(f"后10个: {files[-10:]}")
    
    # 检查文件名格式
    import re
    patterns = {
        'clip_NNNNN': r'^clip_\d{5}\.png$',
        'tile_X_Y': r'^\d+_\d+\.png$',
        'grid_XxY': r'^\d+x\d+\.png$',
    }
    
    for name, pattern in patterns.items():
        match_count = sum(1 for f in files if re.match(pattern, f))
        print(f"  匹配 {name}: {match_count}/{len(files)}")
    
    # 检查是否能提取行列编号
    # 尝试从文件名提取数字
    numbers = []
    for f in files:
        nums = re.findall(r'\d+', f)
        if nums:
            numbers.append(int(nums[-1]))
    
    if numbers:
        print(f"  文件名数字范围: {min(numbers)} - {max(numbers)}")
        print(f"  唯一数字数: {len(set(numbers))}")
        
        # 检查是否能构成网格
        total = len(files)
        # 83x83 = 6889, 41x166 = 6806
        if total == 6889:
            print(f"  可能对应 83x83 网格")
        elif total == 6806:
            print(f"  可能对应 41x166 网格")


def run_full_audit(root_dir):
    """运行完整审计"""
    audit_environment()
    audit_dataset_structure(root_dir)
    audit_image_properties(root_dir)
    audit_labels(root_dir)
    audit_wheat_rape_overlap(root_dir)
    audit_connected_components(root_dir)
    audit_evaluation_script(root_dir)
    audit_sample_submission(root_dir)
    audit_filenames(root_dir)
    
    print("\n" + "=" * 60)
    print("审计完成")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='/root/competition_data/public')
    args = parser.parse_args()
    
    run_full_audit(args.data_dir)
