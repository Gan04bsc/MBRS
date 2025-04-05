import os
import random
from PIL import Image

def select_and_downsample(source_dir, target_dir, num_samples=1000, seed=None):
    """
    source_dir : str - 包含train/test子目录的根目录
    target_dir : str - 输出根目录
    num_samples : int - 每类选取样本数
    seed : int - 随机种子
    """
    random.seed(seed)
    img_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    for split in ['train', 'test']:  # 修改为 train 和 test
        split_src = os.path.join(source_dir, split)
        split_dst = os.path.join(target_dir, split)

        for class_name in os.listdir(split_src):
            class_src = os.path.join(split_src, class_name)
            class_dst = os.path.join(split_dst, class_name)
            
            if not os.path.isdir(class_src):
                continue

            all_images = [f for f in os.listdir(class_src) 
                         if f.lower().endswith(img_exts)]
            if not all_images:
                print(f"跳过空目录：{class_src}")
                continue

            selected = random.sample(all_images, min(num_samples, len(all_images)))
            os.makedirs(class_dst, exist_ok=True)

            for fname in selected:
                src_path = os.path.join(class_src, fname)
                base_name = os.path.splitext(fname)[0]
                dst_path = os.path.join(class_dst, f"{base_name}.png")

                try:
                    with Image.open(src_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        img = img.resize((128, 128), Image.Resampling.LANCZOS)
                        
                        img.save(dst_path, 
                                format='PNG',
                                optimize=True,
                                compress_level=0)
                        
                except Exception as e:
                    print(f"处理失败：{src_path} | 错误：{str(e)}")

            print(f"处理完成：{class_src} -> {len(selected)}个PNG样本")

if __name__ == "__main__":
    select_and_downsample(
        source_dir=r"D:\python\隐写鲁棒\data\experiment\data\gtos-mobile",  # 修改为根目录
        target_dir=r"D:\python\隐写鲁棒\data\experiment\data\gtos128_all",  # 修改为目标目录
        num_samples=1000,  # 每类选取样本数
        seed=42  # 随机种子
    )