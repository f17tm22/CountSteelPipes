from ultralytics import YOLO
import os
import shutil
from pathlib import Path

# 设置工作目录
os.chdir('/Users/merinomo/Documents/Code/python/CountSteelPipe')

# 数据集路径列表（图像文件夹）
dataset_paths = [
    'dataset/aidataset',
    'dataset/steel_pipe_pictures/train/images',
    'dataset/archive/piple0731b',
    'dataset/archive/V3'
]

# 合并数据集的函数
def merge_datasets(dataset_paths, output_dir='dataset/combined'):
    images_dir = Path(output_dir) / 'images'
    labels_dir = Path(output_dir) / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    image_count = 0
    for dataset_path in dataset_paths:
        print(f"处理数据集: {dataset_path}")
        dataset_path = Path(dataset_path)
        
        # 特殊处理 steel_pipe_pictures
        if 'steel_pipe_pictures' in str(dataset_path):
            src_images = dataset_path
            src_labels = dataset_path.parent / 'labels'  # ../labels
        elif (dataset_path / 'images').exists():
            src_images = dataset_path / 'images'
            src_labels = dataset_path / 'labels' if (dataset_path / 'labels').exists() else None
        else:
            src_images = dataset_path
            src_labels = dataset_path  # 假设标签和图像在同一文件夹
        
        print(f"图像源: {src_images}, 存在: {src_images.exists()}")
        print(f"标签源: {src_labels}, 存在: {src_labels.exists() if src_labels else False}")
        
        # 复制图像
        if src_images.exists():
            jpg_count = 0
            png_count = 0
            txt_count = 0
            for img_file in src_images.rglob('*.jpg'):
                new_name = f'{image_count:06d}.jpg'
                shutil.copy(img_file, images_dir / new_name)
                # 复制对应标签
                if src_labels:
                    label_file = src_labels / (img_file.stem + '.txt')
                    if label_file.exists():
                        shutil.copy(label_file, labels_dir / f'{image_count:06d}.txt')
                        txt_count += 1
                    else:
                        # 检查相对路径
                        rel_path = img_file.relative_to(src_images)
                        label_rel = rel_path.with_suffix('.txt')
                        label_file = src_labels / label_rel
                        if label_file.exists():
                            shutil.copy(label_file, labels_dir / f'{image_count:06d}.txt')
                            txt_count += 1
                image_count += 1
                jpg_count += 1
            for img_file in src_images.rglob('*.png'):
                new_name = f'{image_count:06d}.png'
                shutil.copy(img_file, images_dir / new_name)
                # 复制对应标签
                if src_labels:
                    label_file = src_labels / (img_file.stem + '.txt')
                    if label_file.exists():
                        shutil.copy(label_file, labels_dir / f'{image_count:06d}.txt')
                        txt_count += 1
                    else:
                        # 检查相对路径
                        rel_path = img_file.relative_to(src_images)
                        label_rel = rel_path.with_suffix('.txt')
                        label_file = src_labels / label_rel
                        if label_file.exists():
                            shutil.copy(label_file, labels_dir / f'{image_count:06d}.txt')
                            txt_count += 1
                image_count += 1
                png_count += 1
            print(f"复制了 {jpg_count} 个 JPG 和 {png_count} 个 PNG 图像，{txt_count} 个标签")
        
        # 如果标签源是同一文件夹，复制剩余的 txt 文件（如果有未匹配的）
        if src_labels and src_labels == src_images:
            txt_count = 0
            for label_file in src_labels.glob('*.txt'):
                if not (labels_dir / label_file.name).exists():
                    shutil.copy(label_file, labels_dir / label_file.name)
                    txt_count += 1
            if txt_count > 0:
                print(f"额外复制了 {txt_count} 个标签文件")
    
    print(f"总共复制了 {image_count} 个图像")
    return output_dir

# 合并数据集
combined_dataset = merge_datasets(dataset_paths)

# 创建 data.yaml
data_yaml_content = f"""
train: {os.path.abspath(combined_dataset)}/images
val: {os.path.abspath(combined_dataset)}/images  # 注意：需要分离验证集
test: {os.path.abspath(combined_dataset)}/images

nc: 1
names: ['steel_pipe']
"""

data_yaml_path = 'dataset/combined_data.yaml'
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

# 加载预训练的YOLOv10模型
model = YOLO('yolov10b.pt')

# 训练参数
# 注意：训练前确保数据集准备好，包括验证集分离
import time
start_time = time.time()

print("开始训练YOLO模型...")
model.train(
    data=data_yaml_path,
    epochs=100,  # 训练轮数，可调整
    imgsz=640,   # 图像大小
    batch=16,    # 批次大小，根据GPU内存调整
    name='steel_pipe_yolov10_multi',  # 实验名称
    project='runs/train',  # 保存路径
    save=True,   # 保存模型
    save_period=10,  # 每10轮保存一次
    cache=False,  # 是否缓存数据集
    device='cpu'  # 使用CPU，如果有GPU改为0或'cuda'
)

end_time = time.time()
total_time = end_time - start_time
print(f"训练完成！总耗时: {total_time:.2f} 秒 ({total_time/3600:.2f} 小时)")