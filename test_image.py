from ultralytics import YOLO
import os

# 设置工作目录
os.chdir('/Users/merinomo/Documents/Code/python/CountSteelPipe')

# 加载训练好的YOLO模型（使用最新的10轮训练模型）
model = YOLO('runs/train/steel_pipe_yolov10_quick3/weights/best.pt')

# 测试图片路径
test_image = 'image.png'

# 进行推理
results = model.predict(
    source=test_image,
    save=True,  # 保存结果图片
    save_txt=True,  # 保存检测结果到txt文件
    save_conf=True,  # 保存置信度
    conf=0.30,  # 置信度阈值
    iou=0.7,  # IOU阈值
    show=False,  # 不显示窗口
    project='runs/detect',  # 保存路径
    name='test_image',  # 实验名称
    exist_ok=True  # 允许覆盖
)

# 打印检测结果
print("检测结果：")
for result in results:
    print(f"图像: {result.path}")
    print(f"检测到的对象数量: {len(result.boxes)}")
    if len(result.boxes) > 0:
        print("检测详情:")
        for i, box in enumerate(result.boxes):
            cls = int(box.cls.item())
            conf = box.conf.item()
            xyxy = box.xyxy.tolist()[0]
            print(f"  对象 {i+1}: 类 {cls} (steel_pipe), 置信度 {conf:.2f}, 坐标 {xyxy}")
    else:
        print("未检测到钢管")

print(f"\n结果已保存到 runs/detect/test_image/ 文件夹")