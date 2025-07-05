import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from ultralytics.nn.tasks import DetectionModel
import argparse
import time


# 自定义注意力模块
class UnderwaterAttention(nn.Module):
    """水下专用注意力机制"""

    def __init__(self, c1, reduction_ratio=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(c1 // reduction_ratio, c1, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_att(x)
        spatial_att = self.spatial_att(x)
        return x * channel_att * spatial_att


# 集成注意力的C2f模块
class C2f_Underwater(C2f):
    """集成水下注意力机制的C2f模块"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attention = UnderwaterAttention(c2)

    def forward(self, x):
        return self.attention(super().forward(x))


# 集成注意力的SPPF模块
class SPPF_Underwater(SPPF):
    """集成水下注意力机制的SPPF模块"""

    def __init__(self, c1, c2, k=5):
        super().__init__(c1, c2, k)
        self.attention = UnderwaterAttention(c2)

    def forward(self, x):
        return self.attention(super().forward(x))


# 模型配置文件
def create_model_yaml():
    """创建自定义模型配置文件"""
    yaml_content = """
# YOLOv8 水下专用模型
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f_Underwater, [128]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f_Underwater, [256]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f_Underwater, [512]]  # 6
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f_Underwater, [1024]]  # 8
  - [-1, 1, SPPF_Underwater, [1024, 5]]  # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f_Underwater, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f_Underwater, [256]]  # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f_Underwater, [512]]  # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f_Underwater, [1024]]  # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
"""
    model_yaml_path = 'underwater-yolov8.yaml'
    with open(model_yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    return model_yaml_path


class NoiseRobustTrainer:
    def __init__(self, model, data_config, noise_ratio=0.3):
        self.model = model
        self.data_config = data_config
        self.noise_ratio = noise_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = 'runs/detect'

    def train(self, epochs=200, batch_size=16):
        """噪声鲁棒训练循环"""
        # 注册自定义模块
        DetectionModel.modules['C2f_Underwater'] = C2f_Underwater
        DetectionModel.modules['SPPF_Underwater'] = SPPF_Underwater
        DetectionModel.modules['UnderwaterAttention'] = UnderwaterAttention

        # 初始训练阶段（使用所有数据）
        print("阶段1: 初始训练 (10% 轮次)")
        self.model.train(
            data=self.data_config,
            epochs=int(epochs * 0.1),
            imgsz=640,
            batch=batch_size,
            name='underwater_initial',
            project=self.output_dir,
            optimizer='AdamW',
            lr0=0.001,
            cos_lr=True,
            hsv_h=0.015,  # 色调增强
            hsv_s=0.7,  # 饱和度增强
            hsv_v=0.4,  # 亮度增强
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0
        )

        # 主训练阶段
        print("阶段2: 主训练 (90% 轮次)")
        for epoch in range(int(epochs * 0.1), epochs):
            # 每10个epoch保存一次
            save = (epoch % 10 == 0) or (epoch == epochs - 1)

            # 训练一个epoch
            self.model.train(
                data=self.data_config,
                epochs=1,
                imgsz=640,
                batch=batch_size,
                resume=True,
                name=f'underwater_epoch_{epoch}',
                project=self.output_dir,
                optimizer='AdamW',
                lr0=0.001 * (1 - epoch / epochs),  # 学习率衰减
                cos_lr=False,
                save=save,
                hsv_h=0.01,  # 减少增强强度
                hsv_s=0.6,
                hsv_v=0.3
            )

            # 每10个epoch验证一次
            if epoch % 10 == 0:
                metrics = self.model.val()
                print(f"Epoch {epoch}: mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")

                # 在测试集上评估
                self.evaluate_on_test_sets()

    def evaluate_on_test_sets(self):
        """在测试集A和B上评估模型"""
        # 测试集A评估
        print("在测试集A上评估模型...")
        metrics_a = self.model.val(data=os.path.join(os.path.dirname(self.data_config), 'testA.yaml'))
        print(f"测试集A: mAP50={metrics_a.box.map50:.4f}, mAP50-95={metrics_a.box.map:.4f}")

        # 测试集B评估
        print("在测试集B上评估模型...")
        metrics_b = self.model.val(data=os.path.join(os.path.dirname(self.data_config), 'testB.yaml'))
        print(f"测试集B: mAP50={metrics_b.box.map50:.4f}, mAP50-95={metrics_b.box.map:.4f}")

        return metrics_a, metrics_b


def main():
    # 创建模型配置文件
    model_yaml_path = create_model_yaml()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='水下目标检测模型训练')
    parser.add_argument('--data', type=str, default='processed_dataset/underwater.yaml', help='数据集配置')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--noise-ratio', type=float, default=0.3, help='噪声数据比例')
    args = parser.parse_args()

    # 检查数据集配置是否存在
    if not os.path.exists(args.data):
        print(f"错误: 数据集配置文件 {args.data} 不存在!")
        print("请先运行 data_preparation.py 进行数据预处理")
        return

    # 加载模型
    print(f"创建自定义模型: {model_yaml_path}")
    model = YOLO(model_yaml_path)

    # 创建噪声鲁棒训练器
    trainer = NoiseRobustTrainer(
        model=model,
        data_config=args.data,
        noise_ratio=args.noise_ratio
    )

    # 开始训练
    start_time = time.time()
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)

    # 最终评估
    print("\n最终模型评估:")
    metrics_a, metrics_b = trainer.evaluate_on_test_sets()

    # 导出模型
    model.export(format='onnx', simplify=True, dynamic=True)

    training_time = time.time() - start_time
    print(
        f"\n模型训练完成! 总耗时: {training_time // 3600:.0f}h {(training_time % 3600) // 60:.0f}m {training_time % 60:.0f}s")
    print(f"模型已导出为ONNX格式: {os.path.join(trainer.output_dir, 'train', 'weights', 'best.onnx')}")


if __name__ == "__main__":
    main()