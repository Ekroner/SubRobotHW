import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import os

# 水下目标类别
CLASSES = ['holothurian', 'echinus', 'scallop', 'starfish']

# 类别颜色映射
COLOR_MAP = {
    'holothurian': (0, 255, 0),  # 绿色 - 海参
    'echinus': (0, 165, 255),  # 橙色 - 海胆
    'scallop': (255, 0, 0),  # 蓝色 - 扇贝
    'starfish': (0, 255, 255)  # 黄色 - 海星
}


class UnderwaterVideoProcessor:
    def __init__(self, model_path, classes, device='cuda'):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.classes = classes
        self.device = device
        self.frame_count = 0
        self.detection_count = 0
        self.frame_times = []
        self.fps = 0
        self.last_log_time = time.time()

    def underwater_color_correction(self, frame):
        """水下颜色校正"""
        # 分离通道
        b, g, r = cv2.split(frame)

        # 增强红色通道（补偿水下红色衰减）
        r = cv2.add(r, 40)

        # 降低蓝色通道（减少水雾影响）
        b = cv2.add(b, -20)

        # 合并通道
        corrected = cv2.merge([b, g, r])

        # CLAHE对比度增强
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def enhance_blurry_objects(self, frame):
        """增强模糊目标可见性"""
        # 使用锐化滤波器
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        return sharpened

    def process_frame(self, frame):
        """处理单个视频帧"""
        start_time = time.time()

        # 预处理
        color_corrected = self.underwater_color_correction(frame)
        enhanced_frame = self.enhance_blurry_objects(color_corrected)

        # 目标检测
        results = self.model.predict(
            enhanced_frame,
            conf=0.4,
            imgsz=640,
            augment=False,
            verbose=False
        )

        # 可视化结果
        frame_detections = 0
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                if cls_id < len(self.classes):
                    class_name = self.classes[cls_id]
                    color = COLOR_MAP.get(class_name, (0, 255, 255))

                    # 绘制边界框
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    frame_detections += 1

        # 更新检测计数
        self.detection_count += frame_detections

        # 计算处理时间
        process_time = time.time() - start_time
        self.frame_times.append(process_time)

        # 计算平均FPS（使用最近10帧）
        if len(self.frame_times) > 10:
            self.frame_times.pop(0)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0

        # 添加帧信息
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Detections: {frame_detections}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Underwater Detection", (frame.shape[1] - 350, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        self.frame_count += 1
        return frame

    def process_video(self, input_path, output_path):
        """处理整个视频"""
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件: {input_path}")
            return False

        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"错误: 无法创建输出视频文件: {output_path}")
            cap.release()
            return False

        print(f"开始处理视频: {input_path}")
        print(f"视频信息: {width}x{height} @ {fps:.2f} FPS, 共 {total_frames} 帧")
        print(f"输出路径: {output_path}")

        # 重置计数器
        self.frame_count = 0
        self.detection_count = 0
        self.frame_times = []
        self.last_log_time = time.time()
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 处理帧
            processed_frame = self.process_frame(frame)

            # 写入输出视频
            out.write(processed_frame)

            # 显示进度
            current_time = time.time()
            if current_time - self.last_log_time > 5:  # 每5秒打印一次进度
                progress = self.frame_count / total_frames * 100
                print(f"进度: {self.frame_count}/{total_frames} 帧 ({progress:.1f}%), "
                      f"FPS: {self.fps:.1f}, 检测数: {self.detection_count}")
                self.last_log_time = current_time

        # 释放资源
        cap.release()
        out.release()

        # 计算总处理时间
        total_time = time.time() - start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0

        print(f"\n视频处理完成!")
        print(f"总帧数: {self.frame_count}")
        print(f"总检测数: {self.detection_count}")
        print(f"平均FPS: {avg_fps:.1f}")
        print(f"总耗时: {total_time // 60:.0f}m {total_time % 60:.2f}s")
        print(f"输出视频: {output_path}")

        return True


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='水下目标检测视频处理')
    parser.add_argument('--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, required=True, help='输出视频路径')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.onnx',
                        help='模型路径 (默认: runs/detect/train/weights/best.onnx)')

    args = parser.parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print("请先运行 train.py 训练模型")
        exit(1)

    # 创建处理器
    processor = UnderwaterVideoProcessor(
        model_path=args.model,
        classes=CLASSES
    )

    # 处理视频
    success = processor.process_video(args.input, args.output)

    if not success:
        print("视频处理失败!")
        exit(1)