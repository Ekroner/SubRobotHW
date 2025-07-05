import os
import cv2
import xml.etree.ElementTree as ET
import shutil
import yaml
import numpy as np

# 基础路径
BASE_DIR = 'src/data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TESTA_DIR = os.path.join(BASE_DIR, 'testA')
TESTB_DIR = os.path.join(BASE_DIR, 'testB')

# 输出目录
OUTPUT_DIR = 'processed_dataset'

# 类别映射
CLASS_MAP = {
    'holothurian': 0,
    'echinus': 1,
    'scallop': 2,
    'starfish': 3
}

# 有效类别
VALID_CLASSES = set(CLASS_MAP.keys())


def parse_xml_annotation(xml_path, img_width, img_height):
    """解析XML标注文件，过滤无效类别，并返回YOLO格式的标注"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        yolo_annotations = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text

            # 过滤水草类别
            if class_name not in VALID_CLASSES:
                continue

            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))

            # 确保边界框在图像范围内
            xmin = max(0, min(xmin, img_width - 1))
            ymin = max(0, min(ymin, img_height - 1))
            xmax = max(0, min(xmax, img_width - 1))
            ymax = max(0, min(ymax, img_height - 1))

            # 跳过无效边界框
            if xmax <= xmin or ymax <= ymin:
                continue

            # 转换为YOLO格式
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 确保坐标在[0,1]范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            class_id = CLASS_MAP[class_name]
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return yolo_annotations
    except Exception as e:
        print(f"解析XML错误 {xml_path}: {e}")
        return []


def process_dataset(dataset_type, input_image_dir, input_box_dir, output_image_dir, output_label_dir):
    """处理单个数据集（训练集、测试集A或测试集B）"""
    print(f"处理{dataset_type}数据集...")
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith('.jpg')]
    processed_count = 0
    skipped_count = 0

    for img_file in image_files:
        # 图像路径
        img_path = os.path.join(input_image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_file}，已跳过")
            skipped_count += 1
            continue
        img_height, img_width = img.shape[:2]

        # 对应的XML文件
        xml_file = img_file.replace('.jpg', '.xml')
        xml_path = os.path.join(input_box_dir, xml_file)

        if not os.path.exists(xml_path):
            print(f"警告: {img_file} 缺少XML标注文件，已跳过")
            skipped_count += 1
            continue

        # 解析XML并获取YOLO格式标注
        yolo_annotations = parse_xml_annotation(xml_path, img_width, img_height)

        # 保存图像
        shutil.copy(img_path, os.path.join(output_image_dir, img_file))

        # 保存标注
        label_file = img_file.replace('.jpg', '.txt')
        with open(os.path.join(output_label_dir, label_file), 'w') as f:
            f.write("\n".join(yolo_annotations))

        processed_count += 1

    print(f"  -> 已处理 {processed_count} 张图像，跳过 {skipped_count} 张图像")
    return processed_count


def create_dataset_yaml():
    """创建YOLO数据集配置文件"""
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'testA/images',  # 使用testA作为验证集
        'test': 'testB/images',  # 使用testB作为测试集
        'nc': len(CLASS_MAP),
        'names': list(CLASS_MAP.keys())
    }

    with open(os.path.join(OUTPUT_DIR, 'underwater.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    # 创建单独的测试集配置文件
    for test_set in ['testA', 'testB']:
        test_yaml = {
            'path': os.path.abspath(os.path.join(OUTPUT_DIR, test_set)),
            'images': 'images',
            'labels': 'labels',
            'nc': len(CLASS_MAP),
            'names': list(CLASS_MAP.keys())
        }
        with open(os.path.join(OUTPUT_DIR, f'{test_set}.yaml'), 'w') as f:
            yaml.dump(test_yaml, f, sort_keys=False)


def main():
    # 创建输出目录结构
    os.makedirs(os.path.join(OUTPUT_DIR, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'testA', 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'testA', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'testB', 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'testB', 'labels'), exist_ok=True)

    # 处理训练集
    train_count = process_dataset(
        dataset_type='训练集',
        input_image_dir=os.path.join(TRAIN_DIR, 'image'),
        input_box_dir=os.path.join(TRAIN_DIR, 'box'),
        output_image_dir=os.path.join(OUTPUT_DIR, 'train', 'images'),
        output_label_dir=os.path.join(OUTPUT_DIR, 'train', 'labels')
    )

    # 处理测试集A
    testa_count = process_dataset(
        dataset_type='测试集A',
        input_image_dir=os.path.join(TESTA_DIR, 'image'),
        input_box_dir=os.path.join(TESTA_DIR, 'box'),
        output_image_dir=os.path.join(OUTPUT_DIR, 'testA', 'images'),
        output_label_dir=os.path.join(OUTPUT_DIR, 'testA', 'labels')
    )

    # 处理测试集B
    testb_count = process_dataset(
        dataset_type='测试集B',
        input_image_dir=os.path.join(TESTB_DIR, 'image'),
        input_box_dir=os.path.join(TESTB_DIR, 'box'),
        output_image_dir=os.path.join(OUTPUT_DIR, 'testB', 'images'),
        output_label_dir=os.path.join(OUTPUT_DIR, 'testB', 'labels')
    )

    # 创建数据集配置文件
    create_dataset_yaml()

    print("\n数据预处理完成!")
    print(f"训练集: {train_count} 张图像")
    print(f"测试集A: {testa_count} 张图像")
    print(f"测试集B: {testb_count} 张图像")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"数据集配置文件: {os.path.join(OUTPUT_DIR, 'underwater.yaml')}")


if __name__ == "__main__":
    main()