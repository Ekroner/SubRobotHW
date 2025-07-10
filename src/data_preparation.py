import os
import xml.etree.ElementTree as ET

# 目标类别
CLASSES = ['holothurian', 'echinus', 'scallop', 'starfish']

def voc_to_yolo(xml_dir, label_out_dir):
    os.makedirs(label_out_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"Processing {len(xml_files)} files in {xml_dir} ...")

    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        yolo_lines = []

        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in CLASSES:
                # 忽略 waterweeds 或其他类别
                continue
            cls_id = CLASSES.index(cls_name)

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 转为YOLO格式中心点宽高归一化
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        # 输出txt文件
        label_file = os.path.join(label_out_dir, xml_file.replace('.xml', '.txt'))
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
    print(f"Saved labels to {label_out_dir}")

def main():
    base_dir = './data'
    sets = ['train', 'testA', 'testB']

    for s in sets:
        xml_folder = os.path.join(base_dir, s, 'box')
        label_folder = os.path.join(base_dir, 'labels', s)
        voc_to_yolo(xml_folder, label_folder)

if __name__ == '__main__':
    main()
