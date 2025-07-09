import os
import torch
import argparse
import time
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import check_dataset, colorstr, increment_path
from utils.torch_utils import select_device
from utils.loss import ComputeLoss
from utils.plots import plot_results
import yaml


# 自定义注意力模块
class UnderwaterAttention(torch.nn.Module):
    def __init__(self, c1, reduction_ratio=8):
        super().__init__()
        self.channel_att = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(c1, c1 // reduction_ratio, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(c1 // reduction_ratio, c1, 1),
            torch.nn.Sigmoid()
        )
        self.spatial_att = torch.nn.Sequential(
            torch.nn.Conv2d(c1, 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_att(x)
        spatial_att = self.spatial_att(x)
        return x * channel_att * spatial_att


class C2f_Underwater(torch.nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = torch.nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = torch.nn.Conv2d((2 + n) * self.c, c2, 1)
        self.m = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Conv2d(self.c, self.c, 3, 1, 1, groups=g),
                torch.nn.BatchNorm2d(self.c),
                torch.nn.SiLU()
            ) for _ in range(n))
        self.attention = UnderwaterAttention(c2)
        self.shortcut = shortcut

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attention(self.cv2(torch.cat(y, 1)))


class SPPF_Underwater(torch.nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = torch.nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = torch.nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = torch.nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.attention = UnderwaterAttention(c2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.attention(self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1)))


def create_custom_model(cfg='models/yolov5s.yaml', nc=4):
    from models.common import Bottleneck
    from models.yolo import Detect
    setattr(torch.nn, 'C2f_Underwater', C2f_Underwater)
    setattr(torch.nn, 'SPPF_Underwater', SPPF_Underwater)
    setattr(torch.nn, 'UnderwaterAttention', UnderwaterAttention)
    with open(cfg) as f:
        model_yaml = yaml.safe_load(f)
    for i, layer in enumerate(model_yaml['backbone']):
        if layer[-1] == 'C2f':
            model_yaml['backbone'][i][-1] = 'C2f_Underwater'
        elif layer[-1] == 'SPPF':
            model_yaml['backbone'][i][-1] = 'SPPF_Underwater'
    model = Model(model_yaml, ch=3, nc=nc)
    return model


def train(hyp, opt, device):
    save_dir = increment_path('runs/train/exp')
    os.makedirs(save_dir, exist_ok=True)
    data_dict = check_dataset(opt.data)
    train_path = data_dict['train']
    val_path = data_dict['val']
    nc = int(data_dict['nc'])
    names = data_dict['names']

    train_loader, dataset = create_dataloader(train_path, opt.imgsz, opt.batch_size, stride=32,
                                              hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect,
                                              workers=opt.workers, prefix=colorstr('train: '))
    val_loader = create_dataloader(val_path, opt.imgsz, opt.batch_size * 2, stride=32,
                                    hyp=hyp, cache=opt.cache, rect=True,
                                    workers=opt.workers, prefix=colorstr('val: '))[0]

    model = create_custom_model(nc=nc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyp['lr0'], weight_decay=hyp['weight_decay'])
    compute_loss = ComputeLoss(model)

    best_fitness = 0.0
    for epoch in range(opt.epochs):
        model.train()
        lr = hyp['lr0'] * (1 - epoch / opt.epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f'Epoch: {epoch}/{opt.epochs} | Batch: {i}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | lr: {lr:.6f}')

        if epoch % 10 == 0 or epoch == opt.epochs - 1:
            results = validate(model, val_loader, compute_loss, device)
            fitness = results[2]
            if fitness > best_fitness:
                best_fitness = fitness
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            plot_results(save_dir=save_dir)

    test_path = data_dict['test']
    test_loader = create_dataloader(test_path, opt.imgsz, opt.batch_size * 2, stride=32,
                                     hyp=hyp, cache=opt.cache, rect=True,
                                     workers=opt.workers, prefix=colorstr('test: '))[0]
    results = validate(model, test_loader, compute_loss, device)
    print(f"\u6d4b\u8bd5\u96c6\u7ed3\u679c: mAP@0.5={results[2]:.4f}, mAP@0.5:0.95={results[3]:.4f}")
    export_model(model, save_dir)
    return model


def validate(model, dataloader, compute_loss, device):
    model.eval()
    results = [0.0] * 4
    with torch.no_grad():
        for i, (imgs, targets, paths, shapes) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets)
            results[0] = (results[0] * i + loss.item()) / (i + 1)
    return results


def export_model(model, save_dir):
    example = torch.rand(1, 3, 640, 640).to(next(model.parameters()).device)
    torch.onnx.export(model, example, os.path.join(save_dir, 'best.onnx'),
                      opset_version=12, input_names=['images'], output_names=['output'],
                      dynamic_axes={'images': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"\u6a21\u578b\u5df2\u5bfc\u51fa\u4e3aONNX\u683c\u5f0f: {os.path.join(save_dir, 'best.onnx')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='yolov5_dataset/underwater.yaml', help='\u6570\u636e\u96c6\u914d\u7f6e\u8def\u5f84')
    parser.add_argument('--epochs', type=int, default=200, help='\u8bad\u7ec3\u8f6e\u6570')
    parser.add_argument('--batch-size', type=int, default=16, help='\u6279\u6b21\u5927\u5c0f')
    parser.add_argument('--imgsz', '--img', type=int, default=640, help='\u56fe\u50cf\u5927\u5c0f')
    parser.add_argument('--device', default='', help='cuda\u8bbe\u5907')
    parser.add_argument('--workers', type=int, default=8, help='\u6570\u636e\u52a0\u8f7d\u7ebf\u7a0b\u6570')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='\u56fe\u50cf\u7f13\u5b58')
    parser.add_argument('--rect', action='store_true', help='\u77e9\u5f62\u8bad\u7ec3')
    opt = parser.parse_args()
    device = select_device(opt.device)
    hyp = {
        'lr0': 0.001, 'momentum': 0.937, 'weight_decay': 0.0005,
        'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
        'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0,
        'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0,
        'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5,
        'mosaic': 1.0, 'mixup': 0.0,
    }
    start_time = time.time()
    model = train(hyp, opt, device)
    training_time = time.time() - start_time
    print(f"\u8bad\u7ec3\u5b8c\u6210! \u603b\u8017\u65f6: {training_time // 3600:.0f}h {(training_time % 3600) // 60:.0f}m {training_time % 60:.0f}s")


if __name__ == '__main__':
    main()
