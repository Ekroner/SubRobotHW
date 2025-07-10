import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbam import CBAM  # ✅ 导入自定义CBAM模块

# -------------------------------------------------------------
# 常用模块
# -------------------------------------------------------------

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    @property
    def out_channels(self):
        return self.conv.out_channels


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x, self.m(x), self.m(self.m(x)), self.m(self.m(self.m(x)))], 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Detect(nn.Module):
    # YOLOv5 Detect头（简略）
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.stride = torch.tensor([8, 16, 32])
        self.m = nn.ModuleList([nn.Conv2d(x, self.no * self.nl, 1) for x in ch])

    def forward(self, x):
        return x

# 自动填充padding函数
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# -------------------------------------------------------------
# ✅ 关键：模型构造函数，解析模型结构
# -------------------------------------------------------------

def parse_model(d, ch):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    layers, save, c2 = [], [], ch[-1]

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval string modules
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = eval(a)
                except:
                    pass

        n = max(round(n * gd), 1) if n > 1 else n

        if m in [Conv, C3, SPPF, CBAM]:
            c1, c2 = ch[f], args[0]
            args = [c1] + args
            m_ = m(*args) if n == 1 else nn.Sequential(*[m(*args) for _ in range(n)])
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
            m_ = m()
        elif m is Detect:
            args.append([ch[x] for x in f])
            m_ = m(*args)
        elif m is nn.Upsample:
            m_ = m(scale_factor=2, mode='nearest')
        else:
            raise NotImplementedError(f"Unknown module: {m}")

        layers.append(m_)
        if isinstance(f, int):
            ch.append(c2)
        else:
            ch.append(c2)

    return nn.Sequential(*layers), sorted(save)
