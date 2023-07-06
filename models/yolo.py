# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""
'''======================1.导入安装好的python库====================='''
import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

'''===================3..加载自定义模块============================'''
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
'''
models.common：  这个是yolov5的网络结构
models.experimental：  实验性质的代码，包括MixConv2d、跨层权重Sum等
utils.autoanchor：  定义了自动生成锚框的方法
utils.general：   定义了一些常用的工具函数，比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等
utils.plots：    定义了Annotator类，可以在图像上绘制矩形框和标注信息
utils.torch_utils：   定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等'''

# 导入thop包 用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

'''#用这三个1'''
class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    '''===================1.获取预测得到的参数============================'''
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        # nc: 数据集类别数量
        self.nc = nc  # number of classes
        # no: 表示每个anchor的输出数，前nc个01字符对应类别，后5个对应：是否有目标，目标框的中心，目标框的宽高
        self.no = nc + 5  # number of outputs per anchor # nc+5=nc+(x,y,w,h,conf)
        # nl: 表示预测层数，yolov5是3层预测
        self.nl = len(anchors)  # number of detection layers
        # na: 表示anchors的数量，除以2是因为[10,13, 16,30, 33,23]这个长度是6，对应3个anchor
        self.na = len(anchors[0]) // 2  # number of anchors
        # grid: 表示初始化grid列表大小，下面会计算grid，grid就是每个格子的x，y坐标（整数，比如0-19），左上角为(1,1),右下角为(input.w/stride,input.h/stride)
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        # anchor_grid: 表示初始化anchor_grid列表大小，空列表
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        # 注册常量anchor，并将预选框（尺寸）以数对形式存入，并命名为anchors
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # 每一张进行三次预测，每一个预测结果包含nc+5个值
        # (n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20) --> ch=(255, 255, 255)
        # 255 -> (nc+5)*3 ===> 为了提取出预测框的位置信息以及预测框尺寸信息
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 3个输出层最后的1乘1卷积
        # inplace: 一般都是True，默认不使用AWS，Inferential加速
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
    # 如果模型不训练那么将会对这些预测得到的参数进一步处理,然后输出,可以方便后期的直接调用
    # 包含了三个信息pred_box [x,y,w,h] pred_conf[confidence] pre_cls[cls0,cls1,cls2,...clsn]

    '''===================2.向前传播============================'''
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # 维度重排列: bs, 先验框组数, 检测框行数, 检测框列数, 属性数 + 分类数
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            #向前传播时需要将相对坐标转换到grid绝对坐标系中
            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    '''===================3.相对坐标转换到grid绝对坐标系============================'''
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        # grid --> (20, 20, 2), 复制成3倍，因为是三个框 -> (3, 20, 20, 2)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # anchor_grid即每个格子对应的anchor宽高，stride是下采样率，三层分别是8，16，32
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])

'''#用这三个2'''
class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # ====3._forward_once():训练的forward==== #
    def _forward_once(self, x, profile=False, visualize=False):
        # 各网络层输出, 各网络层推导耗时
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        # 遍历model的各个模块
        for m in self.model:
            # m.f 就是该层的输入来源，如果不为-1那就不是从上一层而来
            if m.f != -1:  # if not from previous layer
                # from 参数指向的网络层输出的列表
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 测试该网络层的性能
            if profile:
                self._profile_one_layer(m, x, dt)
            # 使用该网络层进行推导, 得到该网络层的输出
            x = m(x)  # run
            # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到  不在self.save层的输出就为None
            y.append(x if m.i in self.save else None)  # save output
            # 将每一层的输出结果保存到y
            if visualize:
                # 绘制该 batch 中第一张图像的特征图
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    # ====6._profile_one_layer（）:打印日志信息==== #
    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # ====9.fuse（）:将Conv2d+BN进行融合==== #
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            # 如果当前层是卷积层Conv且有bn结构, 那么就调用fuse_conv_and_bn函数讲conv和bn进行融合, 加速推理
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # 更新卷积层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # 移除bn
                delattr(m, 'bn')  # remove batchnorm
                # 更新前向传播
                m.forward = m.forward_fuse  # update forward
        # 打印conv+bn融合后的模型信息
        self.info()
        return self

    # ====11.info():打印模型结构信息==== #
    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    # ====12._apply():将模块转移到 CPU/ GPU上==== #
    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

'''#用这三个3'''
class DetectionModel(BaseModel):
    # YOLOv5 detection model
    '''===================1.__init__函数==========================='''
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # 检查传入的参数格式，如果cfg是加载好的字典结果
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        # 若不是字典 则为yaml文件路径
        else:  # is *.yaml
            import yaml  # for torch hub
            # 保存文件名：cfg file name = yolov5s.yaml
            self.yaml_file = Path(cfg).name
            # 如果配置文件中有中文，打开时要加encoding参数
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        # 搭建模型
        # yaml.get('ch', ch)表示若不存在键'ch',则返回值ch
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # 判断类的通道数和yaml中的通道数是否相等，一般不执行，因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            # 在终端给出提示
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            # 将yaml中的值修改为构造方法中的值
            self.yaml['nc'] = nc  # override yaml value
        # 重写anchor，一般不执行, 因为传进来的anchors一般都是None
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 解析模型，self.model是解析后的模型 self.save是每一层与之相连的层
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist # deepcopy()复杂产生一个新的对象
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # inplace指的是原地操作 如x+=1 有利于节约内存
        # self.inplace=True  默认True  不使用加速推理
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        # 构造步长、先验框
        m = self.model[-1]  # Detect()
        # 判断最后一层是否为Detect层
        if isinstance(m, (Detect, Segment)):
            # 定义一个256 * 256大小的输入
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 保存特征层的stride,并且将anchor处理成相对于特征层的格式
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            # 原始定义的anchor是原始图片上的像素值，要将其缩放至特征图的大小
            m.anchors /= m.stride.view(-1, 1, 1)
            # 将步长保存至模型
            self.stride = m.stride
            # 初始化bias
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')
        '''1.获取结构最后一层Detect层
            2.定义一个256*256大小的输入
            3.将[1, ch, 256, 256]大小的tensor进行一次向前传播，得到3层的输出，用输入大小256分别除以输出大小得到每一层的下采样倍数stride
            4.分别用最初的anchor大小除以stride将anchor线性缩放到对应层上'''

    # ====1.forward():管理前向传播函数==== #
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # ====2._forward_augment():推理的forward==== #
    # 将图片进行裁剪,并分别送入模型进行检测
    def _forward_augment(self, x):
        # 获得图像的高和宽
        img_size = x.shape[-2:]  # height, width
        # s是规模
        s = [1, 0.83, 0.67]  # scales
        # flip是翻转，这里的参数表示沿着哪个轴翻转
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            # scale_img函数的作用就是根据传入的参数缩放和翻转图像
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            # 模型前向传播
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            #  恢复数据增强前的模样
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        # 对不同尺寸进行不同程度的筛选
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train
    '''这个函数只在 val、detect 主函数中使用，用于提高推导的精度。
    对图像进行变换：总共 3 次，分别是 [ 原图 ]，[ 尺寸缩小到原来的 0.83，同时水平翻转 ]，[ 尺寸缩小到原来的 0.67 ]
    对图像使用 _forward_once 函数，得到在 eval 模式下网络模型的推导结果。对原图是 shape 为 [1, 22743, 85] 
    的图像检测框信息 (见 Detect 对象的 forward 函数)根据 尺寸缩小倍数、翻转维度 对检测框信息进行逆变换，添加进列表 y
    截取 y[0] 对大物体的检测结果，保留 y[1] 所有的检测结果，截取 y[2] 对小物体的检测结果，拼接得到新的检测框信息'''

    # ====4._descale_pred():将推理结果恢复到原图尺寸(逆操作)==== #
    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            # 把x,y,w,h恢复成原来的大小
            p[..., :4] /= scale  # de-scale
            # bs c h w  当flips=2是对h进行变换，那就是上下进行翻转
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            # 同理flips=3是对水平进行翻转
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    # ====5._clip_augmented（）:TTA的时候对原图片进行裁剪==== #
    # 也是一种数据增强方式，用在TTA测试的时候
    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    # ====7._initialize_biases（）:初始化偏置biases信息==== #
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None

'''parse_model函数用在DetectionModel模块中，主要作用是解析模型yaml的模块，通过读取yaml文件中的配置，并且到common.py中找到相对于的模块，
然后组成一个完整的模型解析模型文件(字典形式)，并搭建网络结构。简单来说，就是把yaml文件中的网络结构实例化成对应的模型。后续如果需要动模型框架的话，
需要对这个函数做相应的改动。'''
def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    '''===================1. 获取对应参数============================'''
    # 使用 logging 模块输出列标
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    # na: 每组先验框包含的先验框数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no: na * 属性数 (5 + 分类数)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    '''d:  yaml 配置文件（字典形式），yolov5s.yaml中的6个元素 + ch
    ch:  记录模型每一层的输出channel，初始ch=[3]，后面会删除
    na：  判断anchor的数量
    no：  根据anchor数量推断的输出维度'''

    '''===================2. 搭建网络前准备============================'''
    # 网络单元列表, 网络输出引用列表, 当前的输出通道数
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 利用 eval 函数, 读取 model 参数对应的类名 如‘Focus’,'Conv等
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # 利用 eval 函数将字符串转换为变量 如‘None’,‘nc’，‘anchors’等
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        '''===================3. 更新当前层的参数，计算c2============================'''
        # depth gain: 控制深度，如yolov5s: n*0.33
        # n: 当前模块的次数(间接控制深度)
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 当该网络单元的参数含有: 输入通道数, 输出通道数
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            # c1: 当前层的输入channel数; c2: 当前层的输出channel数(初定); ch: 记录着所有层的输出channel数
            c1, c2 = ch[f], args[0]
            # no=75，只有最后一层c2=no，最后一层不用控制宽度，输出channel必须是no
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            '''===================4.使用当前层的参数搭建当前层============================'''
            # 在初始args的基础上更新，加入当前层的输入channel并更新当前层
            # [in_channels, out_channels, *args[1:]]
            args = [c1, c2, *args[1:]]
            # 如果当前层是BottleneckCSP/C3/C3TR/C3Ghost/C3x，则需要在args中加入Bottleneck的个数
            # [in_channels, out_channels, Bottleneck个数, Bool(shortcut有无标记)]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                # 在第二个位置插入bottleneck个数n
                args.insert(2, n)  # number of repeats
                n = 1
        # 判断是否是归一化模块
        elif m is nn.BatchNorm2d:
            # BN层只需要返回上一层的输出channel
            args = [ch[f]]
        # 判断是否是tensor连接模块
        elif m is Concat:
            # Concat层则将f中所有的输出累加得到这层的输出channel
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        # 判断是否是detect模块
        elif m in {Detect, Segment}:
            # 在args中加入三个Detect层的输出channel
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors # number of anchors 几乎不执行
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        '''===================5.打印和保存layers信息============================'''
        # m_: 得到当前层的module，将n个模块组合存放到m_里面
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module

        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 计算这一层的参数量
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print

        # 把所有层结构中的from不是-1的值记下 [6,4,14,10,17,20,23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将当前层结构module加入layers中
        layers.append(m_)
        if i == 0:
            ch = []# 去除输入channel[3]
        # 把当前层的输出channel数加入ch
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
