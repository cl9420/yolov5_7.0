# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""
'''============1.导入安装好的python库=========='''
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
'''首先，导入一下常用的python库：
argparse：  它是一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息
json：  实现字典列表和JSON字符串之间的相互解析
os： 它提供了多种操作系统的接口。通过os模块提供的操作系统接口，我们可以对操作系统里文件、终端、进程等进行操作
sys： 它是与python解释器交互的一个接口，该模块提供对解释器使用或维护的一些变量的访问和获取，它提供了许多函数和变量来处理 Python 运行时环境的不同部分
pathlib：  这个库提供了一种面向对象的方式来与文件系统交互，可以让代码更简洁、更易读
threading：  python中处理多线程的库
然后再导入一些 pytorch库：
numpy：  科学计算库，提供了矩阵，线性代数，傅立叶变换等等的解决方案, 最常用的是它的N维数组对象
torch：   这是主要的Pytorch库。它提供了构建、训练和评估神经网络的工具
tqdm：  就是我们看到的训练时进度条显示'''

'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve()# __file__指的是当前文件(即val.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/val.py
ROOT = FILE.parents[0]  # YOLOv5 root directory ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path: # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH 把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative ROOT设置为相对路径

'''===================3..加载自定义模块============================'''
from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode
'''
models.common：  yolov5的网络结构(yolov5)
utils.callbacks：  定义了回调函数，主要为logger服务
utils.datasets：  dateset和dateloader定义代码
utils.general.py：   定义了一些常用的工具函数，比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等
utils.metrics：   模型验证指标，包括ap，混淆矩阵等
utils.plots.py：    定义了Annotator类，可以在图像上绘制矩形框和标注信息
utils.torch_utils.py：   定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等 通过导入这些模块，
可以更方便地进行目标检测的相关任务，并且减少了代码的复杂度和冗余
'''

'''======================1.保存预测信息到txt文件====================='''
def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    # gn = [w, h, w, h] 对应图片的宽高  用于后面归一化
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id + score + xywh
    for *xyxy, conf, cls in predn.tolist():
        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽高)格式，并归一化，转化为列表再保存
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        # line的形式是： "类别 xywh"，若save_conf为true，则line的形式是："类别 xywh 置信度"
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        # 将上述test得到的信息输出保存 输出为xywh格式 coco数据格式也为xywh格式
        with open(file, 'a') as f:
            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})
'''
image_id：  图片id，即属于哪张图片
category_id：   类别，coco91class()从索引0~79映射到索引0~90
bbox：   预测框坐标
score：  预测得分
之前的的xyxy格式是左上角右下角坐标 ，xywh是中心的坐标和宽高，
而coco的json格式的框坐标是xywh(左上角坐标 + 宽高)，所以 box[:, :2] -= box[:, 2:] / 2 这行代码是将中心点坐标 -> 左上角坐标。'''

'''========================三、计算指标==========================='''
def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    返回每个预测框在10个IoU阈值上是TP还是FP
    Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    # 构建一个[pred_nums, 10]全为False的矩阵
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    # 计算每个gt与每个pred的iou，shape为: [gt_nums, pred_nums]
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        # iou超过阈值而且类别正确，则为True，返回索引
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            # 将符合条件的位置构建成一个新的矩阵，第一列是行索引（表示gt索引），第二列是列索引（表示预测框索引），第三列是iou值
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                # argsort获得有小到大排序的索引, [::-1]相当于取反reserve操作，变成由大到小排序的索引，对matches矩阵进行排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                '''
                参数return_index=True：表示会返回唯一值的索引，[0]返回的是唯一值，[1]返回的是索引
                matches[:, 1]：这里的是获取iou矩阵每个预测框的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
                这个操作的含义：每个预测框最多只能出现一次，如果有一个预测框同时和多个gt匹配，只取其最大iou的一个
                '''
                # matches = matches[matches[:, 2].argsort()[::-1]]
                '''
                matches[:, 0]：这里的是获取iou矩阵gt的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
                这个操作的含义: 每个gt也最多只能出现一次，如果一个gt同时匹配多个预测框，只取其匹配最大的那一个预测框
                '''
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
            '''
            当前获得了gt与预测框的一一对应，其对于的iou可以作为评价指标，构建一个评价矩阵
            需要注意，这里的matches[:, 1]表示的是为对应的预测框来赋予其iou所能达到的程度，也就是iouv的评价指标
            '''
            # 在correct中，只有与gt匹配的预测框才有对应的iou评价指标，其他大多数没有匹配的预测框都是全部为False
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

'''
data：  数据集文件的路径，默认为COCO128数据集的配置文件路径
weights：  模型权重文件的路径，默认为YOLOv5s的权重文件路径
batch_size:   前向传播的批次大小，运行val.py传入默认32 。运行train.py则传入batch_size // WORLD_SIZE * 2
imgsz：  输入图像的大小，默认为640x640
conf_thres：  置信度阈值，默认为0.001
iou_thres：  非极大值抑制的iou阈值，默认为0.6
task:   设置测试的类型 有train, val, test, speed or study几种，默认val
device：  使用的设备类型，默认为空，表示自动选择最合适的设备
single_cls:   数据集是否只用一个类别，运行val.py传入默认False 运行train.py则传入single_cls
augment：  是否使用数据增强的方式进行检测，默认为False
verbose:   是否打印出每个类别的mAP，运行val.py传入默认Fasle。运行train.py则传入nc < 50 and final_epoch
save_txt：  是否将检测结果保存为文本文件，默认为False
save_hybrid:   是否保存 label+prediction hybrid results to *.txt 默认False
save_conf：  是否在保存的文本文件中包含置信度信息，默认为False
save_json：  是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）运行test.py传入默认Fasle。运行train.py则传入is_coco and final_epoch(一般也是False)
project：  结果保存的项目文件夹路径，默认为“runs/val”
name：  结果保存的文件名，默认为“exp”
exist_ok：  如果结果保存的文件夹已存在，是否覆盖，默认为False，即不覆盖
half：  是否使用FP16的半精度推理模式，默认为False
dnn：  是否使用OpenCV DNN作为ONNX推理的后端，默认为False
model:  模型， 如果执行val.py就为None 如果执行train.py就会传入ema.ema(ema模型)
dataloader:  数据加载器， 如果执行val.py就为None 如果执行train.py就会传入testloader
save_dir:  文件保存路径， 如果执行val.py就为‘ ’ ，如果执行train.py就会传入save_dir(runs/train/expn)
plots:  是否可视化，运行val.py传入默认True，运行train.py则传入plots and final_epoch
callback:   回调函数
compute_loss:  损失函数，运行val.py传入默认None，运行train.py则传入compute_loss(train) 
'''
@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    '''======================2.初始化/加载模型以及设置设备====================='''
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py# 通过 train.py 调用的run函数
        # 获得记录在模型中的设备 next为迭代器
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        # 精度减半
        # 如果设备类型不是cpu 则将模型由32位浮点数转换为16位浮点数
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly# 直接通过 val.py 调用 run 函数
        # 调用torch_utils中select_device来选择执行程序时的设备
        device = select_device(device, batch_size=batch_size)

        # Directories# 路径
        # 调用genera.py中的increment_path函数来生成save_dir文件路径  run\test\expn
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # mkdir创建路径最后一级目录
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        # 调用general.py中的check_img_size函数来检查图像分辨率能否被32整除
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        # 调用general.py中的check_dataset函数来检查数据文件是否正常
        data = check_dataset(data)  # check
        '''训练时（train.py）调用：初始化模型参数、训练设备
        验证时（val.py）调用：初始化设备、save_dir文件路径、make dir、加载模型、check imgsz、 加载+check data配置信息'''

    # Configure
    '''======================3.加载配置====================='''
    # 将模型转换为测试模式 固定住dropout层和Batch Normalization层
    model.eval()
    cuda = device.type != 'cpu'
    # 通过 COCO 数据集的文件夹组织结构判断当前数据集是否为 COCO 数据集
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    # 确定检测的类别数目
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 计算mAP相关参数
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    # numel为pytorch预置函数 用来获取张量中的元素个数
    niou = iouv.numel()

    '''======================4.加载val数据集====================='''
    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # 调用datasets.py文件中的create_dataloader函数创建dataloader
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    '''======================5.初始化====================='''
    # 初始化已完成测试的图片数量
    seen = 0
    # 调用matrics中函数 存储混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 获取数据集所有类别的类名
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    # 调用general.py中的函数  获取coco数据集的类别索引
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 设置tqdm进度条的显示信息
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    # 初始化detection中各个指标的值
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    # 初始化网络训练的loss
    loss = torch.zeros(3, device=device)
    # 初始化json文件涉及到的字典、统计信息、AP、每一个类别的AP、图片汇总
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar

    '''=====6.1 开始验证前的预处理====='''
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                # 将图片数据拷贝到device（GPU）上面
                im = im.to(device, non_blocking=True)
                # 对targets也做同样拷贝的操作
                targets = targets.to(device)
            # 将图片从64位精度转换为32位精度
            im = im.half() if half else im.float()  # uint8 to fp16/32
            # 将图像像素值0-255的范围归一化到0-1的范围
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 四个变量分别代表batchsize、通道数目、图像高度、图像宽度
            nb, _, height, width = im.shape  # batch size, channels, height, width

        '''====6.2 前向推理===='''
        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)
            '''out:   推理结果。1个 ，[bs, anchor_num*grid_w*grid_h, xywh+c+20classes] = [1, 19200+4800+1200, 25]
                train_out:   训练结果。3个， [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]。如: [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]'''

        '''====6.3 计算损失===='''
        # Loss
        # compute_loss不为空 说明正在执行train.py  根据传入的compute_loss计算损失值
        if compute_loss:
            # loss 包含bounding box 回归的GIoU、object和class 三者的损失
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls
        '''====6.4 NMS获得预测框===='''
        # NMS
        # targets: [num_target, img_index+class_index+xywh] = [31, 6]
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        # 提取bach中每一张图片的目标的label
        # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)

        '''====6.5 统计真实框、预测框信息===='''
        # Metrics
        # si代表第si张图片，pred是对应图片预测的label信息
        for si, pred in enumerate(preds):
            # 获取第si张图片的gt标签信息 包括class, x, y, w, h    target[:, 0]为标签属于哪张图片的编号
            labels = targets[targets[:, 0] == si, 1:]
            # nl为图片检测到的目标个数
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            # 如果预测为空，则添加空的信息到stats里
            if npr == 0:# 预测为空但同时有label信息
                # stats初始化为一个空列表[] 此处添加一个空信息
                # 添加的每一个元素均为tuple 其中第二第三个变量为一个空的tensor
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            # 对pred进行深复制
            predn = pred.clone()
            # 调用general.py中的函数 将图片调整为原图大小
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # 处理完gt的尺寸信息，重新构建成 (cls, xyxy)的格式
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                # 对当前的预测框与gt进行一一匹配，并且在预测框的对应位置上获取iou的评分信息，其余没有匹配上的预测框设置为False
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                # 保存预测信息到txt文件
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        '''====6.6 画出前三个batch图片的gt和pred框===='''
        # Plot images
        # 画出前三个batch的图片的ground truth和预测框predictions(两个图)一起保存
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    '''====6.7 计算指标===='''
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    # stats[0].any(): stats[0]是否全部为False, 是则返回 False, 如果有一个为 True, 则返回 True
    if len(stats) and stats[0].any():
        # 计算上述测试过程中的各种性能指标
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    '''p:    [nc] 最大平均f1时每个类别的precision
    r:    [nc] 最大平均f1时每个类别的recall
    ap:    [71, 10] 数据集每个类别在10个iou阈值下的mAP
    f1：    [nc] 最大平均f1时每个类别的f1
    ap_class:    [nc] 返回数据集中所有的类别index
    conf [img_sum] ：整个数据集所有图片中所有预测框的conf [1905]
    ap50:   [nc] 所有类别的mAP@0.5   
    ap:   [nc] 所有类别的mAP@0.5:0.95 
    pcls [img_sum] ：整个数据集所有图片中所有预测框的类别 [1905]
    mp:   [1] 所有类别的平均precision(最大f1时)
    mr:   [1] 所有类别的平均recall(最大f1时)
    map50:   [1] 所有类别的平均mAP@0.5
    map:   [1] 所有类别的平均mAP@0.5:0.9'''

    '''====6.8 打印日志===='''
    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    '''====6.9 保存验证结果===='''
    # Plots
    if plots:
        # confusion_matrix.plot（）函数绘制混淆矩阵
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        # 调用Loggers中的on_val_end方法，将日志记录并生成一些记录的图片
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    # 采用之前保存的json文件格式预测结果 通过coco的api评估各个指标
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements('pycocotools')
            # 以下过程为利用官方coco工具进行结果的评测
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    '''====6.10 返回结果===='''
    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
'''mp：  [1] 所有类别的平均precision(最大f1时)
    mr：  [1] 所有类别的平均recall(最大f1时)
    map50：  [1] 所有类别的平均mAP@0.5
    map ： [1] 所有类别的平均mAP@0.5:0.95
    val_box_loss ： [1] 验证集回归损失
    val_obj_loss：  [1] 验证集置信度损失
    val_cls_loss：  [1] 验证集分类损失 maps: [80] 所有类别的mAP@0.5:0.95 t: {tuple: 3}
    0:  打印前向传播耗费的总时间
    1:  nms耗费总时间
    2:  总时间'''


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/mydata.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/best.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt
'''data：  数据集文件的路径，默认为COCO128数据集的配置文件路径
weights：  模型权重文件的路径，默认为YOLOv5s的权重文件路径
batch_size:   前向传播的批次大小，运行val.py传入默认32 。运行train.py则传入batch_size // WORLD_SIZE * 2
imgsz：  输入图像的大小，默认为640x640
conf_thres：  置信度阈值，默认为0.001
iou_thres：  非极大值抑制的iou阈值，默认为0.6
task:   设置测试的类型 有train, val, test, speed or study几种，默认val
device：  使用的设备类型，默认为空，表示自动选择最合适的设备
single_cls:   数据集是否只用一个类别，运行val.py传入默认False 运行train.py则传入single_cls
augment：  是否使用数据增强的方式进行检测，默认为False
verbose:   是否打印出每个类别的mAP，运行val.py传入默认Fasle。运行train.py则传入nc < 50 and final_epoch
save_txt：  是否将检测结果保存为文本文件，默认为False
save_hybrid:   是否保存 label+prediction hybrid results to *.txt 默认False
save_conf：  是否在保存的文本文件中包含置信度信息，默认为False
save_json：  是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）运行test.py传入默认Fasle。运行train.py则传入is_coco and final_epoch(一般也是False)
project：  结果保存的项目文件夹路径，默认为“runs/val”
name：  结果保存的文件名，默认为“exp”
exist_ok：  如果结果保存的文件夹已存在，是否覆盖，默认为False，即不覆盖
half：  是否使用FP16的半精度推理模式，默认为False
dnn：  是否使用OpenCV DNN作为ONNX推理的后端，默认为False'''

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
