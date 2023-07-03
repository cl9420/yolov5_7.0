# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
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
'''===============================================一、导入包==================================================='''
'''====================================1.导入安装好的python库========================================'''
import argparse  # 解析命令行参数的库
import os  # 与操作系统进行交互的文件库 包含文件路径操作与解析
import platform
import sys  # 它是与python解释器交互的一个接口，该模块提供对解释器使用或维护的一些变量的访问和获取，它提供了许多函数和变量来处理 Python 运行时环境的不同部分
from pathlib import Path  # 这个库提供了一种面向对象的方式来与文件系统交互，可以让代码更简洁、更易读

import torch

'''==============================================2.获取当前文件的绝对路径================================================'''
FILE = Path(__file__).resolve() # __file__指的是当前文件(即detect.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/detect.py
ROOT = FILE.parents[0]  # YOLOv5 root directory  ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:  # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH  把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative  relative ROOT设置为相对路径

#将当前项目添加到系统路径上，以使得项目中的模块可以调用。
#将当前项目的相对路径保存在ROOT中，便于寻找项目中的文件。

'''==================================================3..加载自定义模块===================================================='''
from models.common import DetectMultiBackend  #这个文件定义了一些通用的函数和类，比如图像的处理、非极大值抑制等等。
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams #这个文件定义了两个类，LoadImages和LoadStreams，它们可以加载图像或视频帧，并对它们进行一些预处理，以便进行物体检测或识别。
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh) #这个文件定义了一些常用的工具函数，比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等。
from utils.plots import Annotator, colors, save_one_box # 这个文件定义了Annotator类，可以在图像上绘制矩形框和标注信息。
from utils.torch_utils import select_device, smart_inference_mode #这个文件定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等等。


'''==================================================二、run函数——传入参数===================================================='''
'''====================================1.载入参数========================================'''
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):

    '''====================================2.初始化配置========================================'''
    # 输入的路径变为字符串
    source = str(source)
    # 是否保存图片和txt文件，如果nosave(传入的参数)为false且source的结尾不是txt则保存图片
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 判断source是不是视频/图像文件路径
    # Path()提取文件名。suffix：最后一个组件的文件扩展名。若source是"D://YOLOv5/data/1.jpg"， 则Path(source).suffix是".jpg"， Path(source).suffix[1:]是"jpg"
    # 而IMG_FORMATS 和 VID_FORMATS两个变量保存的是所有的视频和图片的格式后缀。
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 判断source是否是链接
    # .lower()转化成小写 .upper()转化成大写 .title()首字符转化成大写，其余为小写, .startswith('http://')返回True or Flase
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 判断是source是否是摄像头
    # .isnumeric()是否是由数字组成，返回True or False
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        # 返回文件。如果source是一个指向图片/视频的链接,则下载输入数据
        source = check_file(source)  # download
    '''====================================3.保存结果========================================'''
    # Directories
    # save_dir是保存运行结果的文件夹名，是通过递增的方式来命名的。第一次运行时路径是“runs\detect\exp”，第二次运行时路径是“runs\detect\exp1”
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 根据前面生成的路径创建文件夹
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    '''====================================4.加载模型========================================'''
    # Load model
    device = select_device(device)
    # DetectMultiBackend定义在models.common模块中，是我们要加载的网络，其中weights参数就是输入时指定的权重文件（比如yolov5s.pt）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    '''
           stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
           names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...] 
           pt: 加载的是否是pytorch模型（也就是pt格式的文件）
           jit：当某段代码即将第一次被执行时进行编译，因而叫“即时编译”
           onnx：利用Pytorch我们可以将model.pt转化为model.onnx格式的权重，在这里onnx充当一个后缀名称，
                 model.onnx就代表ONNX格式的权重文件，这个权重文件不仅包含了权重值，也包含了神经网络的网络流动信息以及每一层网络的输入输出信息和一些其他的辅助信息。
    '''
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回

    '''====================================5.加载数据========================================'''
    # Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    bs = 1  # batch_size
    if webcam: # 使用摄像头作为输入
        view_img = check_imshow(warn=True)# 检测cv2.imshow()方法是否可以执行，不能执行则抛出异常
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)# 加载输入数据流
        '''
                 source：输入数据源；image_size 图片识别前被放缩的大小；stride：识别时的步长， 
                 auto的作用可以看utils.augmentations.letterbox方法，它决定了是否需要将图片填充为正方形，如果auto=True则不需要
        '''
        bs = len(dataset)# batch_size 批大小
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs # 前者是视频路径,后者是一个cv2.VideoWriter对象

    '''====================================6.推理部分========================================'''
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        '''
                在dataset中，每次迭代的返回值是self.sources, img, img0, None, ''
                 path：文件路径（即source）
                 im: resize后的图片（经过了放缩操作）
                 im0s: 原始图片
                 vid_cap=none
                 s： 图片的基本信息，比如路径，大小
        '''
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)# 将图片放到指定设备(如GPU)上识别。#torch.size=[3,640,480]
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32# 把输入从整型转化为半精度/全精度浮点数。
            im /= 255  # 0 - 255 to 0.0 - 1.0归一化，所有像素点除以255
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim 添加一个第0维。缺少batch这个尺寸，所以将它扩充一下，变成[1，3,640,480]

        # Inference
        with dt[1]:
            # 可视化文件路径。如果为True则保留推理过程中的特征图，保存在runs文件夹中
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 推理结果，pred保存的是所有的bound_box的信息
            pred = model(im, augment=augment, visualize=visualize)#模型预测出来的所有检测框，torch.size=[1,18900,85]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            '''
                     pred: 网络的输出结果
                     conf_thres： 置信度阈值
                     iou_thres： iou阈值
                     classes: 是否只保留特定的类别 默认为None
                     agnostic_nms： 进行nms是否也去除不同类别之间的框
                     max_det: 检测框结果的最大数量 默认1000
            '''
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # 把所有的检测框画到原图中
        for i, det in enumerate(pred):  # per image\
            '''
                       i：每个batch的信息
                       det:表示5个检测框的信息
            '''
            seen += 1 #seen是一个计数的功能
            if webcam:  # batch_size >= 1
                # 如果输入源是webcam则batch_size>=1 取出dataset中的一张图片
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                '''
                大部分我们一般都是从LoadImages流读取本都文件中的照片或者视频 所以batch_size=1
                p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                s: 输出信息 初始为 ''
                im0: 原始图片 letterbox + pad 之前的图片
                frame: 视频流,此次取的是第几张图片
                '''

            p = Path(p)  # to Path
            # 图片/视频的保存路径save_path 如 runs\\detect\\exp8\\fire.jpg
            save_path = str(save_dir / p.name)  # im.jpg
            # 设置保存框坐标的txt文件路径，每张图片对应一个框坐标信息
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # 设置输出图片信息。图片shape (w, h)
            s += '%gx%g ' % im.shape[2:]  # print string
            # 得到原图的宽和高
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 保存截图。如果save_crop的值为true，则将检测到的bounding_box单独保存成一张图片。
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 得到一个绘图的类，类中预先存储了原图、线条宽度、类名
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息映射到原图
                # 将标注的bounding_box大小调整为和原图一致（因为训练时原图经过了放缩）此时坐标格式为xyxy
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 保存预测结果：txt/图片画框/crop-image
                for *xyxy, conf, cls in reversed(det):
                    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id + score + xywh
                    if save_txt:  # Write to file
                        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # line的形式是： ”类别 x y w h“，若save_conf为true，则line的形式是：”类别 x y w h 置信度“
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在原图上画框+将预测到的目标剪切出来保存成图片，保存在save_dir/crops下，在原图像画图或者保存结果
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class# 类别标号
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')# 类别名
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # 在原图上画框+将预测到的目标剪切出来保存成图片，保存在save_dir/crops下（单独保存）
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result() # im0是绘制好的图片
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 设置保存图片/视频
            if save_img:# 如果save_img为true,则保存绘制完的图片
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video  vid_path[i] != save_path,说明这张图片属于一段新的视频,需要重新创建视频文件
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    '''====================================7.在终端里打印出运行的结果========================================'''
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image平均每张图片所耗费时间
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''# 标签保存的路径
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

# --weights：  训练的权重路径，可以使用自己训练的权重，也可以使用官网提供的权重。默认官网的权重yolov5s.pt(yolov5n.pt/yolov5s.pt/yolov5m.pt/yolov5l.pt/yolov5x.pt/区别在于网络的宽度和深度以此增加)
# --source：  测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头)，也可以是rtsp等视频流, 默认data/images
# --data：  配置数据文件路径，包括image/label/classes等信息，训练自己的文件，需要作相应更改，可以不用管
# --imgsz：  预测时网络输入图片的尺寸，默认值为 [640]
# --conf-thres：  置信度阈值，默认为 0.50
# --iou-thres：  非极大抑制时的 IoU 阈值，默认为 0.45
# --max-det：  保留的最大检测框数量，每张图片中检测目标的个数最多为1000类
# --device：  使用的设备，可以是 cuda 设备的 ID（例如 0、0,1,2,3）或者是 'cpu'，默认为 '0'
# --view-img：  是否展示预测之后的图片/视频，默认False
# --save-txt：  是否将预测的框坐标以txt文件形式保存，默认False，使用--save-txt 在路径runs/detect/exp*/labels/*.txt下生成每张图片预测的txt文件
# --save-conf：  是否保存检测结果的置信度到 txt文件，默认为 False
# --save-crop：  是否保存裁剪预测框图片，默认为False，使用--save-crop 在runs/detect/exp*/crop/剪切类别文件夹/ 路径下会保存每个接下来的目标
# --nosave：  不保存图片、视频，要保存图片，不设置--nosave 在runs/detect/exp*/会出现预测的结果
# --classes：  仅检测指定类别，默认为 None
# --agnostic-nms：  是否使用类别不敏感的非极大抑制（即不考虑类别信息），默认为 False
# --augment：  是否使用数据增强进行推理，默认为 False
# --visualize：  是否可视化特征图，默认为 False
# --update：  如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
# --project：  结果保存的项目目录路径，默认为 'ROOT/runs/detect'
# --name：  结果保存的子目录名称，默认为 'exp'
# --exist-ok：  是否覆盖已有结果，默认为 False
# --line-thickness：  画 bounding box 时的线条宽度，默认为 3
# --hide-labels：  是否隐藏标签信息，默认为 False
# --hide-conf：  是否隐藏置信度信息，默认为 False
# --half：  是否使用 FP16 半精度进行推理，默认为 False
# --dnn：  是否使用 OpenCV DNN 进行 ONNX 推理，默认为 False


'''=======================================三、Parse_opt()用来设置输入参数的子函数===================================='''
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp2/weights/best.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/mydata.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')#与上面的配合使用
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')#裁剪检测物体
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')#可以指定检测什么
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')#特征图可视化
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

'''==================================================四、设置main函数===================================================='''
def main(opt):
    # 检查环境/打印参数,主要是requrement.txt的包是否安装，用彩色显示设置的参数
    check_requirements(exclude=('tensorboard', 'thop'))
    # 执行run()函数
    run(**vars(opt))

# 命令使用
# python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source  data/images/xx.jpg

if __name__ == "__main__":
    opt = parse_opt()
    #解析命令行传进的参数。该段代码分为三部分，第一部分定义了一些可以传导的参数类型，第二部分对于imgsize部分进行了额外的判断（640*640），第三部分打印所有参数信息，opt变量存储所有的参数信息，并返回。
    main(opt)
    # 执行命令行参数。该段代码分为两部分，第一部分首先完成对于requirements.txt的检查，检测这些依赖包有没有安装；第二部分，将opt变量参数传入，执行run函数。
