from yolo_model import get_class, get_anchors
from yolov3_tf2.Drank53 import yolo_body
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,TensorBoard)
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from dataset import yolo_loss
import numpy as np

from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

#####改了float

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,random=True):
    '''data generator for fit_generator,annotation_lines: 所有的图片名称,batch_size：每批图片的大小
    input_shape： 图片的输入尺寸,anchors: 大小,num_classes： 类别数'''
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box =get_random_data(annotation_lines[i], input_shape, random=random)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors,num_classes)
        # y_true的第0和1位是中心点xy，范围是(0~13/26/52)，第2和3位是宽高wh，范围是0~1，
        # 第4位是置信度1或0，第5~n位是类别为1其余为0。
        # [(16, 13, 13, 3, 6), (16, 26, 26, 3, 6), (16, 52, 52, 3, 6)]
        yield [image_data, *y_true], np.zeros((batch_size))


def preprocess_true_boxes(true_boxes, input_shape, anchors,num_classes):
    '''Preprocess true boxes to training input format
       y_true的第0和1位是中心点xy，范围是(0~13/26/52)，第2和3位是宽高wh，范围是0~1，
       第4位是置信度1或0，第5~n位是类别为1其余为0。
       Parameters----------
       true_boxes: array, shape=(m, T, 5)
           Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
       input_shape: array-like, hw, multiples of 32
       anchors: array, shape=(N, 2), wh
       num_classes: integer
       Returns-------
       y_true: list of array, shape like yolo_outputs, xywh are reletive value'''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    true_boxes = np.array(true_boxes,dtype='float32')#######类型
    input_shape = np.array(input_shape,dtype='int32')
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4])//2
    true_boxes[..., 2:4] = boxes_wh/input_shape
    true_boxes[..., 0:2] = boxes_xy/input_shape
    # 得到模型每一个输出y1-y3的下采样之后的特征图片大小
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    # 获取三组y_true # [(16, 13, 13, 3, 6), (16, 26, 26, 3, 6), (16, 52, 52, 3, 6)],
    y_true = [np.zeros((true_boxes.shape[0], grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]
    anchors = tf.expand_dims(anchors,0) #shape=[1,9,2]
    max_anchor = anchors/2.
    min_anchor = -max_anchor
    valid_mask = boxes_wh[..., 0] > 0
    for b in range(true_boxes.shape[0]):
        # 取第b个boxes选取wh大于0的anchors
        wh = boxes_wh[b, valid_mask[b]]#######记得该方式
        if len(wh)==0: continue
        wh = tf.expand_dims(wh,-2)  #shape=[T,1,2]
        box_maxs = wh/2.
        box_mins = -box_maxs
        # 求目标的范围，和anchors的iou值，查看目标的标记值与9个anchors哪个iou最大
        intersect_mins = np.maximum(box_mins, min_anchor)
        intersect_maxes = np.minimum(box_maxs, max_anchor)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area =  tf.cast(intersect_wh[..., 0] * intersect_wh[..., 1], tf.float32)
        box_area = tf.cast(wh[..., 0] * wh[..., 1],tf.float32)
        anchor_area =tf.cast(anchors[..., 0] * anchors[..., 1], tf.float32)
        iou = intersect_area / (box_area + anchor_area - intersect_area) ##shape=[T,9]
        # 从每个iou值中，找到iou值最大的目标# Find best anchor for each true box
        # 得到9个anchos的一个值
        best_anchor = np.argmax(iou, axis=-1) #shape=T
        for t, n in enumerate(best_anchor):# index，t为9个最大之中的某一个
            for l in range(num_layers):##层数
                if n in anchor_mask[l]:
                    # i就是在对应的特征图上的实际尺寸的宽，就是高
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')  ##在特征图中grid的x位置
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')  ##在特征图中grid的y位置
                    k = anchor_mask[l].index(n) ###在第几层的第几个锚框的索引
                    c = true_boxes[b, t, 4].astype('int32')###类别索引
                    # y_true的第0和1位是中心点xy，范围是(0~13/26/52)，
                    # 第2和3位是宽高wh，范围是0~1，
                    # 第4位是置信度1或0，
                    # 第5~n位是类别为1其余为0。
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1
                    # [(16, 13, 13, 3, 6), (16, 26, 26, 3, 6), (16, 52, 52, 3, 6)]
    return y_true

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

# 数据增强
def get_random_data(annotation_line, input_shape, max_boxes=20, jitter=.3,
                    proc_img=True, hue=.1, sat=1.5, val=1.5, random=True):
    '''获取真实的数据根据输入的尺寸对原始数据进行缩放处理得到input_shape大小的数据图片，
    随机进行图片的翻转，标记数据数据也根据比例改变
    annotation_line： 单条图片的信息的列表,input_shape：输入的尺寸'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    w, h = input_shape
    box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])
    if not random:
        scale = tf.minimum(float(w)/float(iw),float(h)/float(ih))
        new_w = int(iw* scale)
        new_h = int(ih* scale)
        image_data = 0
        if proc_img:
            image = image.resize((new_w,new_h), Image.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, ((w-iw)/2, (h-ih)/2))
            image_data = np.array(new_image)/255.
        box_data = np.zeros((max_boxes,5))
        if len(box) > 0:
            np.random.shuffle(box)
            box = box[:max_boxes] if len(box) > max_boxes else box
            box[:, [0, 2]] = box[:, [0, 2]] * scale + (w - iw) / 2
            box[:, [1, 3]] = box[:, [1, 3]] * scale + (h - ih) / 2
            box_data[:len(box)] = box
        return image_data,box_data
        # resize image
        # 随机的图片比例变换
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2.)
    # 计算新的图片尺寸
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    # 改变图片尺寸
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    # 随机把图片摆放在灰度图片上
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image
    # 是否反转图片# flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # distort image
    # 在HSV坐标域中，改变图片的颜色范围，hue值相加，sat和vat相乘，
    # 先由RGB转为HSV，再由HSV转为RGB，添加若干错误判断，避免范围过大。
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
    # correct boxes
    # 将所有的图片变换，增加至检测框中，并且包含若干异常处理，
    # 避免变换之后的值过大或过小，去除异常的box。
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        # 变换所有目标的尺寸
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        # 如果已经翻转了需要进行坐标变换，并且把坐标限制在图片内
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        # 最大的目标数不能超过超参数
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['AUTOGRAPH_VERBOSITY'] =  "1"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7 # maximun alloc gpu70% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
session = tf.compat.v1.Session(config=config)
#-----------------------------------------------
if __name__ == "__main__":
    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    annotation_path = '2012_train.txt'
    # ------------------------------------------------------#
    #   训练后的模型保存的位置，保存在logs文件夹里面
    # ------------------------------------------------------#
    log_dir = 'logs/'
    # ----------------------------------------------------#
    #   classes和anchor的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    # ----------------------------------------------------#
    classes_path = 'data/voc_classes.txt'
    anchors_path = 'data/VOC2012_anchors.txt'
    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    # ------------------------------------------------------#
    weights_path = 'data/yolo_weights.h5'
    # ------------------------------------------------------#
    #   输入的shape大小
    # ------------------------------------------------------#
    input_shape = (416, 416)
    # ------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    # ------------------------------------------------------#
    normalize = False
    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names = get_class(classes_path)
    anchors = get_anchors(anchors_path)
    # ------------------------------------------------------#
    #   一共有多少类和多少先验框
    # ------------------------------------------------------#
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    ###创建yolo模型，输入下面几个参数，就可以创建出模型
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)

    # ------------------------------------------------------#
    #   载入预训练权重  用的是迁移模型的思想
    # ------------------------------------------------------#
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True,  # 从hdf5文件中加载权重到当前模型，默认情况模型结构保持不变。
                            # 如果想要将权重载入不同模型（部分层不相同），则设置by_name=True，只有名字匹配的层才会载入，为true时更适合微调和迁移学习
                            skip_mismatch=True)  # 当使用权重时，遇到不匹配特征层可以直接绕过
    """把目标当成一个输入，构成多输入模型，把loss写成一个层，作为最后的输出，搭建模型的时候，
          就只需要将模型的output定义为loss，而compile的时候，
          直接将loss设置为y_pred（因为模型的输出就是loss，所以y_pred就是loss），
          无视y_true，训练的时候，y_true随便扔一个符合形状的数组进去就行了。"""
    # ------------------------------------------------------#
    # y_true的shape为【grid,grid,3,25】
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    # model_loss =yolo_loss((*model_body.output, *y_true), anchors, num_classes)

    model_loss = Lambda(yolo_loss,
                        output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'ignore_thresh': 0.5}
                        )([*model_body.output, *y_true])
    # --model_body.input是任意(?)个(416, 416, 3)y_true是已标注数据所转换的真值结构
    model = Model(inputs=[model_body.input, *y_true], outputs=model_loss)  # 模型，inputs和outputs
    # plot_model(model, to_file=os.path.join('data', 'model.png'), show_shapes=True, show_layer_names=True)
    model.summary()

    # 冻结训练：这是迁移学习的思想，因为神经网络主干特征提取部分所提取到的特征是通用的，
    # 我们冻结起来训练可以加快训练效率，也可以防止权值被破坏
    freeze_layers = 184  ##冻结，不让训练(特征提取层)
    for i in range(freeze_layers): model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    # -------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    # -------------------------------------------------------------------------------#
    # https://www.cnblogs.com/coolqiyu/p/9092807.html 进入TensorBoard界面指导
    logging = TensorBoard(log_dir=log_dir)  # 使用TensorBoard将keras的训练过程显示出来
    # 每两个周期进行保存，定时储存数据
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=2)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # monitor监测的量

                                  factor=0.5,  # 每次减小学习中的因子，学习率将以Lr=lr*factor的形式减小
                                  patience=5,  # 每3个epoch过程模型性能不提升，便触发Lr
                                  verbose=1)
    # 当6个epoch没有降低loss，则停止
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

    # ----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    # ----------------------------------------------------------------------#
    val_split = 0.1  # 使用10%的数据做验证集
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        Init_epoch = 0
        Freeze_epoch = 50
        batch_size = 8
        learning_rate_base = 1e-3

        model.compile(optimizer=Adam(lr=learning_rate_base), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})
        # name= yolo_loss，然后传入y_true, y_pred，函数返回的是即为预测值与真实值的某种误差函数

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, random=True),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                           random=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=Freeze_epoch,
            initial_epoch=Init_epoch,  # 从0开始训练
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])  # 在训练时调用一系列回调函数
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
    # 之前冻结了184层，现在需要加进来一起训练#
    for i in range(freeze_layers): model_body.layers[i].trainable = True

    if True:
        Freeze_epoch = 50
        Epoch = 100
        batch_size = 4
        learning_rate_base = 5e-5

        model.compile(optimizer=Adam(lr=learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, random=True),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                           random=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=Epoch,
            initial_epoch=Freeze_epoch,  # 从之前训练结束出开始训练
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'VOC2012.h5')

