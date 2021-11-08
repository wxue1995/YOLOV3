import tensorflow as tf
import numpy as np
from yolov3_tf2.Drank53 import yolo_body
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
import os, cv2
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import tensorflow.keras.backend as K
from yolo_model import yolo_box
from yolo_model import get_class, get_anchors




def box_giou(b1, b2):
    '''Return iou tensor,
    Parameters----------b1: tensor, shape=(i1,...,iN, 4), xywh,b2: tensor, shape=(j, 4), xywh
    Returns-------iou: tensor, shape=(i1,...,iN, j)'''
    # Expand dim to apply broadcasting.
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half
    #===========cal IOU ======================
    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    # ===========cal enclose area for GIOU=============#
    enclose_left_up = tf.minimum(b1_mins, b2_mins)
    enclose_right_down = tf.maximum(b1_maxes, b2_maxes)
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    # cal GIOU
    gious = iou - 1.0 * (enclose_area - (b1_area + b2_area - intersect_area)) / enclose_area
    return gious


    # return iou



def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False,normalize=True):
    """Return yolo_loss tensor,Parameters----------
        yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
        y_true: list of array, the output of preprocess_true_boxes
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        ignore_thresh: float, the iou threshold whether to ignore object confidence loss
        Returns------- loss: tensor, shape=(1,)"""
    num_layers = len(anchors)//3
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    # input_shape是输出的尺寸*32, 就是原始的输入尺寸，[1:3]是尺寸的位置
    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3]*32, K.dtype(y_true[0]))
    grid_shape = [tf.cast(tf.shape(yolo_outputs[l])[1:3], K.dtype(y_true[l])) for l in range(num_layers)]
    loss = 0
    num_pos = 0
    m = tf.shape(yolo_outputs[0])[0]
    for i in range(num_layers):
        true_object_mask = y_true[i][..., 4:5]
        true_class_index = y_true[i][..., 5:]
        # 这是yolo_outputs的后处理程序
        grid, raw_pred, pred_xy, pred_wh = yolo_box(yolo_outputs[i], anchors[anchor_mask[i]],
                      num_classes, calc_loss = True)
        pred_box = tf.concat([pred_xy, pred_wh],axis=-1)
        # 把真实的坐标转换到预测坐标系
        raw_true_xy = y_true[i][..., 0:2] * grid_shape[i] - grid
        raw_true_wh = tf.math.log(y_true[i][..., 2:4] * input_shape / anchors[anchor_mask[i]])
        raw_true_wh = K.switch(true_object_mask, raw_true_wh, K.zeros_like(raw_true_wh))
        # give higher weights to small boxes
        box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]  # 2-w*h

        ##tf.TensorArray可以看作是具有动态size功能的Tensor数组，可以动态扩展，因此常用于储存中间结果
        # 对于每一张图片计算ignore_mask（负样本参数），ignore_mask这里是为了防止正负样本失衡，要进行筛选
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        # 将真实标定的数据置信率转换为bool类型
        object_mask_bool = K.cast(true_object_mask, 'bool')

        # -----------------------------------------------------------#
        #   对每一张图片计算ignore_mask
        # -----------------------------------------------------------#

        def loop_body(b, ignore_mask):
            # -----------------------------------------------------------#
            #   取出n个真实框：n,4
            # -----------------------------------------------------------#
            # tf.boolean_mask的作用是通过bool值过滤元素，利用T or F ，先拿到存在的box的四个坐标
            true_box = tf.boolean_mask(y_true[i][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # -----------------------------------------------------------#
            #   计算预测框与真实框的iou
            #   pred_box    13,13,3,4 预测框的坐标
            #   true_box    n,4 真实框的坐标
            #   iou         13,13,3,n 预测框和真实框的iou
            # -----------------------------------------------------------#
            iou = box_giou(pred_box[b], true_box)

            # -----------------------------------------------------------#
            #   best_iou    13,13,3 每个特征点与真实框的最大重合程度
            # -----------------------------------------------------------#
            best_iou = K.max(iou, axis=-1)

            # -----------------------------------------------------------#
            #   判断预测框和真实框的最大iou小于ignore_thresh
            #   则认为该预测框没有与之对应的真实框
            #   该操作的目的是：
            #   忽略预测结果与真实框非对应特征点，因为这些框已经比较准了
            #   不适合当作负样本，所以忽略掉。
            # -----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        # -----------------------------------------------------------#
        #   在这个地方进行一个循环、循环是对每一张图片进行的
        # -----------------------------------------------------------#
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

        # -----------------------------------------------------------#
        #   ignore_mask用于提取出作为负样本的特征点
        #   (m,13,13,3)
        # -----------------------------------------------------------#
        ignore_mask = ignore_mask.stack()
        #   (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)
        # true_object_mask = tf.squeeze(true_object_mask,-1)
        # y_true_flat = tf.boolean_mask(y_true[i][..., 0:4], tf.cast(true_object_mask, tf.bool))
        # # y_true_flat = tf.expand_dims(y_true_flat,axis=-1)
        # best_iou = tf.reduce_max(box_giou(pred_box, y_true_flat),axis=-1)
        # ignore_mask = tf.cast(best_iou<ignore_thresh, tf.float32)
        # -----------------------------------------------------------#
        #   xy_loss中心点误差，利用binary_crossentropy计算中心点偏移情况，效果更好
        #   from_logits=True时，网络预测值y_pred 表示必须为还没经过sofmax处理的函数变量
        # -----------------------------------------------------------#
        xy_loss = true_object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        # -----------------------------------------------------------#
        #   wh_loss用于计算宽高损失
        # -----------------------------------------------------------#
        wh_loss = true_object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])

        # ------------------------------------------------------------------------------#
        #   如果该位置本来有框，那么计算1与置信度的交叉熵
        #   如果该位置本来没有框，那么计算0与置信度的交叉熵
        #   在这其中会忽略一部分样本，这些被忽略的样本满足条件best_iou<ignore_thresh
        #   该操作的目的是：
        #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
        #   不适合当作负样本，所以忽略掉。
        # ------------------------------------------------------------------------------#
        confidence_loss = true_object_mask * K.binary_crossentropy(true_object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - true_object_mask) * K.binary_crossentropy(true_object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask

        class_loss = true_object_mask * K.binary_crossentropy(true_class_index, raw_pred[..., 5:], from_logits=True)
        # -----------------------------------------------------------#
        #   将所有损失求和
        # -----------------------------------------------------------#
        xy_loss = K.sum(xy_loss)
        wh_loss = K.sum(wh_loss)
        confidence_loss = K.sum(confidence_loss)
        class_loss = K.sum(class_loss)
        # -----------------------------------------------------------#
        # -----------------------------------------------------------#
        #   计算正样本数量
        # -----------------------------------------------------------#
        num_pos += tf.maximum(K.sum(K.cast(true_object_mask, tf.float32)), 1)
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        #
        if print_loss:
            loss = tf.compat.v1.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='loss: ')
    if normalize:
        loss = loss / num_pos
    else:
        loss = loss / K.cast(m, K.dtype(yolo_outputs[0]))##转为float类型
    return loss






















