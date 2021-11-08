from yolov3_tf2.Drank53 import yolo_body
import os
from tensorflow.keras.models import load_model
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image, ImageDraw, ImageFont
import colorsys
from tensorflow.keras.models import Model
import cv2

classes_path = "./data/voc_classes.txt"
anchors_path = "./data/VOC2012_anchors.txt"


def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


anchors = get_anchors(anchors_path)
class_names = get_class(classes_path)
num_anchors = len(anchors)
num_classes = len(class_names)


def yolo_head(inputs,max_boxes=20,score_threshold=.7,iou_threshold=.5):

    yolo_model = yolo_body(inputs, num_anchors//3, num_classes)

    num_layers = len(yolo_model.output)
    boxes = []
    box_scores = []
    # -----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
    #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
    #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
    # -----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    for i in range(num_layers):
        lrrbox, box_score = yolo_box(yolo_model.output[i], anchors[anchor_mask[i]], num_classes)
        boxes.append(lrrbox)
        box_scores.append(box_score)
    boxes = tf.concat(boxes,axis=0)
    box_scores = tf.concat(box_scores,axis=0)
    # -----------------------------------------------------------#
    #   判断得分是否大于score_threshold,#可能会产生很多个预选框，
    #   需要经过（1）阈值的删选，（2）非极大值抑制的删选
    # ---------------------------------------------------------
    mask = box_scores >= score_threshold  # 得分大于置信度为True,否则为Flase
    max_boxes_tensor = tf.cast(max_boxes, tf.int32)
    boxes_ = []
    scores_ = []
    classes_ = []

    for c in range(num_classes):
        # -----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成绩，#tf.boolean_mask 的作用是 通过布尔值 过滤元素
        # -----------------------------------------------------------#
        class_boxes = tf.boolean_mask(boxes, mask[:, c])  # 将输入的数组挑出想要的数据输出，将得分大于阈值的坐标挑选出来
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # 将第c类中得分大于阈值的框挑选出来
        # -----------------------------------------------------------#
        #   非极大抑制
        #   保留一定区域内得分最大的框
        """原理：(1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;

                (2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。

                (3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。

                就这样一直重复，找到所有被保留下来的矩形框。"""
        # -----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                 score_threshold=score_threshold, iou_threshold=iou_threshold)
        # -----------------------------------------------------------#
        #   获取非极大抑制后的结果
        #   下列三个分别是
        #   框的位置，得分与种类
        # -----------------------------------------------------------#
        class_boxes = K.gather(class_boxes, nms_index)  # 从第二个参数索引出第一个的值
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c  # 将class_box_scores中的数变成1
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)
    # return 经过非极大值抑制保留下来的一个框
    return  Model(inputs,[boxes_, scores_, classes_])
# def yolo_head(inputs,max_boxes=20,score_threshold=.6,iou_threshold=.5):
#
#     # model_path = "./data/yolo_weights.h5"
#     # model_path = os.path.expanduser(model_path)
#     # assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
#     yolo_model = yolo_body(inputs, num_anchors//3, num_classes)
#     # yolo_model.load_weights(model_path)
#     # print('{} model, anchors, and classes loaded.'.format(model_path))
#     num_layers = len(yolo_model.output)
#     boxes = []
#     box_scores = []
#     # -----------------------------------------------------------#
#     #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
#     #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
#     #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
#     # -----------------------------------------------------------#
#     anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
#     for i in range(num_layers):
#         lrrbox, box_score = yolo_box(yolo_model.output[i], anchors[anchor_mask[i]], num_classes)
#         boxes.append(tf.reshape(lrrbox,(tf.shape(lrrbox)[0],-1,tf.shape(lrrbox)[-1])))
#         box_scores.append(tf.reshape(box_score,(tf.shape(box_score)[0],-1,tf.shape(box_score)[-1])))
#     boxes = tf.concat(boxes,axis=1)
#     box_scores = tf.concat(box_scores,axis=1)
#     # -----------------------------------------------------------#
#     #   判断得分是否大于score_threshold,#可能会产生很多个预选框，
#     #   需要经过（1）阈值的删选，（2）非极大值抑制的删选
#     # ---------------------------------------------------------
#
#     """原理：(1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;
#
#             (2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。
#
#             (3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。
#
#             就这样一直重复，找到所有被保留下来的矩形框。"""
#         # -----------------------------------------------------------#
#     boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#     boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
#     scores=tf.reshape(
#         box_scores, (tf.shape(box_scores)[0], -1, tf.shape(box_scores)[-1])),
#     max_output_size_per_class=max_boxes,
#     max_total_size=max_boxes,
#     iou_threshold=iou_threshold,
#     score_threshold=score_threshold)  # boxes shape=(None, 100, 4), dtype=float32)
#     return Model(inputs,[boxes, scores, classes, valid_detections])

def yolo_box(feat, anchors, num_classes, calc_loss=False):
    batch_size = feat.shape[0]
    grid_size = tf.cast(tf.shape(feat)[1], tf.float32)
    image_size = tf.cast(tf.shape(feat)[1:3] * 32, tf.float32)
    anchors_tensor = tf.reshape(K.constant(anchors), [1, 1, 1, 3, 2])
    feat = tf.reshape(feat, [-1, grid_size, grid_size, 3, 5 + num_classes])
    box_xy, box_wh, objectness, class_probs = tf.split(feat, (2, 2, 1, num_classes), axis=-1)
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_size), [-1, 1, 1, 1]), [1, grid_size, 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_size), [1, -1, 1, 1]), [grid_size, 1, 1, 1])
    grid_xy = K.concatenate([grid_x, grid_y])
    grid_xy = K.cast(grid_xy, K.dtype(feat))
    box_xy = (tf.sigmoid(box_xy) + grid_xy) / grid_size
    box_wh = tf.exp(box_wh) * anchors_tensor / image_size
    pred_cbox = tf.concat([box_xy, box_wh], axis=-1)
    box_min = box_xy - box_wh / 2
    box_max = box_xy + box_wh / 2
    objectness = tf.sigmoid(objectness)  # [batch_size, grid, grid, anchors, obj]
    class_probs = tf.sigmoid(class_probs)  # [batch_size,grid, grid, anchors, classes]
    lrrbox = tf.reshape(tf.concat([box_min, box_max], axis=-1), [-1, 4])
    box_score = tf.reshape(objectness * class_probs, [-1, num_classes])
    if calc_loss == True:
        return grid_xy, feat, box_xy, box_wh
    return lrrbox, box_score
#
# def yolo_box(feat, anchors, num_classes, calc_loss=False):
#     batch_size = feat.shape[0]
#     grid_size = tf.cast(tf.shape(feat)[1], tf.float32)
#     image_size = tf.cast(tf.shape(feat)[1:3] * 32, tf.float32)
#     anchors_tensor = tf.reshape(K.constant(anchors), [1, 1, 1, 3, 2])
#     feat = tf.reshape(feat, [-1, grid_size, grid_size, 3, 5 + num_classes])
#     box_xy, box_wh, objectness, class_probs = tf.split(feat, (2, 2, 1, num_classes), axis=-1)
#     grid_y = K.tile(K.reshape(K.arange(0, stop=grid_size), [-1, 1, 1, 1]), [1, grid_size, 1, 1])
#     grid_x = K.tile(K.reshape(K.arange(0, stop=grid_size), [1, -1, 1, 1]), [grid_size, 1, 1, 1])
#     grid_xy = K.concatenate([grid_x, grid_y])
#     grid_xy = K.cast(grid_xy, K.dtype(feat))
#     box_xy = (tf.sigmoid(box_xy) + grid_xy) / grid_size
#     box_wh = tf.exp(box_wh) * anchors_tensor / image_size
#     pred_cbox = tf.concat([box_xy, box_wh], axis=-1)
#     box_min = box_xy - box_wh / 2
#     box_max = box_xy + box_wh / 2
#     objectness = tf.sigmoid(objectness)  # [batch_size, grid, grid, anchors, obj]
#     class_probs = tf.sigmoid(class_probs)  # [batch_size,grid, grid, anchors, classes]
#     lrrbox = tf.concat([box_min, box_max], axis=-1)
#     box_score = objectness * class_probs
#     if calc_loss == True:
#         return grid_xy, feat, box_xy, box_wh
#     return lrrbox, box_score

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes= outputs
    print(boxes,boxes.shape)
    # boxes, objectness, classes = boxes[0], objectness[0], classes[0]
    wh = np.flip(img.shape[0:2])
    # 画框设置不同的颜色
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
    # 打乱颜色
    np.random.seed(10101)
    np.random.shuffle(colors)
    np.random.seed(None)
    for i,c in enumerate(classes):
        x1y1 = tuple((np.array(boxes[i][0:2]*wh)).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]*wh)).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, colors[c], 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
    return img

def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255.
    return x_train





