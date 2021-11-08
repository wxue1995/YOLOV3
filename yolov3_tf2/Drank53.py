from functools import wraps
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import (Add, Concatenate, Conv2D, UpSampling2D,
                          ZeroPadding2D)
from tensorflow.keras.layers import LeakyReLU,Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from functools import reduce
from tensorflow.keras.utils import plot_model

# 1、functions代表输出参数有N函数， f第一次是取第N-1个函数，g是取第N个函数，
# 2、step1 计算完结果作为第N-2个函数的输入，依次类推。当 f(g(x)) 改为g(f(x))时f,g从第0个函数开始取，相当于倒过来。
def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
#--------------------------------------------------#
#   单次卷积DarknetConv2D
#   如果步长为2则自己设定padding方式。
#   测试中发现没有l2正则化效果更好，所以去掉了l2正则化
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------------------------#
#   残差结构
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后对num_blocks进行循环，循环内部是残差结构。
#---------------------------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)
        x = Add()([x,y])
    return x

#---------------------------------------------------#
#   darknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
def darknet_body(x):
    # 416,416,3 -> 416,416,32
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    # 416,416,32 -> 208,208,64
    x = resblock_body(x, 64, 1)
    # 208,208,64 -> 104,104,128
    x = resblock_body(x, 128, 2)
    # 104,104,128 -> 52,52,256
    x = resblock_body(x, 256, 8)
    feat1 = x
    # 52,52,256 -> 26,26,512
    x = resblock_body(x, 512, 8)
    feat2 = x
    # 26,26,512 -> 13,13,1024
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1,feat2,feat3


def make_last_layers(x, num_filters, out_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)

    # 将最后的通道数调整为outfilter
    y = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    y = DarknetConv2D(out_filters, (1, 1))(y)

    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    # ---------------------------------------------------#
    #   生成darknet53的主干模型
    #   获得三个有效特征层，他们的shape分别是：
    #   52,52,256
    #   26,26,512
    #   13,13,1024
    # ---------------------------------------------------#
    feat1, feat2, feat3 = darknet_body(inputs)

    # ---------------------------------------------------#
    #   第一个特征层
    #   y1=(batch_size,13,13,3,85)
    # ---------------------------------------------------#
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    x, y1 = make_last_layers( feat3, 512, num_anchors * (num_classes + 5))

    # 13,13,512 -> 13,13,256 -> 26,26,256
    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)

    # 26,26,256 + 26,26,512 -> 26,26,768
    x = Concatenate()([x,  feat2])
    # ---------------------------------------------------#
    #   第二个特征层
    #   y2=(batch_size,26,26,3,85)
    # ---------------------------------------------------#
    # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    # 26,26,256 -> 26,26,128 -> 52,52,128
    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    # 52,52,128 + 52,52,256 -> 52,52,384
    x = Concatenate()([x, feat1])
    # ---------------------------------------------------#
    #   第三个特征层
    #   y3=(batch_size,52,52,3,85)
    # ---------------------------------------------------#
    # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))
    return Model(inputs,[y1, y2, y3])


model_inputs = Input(shape=(416,416,3))
model_outputs =  yolo_body(model_inputs,3,20)
model_outputs.summary()
plot_model(model_outputs, to_file='./model.png',show_shapes=True)
print("the num of layers:", len(model_outputs.layers))
print("倒数第二层参数：",model_outputs.variables[-2])