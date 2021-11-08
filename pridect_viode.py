from yolo_model import yolo_head,transform_images,draw_outputs,get_class
import cv2 ,time
import tensorflow as tf
from tensorflow.keras.layers import Input
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
classes_path = "data/voc_classes.txt"
class_names = get_class(classes_path)

inputs = Input(shape=(416, 416, 3))
yolo = yolo_head(inputs)
model_path = "data/VOC2012.h5"
# model_path = os.path.expanduser(model_path)
assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
yolo.load_weights(model_path)
print('{} model, anchors, and classes loaded.'.format(model_path))
times = []
try:
    # vid = cv2.VideoCapture(int(FLAGS.video))
    vid = cv2.VideoCapture(0)
except:
    vid = cv2.VideoCapture("./data/Mojito.mkv")
while True:
    fps=0.0
    _, img = vid.read()
    if img is None:
        print("Empty Frame")
        time.sleep(2)
        break

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print("img_in \n", img_in.shape)
    img_in = tf.expand_dims(img_in, 0)
    # print("img_in",img_in.shape)
    img_in = transform_images(img_in, 416)
    # print(img_in.shape)

    t1 = time.time()
    boxes, scores, classes= yolo.predict(img_in)
    # print("----------------",boxes.shape,scores.shape, classes.shape,nums.shape)
    t2 = time.time()
    times.append(t2-t1)
    times = times[-20:]

    img = draw_outputs(img, (boxes, scores, classes), class_names)
    img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    fps = (fps + (1. / (t2 - t1)))
    # print("fps= %.2f" % (fps))
    img = cv2.putText(img, "fps= %.2f" % (fps), (0, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

    cv2.imshow('output', img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
