import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from seaborn import color_palette

from YOLOModel import Yolo_v3

#https://www.kaggle.com/code/aruchomu/yolo-v3-object-detection-in-tensorflow

_MODEL_INPUT_SIZE = (416, 416)


""" Cargar imagenes usando PIL """
def load_images(model_size, img_names, img_path = "./"):
    imgs = []

    for img_name in img_names:
        img = Image.open(img_path + img_name)
        img = img.resize(size = model_size)
        img = np.array(img, dtype = np.float32)
        img = np.expand_dims(img, axis = 0)
        imgs.append(img)
    
    imgs = np.concatenate(imgs)

    return imgs

""" Cargar las clases de objetos """
def load_class_names(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    
    return class_names

""" Cargar los pesos """
def load_weights(variables, file_name):
    """Reshapes and loads official pretrained Yolo weights.

    Args:
        variables: A list of tf.Variable to be assigned.
        file_name: A name of a file containing weights.

    Returns:
        A list of assign operations.
    """
    with open(file_name, "rb") as f:
        # Skip first 5 values containing irrelevant info
        np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

        assign_ops = []
        ptr = 0

        # Load weights for Darknet part.
        # Each convolution layer has batch normalization.
        for i in range(52):
            conv_var = variables[5 * i]
            gamma, beta, mean, variance = variables[5 * i + 1:5 * i + 5]
            batch_norm_vars = [beta, gamma, mean, variance]

            for var in batch_norm_vars:
                shape = var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(shape)
                ptr += num_params
                assign_ops.append(tf.assign(var, var_weights))

            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

        # Loading weights for Yolo part.
        # 7th, 15th and 23rd convolution layer has biases and no batch norm.
        ranges = [range(0, 6), range(6, 13), range(13, 20)]
        unnormalized = [6, 13, 20]
        for j in range(3):
            for i in ranges[j]:
                current = 52 * 5 + 5 * i + j * 2
                conv_var = variables[current]
                gamma, beta, mean, variance = variables[current + 1:current + 5]
                batch_norm_vars = [beta, gamma, mean, variance]

                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights))

                shape = conv_var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(
                    (shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.assign(conv_var, var_weights))

            bias = variables[52 * 5 + unnormalized[j] * 5 + j * 2 + 1]
            shape = bias.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            assign_ops.append(tf.assign(bias, var_weights))

            conv_var = variables[52 * 5 + unnormalized[j] * 5 + j * 2]
            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

    return assign_ops

""" EntryPoint """
img_names = [
    "testImage.jpg"
]
batch_size = len(img_names)
batch = load_images(model_size = _MODEL_INPUT_SIZE, img_names = img_names, img_path = "./datasetLocal/")
class_names = load_class_names('./classnames/ms_coco_classnames.txt')
n_classes = len(class_names)
max_output_size = 10
iou_threshold = 0.5
confidence_threshold = 0.5

model = Yolo_v3(n_classes = n_classes, model_size = _MODEL_INPUT_SIZE,
                max_output_size = max_output_size, iou_threshold = iou_threshold,
                confidence_threshold = confidence_threshold)

inputs = tf.keras.Input(shape = (416, 416, 3), batch_size = batch_size, dtype = tf.float32)

detections = model(inputs, training=False)
"""
model_vars = tf.global_variables(scope='yolo_v3_model')
assign_ops = load_weights(model_vars, '../input/yolov3.weights')

with tf.Session() as sess:
    sess.run(assign_ops)
    detection_result = sess.run(detections, feed_dict={inputs: batch})
    
Yolo_v3.draw_boxes(img_names, detection_result, class_names, _MODEL_SIZE)"""