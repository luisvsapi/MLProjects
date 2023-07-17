import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from seaborn import color_palette
import cv2

_batch_norm_DECAY = 0.9
_batch_norm_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]

class Yolo_v3(tf.Module):

    def __init__(self, n_classes, model_size, max_output_size, iou_threshold, confidence_threshold, data_format = None):

        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'
        
        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_treshold = iou_threshold
        self.confidence_treshold = confidence_threshold
        self.data_format = data_format
    
    """ Realiza una normalizacion de las entrada """
    def batch_norm(self, inputs, training, data_format):
        return tf.keras.layers.BatchNormalization(axis = 1 if data_format == 'channels_first' else 3,
            momentum = _batch_norm_DECAY, epsilon = _batch_norm_EPSILON)(inputs)
    
    """ Realiza un padding a su entrada """
    def fixed_padding(self, inputs, kernel_size, data_format):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [
                [0,0],[0,0],[pad_beg, pad_end],[pad_beg, pad_end]
            ])
        else:
            padded_inputs = tf.pad(inputs, [
                [0,0],[pad_beg,pad_end],[pad_beg, pad_end],[0,0]
            ])
        
        return padded_inputs

    """ Empacar varias capas 2D"""
    def conv2d_fixed_padding(self, inputs, filters, kernel_size, data_format, strides = 1):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format)
        
        return tf.keras.layers.Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = ('SAME' if strides == 1 else 'VALID'),
            use_bias = False,
            data_format = data_format
        )(inputs)

    """ El bloque residual de darknet53 el clasificador"""
    def darknet53_residual_block(self, inputs, filters, training, data_format, strides = 1):
        shorcut = inputs

        inputs = self.conv2d_fixed_padding(
            inputs, filters = filters, kernel_size = 1, strides = strides, data_format = data_format
        )
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        inputs = self.conv2d_fixed_padding(
            inputs, filters = 2 * filters, kernel_size = 3, strides = strides, data_format = data_format
        )
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        inputs += shorcut

        return inputs

    """ Estructura de la red neuronal Darknet53 para la extraccion de caracteristicas"""
    def darknet53(self, inputs, training, data_format):
        inputs = self.conv2d_fixed_padding(inputs, filters = 32, kernel_size = 3, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        inputs = self.conv2d_fixed_padding(inputs, filters = 64, kernel_size = 3, data_format = data_format, strides = 2)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        inputs = self.darknet53_residual_block(inputs, filters = 32, training = training, data_format = data_format)

        inputs = self.conv2d_fixed_padding(inputs, filters = 128, kernel_size = 3, strides = 2, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        for _ in range(2):
            inputs = self.darknet53_residual_block(inputs, filters = 64, training = training, data_format = data_format)
        
        inputs = self.conv2d_fixed_padding(inputs, filters = 256, kernel_size = 3, strides = 2, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        for _ in range(8):
            inputs = self.darknet53_residual_block(inputs, filters = 128, training = training, data_format = data_format)
        
        route1 = inputs

        inputs = self.conv2d_fixed_padding(inputs, filters = 512, kernel_size = 3, strides = 2, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        for _ in range(8):
            inputs = self.darknet53_residual_block(inputs, filters = 256, training = training, data_format = data_format)
        
        route2 = inputs

        inputs = self.conv2d_fixed_padding(inputs, filters = 1024, kernel_size = 3, strides = 2, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        for _ in range(4):
            inputs = self.darknet53_residual_block(inputs, filters = 512, training = training, data_format = data_format)
        
        return route1, route2, inputs

    """ Bloque convolucional propio de YOLO """
    def yolo_convolution_block(self, inputs, filters, training, data_format):
        inputs = self.conv2d_fixed_padding(inputs, filters = filters, kernel_size = 1, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        inputs = self.conv2d_fixed_padding(inputs, filters = 2 * filters, kernel_size = 3, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        inputs = self.conv2d_fixed_padding(inputs, filters = filters, kernel_size = 1, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        inputs = self.conv2d_fixed_padding(inputs, filters = 2 * filters, kernel_size = 3, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        inputs = self.conv2d_fixed_padding(inputs, filters = filters, kernel_size = 1, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        route = inputs

        inputs = self.conv2d_fixed_padding(inputs, filters = 2 * filters, kernel_size = 3, data_format = data_format)
        inputs = self.batch_norm(inputs, training = training, data_format = data_format)
        inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

        return route, inputs

    """ Capa de YOLO """
    def yolo_layer(self, inputs, n_classes, anchors, img_size, data_format):
        n_anchors = len(anchors)

        inputs = tf.keras.layers.Conv2D(filters = n_anchors * (5 + n_classes), kernel_size = 1, strides = 1, use_bias = True, data_format = data_format)(inputs)
        shape = inputs.get_shape().as_list()
        grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]

        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0,2,3,1])
        
        inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1], 5 + n_classes])

        strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

        box_centers, box_shapes, confidence, classes = tf.split(inputs, [2, 2, 1, n_classes], axis = -1)

        x = tf.range(grid_shape[0], dtype = tf.float32)
        y = tf.range(grid_shape[1], dtype = tf.float32)
        x_offset, y_offset = tf.meshgrid(x,y)
        x_offset = tf.reshape(x_offset, (-1, 1))
        y_offset = tf.reshape(y_offset, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis = -1)
        x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
        x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
        box_centers = tf.nn.sigmoid(box_centers)
        box_centers = (box_centers + x_y_offset) * strides

        anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
        box_shapes = tf.exp(box_shapes) * tf.cast(anchors, tf.float32)

        confidence = tf.nn.sigmoid(confidence)

        classes = tf.nn.sigmoid(classes)

        inputs = tf.concat([box_centers, box_shapes, confidence, classes], axis = -1)

        return inputs

    """ Upsample to 'out_shape'  """
    def upsample(self, inputs, out_shape, data_format):
        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
            new_height = out_shape[3]
            new_width = out_shape[2]
        else:
            new_height = out_shape[2]
            new_width = out_shape[1]
        
        inputs = tf.image.resize(inputs, (new_height, new_width), method = 'nearest')

        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        
        return inputs

    """ Construccion de cajas """
    def build_boxes(self, inputs):
        center_x, center_y, width, height, confidence, classes = tf.split(inputs, [1, 1, 1, 1, 1, -1], axis = -1)

        top_left_x = center_x - width / 2
        top_left_y = center_y - height / 2
        bottom_right_x = center_x + width / 2
        bottom_right_y = center_y + height / 2 

        boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, classes], axis = -1)

        return boxes

    """ Eliminacion de cajas de poca probabilidad """
    def non_max_suppression(self, inputs, n_classes, max_output_size, iou_treshold, confidence_treshold):
        batch = tf.unstack(inputs)
        boxes_dicts = []

        for boxes in batch:
            boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_treshold)
            classes = tf.argmax(boxes[: , 5:], axis = -1)
            classes = tf.expand_dims(tf.cast(classes, dtype = tf.float32), axis = -1)
            boxes = tf.concat([boxes[: , :5], classes], axis = -1)

            boxes_dict = dict()
            for cls in range(n_classes):
                mask = tf.equal(boxes[:, 5], cls)
                mask_shape = mask.get_shape()
                if mask_shape.ndims != 0:
                    class_boxes = tf.boolean_mask(boxes, mask)
                    boxes_coords, boxes_conf_scores, _ = tf.split(
                        class_boxes, [4, 1, -1], axis = -1
                    )
                    boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                    indices = tf.image.non_max_suppression(
                        boxes_coords,boxes_conf_scores, max_output_size, iou_treshold
                    )

                    class_boxes = tf.gather(class_boxes, indices)
                    boxes_dict[cls] = class_boxes[:, :5]
            
            boxes_dicts.append(boxes_dict)

        return boxes_dicts

    """ Presentacion de cajas """
    def draw_boxes(self, img_names, boxes_dicts, class_names, model_size):
        colors = ((np.array(color_palette("hls",80)) * 255)).astype(np.uint8)

        for num, img_name, boxes_dict in zip(range(len(img_names)), img_names, boxes_dicts):
            img = Image.open(img_name)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font = '../input/futur.ttf', size = (img.size[0] + img.size[1]) // 100)

            resize_factor = (img.size[0] / model_size[0], img.size[1] / model_size[1])

            for cls in range(len(class_names)):
                boxes = boxes_dict[cls]
                if np.size(boxes) != 0:
                    color = colors[cls]
                    for box in boxes:
                        xy, confidence = box[:4], box[4]
                        xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                        x0, y0 = xy[0], xy[1]
                        thickness = (img.size[0] + img.size[1]) // 200
                        for t in np.linspace(0, 1, thickness):
                            xy[0], xy[1] = xy[0] + t, xy[1] + t
                            xy[2], xy[3] = xy[2] - t, xy[3] - t
                            draw.rectangle(xy, outline = tuple(color))
                        text = '{} {:.1f}%'.format(class_names[cls], confidence * 100)
                        text_size = draw.textsize(text, font = font)
                        draw.rectangle(
                            [x0, y0 - text_size[1], x0 + text_size[0], y0],
                            fill = tuple(color)
                        )
                        draw.text((x0, y0 - text_size[1]), text, fill = 'black', font = font)
            
            #Mostrar
    
    def __call__(self, inputs, training):
        with tf.compat.v1.variable_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])
            
            inputs = inputs / 255

            route1, route2, inputs = self.darknet53(inputs, training = training, data_format = self.data_format)

            route, inputs = self.yolo_convolution_block(
                inputs, filters = 5121, training = training, data_format = self.data_format, 
            )

            detect1 = self.yolo_layer(
                inputs, n_classes = self.n_classes, anchors = _ANCHORS[6:9], img_size = self.model_size,
                data_format = self.data_format
            )

            inputs = self.conv2d_fixed_padding(route, filters = 256, kernel_size = 1, data_format = self.data_format)
            inputs = self.batch_norm(inputs, training = training, data_format = self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha = _LEAKY_RELU)

            upsample_size = route2.get_shape().as_list()
            inputs = self.upsample(inputs, out_shape = upsample_size, data_format = self.data_format)
            
            axis = 1 if self.data_format == 'channels_first' else 3

            inputs = tf.concat([inputs, route2], axis = axis)
            route, inputs = self.yolo_convolution_block(
                inputs, filters = 256, training = training, data_format = self.data_format
            )

            detect2 = self.yolo_layer(
                inputs, n_classes = self.n_classes, anchors = _ANCHORS[3:6],img_size = self.model_size,
                data_format = self.data_format
            )

            inputs = self.conv2d_fixed_padding(route, filters=128, kernel_size=1,
                                          data_format=self.data_format)
            inputs = self.batch_norm(inputs, training = training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha =_LEAKY_RELU)

            upsample_size = route1.get_shape().as_list()
            inputs = self.upsample(inputs, out_shape = upsample_size, data_format = self.data_format)
            inputs = tf.concat([inputs, route1], axis = axis)

            route, inputs = self.yolo_convolution_block(
                inputs, filters = 128, training = training, data_format = self.data_format
            )
            detect3 = self.yolo_layer(
                inputs, n_classes = self.n_classes, anchors = _ANCHORS[0:3], img_size = self.model_size, data_format = self.data_format
            )

            inputs = tf.concat([detect1, detect2, detect3], axis = 1)

            inputs = self.build_boxes(inputs)

            boxes_dicts = self.non_max_suppression(
                inputs, n_classes = self.n_classes,
                max_output_size = self.max_output_size,
                iou_treshold = self.iou_treshold,
                confidence_treshold = self.confidence_treshold
            )

            return boxes_dicts