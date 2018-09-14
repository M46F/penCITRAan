import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from keras import Input

from .yad2k.models.keras_yolo import yolo_head
from .yad2k.models.keras_yolo import yolo_boxes_to_corners

import cv2
import numpy as np

import time

LABELS = ['bicycle', 'bus', 'car', 'motorbike']

ANCHORS = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]


class Yolo(object):
    def __init__(self,
                 model_dir=None,
                 custom_objects=None,
                 anchors=ANCHORS,
                 labels=LABELS,
                 yolo_model=None):
        self.sess = K.get_session()
        self.class_names = labels
        self.anchors = np.array(anchors)

        if yolo_model is None and model_dir is None:
            raise ValueError("please specify yolo_model or model_idr")

        if yolo_model is None:
            self.yolo_model = load_model(
                model_dir, custom_objects=custom_objects)

        if model_dir is None:
            self.yolo_model = yolo_model

        self.yolo_outputs = yolo_head(self.yolo_model.output, self.anchors,
                                      len(self.class_names))

    def yolo_eval(self,
                  yolo_outputs,
                  image_shape=(720., 1280.),
                  max_boxes=25,
                  score_threshold=.3,
                  iou_threshold=.4):

        box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

        boxes = yolo_boxes_to_corners(box_xy, box_wh)

        scores, boxes, classes = self.yolo_filter_boxes(
            box_confidence, boxes, box_class_probs, threshold=score_threshold)

        boxes = scale_boxes(boxes, image_shape)

        scores, boxes, classes = self.yolo_non_max_suppression(
            scores, boxes, classes, max_boxes, iou_threshold)

        return scores, boxes, classes

    def yolo_filter_boxes(self,
                          box_confidence,
                          boxes,
                          box_class_probs,
                          threshold=.6):

        box_scores = box_confidence * box_class_probs

        box_classes = K.argmax(box_scores, axis=-1)
        box_class_scores = K.max(box_scores, axis=-1)

        filtering_mask = box_class_scores >= threshold

        scores = tf.boolean_mask(box_class_scores, filtering_mask)
        boxes = tf.boolean_mask(boxes, filtering_mask)
        classes = tf.boolean_mask(box_classes, filtering_mask)

        return scores, boxes, classes

    def yolo_non_max_suppression(self,
                                 scores,
                                 boxes,
                                 classes,
                                 max_boxes=25,
                                 iou_threshold=0.5):

        max_boxes_tensor = K.variable(
            max_boxes, dtype='int32'
        )  # tensor to be used in tf.image.non_max_suppression()
        K.get_session().run(tf.variables_initializer(
            [max_boxes_tensor]))  # initialize variable max_boxes_tensor

        nms_indices = tf.image.non_max_suppression(
            boxes,
            scores,
            max_output_size=max_boxes,
            iou_threshold=iou_threshold)

        scores = K.gather(scores, nms_indices)
        boxes = K.gather(boxes, nms_indices)
        classes = K.gather(classes, nms_indices)

        return scores, boxes, classes

    def predict_single(self, img, score_threshold, special_mode=True):
        if special_mode:
            image_shape, image_data = preprocess_specialy(img)
            print(image_data.shape)
        else:
            image_shape, image_data = preprocess(img)
        scores, boxes, classes = self.yolo_eval(
            self.yolo_outputs,
            image_shape=image_shape,
            score_threshold=score_threshold)

        out_scores, out_boxes, out_classes = self.sess.run(
            [scores, boxes, classes],
            feed_dict={
                self.yolo_model.input: image_data,
                K.learning_phase(): 0
            })
        return out_scores, out_boxes, out_classes

    def get_localization(self,
                         img,
                         result_mapping={
                             0: 'bicycle',
                             1: 'bus',
                             2: 'car',
                             3: 'motorbike'
                         },
                         score_threshold=0.3,
                         class_threshold={
                             'bicycle': 0.9,
                             'bus': 0.3,
                             'car': 0.6,
                             'motorbike': 0.3
                         },
                         special_mode=True):
        out_scores, out_boxes, out_classes = self.predict_single(
            img, score_threshold)
        results = {key:{"boxes":[], "scores":[]} for key in self.class_names}
        for i, predicted_class in enumerate(out_classes):
            if predicted_class in result_mapping:
                if out_scores[i] >= class_threshold[self.class_names[predicted_class]]:
                    results[self.class_names[predicted_class]]["boxes"].append(out_boxes[i])
                    results[self.class_names[predicted_class]]["scores"].append(out_scores[i])
        return results

    def get_batch_localization(
                        self,
                        images,
                        result_mapping={
                            0: 'bicycle',
                            1: 'bus',
                            2: 'car',
                            3: 'motorbike'
                        },
                        score_threshold=0.3,
                        class_threshold={
                            'bicycle': 0.9,
                            'bus': 0.3,
                            'car': 0.6,
                            'motorbike': 0.3
                        },
                        special_mode=True):

        image_shape = (float(images[0].shape[0]), float(images[0].shape[1]))
        image_datas = np.array([_preprocess_helper(img) for img in images])

        start_time = time.time()
        feats = self.sess.run(
            self.yolo_model.output,
            feed_dict={
                self.yolo_model.input: image_datas,
                K.learning_phase(): 0
            })

        feat_input = Input((19, 19, 45))
        scores, boxes, classes = self.yolo_eval(
            yolo_head(feat_input, self.anchors, len(self.class_names)),
            image_shape=image_shape,
            score_threshold=score_threshold
        )
        feat_extract_time = time.time() - start_time

        results = []
        for feat in feats:
            out_scores, out_boxes, out_classes = self.sess.run(
                [scores, boxes, classes],
                feed_dict={
                    feat_input: np.expand_dims(feat, 0),
                    K.learning_phase(): 0
                }
            )

            result = {key: {"boxes": [], "scores": []} for key in self.class_names}
            for i, predicted_class in enumerate(out_classes):
                if predicted_class in result_mapping:
                    if out_scores[i] >= class_threshold[self.class_names[predicted_class]]:
                        result[self.class_names[predicted_class]]["boxes"].append(out_boxes[i])
                        result[self.class_names[predicted_class]]["scores"].append(out_scores[i])
            results.append(result)

        tidying_time = time.time() - start_time
        return results, feat_extract_time, tidying_time


def preprocess(img):
    image_shape = (float(img.shape[0]), float(img.shape[1]))
    image_data = cv2.resize(img, (608, 608), interpolation=cv2.INTER_CUBIC)
    image_data = image_data.astype(np.float32)
    image_data /= 255.0
    image_data = np.expand_dims(image_data, axis=0)
    return image_shape, image_data


def _preprocess_helper(img):
    test_crop_atas = img[0:140, :]
    test_crop_bawah = img[140:img.shape[1], :]
    black = np.zeros((test_crop_atas.shape[0], test_crop_bawah.shape[1], 3), np.uint8)
    vis = np.concatenate((black, test_crop_bawah), axis=0)
    image_data = cv2.resize(vis, (608, 608), interpolation=cv2.INTER_CUBIC)
    image_data = image_data.astype(np.float32)
    image_data /= 255.0
    return image_data


def preprocess_specialy(img):
    image_shape = (float(img.shape[0]), float(img.shape[1]))
    image_data = _preprocess_helper(img)
    image_data = np.expand_dims(image_data, axis=0)
    return image_shape, image_data


def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes
