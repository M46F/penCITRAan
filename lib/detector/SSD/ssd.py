'''
Script to test traffic light localization and detection
'''

import numpy as np
import tensorflow as tf
from PIL import Image
#import os
from matplotlib import pyplot as plt
import time
from glob import glob
#cwd = os.path.dirname(os.path.realpath(__file__))

# Uncomment the following two lines if need to use the visualization_tunitls
#os.chdir(cwd+'/models')
#from object_detection.utils import visualization_utils as vis_util

class Detector(object):
    def __init__(self, path_to_ckpt):

        self.car_boxes = []
        
        #os.chdir(cwd)
        
        #Tensorflow localization/detection model
        # Single-shot-dectection with mobile net architecture trained on COCO
        # dataset
        PATH_TO_CKPT = path_to_ckpt
        
        # setup tensorflow graph
        self.detection_graph = tf.Graph()
        
        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize 
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')
               
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')
    
    # Helper function to convert image into numpy array    
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)       
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
    
        height, width = dim[0], dim[1]
        box_pixel = [float(box[0]*height), float(box[1]*width), float(box[2]*height), float(box[3]*width)]
        return np.array(box_pixel)       
        
    def get_localization(self, image, visual=False):  
        """Determines the locations of the traffic light in the image

        Args:
            image: camera image

        Returns:
            list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]

        """
        category_index={1: {'id': 1, 'name': u'person'},
                        2: {'id': 2, 'name': u'bicycle'},
                        3: {'id': 3, 'name': u'car'},
                        4: {'id': 4, 'name': u'motorbike'},
                        5: {'id': 5, 'name': u'airplane'},
                        6: {'id': 6, 'name': u'bus'},
                        7: {'id': 7, 'name': u'train'},
                        8: {'id': 8, 'name': u'truck'},
                        9: {'id': 9, 'name': u'boat'},
                        10: {'id': 10, 'name': u'traffic light'},
                        11: {'id': 11, 'name': u'fire hydrant'},
                        13: {'id': 13, 'name': u'stop sign'},
                        14: {'id': 14, 'name': u'parking meter'}}  
        
        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})
          
              if visual == True:
                  vis_util.visualize_boxes_and_labels_on_image_array(
                      image,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,min_score_thresh=.4,
                      line_thickness=3)
    
                  plt.figure(figsize=(9,6))
                  plt.imshow(image)
                  plt.show()  
              
              boxes=np.squeeze(boxes)
              classes =np.squeeze(classes)
              scores = np.squeeze(scores)
    
              cls = classes.tolist()
              
              # The ID for car is 3 
              idx_vec = [i for i, v in enumerate(cls) if (((v==3) or (v==4) or (v==2) or (v==6)) and (scores[i]>=0.25))]
              tmp_boxes={category_index[3]['name']:{'boxes':[], 'scores':[]}, category_index[4]['name']:{'boxes':[], 'scores':[]}, category_index[2]['name']:{'boxes':[], 'scores':[]}, category_index[6]['name']:{'boxes':[], 'scores':[]}}
              if len(idx_vec) ==0:
                  print('no detection!')
              else:
#                  tmp_boxes={category_index[3]['name']:{'boxes':[], 'scores':[]}, category_index[4]['name']:{'boxes':[], 'scores':[]}, category_index[2]['name']:{'boxes':[], 'scores':[]}, category_index[6]['name']:{'boxes':[], 'scores':[]}}
                  for idx in idx_vec:
                      dim = image.shape[0:2]
                      box = self.box_normal_to_pixel(boxes[idx], dim)
                      box_h = box[2] - box[0]
                      box_w = box[3] - box[1]
                      ratio = box_h/(box_w + 0.01)
                      
                      if (ratio < 0.99999):
                          tmp_boxes[category_index[cls[idx]]['name']]['boxes'].append([int(box[0]),int(box[1]),int(box[2]),int(box[3]) ])
                          tmp_boxes[category_index[cls[idx]]['name']]['scores'].append(scores[idx])
                          print(box, ', confidence: ', scores[idx], 'ratio:', ratio)
                          
                          
                  
#                  self.car_boxes = tmp_car_boxes
             
        return tmp_boxes
