import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util
import glob, os
from utils import visualization_utils as vis_util
import time
from moviepy.editor import VideoFileClip, ImageSequenceClip
import cv2

MODEL_NAME='./bosch3'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './bosch_label_map.pbtxt'

NUM_CLASSES = 13

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# TEST_IMAGE_PATHS = glob.glob('/home/faraz/Code/bosch-to-tfrecord/rgb/test/*.png')
TEST_IMAGE_PATHS = sorted(glob.glob('/tmp/loop/camera/*.png'))
# Size, in inches, of the output images.
IMAGE_SIZE = (48, 32)
frames = []
i=300
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS[i:]:
      start = time.time()
      image = Image.open(image_path)
      image_np = cv2.imread(image_path)
      image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
      # image = image.resize((1200,900))
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      diff1 = time.time() - start
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      diff2 = time.time() - start
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4,
          min_score_thresh=0.4)
      # plt.figure(figsize=IMAGE_SIZE)
      # plt.imshow(image_np)
      frames.append(image_np)
      diff3 = time.time() - start
      print('Saved {} image, diff1: {} diff2: {}, diff3: {}'.format(i, diff1, diff2, diff3))
      i+=1
      if i > 500:
        break
      # plt.show()

clip = ImageSequenceClip(frames, fps=20)
clip.write_videofile('output.mp4')

