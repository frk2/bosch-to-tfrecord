import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import yaml
from pathlib import Path

def clamp(value):
  return max(min(value, 1), 0)

def udacity_to_tf_example(img_path, data, force_dir=None):
  if (force_dir):
    image = Path(img_path).name
    img_path = force_dir + '/' + image

  filename = img_path
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()

  encoded_image_data = io.BytesIO(encoded_jpg)

  print("Read in {}, {}".format(img_path, data))
  image_format = b'jpg'

  width = 1920
  height = 1200

  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  classes = []
  classes_text = []
  for boxes in data:
    # mark state
    state = 0
    if (len(boxes) == 7 and boxes[4] == '0'):
      state_txt = boxes[6]
      # if ('Red' in state_txt):
      #   state_txt = 'Red'
      #   state = 1
      # elif ('Yellow' in state_txt):
      #   state_txt = 'Yellow'
      #   state = 2
      # elif ('Green' in state_txt):
      #   state_txt = 'Green'
      #   state = 3

      if (state_txt == 'Red'):
        state = 1
      elif (state_txt == 'Yellow'):
        state = 2
      elif (state_txt == 'Green'):
        state = 3
      else:
        state = -1

      #1-class
      state_txt = 'light'
      state = 1

      # elif (state_txt == 'off'):
      #   state = 4
      # elif (state_txt == 'GreenStraightRight'):
      #   state = 5
      # elif (state_txt == 'GreenStraightLeft'):
      #   state = 6
      # elif (state_txt == 'GreenStraight'):
      #   state = 7
      # elif (state_txt == 'GreenRight'):
      #   state = 8
      # elif (state_txt == 'RedStraightLeft'):
      #   state = 9
      # elif (state_txt == 'RedStraight'):
      #   state = 10
      # elif (state_txt == 'GreenLeft'):
      #   state = 11
      # elif (state_txt == 'RedRight'):
      #   state = 12
      # elif (state_txt == 'RedLeft'):
      #   state = 13

      if (state == 0):
        raise Exception('state cant be zero bro')

      if (state > 0):
        xmins.append(clamp(float(boxes[0]) / width))
        xmaxs.append(clamp(float(boxes[2]) / width))
        ymins.append(clamp(float(boxes[1]) / height))
        ymaxs.append(clamp(float(boxes[3]) / height))

        classes_text.append(state_txt)
        classes.append(state)

  if (len(classes) > 0):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data.getvalue()),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
  return None


def dict_to_tf_example(data, force_dir=None):
  img_path = data['path']
  if (force_dir):
    image = Path(img_path).name
    img_path = force_dir + '/' + image

  filename = img_path
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()

  encoded_image_data = io.BytesIO(encoded_jpg)

  print("Read in {}, {}".format(img_path, encoded_image_data))
  key = hashlib.sha256(encoded_jpg).hexdigest()
  image_format = b'png'

  width = 1280
  height = 720

  current_label = []
  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  classes = []
  classes_text = []
  for boxes in data['boxes']:
    label = []
    # mark state
    state = 0
    state_txt = boxes['label']
    if (boxes['occluded'] == False):
      # if ('Red' in state_txt):
      #   state_txt = 'Red'
      #   state = 1
      # elif ('Yellow' in state_txt):
      #   state_txt = 'Yellow'
      #   state = 2
      # elif ('Green' in state_txt):
      #   state_txt = 'Green'
      #   state = 3


      if (boxes['label'] == 'Red'):
        state = 1
      elif (boxes['label'] == 'Yellow'):
        state = 2
      elif (boxes['label'] == 'Green'):
        state = 3
      elif (boxes['label'] == 'off'):
        state = 4
      elif (boxes['label'] == 'GreenStraightRight'):
        state = 5
      elif (boxes['label'] == 'GreenStraightLeft'):
        state = 6
      elif (boxes['label'] == 'GreenStraight'):
        state = 7
      elif (boxes['label'] == 'GreenRight'):
        state = 8
      elif (boxes['label'] == 'RedStraightLeft'):
        state = 9
      elif (boxes['label'] == 'RedStraight'):
        state = 10
      elif (boxes['label'] == 'GreenLeft'):
        state = 11
      elif (boxes['label'] == 'RedRight'):
        state = 12
      elif (boxes['label'] == 'RedLeft'):
        state = 13

      # 1-class
      state_txt = 'light'
      state = 1

      if (state == 0):
        raise Exception('state cant be zero, bro')

      if (state > 0):
        xmins.append(clamp(float(boxes['x_min']) / width))
        xmaxs.append(clamp(float(boxes['x_max']) / width))
        ymins.append(clamp(float(boxes['y_min']) / height))
        ymaxs.append(clamp(float(boxes['y_max']) / height))

        classes_text.append(state_txt)
        classes.append(state)

  if (len(classes) > 0):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data.getvalue()),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
  return None


def create_tf_record(writer,
                     yaml_file, forced_dir=None, max_len=None):
  
  images = yaml.load(open(yaml_file, 'rb').read())
  max = len(images)
  if (max_len):
    max = max_len
  for i in range(max):
    tf_example = dict_to_tf_example(images[i], forced_dir)
    if (tf_example):
      writer.write(tf_example.SerializeToString())
    else:
      i -= 1

def create_udacity_tf_record(writer, file_path, forced_dir=None):
  curr_file_path = None
  curr_data = []
  with open(file_path) as f:
    content = [x.strip() for x in f.readlines()]
    for line in content:
      line = line.replace('"', '')
      splitlines = line.split()

      if not curr_file_path:
        curr_file_path = splitlines[0]

      if curr_file_path != splitlines[0]:
        #got ourselves a new one, process the last!
        tf_example = udacity_to_tf_example(curr_file_path, curr_data, forced_dir)
        if (tf_example):
          writer.write(tf_example.SerializeToString())
        curr_data = []

      curr_file_path = splitlines[0]
      curr_data.append(splitlines[1:])



if __name__ == '__main__':
  testwriter = tf.python_io.TFRecordWriter('bosch_test_1class.record')
  trainwriter = tf.python_io.TFRecordWriter('bosch_train_1class.record')
  # create_udacity_tf_record(trainwriter, 'udacity/labels_traffic.csv', 'udacity/')
  create_tf_record(trainwriter, 'combined_train.yaml')
  create_tf_record(testwriter, 'test.yaml', 'rgb/test/',1000)
  trainwriter.close()
  testwriter.close()

