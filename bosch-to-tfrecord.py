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
  return max(min(value, 0.99), 0.01)

def dict_to_tf_example(data, force_dir=None):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
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

    if (state == 0):
      raise Exception('state cant be zero, bro')

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


def create_tf_record(output_filename,
                     yaml_file, forced_dir=None, max_len=None):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  images = yaml.load(open(yaml_file, 'rb').read())
  writer = tf.python_io.TFRecordWriter(output_filename)
  max = len(images)
  if (max_len):
    max = max_len
  for i in range(max):
    tf_example = dict_to_tf_example(images[i], forced_dir)
    if (tf_example):
      writer.write(tf_example.SerializeToString())
    else:
      i -= 1

  writer.close()


# TODO: Add test for pet/PASCAL main files.


if __name__ == '__main__':
  create_tf_record('bosch-train.pb', 'combined_test_train.yaml')
  create_tf_record('bosch-test.pb', 'test.yaml', 'rgb/test/', 500)
