# bosch-to-tfrecord

Converts the bosch traffic light dataset (https://hci.iwr.uni-heidelberg.de/node/6132) to TFRecords for use with the tensorflow object detection API

Make sure your PYTHONPATH is set correctly and fix the paths for the train/test yaml files in the main function.


Currently expects a faster RCNN saved model present in fasterrcnn/ (you can get the model from http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz)

Feel free to adjust the TEST_IMAGE_PATHS variable to point to test images.

currently its going to write a moviepy movie. If you just want to test intermediate image generation, search for the 'plot' variable and mark it True
