import os
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
classPath="H:\\1"
write = tf.python_io.TFRecordWriter("train.tfrecords")
def init64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def byte64_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
j=1
for i in os.listdir(classPath):
    img_abs_path = classPath + "\\" + i
    img = Image.open(img_abs_path)
    img = img.resize((280,280))
    img_raw = img.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": init64_feature(j),
        "img_raw": byte64_feature(img_raw),
    }))
    write.write(example.SerializeToString())
    j+=1
write.close()