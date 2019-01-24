from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
filename_queue = tf.train.string_input_producer(["train.tfrecords"])
read = tf.TFRecordReader()
_, serialized_example = read.read(filename_queue)
features = tf.parse_single_example(serialized_example,features={ 'label': tf.FixedLenFeature([], tf.int64),'img_raw' : tf.FixedLenFeature([], tf.string),})
img = tf.decode_raw(features['img_raw'], tf.uint8)
img = tf.reshape(img, [280, 280, 3])
label = tf.cast(features['label'], tf.int32)
img_batch, label_batch = tf.train.shuffle_batch([img, label],batch_size= 20, num_threads=64,capacity=2000, min_after_dequeue=1500)
with tf.Session()  as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    i = 0
    # just plot one batch size
    try:
        for k in range(10):
            image, label = sess.run([img_batch, label_batch])
            for j in np.arange(20):
                print('label: %d' % label[j])
                plt.imshow(image[j])
                plt.show()
        i += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
