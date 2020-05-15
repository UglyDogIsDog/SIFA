import tensorflow as tf
import json


def _decode_samples(image_list, is_training_value=True):
    decomp_feature = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.FixedLenFeature([], tf.int64),  # 256
        'dsize_dim1': tf.FixedLenFeature([], tf.int64),  # 256
        'dsize_dim2': tf.FixedLenFeature([], tf.int64),  # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.FixedLenFeature([], tf.int64),  # 256
        'lsize_dim1': tf.FixedLenFeature([], tf.int64),  # 256
        'lsize_dim2': tf.FixedLenFeature([], tf.int64),  # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 1]
        'label_vol': tf.FixedLenFeature([], tf.string)}

    raw_size = [256, 256, 3]
    volume_size = [256, 256, 3]
    label_size = [256, 256, 1]

    if is_training_value:
        data_queue = tf.train.string_input_producer(image_list)
    else:
        data_queue = tf.train.string_input_producer(image_list, num_epochs=1)
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(data_queue)
    parser = tf.parse_single_example(
        serialized_example, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, raw_size)
    label_vol = tf.slice(label_vol, [0, 0, 1], label_size)

    batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)

    return tf.expand_dims(data_vol[:, :, 1], axis=2), batch_y


def _load_samples(source_pth, target_pth, is_training_value=True):

    with open(source_pth, 'r') as fp:
        rows = fp.readlines()
    imagea_list = [row[:-1] for row in rows]

    with open(target_pth, 'r') as fp:
        rows = fp.readlines()
    imageb_list = [row[:-1] for row in rows]

    data_vola, label_vola = _decode_samples(imagea_list, is_training_value)
    data_volb, label_volb = _decode_samples(imageb_list, is_training_value)

    return data_vola, data_volb, label_vola, label_volb


def load_data(source_pth, target_pth, batch_size, do_shuffle=True, is_training_value=True):

    image_i, image_j, gt_i, gt_j = _load_samples(
        source_pth, target_pth, is_training_value)

    # For converting the value range to be [-1 1] using the equation 2*[(x-x_min)/(x_max-x_min)]-1.
    # The values {-1.8, 4.4, -2.8, 3.2} need to be changed according to the statistics of specific datasets
    if 'mr' in source_pth:
        image_i = tf.subtract(tf.multiply(
            tf.div(tf.subtract(image_i, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)
    elif 'ct' in source_pth:
        image_i = tf.subtract(tf.multiply(
            tf.div(tf.subtract(image_i, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)

    if 'ct' in target_pth:
        image_j = tf.subtract(tf.multiply(
            tf.div(tf.subtract(image_j, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)
    elif 'mr' in target_pth:
        image_j = tf.subtract(tf.multiply(
            tf.div(tf.subtract(image_j, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)

    # Batch
    if do_shuffle is True:
        images_i, images_j, gt_i, gt_j = tf.train.shuffle_batch(
            [image_i, image_j, gt_i, gt_j], batch_size, 500, 100)
    else:
        images_i, images_j, gt_i, gt_j = tf.train.batch(
            [image_i, image_j, gt_i, gt_j], batch_size=batch_size, num_threads=1, capacity=500, allow_smaller_final_batch=True)

    return images_i, images_j, gt_i, gt_j
