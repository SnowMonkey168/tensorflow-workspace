# -*- coding: UTF-8 -*-
import cPickle
import os
import sys
import tarfile

import cv2
import numpy as np
from six.moves import urllib

train_data_dir = 'cifar10_data'
full_train_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_'
valid_data_dir = 'cifar10_data/cifar-10-batches-py/test_batch'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10
TRAIN_RANDOM_LABEL = False
VALID_RANDOM_LABEL = False
NUM_TRAIN_BATCH = 5
EPOCH_SIZE = 10000 * NUM_TRAIN_BATCH


def maybe_download_and_extract():
    dest_directory = train_data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        print('Successfully downloaded ', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _read_one_batch(path, is_random_label):
    fo = open(path, 'rb')
    dicts = cPickle.load(fo)
    fo.close()
    data = dicts['data']
    if is_random_label is False:
        label = np.array(dicts['labels'])
    else:
        labels = np.random.randint(low=0, high=10, size=10000)
        label = np.array(labels)
    return data, label


def read_in_all_image(address_list, shuffle=True, is_random_label=False):
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])
    for address in address_list:
        print 'Reading images from ' + address
        batch_data, batch_label = _read_one_batch(address, is_random_label)
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))
    num_data = len(label)
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    if shuffle is True:
        print('Shuffling')
        # permutation 不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]
    data = data.astype(np.float32)
    return data, label


def horizontal_flip(image, axis):
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)
    return image


def whitening_image(image_np):
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        std = np.max([np.std(image_np[i, ...]), 1.0 / np.sqrt(IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH)])
        image_np[i, ...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):
    cropped_batch = np.zeros(len(batch_data) * IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH) \
        .reshape(len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)
    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset + IMG_HEIGHT, y_offset:y_offset + IMG_WIDTH, :]
        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)
    return cropped_batch


def prepare_train_data(padding_size):
    path_list = []
    for i in range(1, NUM_TRAIN_BATCH + 1):
        path_list.append(full_train_data_dir + str(i))
    data, label = read_in_all_image(path_list, is_random_label=TRAIN_RANDOM_LABEL)
    # 设置边框大小
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    # 给图像数据添加边框
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    return data, label


def read_validation_data():
    validation_array, validation_labels = read_in_all_image([valid_data_dir], is_random_label=VALID_RANDOM_LABEL)
    validation_array = whitening_image(validation_array)
    return validation_array, validation_labels
