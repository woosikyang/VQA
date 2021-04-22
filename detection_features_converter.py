"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import base64
import csv
import h5py
import pickle as cPickle
import numpy as np
import utils

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infile = 'C:\\Users\\yang\\Desktop\\VQA_Example\\v2_data\\trainval_resnet101_faster_rcnn_genome_36.tsv'
train_data_file = 'data/train36.hdf5'
val_data_file = 'data/val36.hdf5'
train_indices_file = 'data/train36_imgid2idx.pkl'
val_indices_file = 'data/val36_imgid2idx.pkl'
train_ids_file = 'data/train_ids.pkl'
val_ids_file = 'data/val_ids.pkl'

feature_length = 2048
num_fixed_boxes = 36
train_path = os.path.join(config.data_path,'train2014')
val_path = os.path.join(config.data_path,'val2014')


if __name__ == '__main__':
    h_train = h5py.File(train_data_file, "w")
    h_val = h5py.File(val_data_file, "w")

    if os.path.exists(train_ids_file) and os.path.exists(val_ids_file):
        train_imgids = cPickle.load(open(train_ids_file))
        val_imgids = cPickle.load(open(val_ids_file))
    else:
        train_imgids = utils.load_imageid(train_path)
        val_imgids = utils.load_imageid(val_path)
        cPickle.dump(train_imgids, open(train_ids_file, 'wb'))
        cPickle.dump(val_imgids, open(val_ids_file, 'wb'))

    train_indices = {}
    val_indices = {}

    train_img_features = h_train.create_dataset(
        'image_features', (len(train_imgids), num_fixed_boxes, feature_length), 'f')
    train_img_bb = h_train.create_dataset(
        'image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
    train_spatial_img_features = h_train.create_dataset(
        'spatial_features', (len(train_imgids), num_fixed_boxes, 6), 'f')

    val_img_bb = h_val.create_dataset(
        'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (len(val_imgids), num_fixed_boxes, feature_length), 'f')
    val_spatial_img_features = h_val.create_dataset(
        'spatial_features', (len(val_imgids), num_fixed_boxes, 6), 'f')

    train_counter = 0
    val_counter = 0

    print("reading tsv...")
    #with open(infile, "r+b") as tsv_in_file:
    with open(infile, "rt",encoding='utf-8') as tsv_in_file:

        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.decodestring(item['boxes'].encode()),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            if image_id in train_imgids:
                train_imgids.remove(image_id)
                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :, :] = bboxes
                train_img_features[train_counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features'].encode()),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                train_spatial_img_features[train_counter, :, :] = spatial_features
                train_counter += 1
            elif image_id in val_imgids:
                val_imgids.remove(image_id)
                val_indices[image_id] = val_counter
                val_img_bb[val_counter, :, :] = bboxes
                val_img_features[val_counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features'].encode()),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                val_spatial_img_features[val_counter, :, :] = spatial_features
                val_counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    if len(train_imgids) != 0:
        print('Warning: train_image_ids is not empty')

    if len(val_imgids) != 0:
        print('Warning: val_image_ids is not empty')

    cPickle.dump(train_indices, open(train_indices_file, 'wb'))
    cPickle.dump(val_indices, open(val_indices_file, 'wb'))
    h_train.close()
    h_val.close()
    print("done!")
#C:\Program Files\JetBrains\PyCharm 2018.3.4\helpers\pydev\pydevconsole.py:50:
#DeprecationWarning: decodestring() is a deprecated alias since Python 3.1,
#use decodebytes() from _pydev_bundle.pydev_umd import runfile, _set_globals_function
