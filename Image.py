import os
import pickle as cPickle

dataroot = 'data'
ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
ans2label = cPickle.load(open(ans2label_path, 'rb'))
label2ans = cPickle.load(open(label2ans_path, 'rb'))
num_ans_candidates = len(ans2label)


ans_name = list(ans2label.keys())

import pandas as pd
pd.DataFrame(ans_name).to_csv('answer.csv')


# q_repr 에 attention 추가

"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
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
train_data_file = 'data/train36_2.hdf5'
val_data_file = 'data/val36_2.hdf5'
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
        tmp_data = []
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            tmp_data.append(item)
            if len(tmp_data) == 5 :
                break
        len(tmp_data)

        item = tmp_data[0]
        bboxes = np.frombuffer(
            base64.decodestring(item['boxes'].encode()),
            dtype=np.float32).reshape((int(item['num_boxes']), -1))

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


import pickle

train_object = pickle.load(open('data/train_object.pkl','rb'))
val_object = pickle.load(open('data/val_object.pkl','rb'))
test_object = pickle.load(open('data/test_object.pkl','rb'))

train_object[9]

CATEGORIES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
name = 'train'
dataroot = 'data'

import json
def _create_entry_test(img, question):
    #answer.pop('image_id')
    #answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question']}
    return entry

def _load_dataset_test(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if name == 'train' or name == 'val' :
        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    else :
        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s2015_questions.json' % name)

    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    entries = []
    #answer = None
    for question in (questions):
        img_id = question['image_id']
        entries.append(_create_entry_test(img_id2val[img_id], question))

    return entries


import h5py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp1')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

torch.manual_seed(1111)
#torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed(1111)
torch.backends.cudnn.benchmark = True
dictionary = Dictionary.load_from_file('data/dictionary.pkl')
train_dset = VQAFeatureDataset('train', dictionary)
eval_dset = VQAFeatureDataset('val', dictionary)
test_dset = VQAFeatureDataset('test', dictionary)
test_dset.img_id2idx




#batch_size = args.batch_size
batch_size = 512
#constructor = 'build_%s' % args.model
#constructor = 'build_%s' % 'baseline0_newatt'
constructor = 'build_%s' % 'baseline0_newatt_lstm_bidirection'
#model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
model = getattr(base_model, constructor)(train_dset, 1024).cuda()
model.w_emb.init_embedding('data/glove6b_init_300d.npy')
device = torch.device('cuda:0')
model.to(device)
#model = nn.DataParallel(model).cuda()
train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=0)
#train(model, train_loader, eval_loader, args.epochs, args.output)
train(model, train_loader, eval_loader, 30, 'saved_models/exp4')



def _create_entry2(img, question, objects):
    #answer.pop('image_id')
    #answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'object' : list(objects)}
    return entry




def _load_dataset2(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if name == 'train' or name == 'val' :
        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    else :
        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s2015_questions.json' % name)

    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    #answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    #answers = cPickle.load(open(answer_path, 'rb'))
    #answers = sorted(answers, key=lambda x: x['question_id'])
    objects_path = os.path.join(dataroot,'%s_object.pkl' % name)
    objects = cPickle.load(open(objects_path,'rb'))
    #utils.assert_eq(len(questions), len(answers))
    entries = []
    #answer = None
    for question  in questions:
        img_id = question['image_id']
        entries.append(_create_entry2(img_id2val[img_id], question, objects[img_id]))

    return entries






