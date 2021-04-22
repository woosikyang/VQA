from config import *
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle as cPickle
from dataset import Dictionary, VQAFeatureDataset4
#import base_model
import final_base_model
#from train import train
from final_train import train2, train_all2

# torch.manual_seed(args.seed)
torch.manual_seed(1111)

# torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed(1111)

torch.backends.cudnn.benchmark = True

dictionary = Dictionary.load_from_file(os.path.join(saved_data_path,'dictionary.pkl'))
train_dset = VQAFeatureDataset4('train', dictionary)


dataroot=saved_data_path
ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
ans2label = cPickle.load(open(ans2label_path, 'rb'))
len(ans2label)
ans2label['net']
class VQAFeatureDataset_Relation(Dataset):
    def __init__(self, name, dictionary, dataroot=saved_data_path):
        super(VQAFeatureDataset4, self).__init__()
        assert name in ['train', 'val', 'test']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.question_type = cPickle.load(open('data/total_question_type.pkl','rb'))

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
        self.CATEGORIES = CATEGORIES
        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name), 'rb'))
        img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % 'train'), 'rb'))
        # cPickle.load(
        # open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))

        #objects_path = os.path.join(dataroot, '%s_object.pkl' % name)
        #self.objects = cPickle.load(open(objects_path, 'rb'))
        self.entries = _load_dataset4(dataroot, name, self.img_id2idx,self.question_type)

        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)
        self.s_dim = self.spatials.size(2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            labels = labels.long()
            target.scatter_(0, labels, scores)
        objects = entry['object']
        objects_oh = torch.zeros(len(self.CATEGORIES))
        tmp_oh = [self.CATEGORIES.index(v)for i , v in enumerate(objects)]
        for oh_ in tmp_oh :
            objects_oh[oh_] = 1

        question_type = entry['question_type']
        qt_oh = torch.zeros(3)
        qt_oh[question_type] = 1

        return features, spatials, question, target , objects_oh,qt_oh

    def __len__(self):
        return len(self.entries)
