from __future__ import print_function
import os
import json
import pickle as cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset


class Dictionary_test(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


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

def _create_entry3(img, question, question_type):
    #answer.pop('image_id')
    #answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'question_type' : question_type[question['question']]}
    return entry

def _create_entry4(img, question, objects,question_type):
    #answer.pop('image_id')
    #answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'object' : list(objects),
        'question_type' : question_type[question['question']]}
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



def _load_dataset3(dataroot, name, img_id2val,question_type):
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
    #objects_path = os.path.join(dataroot,'%s_object.pkl' % name)
    #objects = cPickle.load(open(objects_path,'rb'))
    #utils.assert_eq(len(questions), len(answers))
    entries = []
    #answer = None
    for question  in questions:
        img_id = question['image_id']
        entries.append(_create_entry3(img_id2val[img_id], question, question_type))

    return entries



def _load_dataset4(dataroot, name, img_id2val,question_type):
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
        entries.append(_create_entry4(img_id2val[img_id], question, objects[img_id],question_type))

    return entries



class VQAFeatureDatasettest(Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDatasettest, self).__init__()
        assert name in ['train', 'val', 'test']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name), 'rb'))
        # cPickle.load(
        # open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))

        self.entries = _load_dataset_test(dataroot, name, self.img_id2idx)

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

    def tokenize2(self, max_length=14):
        question_dict = {}
        for entry in self.entries :
            tmp = entry['question_id']
            tmp2 = entry['q_token']
            question_dict[tmp] = tmp2

        return question_dict

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            #answer = entry['answer']
            #labels = np.array(answer['labels'])
            #scores = np.array(answer['scores'], dtype=np.float32)
            #if len(labels):
            #    labels = torch.from_numpy(labels)
            #    scores = torch.from_numpy(scores)
            #    entry['answer']['labels'] = labels
            #    entry['answer']['scores'] = scores

            #entry['answer']['labels'] = None
            #entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]

        question = entry['q_token']
        #answer = entry['answer']
        #labels = answer['labels']
        #scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        #if labels is not None:
        #    labels = labels.long()
        #    target.scatter_(0, labels, scores)

        return features, spatials, question, target

    def __len__(self):
        return len(self.entries)



class VQAFeatureDatasettest2(Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDatasettest2, self).__init__()
        assert name in ['train', 'val', 'test']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        #self.question_type = cPickle.load(open('data/total_question_type.pkl','rb'))

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
        # cPickle.load(
        # open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))

        #objects_path = os.path.join(dataroot, '%s_object.pkl' % name)
        #self.objects = cPickle.load(open(objects_path, 'rb'))
        self.entries = _load_dataset2(dataroot, name, self.img_id2idx)

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

            #answer = entry['answer']
            #labels = np.array(answer['labels'])
            #scores = np.array(answer['scores'], dtype=np.float32)
            #if len(labels):
            #    labels = torch.from_numpy(labels)
            #    scores = torch.from_numpy(scores)
            #    entry['answer']['labels'] = labels
            #    entry['answer']['scores'] = scores
            #else:
            #    entry['answer']['labels'] = None
            #    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]

        question = entry['q_token']
        #answer = entry['answer']
        #labels = answer['labels']
        #scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        #if labels is not None:
        #    labels = labels.long()
        #    target.scatter_(0, labels, scores)
        objects = entry['object']
        objects_oh = torch.zeros(len(self.CATEGORIES))
        tmp_oh = [self.CATEGORIES.index(v)for i, v in enumerate(objects)]
        for oh_ in tmp_oh :
            objects_oh[oh_] = 1

        return features, spatials, question, target, objects_oh

    def __len__(self):
        return len(self.entries)



class VQAFeatureDatasettest3(Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDatasettest3, self).__init__()
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
        # cPickle.load(
        # open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))

        #objects_path = os.path.join(dataroot, '%s_object.pkl' % name)
        #self.objects = cPickle.load(open(objects_path, 'rb'))
        self.entries = _load_dataset3(dataroot, name, self.img_id2idx,self.question_type)

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

            #answer = entry['answer']
            #labels = np.array(answer['labels'])
            #scores = np.array(answer['scores'], dtype=np.float32)
            #if len(labels):
            #    labels = torch.from_numpy(labels)
            #    scores = torch.from_numpy(scores)
            #    entry['answer']['labels'] = labels
            #    entry['answer']['scores'] = scores
            #else:
            #    entry['answer']['labels'] = None
            #    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]

        question = entry['q_token']
        #answer = entry['answer']
        #labels = answer['labels']
        #scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        #if labels is not None:
        #    labels = labels.long()
        #    target.scatter_(0, labels, scores)
        question_type = entry['question_type']
        qt_oh = torch.zeros(3)
        qt_oh[question_type] = 1
        return features, spatials, question, target,  qt_oh

    def __len__(self):
        return len(self.entries)


class VQAFeatureDatasettest4(Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDatasettest4, self).__init__()
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

            #answer = entry['answer']
            #labels = np.array(answer['labels'])
            #scores = np.array(answer['scores'], dtype=np.float32)
            #if len(labels):
            #    labels = torch.from_numpy(labels)
            #    scores = torch.from_numpy(scores)
            #    entry['answer']['labels'] = labels
            #    entry['answer']['scores'] = scores
            #else:
            #    entry['answer']['labels'] = None
            #    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]

        question = entry['q_token']
        #answer = entry['answer']
        #labels = answer['labels']
        #scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        #if labels is not None:
        #    labels = labels.long()
        #    target.scatter_(0, labels, scores)
        objects = entry['object']
        objects_oh = torch.zeros(len(self.CATEGORIES))
        tmp_oh = [self.CATEGORIES.index(v)for i, v in enumerate(objects)]
        for oh_ in tmp_oh :
            objects_oh[oh_] = 1
        question_type = entry['question_type']
        qt_oh = torch.zeros(3)
        qt_oh[question_type] = 1
        return features, spatials, question, target, objects_oh, qt_oh

    def __len__(self):
        return len(self.entries)
