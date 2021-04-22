import os
import json
import pickle
from config import *
vg_data_path = '/home/woosik/PycharmProjects/Datasets/visual_genome'

image_data = json.load(open(os.path.join(vg_data_path,'image_data.json')))
# 108077
# dict_keys(['width', 'url', 'height', 'image_id', 'coco_id', 'flickr_id'])

image_data[5]['coco_id']
coco_img = [v for i, v in enumerate(image_data) if v['coco_id'] != None]
len(coco_img)
coco_img[0]
region_desc = json.load(open(os.path.join(vg_data_path,'region_descriptions.json')))
# 108077
# dict_keys(['regions', 'id'])
# different # of regions for each id
# region -> dict_keys(['region_id', 'width', 'height', 'image_id', 'phrase', 'y', 'x'])
q_a = json.load(open(os.path.join(vg_data_path,'question_answers.json')))
# 108077
# dict_keys(['id', 'qas'])
# different # of qas for each id
# dict_keys(['a_objects', 'question', 'image_id', 'qa_id', 'answer', 'q_objects'])

coco_train_id = pickle.load(open(os.path.join(saved_data_path,'train_ids.pkl'),'rb'))
coco_val_id = pickle.load(open(os.path.join(saved_data_path,'val_ids.pkl'),'rb'))
coco_test_id = pickle.load(open(os.path.join(saved_data_path,'test_ids.pkl'),'rb'))
len(coco_train_id)
list(coco_val_id).index(coco_img[1]['coco_id'])