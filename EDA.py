import os
from glob import glob
import json
import cv2


train_list = os.listdir('data/train2014')
val_list = os.listdir('data/val2014')
test_list = os.listdir('data/test2015')

print(f'train : {len(train_list)}')
print(f'val : {len(val_list)}')
print(f'test : {len(test_list)}')

path_name = {'tr' : 'train2014', 'val' : 'val2014', 'te' : 'test2015'}

with open(f'data/v2_OpenEnded_mscoco_{path_name["tr"]}_questions.json','r') as f :
    tr_question = json.load(f)
tr_question = tr_question['questions']

with open(f'data/v2_OpenEnded_mscoco_{path_name["val"]}_questions.json','r') as f :
    val_question = json.load(f)
val_question = val_question['questions']

with open(f'data/v2_OpenEnded_mscoco_{path_name["te"]}_questions.json','r') as f :
    te_question = json.load(f)
te_question = te_question['questions']

print(f'train question : {len(tr_question)}')
print(f'val question : {len(val_question)}')
print(f'test question : {len(te_question)}')

with open(f'data/v2_mscoco_{path_name["tr"]}_annotations.json','r') as f :
    tr_annotation = json.load(f)
tr_annotation = tr_annotation['annotations']

with open(f'data/v2_mscoco_{path_name["val"]}_annotations.json','r') as f :
    val_annotation = json.load(f)
val_annotation = val_annotation['annotations']

print(f'train annotations : {len(tr_annotation)}')
print(f'val annotations : {len(val_annotation)}')

question_type = [v['question_type'] for v in tr_annotation]
answer_type = [v['answer_type'] for v in tr_annotation]
set(question_type)
set(answer_type)
