import torch
import base_model
import time
import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import train
device = torch.device('cuda:0')
constructor = 'build_%s' % 'baseline0'
model2 = getattr(base_model, constructor)
model3 = (base_model, constructor)
model3
model3.load_state_dict(torch.load('./saved_models/exp1/model.pth'))
model2.to(device)

tmp = torch.load('./saved_models/exp1/model.pth')
type(tmp)
tmp.keys()
len(tmp)


for i, (v, b, q, a) in enumerate(test_loader):
    v = v.cuda()
    b = b.cuda()
    q = q.long().cuda()
    a = a.cuda()
    model =  model(v, b, q, a)
    loss = train.instance_bce_with_logits(pred, a)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optim.step()
    optim.zero_grad()
    batch_score = train.compute_score_with_logits(pred, a.data).sum()
    total_loss += loss.data * v.size(0)
    train_score += batch_score
    if i % 5000 == 0 :
        print('{}_iteration_done'.format(i))
total_loss /= len(train_loader.dataset)
train_score = 100 * train_score / len(train_loader.dataset)
model.train(False)
eval_score, bound = train.evaluate(model, eval_loader)



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


if __name__ == '__main__':
    args = parse_args()

    #torch.manual_seed(args.seed)
    torch.manual_seed(1111)

    #torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed(1111)

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    #batch_size = args.batch_size
    batch_size = 512

    #constructor = 'build_%s' % args.model
    constructor = 'build_%s' % 'baseline0'

    #model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model = getattr(base_model, constructor)(train_dset, 1024).cuda()

    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    device = torch.device('cuda:0')
    model.to(device)
    #model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dset, batch_size, shuffle=True, num_workers=0)
    #train(model, train_loader, eval_loader, args.epochs, args.output)

    train(model, train_loader, eval_loader, 30, 'saved_models/exp1')




def test(model, dataloader):
    with torch.no_grad() :
        for v, b, q, a in iter(dataloader):
            v = v.cuda()
            b = b.cuda()
            q = q.long().cuda()
            pred = model(v, b, q, None).load_sate_dict(torch.load('./saved_models/exp1/model.pth'))

    return pred


import json
import config
import os
import pickle
data_path = os.path.join(config.data_path, 'v2_OpenEnded_mscoco_test2015_questions.json')

tmp = json.load(open(data_path))
tmp.keys()
len(list(tmp.values()))
tmp['questions'][0]
len(tmp['questions'])


tmp2 = pickle.load(open('data/dictionary.pkl','rb'))

tmp3 = pickle.load(open('data/train36_imgid2idx.pkl','rb'))
tmp3[list(tmp3.keys())[4]]


