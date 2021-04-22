import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset4, VQAFeatureDataset_Relation
#import base_model
import final_base_model
#from train import train
from final_train import train2, train_all2
import os
import utils
from config import *

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

#args = parse_args()

#torch.manual_seed(args.seed)
torch.manual_seed(1111)

#torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed(1111)

torch.backends.cudnn.benchmark = True

dictionary = Dictionary.load_from_file(os.path.join(saved_data_path,'dictionary.pkl'))
train_dset = VQAFeatureDataset4('train', dictionary, saved_data_path)

len(train_dset)
train_dset[0][1].shape



#eval_dset = VQAFeatureDataset3('val', dictionary)
#test_dset = VQAFeatureDataset('test', dictionary)

#batch_size = args.batch_size
batch_size = 512

#constructor = 'build_%s' % args.model

#constructor = 'build_%s' % 'baseline0_newatt'
constructor = 'build_%s' % 'baseline0_both_guided_newatt'

#model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
model = getattr(final_base_model, constructor)(train_dset, 1024).cuda()
#model = getattr(final_base_model, constructor)(eval_dset, 1024).cuda()


model.w_emb.init_embedding(os.path.join(saved_data_path,'glove6b_init_300d.npy'))
device = torch.device('cuda:0')
model.to(device)
#model = nn.DataParallel(model).cuda()

train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
#eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=0)
#train(model, train_loader, eval_loader, args.epochs, args.output)

#train(model, train_loader, eval_loader, 30, 'saved_models/exp0609')
#train2(model, train_loader, 30, 'saved_models/exp0609_q')
train_all2(model, train_loader, 30, 'saved_models/exp0609_both')


# For trial
optim = torch.optim.Adamax(model.parameters())
best_eval_score = 0

for i, (v, b, q, a,o,qt) in enumerate(train_loader):
    if i == 1 :
        break
    v = v.cuda()
    b = b.cuda()
    q = q.long().cuda()
    a = a.cuda()
    o = o.cuda()
    qt = qt.cuda()
    pred = model(v, b, q, a, o,qt)
pred.shape
pred2 =pred[:2]
pred2.size()
v.shape
b.shape
q.shape
a.shape
a2 = a[:2]
a2.shape
torch.argmax((a2))
a2[5]

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores
pred2.shape
tmp = instance_bce_with_logits(pred2.to('cpu'), a2.to('cpu'))
tmp.size()
tmp.shape
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
loss = torch.nn.functional.binary_cross_entropy_with_logits(input, target)
loss.backward()
len(tmp[0])
tmp
from language_model import QuestionEmbedding, WordEmbedding, QuestionEmbedding2
num_hid = 1024
w_emb = WordEmbedding(dictionary.ntoken, 300, 0.0)
q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
q_emb2 = QuestionEmbedding2(300, num_hid, 1, True, 0.0)
w_emb = w_emb(q)
q_emb = q_emb(w_emb)  # [batch, q_dim]
q_emb2 = q_emb2(w_emb)
q_emb.shape
q.shape
q_emb2.shape
type(q_emb2)