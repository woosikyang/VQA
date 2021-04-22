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
    #args = parse_args()

    #torch.manual_seed(args.seed)
    torch.manual_seed(1111)

    #torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed(1111)

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    test_dset = VQAFeatureDataset('test', dictionary)

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
