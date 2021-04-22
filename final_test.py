import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from dataset import Dictionary, VQAFeatureDataset
from dataset_test import VQAFeatureDatasettest4
import base_model
from train import train
import utils
import config
import json
import pickle
from tqdm import tqdm
import final_base_model


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

    #train
    #train_dset = VQAFeatureDataset('train', dictionary)
    #eval_dset = VQAFeatureDataset('val', dictionary)

    #test
    test_dset = VQAFeatureDatasettest4('test',dictionary)


    batch_size = 512
    #constructor = 'build_%s' % 'baseline0_newatt'
    #0606
    #constructor = 'build_%s' % 'baseline0_newatt_lstm_bidirection'
    constructor = 'build_%s' % 'baseline0_both_guided_newatt'

    #model = getattr(base_model, constructor)(train_dset, 1024).cuda()
    model = getattr(final_base_model, constructor)(test_dset, 1024).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    device = torch.device('cuda:0')
    model.to(device)

    ###
    #if test
    state_dict = torch.load('saved_models/exp0609_both/model.pth')
    #load saved model
    model.load_state_dict(state_dict)

    #train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    #eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dset, batch_size, shuffle=False, num_workers=0)

    ###

    #train
    #train(model, train_loader, eval_loader, 30, 'saved_models/exp2')

    #test
    max_length = 14
    num_ans = 3129
    output = 'result'
    utils.create_dir(output)
    logger = utils.Logger(os.path.join(output, 'log_0609_both.txt'))
    #for i, (v, b, q, a) in enumerate(test_loader):
    #q3 = np.empty([0, max_length])
    pred3 = np.empty([0, num_ans])
    for i, (v, b, q, a,o,qt) in enumerate(tqdm(test_loader)):
        if i == 600 :
            break
        with torch.no_grad() :
            v = v.cuda()
            b = b.cuda()
            q = q.long().cuda()
            o = o.cuda()
            qt = qt.cuda()
            pred = model(v, b, q, None, o, qt)
            #q2 = q.cpu().numpy()
            pred2 = pred.cpu().numpy()
            #q3 = np.vstack([q3,q2])
            pred3 = np.vstack([pred3,pred2])
            #del q2
            del pred2

    np.save('data/test_pred_answer_0609_both_0.npy',pred3)


    pred3 = np.empty([0, num_ans])
    for i, (v, b, q, a, o,qt) in enumerate(tqdm(test_loader)):
        if i >= 600 :

            with torch.no_grad() :
                v = v.cuda()
                b = b.cuda()
                q = q.long().cuda()
                o = o.cuda()
                qt = qt.cuda()
                pred = model(v, b, q, None, o,qt)
                #q2 = q.cpu().numpy()
                pred2 = pred.cpu().numpy()
                #q3 = np.vstack([q3,q2])
                pred3 = np.vstack([pred3,pred2])
                #del q2
                del pred2
    np.save('data/test_pred_answer_0609_both_1.npy', pred3)
    #np.save('data/test_question_0606.npy',q3)

    ##Question id : tokenize result
    #question_dict = test_dset.tokenize2()
    ## save
    #with open('question_dict_0606.pickle', 'wb') as f:
    #    pickle.dump(question_dict, f, pickle.HIGHEST_PROTOCOL)

    import tqdm
    import os
    import numpy as np
    import json
    import pickle
    import config
    from collections import OrderedDict
    from tqdm import tqdm
    import tqdm

    #question_dict = pickle.load(open('question_dict.pickle', 'rb'))
    #test_question_token = np.load('data/test_question.npy')
    test_pred_answer_0 = np.load('data/test_pred_answer_0609_both_0.npy')
    test_pred_answer_1 = np.load('data/test_pred_answer_0609_both_1.npy')

    test_pred_answer = np.concatenate((test_pred_answer_0,test_pred_answer_1),axis=0)

    #test_question_file = 'v2_OpenEnded_mscoco_val2014_questions.json'
    #test_question_file = os.path.join(config.data_path, test_question_file)
    #test_questions = json.load(open(test_question_file))['questions']

    question_dict = pickle.load(open('data/test2015_1st/question_dict.pickle','rb'))
    question_id = list(question_dict.keys())

    answer_label = pickle.load(open(os.path.join(os.getcwd(), 'data', 'cache', 'trainval_label2ans.pkl'), 'rb'))

    result = []
    for i in tqdm.tqdm(range(447793)):
        result_tmp = {}
        result_tmp['answer'] = answer_label[np.argmax(test_pred_answer[i])]
        result_tmp['question_id'] = question_id[i]
        result.append(result_tmp)

    #result_json = json.dumps(result)
    trial = 'base_both'
    with open('data/final_result_{}_0609.json'.format(trial), 'w') as f:
        json.dump(result, f)





