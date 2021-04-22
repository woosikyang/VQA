import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable


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


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        print('{}_epoch_starts'.format(epoch))

        for i, (v, b, q, a,o) in enumerate(train_loader):
            v = v.cuda()
            b = b.cuda()
            q = q.long().cuda()
            a = a.cuda()
            o = o.cuda()
            pred = model(v, b, q, a,o)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data * v.size(0)
            train_score += batch_score
            if i % 5000 == 0 :
                print('{}_iteration_done'.format(i))
        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
            print('{}epoch_best_eveal_score: {}'.format(epoch, eval_score))


def train2(model, train_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        print('{}_epoch_starts'.format(epoch))

        for i, (v, b, q, a,o) in enumerate(train_loader):
            v = v.cuda()
            b = b.cuda()
            q = q.long().cuda()
            a = a.cuda()
            o = o.cuda()
            pred = model(v, b, q, a,o)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data * v.size(0)
            train_score += batch_score
            if i % 5000 == 0 :
                print('{}_iteration_done'.format(i))
        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        model_path = os.path.join(output, 'model.pth')
        torch.save(model.state_dict(), model_path)


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    with torch.no_grad() :
        for v, b, q, a,o in iter(dataloader):
            v = v.cuda()
            b = b.cuda()
            q = q.long().cuda()
            o = o.cuda()
            pred = model(v, b, q, None,o)
            batch_score = compute_score_with_logits(pred, a.cuda()).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound


def train_all(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        print('{}_epoch_starts'.format(epoch))

        for i, (v, b, q, a,o,qt) in enumerate(train_loader):
            v = v.cuda()
            b = b.cuda()
            q = q.long().cuda()
            a = a.cuda()
            o = o.cuda()
            qt = qt.cuda()
            pred = model(v, b, q, a,o,qt)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data * v.size(0)
            train_score += batch_score
            if i % 5000 == 0 :
                print('{}_iteration_done'.format(i))
        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate_all(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
            print('{}epoch_best_eveal_score: {}'.format(epoch, eval_score))

def train_all2(model, train_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        print('{}_epoch_starts'.format(epoch))

        for i, (v, b, q, a,o,qt) in enumerate(train_loader):
            v = v.cuda()
            b = b.cuda()
            q = q.long().cuda()
            a = a.cuda()
            o = o.cuda()
            qt = qt.cuda()
            pred = model(v, b, q, a,o,qt)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data * v.size(0)
            train_score += batch_score
            if i % 5000 == 0 :
                print('{}_iteration_done'.format(i))
        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        model_path = os.path.join(output, 'model.pth')
        torch.save(model.state_dict(), model_path)


def evaluate_all(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    with torch.no_grad() :
        for v, b, q, a,o,qt in iter(dataloader):
            v = v.cuda()
            b = b.cuda()
            q = q.long().cuda()
            o = o.cuda()
            qt = qt.cuda()
            pred = model(v, b, q, None,o,qt)
            batch_score = compute_score_with_logits(pred, a.cuda()).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound


def test(model, dataloader):
    with torch.no_grad() :
        for v, b, q, a,o in iter(dataloader):
            v = v.cuda()
            b = b.cuda()
            q = q.long().cuda()
            o = o.cuda()
            pred = model(v, b, q, None,o)

    return pred