# -*- coding:utf-8 -*-
from __future__ import print_function
import time
import codecs
from src.model.crf import *
from src.model.lstm_crf import *
import src.model.utils as utils
from src.model.evaluator import eval_w

import argparse
import numpy as np
import os
import sys
import functools


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with BLSTM-CRF')
    parser.add_argument('--rand_char_embedding', type=bool, default=True, help='random initialize char embedding')
    parser.add_argument('--rand_word_embedding', type=bool, default=True, help='random initialize word embedding')
    parser.add_argument('--rand_bichar_embedding', type=bool, default=True, help='random initialize bichar embedding')
    parser.add_argument('--emb_file', default='/data/disk1/zhangwenjing/chinesegigawordv5/chinesegigawordv5.mws.structured_skipngram.50d.txt', help='path to pre-trained embedding')
    parser.add_argument('--train_file', default='../data/mws_dict/multigrain.alltrain.hwc.BIES.all.txt', help='train file path')
    parser.add_argument('--dev_file', default='../data/mws_dict/multigrain.alldev.hwc.BIES.txt', help='dev file path')
    parser.add_argument('--test_file', default='../data/mws_dict/mannual-test-1500.BIES.txt', help='test file path')
    parser.add_argument('--lexicon_train_dir', default='../data/mws_dict/mws.train.dict.lexicon', help='train lexicon file path')
    parser.add_argument('--lexicon_dev_dir', default='../data/mws_dict/mws.dev.dict.lexicon', help='dev lexicon file path')
    parser.add_argument('--lexicon_test_dir', default='../data/mws_dict/mws.test.dict.lexicon', help='test lexicon file path')

    parser.add_argument('--gpu', type=int, default=0, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--unk', default='unk', help='unknow-token in pre-trained embedding')
    parser.add_argument('--checkpoint', default='../checkpoint/', help='path to checkpoint prefix')
    parser.add_argument('--hidden', type=int, default=200, help='hidden dimension')
    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=200, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch idx')
    parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--embedding_dim', type=int, default=50, help='dimension for word embedding')
    parser.add_argument('--layers', type=int, default=1, help='number of lstm layers')
    parser.add_argument('--lr', type=float, default=0.015, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='decay ratio of learning rate')
    parser.add_argument('--fine_tune', action='store_true', help='fine tune pre-trained embedding dictionary')
    parser.add_argument('--load_check_point', default='', help='path of checkpoint')
    parser.add_argument('--load_opt', action='store_true', help='load optimizer from ')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='sgd', help='optimizer method')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='grad clip at')
    parser.add_argument('--small_crf', action='store_false', help='use small crf instead of large crf, refer model.crf module for more details')
    parser.add_argument('--mini_count', type=float, default=1, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
    parser.add_argument('--least_iters', type=int, default=50, help='at least train how many epochs before stop')
    parser.add_argument('--shrink_embedding', type=bool, default=True, help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    parser.add_argument('--bichar', type=bool, default=False, help='use bichar or not')
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)


    print('setting:')
    print(args)

    # load corpus
    print('loading corpus')
    with codecs.open(args.train_file, 'r', 'utf-8') as f:
        lines = f.readlines()
    with codecs.open(args.dev_file, 'r', 'utf-8') as f:
        dev_lines = f.readlines()


    # converting format
    dev_features, dev_labels, dev_bichar_features = utils.read_corpus(dev_lines)
    with codecs.open(args.test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()
    test_features, test_labels, test_bichar_features = utils.read_corpus(test_lines)

    with codecs.open(args.lexicon_train_dir, 'r', 'utf-8') as f:
        lexicon_train_lines = f.readlines()
    with codecs.open(args.lexicon_dev_dir, 'r', 'utf-8') as f:
        lexicon_dev_lines = f.readlines()

    # converting format
    lexicon_f_map = dict()
    lexicon_dev_features, lexicon_feature_map = utils.read_corpus_lexicon(lexicon_dev_lines, dev_features, lexicon_f_map)
    with codecs.open(args.lexicon_test_dir, 'r', 'utf-8') as f:
        lexicon_test_lines = f.readlines()
    lexicon_test_features, lexicon_feature_map = utils.read_corpus_lexicon(lexicon_test_lines, test_features, lexicon_f_map)


    if args.load_check_point:
        if os.path.isfile(args.load_check_point):
            print("loading checkpoint: '{}'".format(args.load_check_point))
            checkpoint_file = torch.load(args.load_check_point)
            args.start_epoch = checkpoint_file['epoch']
            f_map = checkpoint_file['f_map']
            l_map = checkpoint_file['l_map']
            lexicon_f_map = checkpoint_file['lexicon_f_map']
            train_features, train_labels = utils.read_corpus(lines)
            lexicon_train_features, lexicon_feature_map = utils.read_corpus_lexicon(lexicon_train_lines, train_features,
                                                                                  lexicon_feature_map)
        else:
            print("no checkpoint found at: '{}'".format(args.load_check_point))
    else:
        print('constructing coding table')

        # converting format

        train_features, train_labels, f_map, l_map, train_bichar_features, bichar_f_map = utils.generate_corpus(lines, if_shrink_feature=False, thresholds=0)

        f_set = {v for v in f_map}
        f_map = utils.shrink_features(f_map, train_features, args.mini_count)

        bichar_f_set = {v for v in bichar_f_map}
        bichar_f_map = utils.shrink_features(bichar_f_map, train_bichar_features, args.mini_count)

        lexicon_train_features, lexicon_f_map = utils.generous_corpus_lexicon(lexicon_train_lines, train_features, lexicon_feature_map)
        lexicon_f_set = {v for v in lexicon_f_map}

        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_features), f_set)
        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_labels))

        bichar_dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_bichar_features), bichar_f_set)

        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features), dt_f_set)
        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_labels), l_set)

        bichar_dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_bichar_features), bichar_dt_f_set)

        for label in l_set:
            if label not in l_map:
                if '<start>' in l_map.keys():
                    l_map.pop('<start>')
                if '<pad>' in l_map.keys():
                    l_map.pop('<pad>')
                l_map[label] = len(l_map)
                l_map['<start>'] = len(l_map)
                l_map['<pad>'] = len(l_map)

        if not args.rand_char_embedding:
            print("feature size: '{}'".format(len(f_map)))
            print('loading embedding')
            if args.fine_tune:  # which means does not do fine-tune
                f_map = {'<eof>': 0}
            f_map, embedding_tensor, in_doc_words = utils.load_embedding_wlm(args.emb_file, ' ', f_map, dt_f_set, args.caseless,args.unk, args.embedding_dim, shrink_to_corpus=args.shrink_embedding)
            print("embedding size: '{}'".format(len(f_map)))

        if not args.rand_word_embedding:
            print("feature size: '{}'".format(len(lexicon_f_map)))
            print('loading embedding')
            if args.fine_tune:  # which means does not do fine-tune
                lexicon_f_map = {'<eof>': 0}
            lexicon_f_map, word_embedding_tensor, word_in_doc_words = utils.load_embedding_wlm(args.emb_file, ' ', lexicon_f_map, lexicon_f_set, args.caseless, args.unk, args.embedding_dim, shrink_to_corpus=True)
            print("embedding size: '{}'".format(len(lexicon_f_map)))

        if not args.rand_bichar_embedding:
            print("feature size: '{}'".format(len(bichar_f_map)))
            print('loading embedding')
            if args.fine_tune:  # which means does not do fine-tune
                bichar_f_map = {'<eof>': 0}
            bichar_f_map, bichar_embedding_tensor, bichar_in_doc_words = utils.load_embedding_wlm(args.emb_file, ' ', bichar_f_map, bichar_dt_f_set, args.caseless,args.unk, args.embedding_dim, shrink_to_corpus=True)
            print("embedding size: '{}'".format(len(bichar_f_map)))

    # construct dataset
    lexicon_train_dataset = utils.padding_lexicon_bucket(lexicon_train_features, lexicon_f_map, args.gpu)
    lexicon_dev_dataset = utils.padding_lexicon_bucket(lexicon_dev_features, lexicon_f_map, args.gpu)

    dataset = utils.construct_bucket_mean_vb(train_features, train_labels, lexicon_train_dataset, f_map, l_map, train_bichar_features, bichar_f_map, args.caseless)
    dev_dataset = utils.construct_bucket_mean_vb(dev_features, dev_labels, lexicon_dev_dataset, f_map, l_map, dev_bichar_features, bichar_f_map,args.caseless)
    lexicon_test_dataset = utils.padding_lexicon_bucket(lexicon_test_features, lexicon_f_map, args.gpu)
    test_dataset = utils.construct_bucket_mean_vb(test_features, test_labels, lexicon_test_dataset, f_map, l_map, test_bichar_features, bichar_f_map, args.caseless)

    # build model
    print('building model')
    ner_model = LSTM_CRF(len(f_map), len(bichar_f_map), len(lexicon_f_map), len(l_map), args.embedding_dim, args.hidden, args.layers, args.drop_out, args.gpu, args.bichar, large_CRF=args.small_crf)

    if args.load_check_point:
            ner_model.load_state_dict(checkpoint_file['state_dict'])
    else:
        pre_train = []
        if not args.rand_char_embedding:
            pre_train.append(embedding_tensor)
        if not args.rand_word_embedding:
            pre_train.append(word_embedding_tensor)
        if not args.rand_bichar_embedding:
            pre_train.append(bichar_embedding_tensor)
        if len(pre_train) > 0 :
            ner_model.load_pretrained_embedding(pre_train)
        print('random initialization')
        ner_model.rand_init(init_char_embedding=args.rand_char_embedding, init_word_embedding=args.rand_word_embedding, init_bichar_embedding=args.rand_bichar_embedding)

    if args.update == 'sgd':
        optimizer = optim.SGD(ner_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.update == 'adam':
        optimizer = optim.Adam(ner_model.parameters(), lr=args.lr)

    if args.load_check_point and args.load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    crit = CRFLoss_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

    if args.gpu >= 0:
        if_cuda = True
        print('device: ' + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        crit.cuda()
        ner_model.cuda()
        packer = CRFRepack(len(l_map), True)
    else:
        if_cuda = False
        packer = CRFRepack(len(l_map), False)

    tot_length = sum(map(lambda t: len(t), dataset[0]))
    best_f1 = float('-inf')
    best_acc = float('-inf')
    track_list = list()
    start_time = time.time()
    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)
    patience_count = 0

    evaluator = eval_w(packer, l_map, args.eva_matrix)
    r_l_map = {v: k for k, v in l_map.items()}
    matching_tag = utils.create_matching_labels_matrix(r_l_map).tolist()
    illegal_idx = []
    for i in range(len(matching_tag)):
        for j in range(len(matching_tag[i])):
            if matching_tag[i][j] == 0:
                illegal_idx.append(str(i) + '_' + str(j))
    for epoch_idx, args.start_epoch in enumerate(epoch_list):

        epoch_loss = 0
        ner_model.train()
        shuffle_index = np.random.permutation(len(dataset[0]))

        for i in shuffle_index:

            fea_v, tg_v, mask_v, bi_fea_v = packer.repack_vb(np.asarray(dataset[0][i]), np.asarray(dataset[1][i]), np.asarray(dataset[2][i]), np.asarray(dataset[4][i]))
            ner_model.zero_grad()
            scores, hidden = ner_model.forward(fea_v, bi_fea_v, dataset[3][i], illegal_idx, l_map, args.bichar)
            loss = crit.forward(scores, tg_v, mask_v)
            loss.backward()
            nn.utils.clip_grad_norm(ner_model.parameters(), args.clip_grad)
            optimizer.step()
            epoch_loss += utils.to_scalar(loss)

        # update lr
        utils.adjust_learning_rate(optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

        # average
        epoch_loss /= tot_length

        # eval & save check_point
        dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(ner_model, dev_dataset, illegal_idx, args.bichar)
        if dev_f1 > best_f1:
            patience_count = 0
            best_f1 = dev_f1

            test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(ner_model, test_dataset, illegal_idx, args.bichar)
            track_list.append(
                {'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc, 'dev_pre': dev_pre, 'dev_rec': dev_rec, 'test_f1': test_f1,
                 'test_pre': test_pre, 'test_rec': test_rec, 'test_acc': test_acc})
            print(
                '(loss: %.4f, epoch: %d, dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f, dev acc = %.4f, F1 on test = %.4f, pre on test= %.4f, rec on test= %.4f, acc on test= %.4f), saving...' %
                (epoch_loss,
                 args.start_epoch,
                 dev_f1,
                 dev_pre,
                 dev_rec,
                 dev_acc,
                 test_f1,
                 test_pre,
                 test_rec,
                 test_acc))
            try:
                utils.save_checkpoint({
                    'epoch': args.start_epoch,
                    'state_dict': ner_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'f_map': f_map,
                    'lexicon_f_map': lexicon_f_map,
                    'bichar_f_map': bichar_f_map,
                    'l_map': l_map,
                    'bichar': args.bichar,
                }, {'track_list': track_list,
                    'args': vars(args)
                    }, args.checkpoint + 'lattice_word_seg')
            except Exception as inst:
                print(inst)
        else:
            patience_count += 1
            print('(loss: %.4f, epoch: %d, dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f, dev acc = %.4f)' %
                  (epoch_loss,
                   args.start_epoch,
                   dev_f1,
                   dev_pre,
                   dev_rec,
                   dev_acc))
            track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc, 'dev_pre': dev_pre, 'dev_rec': dev_rec})

        print('epoch: ' + str(args.start_epoch) + '\t in ' + str(args.epoch) + ' take: ' + str(
            time.time() - start_time) + ' s')

        if patience_count >= args.patience and args.start_epoch >= args.least_iters:
            break

    # print best

    eprint(args.checkpoint + ' dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (dev_f1, dev_rec, dev_pre, dev_acc, test_f1, test_rec, test_pre, test_acc))

    # printing summary
    print('setting:')
    print(args)
